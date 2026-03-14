"""
mtl.py - REDESIGNED Multi-Task Learning
========================================
✅ LOCAL + GLOBAL dual-task approach
   - Local: Segment-level predictions (fine-grained)
   - Global: Station-to-station predictions (coarse-grained)

Masking modes (mask_type parameter on DualTaskMTLHead / MTLHead)
----------------------------------------------------------------
  'hard'  (default) — Binary masking.
            Attention : key_padding_mask=~mask forces padding positions to
                        -inf before softmax → zero attention weight.
            Pooling   : padding positions receive weight 0 in the mean.
            Identical to the original HDBSCAN behaviour.

  'soft'  — Learned continuous gating via SoftMaskGate.
            A small linear+sigmoid network produces a per-position weight
            w ∈ (0, 1) for every token.  Weights are initialised near 1.0
            so early training resembles full attention, and the gate learns
            to suppress unimportant or padding positions gradually.
            Attention : input x is scaled by the gate weights before being
                        passed to MultiheadAttention (no key_padding_mask),
                        so gradients can flow through padding positions at a
                        reduced magnitude rather than being blocked entirely.
            Pooling   : soft gate weights are used directly for weighted mean;
                        structurally invalid positions (hard mask = False) are
                        still hard-zeroed so they cannot dominate.
            Use soft masking when you want the model to learn which segments
            are most informative rather than relying on a pre-defined binary
            validity signal.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# SOFT MASK GATE
# =============================================================================

class SoftMaskGate(nn.Module):
    """
    Learns a continuous importance weight in (0, 1) for each sequence position.

    Architecture: Linear(feature_dim → 1) → Sigmoid
    Initialisation: bias = +2.0 so the gate starts near 0.88 (almost fully
    open), matching the behaviour of no masking at epoch 0 and gradually
    learning to suppress unimportant positions.

    Args:
        feature_dim (int): Dimensionality of the input token embeddings.

    Forward:
        x          : [batch, seq_len, feature_dim]
        hard_mask  : [batch, seq_len] bool — positions that are structurally
                     invalid (e.g. sequence padding) are hard-zeroed after the
                     sigmoid so they can never dominate the weighted mean.
    Returns:
        soft_weights : [batch, seq_len, 1]  values in [0, 1)
    """

    def __init__(self, feature_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )
        # Initialise near 1.0: weight ≈ 0, bias = +2 → sigmoid(2) ≈ 0.88
        nn.init.zeros_(self.gate[0].weight)
        nn.init.constant_(self.gate[0].bias, 2.0)

    def forward(self, x: torch.Tensor, hard_mask: torch.Tensor = None) -> torch.Tensor:
        soft_weights = self.gate(x)                          # [B, L, 1]
        if hard_mask is not None:
            # Hard-zero structurally invalid positions
            soft_weights = soft_weights * hard_mask.unsqueeze(-1).float()
        return soft_weights


# =============================================================================
# DUAL-TASK MTL HEAD
# =============================================================================

class DualTaskMTLHead(nn.Module):
    """
    Dual-Task MTL: Local segments + Global station-to-station.

    Args:
        feature_dim  : Dimensionality of input embeddings.
        local_hidden : Hidden size for the local (per-segment) head.
        global_hidden: Hidden size for the global (aggregate) head.
        dropout      : Dropout probability.
        mask_type    : 'hard' (default) or 'soft' — see module docstring.
    """

    def __init__(self, feature_dim, local_hidden=64, global_hidden=128,
                 dropout=0.2, mask_type: str = 'hard'):
        super().__init__()

        assert mask_type in ('hard', 'soft'), \
            f"mask_type must be 'hard' or 'soft', got '{mask_type}'"
        self.mask_type = mask_type

        # Local task: Individual segment prediction
        self.local_head = nn.Sequential(
            nn.Linear(feature_dim, local_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(local_hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

        # Global task: Station-to-station aggregate
        self.global_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )

        self.global_head = nn.Sequential(
            nn.Linear(feature_dim, global_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(global_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        # Soft mask gate (only instantiated when mask_type='soft')
        if mask_type == 'soft':
            self.soft_gate = SoftMaskGate(feature_dim)

        # Task weighting (learnable uncertainty — Kendall et al. 2018)
        self.log_var_local  = nn.Parameter(torch.zeros(1))
        self.log_var_global = nn.Parameter(torch.zeros(1))

        print(f"   DualTaskMTLHead  mask_type='{mask_type}'")

    # ------------------------------------------------------------------
    def forward(self, x, mask=None):
        """
        Args:
            x    : [batch, seq_len, feature_dim]
            mask : [batch, seq_len] bool — True = valid position
        Returns:
            local_preds : [batch, seq_len, 1]
            global_pred : [batch, 1]
        """
        # ── Local task (same for both mask types) ────────────────────────
        local_preds = self.local_head(x)                     # [B, L, 1]

        # ── Global task ───────────────────────────────────────────────────
        if self.mask_type == 'hard':
            # Hard masking: padding positions → -inf in softmax → zero weight
            key_padding_mask = ~mask if mask is not None else None
            attn_out, _ = self.global_attention(
                x, x, x, key_padding_mask=key_padding_mask
            )
            # Weighted mean — padding positions have zero weight
            if mask is not None:
                mask_f = mask.unsqueeze(-1).float()          # [B, L, 1]
                global_repr = (
                    (attn_out * mask_f).sum(dim=1) /
                    mask_f.sum(dim=1).clamp(min=1)
                )
            else:
                global_repr = attn_out.mean(dim=1)

        else:  # soft masking
            # Soft gate: learned continuous weights per position
            soft_w = self.soft_gate(x, mask)                 # [B, L, 1]

            # Scale input by soft weights before attention so the attention
            # mechanism naturally learns to focus on high-weight tokens.
            # No key_padding_mask — gradients flow through padding at reduced
            # magnitude rather than being fully blocked.
            x_gated = x * soft_w
            attn_out, _ = self.global_attention(x_gated, x_gated, x_gated)

            # Soft-weighted mean pooling
            weight_sum = soft_w.sum(dim=1).clamp(min=1e-6)  # [B, 1]
            global_repr = (attn_out * soft_w).sum(dim=1) / weight_sum

        global_pred = self.global_head(global_repr)          # [B, 1]

        return local_preds, global_pred


# =============================================================================
# DUAL-TASK MTL LOSS
# =============================================================================

class DualTaskMTLLoss(nn.Module):
    """
    Loss for dual-task MTL with uncertainty weighting (Kendall et al. 2018).

    The loss always uses the *binary* mask for selecting which segment
    positions contribute to the local loss — independent of whether the
    head uses hard or soft masking.  Padding positions are never penalised.
    """

    def __init__(self, criterion=None):
        super().__init__()
        self.criterion = criterion if criterion is not None else nn.SmoothL1Loss(reduction='none')

    def forward(self, local_preds, global_pred, segment_targets, global_target,
                mask, log_var_local, log_var_global):
        """
        Args:
            local_preds     : [batch, seq_len, 1]
            global_pred     : [batch, 1]
            segment_targets : [batch, seq_len, 1]
            global_target   : [batch, 1]
            mask            : [batch, seq_len] bool — valid segments
            log_var_local   : learnable log-variance for local task
            log_var_global  : learnable log-variance for global task
        """
        # Local loss — only valid (non-padding) positions count
        local_loss = self.criterion(local_preds, segment_targets)
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            local_loss = (
                (local_loss * mask_expanded).sum() /
                mask_expanded.sum().clamp(min=1)
            )
        else:
            local_loss = local_loss.mean()

        # Global loss
        global_loss = self.criterion(global_pred, global_target).mean()

        # Uncertainty weighting
        precision_local  = torch.exp(-log_var_local)
        precision_global = torch.exp(-log_var_global)

        total_loss = (
            precision_local  * local_loss  + log_var_local  +
            precision_global * global_loss + log_var_global
        )

        return total_loss, {
            'total':         total_loss.item(),
            'local':         local_loss.item(),
            'global':        global_loss.item(),
            'weight_local':  precision_local.item(),
            'weight_global': precision_global.item()
        }


# =============================================================================
# LEGACY MTL HEAD  (kept for backward compatibility)
# =============================================================================

class MTLHead(nn.Module):
    """
    Original MTL head (deprecated — use DualTaskMTLHead).

    Args:
        mask_type : 'hard' (default) or 'soft' — see module docstring.
    """

    def __init__(self, feature_dim, segment_hidden=64, path_hidden=128,
                 dropout=0.2, mask_type: str = 'hard'):
        super().__init__()

        assert mask_type in ('hard', 'soft'), \
            f"mask_type must be 'hard' or 'soft', got '{mask_type}'"
        self.mask_type = mask_type

        self.segment_predictor = nn.Sequential(
            nn.Linear(feature_dim, segment_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(segment_hidden, 1)
        )
        self.path_attention = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=4, dropout=dropout, batch_first=True
        )
        self.path_predictor = nn.Sequential(
            nn.Linear(feature_dim, path_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(path_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        if mask_type == 'soft':
            self.soft_gate = SoftMaskGate(feature_dim)

    def forward(self, x, mask=None, return_components=False):
        individual_preds = self.segment_predictor(x)

        if self.mask_type == 'hard':
            attn_out, attn_weights = self.path_attention(
                x, x, x,
                key_padding_mask=~mask if mask is not None else None
            )
            if mask is not None:
                mask_f = mask.unsqueeze(-1).float()
                pooled = (
                    (attn_out * mask_f).sum(dim=1) /
                    mask_f.sum(dim=1).clamp(min=1)
                )
            else:
                pooled = attn_out.mean(dim=1)

        else:  # soft
            soft_w = self.soft_gate(x, mask)                 # [B, L, 1]
            x_gated = x * soft_w
            attn_out, attn_weights = self.path_attention(x_gated, x_gated, x_gated)
            weight_sum = soft_w.sum(dim=1).clamp(min=1e-6)
            pooled = (attn_out * soft_w).sum(dim=1) / weight_sum

        collective_pred = self.path_predictor(pooled)

        if return_components:
            return individual_preds, collective_pred, attn_weights
        return collective_pred


# =============================================================================
# LEGACY MTL LOSS  (kept for backward compatibility)
# =============================================================================

class MTLLoss(nn.Module):
    """Original MTL loss (deprecated)."""

    def __init__(self, lambda_weight=0.5, criterion=None):
        super().__init__()
        self.lambda_weight = lambda_weight
        self.criterion = criterion if criterion is not None else nn.SmoothL1Loss(reduction='none')

    def forward(self, individual_preds, collective_pred, target, mask):
        individual_loss = self.criterion(
            individual_preds.squeeze(-1),
            target.expand_as(individual_preds).squeeze(-1)
        )
        if mask is not None:
            individual_loss = (
                (individual_loss * mask.float()).sum() /
                mask.float().sum().clamp(min=1)
            )
        else:
            individual_loss = individual_loss.mean()

        collective_loss = self.criterion(collective_pred, target).mean()
        total_loss = (
            self.lambda_weight * individual_loss +
            (1 - self.lambda_weight) * collective_loss
        )
        return total_loss, {
            'total':      total_loss.item(),
            'individual': individual_loss.item(),
            'collective': collective_loss.item()
        }