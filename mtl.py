"""
<<<<<<< Updated upstream
mtl.py
======
Multi-Task Learning for MAGNN-LSTM-MTL travel time prediction.

Architecture: MAGNN-LSTM-Seq2Seq

Key upgrades over the original:
  1.  MTLHead        — per-step decoder + path-level aggregation head, both
                       derived from the same LSTM hidden states.
  2.  MTLLoss        — three-task loss that directly attacks error accumulation:
        Task A  individual segment loss  (local accuracy)
        Task B  collective path loss     (global accuracy)
        Task C  sequential consistency   (Σ individual ≈ collective target)
                This penalises the model when its segment predictions do NOT
                sum to the trip total — the root cause of error accumulation.
  3.  Uncertainty weighting (Kendall et al., 2018) — each task has a learnable
      log-variance (log_σ²).  Tasks that are harder to fit automatically
      receive less weight; no manual α tuning needed.

Signature compatibility with model.py
--------------------------------------
  from mtl import MTLHead, MTLLoss         ← same as before
  MTLLoss(lambda_weight=cfg.mtl_lambda,
          criterion=nn.SmoothL1Loss())     ← matches model.py call site

MTLHead.forward() returns:
  individual_preds  [batch, seq_len, 1]   per-segment predictions
  collective_pred   [batch, 1]            path-level prediction
  attention         [batch, seq_len]      attention weights (interpretable)
=======
mtl.py — REDESIGNED Dual-Task MTL
===================================

Architecture
------------
The DualTaskMTL model no longer re-runs MAGNN from scratch.
Instead it wraps a frozen MAGNN_LSTM_Residual and uses its
hidden-state representations as the per-segment encoder.

  LOCAL  task : predict each individual segment duration
                (one output per segment in the trip sequence)
                supervised by each segment's own duration_sec

  GLOBAL task : predict the total trip travel time
                (one output per trip)
                supervised by sum(segment durations) for that trip
                — trip = same trip_id on the same calendar day

Data flow
---------
  TripDataset  →  trip_collate_fn
      │
      ▼  for each segment in the trip (seq_len steps, chronological)
  MAGNN_LSTM_Residual.forward(..., return_components=True)
      │  returns per-segment hidden state h_t  (spatial_emb after gate)
      ▼
  TripSequenceEncoder  (LSTM over the trip sequence of h_t vectors)
      │  output: (B, T, enc_hidden)
      ▼
  LocalHead   → local_preds  (B, T, 1)   per-segment durations
  GlobalHead  ← cross-segment attention + mean pool
              → global_pred  (B, 1)      total trip time

Loss
----
  Uncertainty-weighted (Kendall et al. 2018):
      L = exp(-s_local)*L_local + s_local
        + exp(-s_global)*L_global + s_global
  where s_local, s_global are learned log-variances.
>>>>>>> Stashed changes
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
<<<<<<< Updated upstream
# MTL PREDICTION HEAD
# =============================================================================

class MTLHead(nn.Module):
    """
    Dual-output prediction head attached to the shared LSTM hidden states.

    Individual (per-segment) branch
    ─────────────────────────────────
    A small MLP applied at every time step independently.
    Uses the LSTM hidden state h_t to predict duration of segment t.

    Collective (path-level) branch
    ────────────────────────────────
    Attention-weighted sum over valid hidden states → single trip prediction.
    The attention weights are exposed so the caller can inspect which
    segments dominate the path estimate.

    Parameters
    ──────────
    feature_dim      : size of LSTM hidden state (= lstm_hidden in MAGNN_LSTM_MTL)
    segment_hidden   : hidden units in the per-segment MLP
    path_hidden      : hidden units in the path-level MLP
    dropout          : dropout rate
    """

    def __init__(self, feature_dim: int, segment_hidden: int = 64,
                 path_hidden: int = 128, dropout: float = 0.2):
        super().__init__()

        # ── Individual branch (applied per timestep) ──────────────────────
        self.segment_head = nn.Sequential(
            nn.Linear(feature_dim, segment_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(segment_hidden, 1)
        )

        # ── Attention for collective branch ────────────────────────────────
        # Scalar attention score per hidden state → softmax over valid steps
        self.attn_score = nn.Linear(feature_dim, 1, bias=False)

        # ── Collective branch ──────────────────────────────────────────────
        self.path_head = nn.Sequential(
            nn.Linear(feature_dim, path_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(path_hidden, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, hidden: torch.Tensor,
                mask: torch.Tensor = None,
                return_components: bool = True):
        """
        Args
        ────
        hidden    : [batch, seq_len, feature_dim]   LSTM outputs after attention
        mask      : [batch, seq_len] bool/float      1 = valid, 0 = padding
        return_components : if False, return only collective_pred (inference mode)

        Returns (when return_components=True)
        ──────────────────────────────────────
        individual_preds  : [batch, seq_len, 1]
        collective_pred   : [batch, 1]
        attn_weights      : [batch, seq_len]
        """
        batch, seq_len, _ = hidden.size()

        # ── Individual predictions ─────────────────────────────────────────
        # Apply the same small MLP at every time step
        individual_preds = self.segment_head(hidden)     # [batch, seq_len, 1]

        # ── Attention weights (mask out padding before softmax) ────────────
        scores = self.attn_score(hidden).squeeze(-1)     # [batch, seq_len]

        if mask is not None:
            float_mask = mask.float() if mask.dtype == torch.bool else mask
            # Set padding positions to -inf so they get ~0 attention
            scores = scores.masked_fill(float_mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)         # [batch, seq_len]
=======
# TRIP SEQUENCE ENCODER
# =============================================================================

class TripSequenceEncoder(nn.Module):
    """
    Takes the per-segment representations produced by MAGNN_LSTM_Residual
    and models their sequential dependencies within a trip.

    Input : (B, T, residual_repr_dim)   — one vector per segment
    Output: (B, T, hidden_dim)          — contextualised hidden states
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 n_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,        # causal — we know past, not future
        )
        self.norm = nn.LayerNorm(hidden_dim)
        nn.init.orthogonal_(self.lstm.weight_hh_l0)
        nn.init.xavier_uniform_(self.lstm.weight_ih_l0)

    def forward(self, x: torch.Tensor,
                lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x       : (B, T, input_dim)
        lengths : (B,) actual sequence lengths before padding  [optional]
        returns : (B, T, hidden_dim)
        """
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True,
                                                       total_length=x.size(1))
        else:
            out, _ = self.lstm(x)
        return self.norm(out)


# =============================================================================
# DUAL-TASK HEAD
# =============================================================================

class DualTaskMTLHead(nn.Module):
    """
    LOCAL  → predict each segment's duration independently.
    GLOBAL → cross-segment attention then MLP to predict total trip time.

    Both tasks are supervised; the model learns the correct balance
    via uncertainty-weighted loss (see DualTaskMTLLoss).
    """

    def __init__(self, feature_dim: int,
                 local_hidden: int = 64,
                 global_hidden: int = 128,
                 dropout: float = 0.2):
        super().__init__()

        # Local head — independent per-step regression
        self.local_head = nn.Sequential(
            nn.Linear(feature_dim, local_hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(local_hidden, 32), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

        # Global head — attend over the whole trip then regress
        self.global_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=max(1, feature_dim // 32),   # safe for small dims
            dropout=dropout,
            batch_first=True,
        )
        self.global_head = nn.Sequential(
            nn.Linear(feature_dim, global_hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(global_hidden, 64), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        # Learnable task-uncertainty weights (Kendall et al. 2018)
        self.log_var_local  = nn.Parameter(torch.zeros(1))
        self.log_var_global = nn.Parameter(torch.zeros(1))

        # Xavier init
        for seq in (self.local_head, self.global_head):
            for m in seq:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x    : (B, T, feature_dim)
        mask : (B, T)  True = valid step

        returns
        -------
        local_preds  : (B, T, 1)
        global_pred  : (B, 1)
        """
        # LOCAL — each step independently
        local_preds = self.local_head(x)                        # (B, T, 1)

        # GLOBAL — attend across steps, then pool valid steps only
        key_pad = (~mask) if mask is not None else None         # MHA wants True=ignore
        attn_out, _ = self.global_attention(x, x, x,
                                            key_padding_mask=key_pad)

        if mask is not None:
            m = mask.unsqueeze(-1).float()                      # (B, T, 1)
            global_repr = (attn_out * m).sum(1) / m.sum(1).clamp(min=1)
        else:
            global_repr = attn_out.mean(1)                      # (B, feature_dim)

        global_pred = self.global_head(global_repr)             # (B, 1)
        return local_preds, global_pred


# =============================================================================
# DUAL-TASK LOSS
# =============================================================================

class DualTaskMTLLoss(nn.Module):
    """
    Uncertainty-weighted multi-task loss (Kendall et al. 2018).

      L = precision_local  * L_local  + log_var_local
        + precision_global * L_global + log_var_global

    where precision = exp(-log_var).
    """

    def __init__(self):
        super().__init__()
        self._base = nn.SmoothL1Loss(reduction='none')

    def forward(
        self,
        local_preds:     torch.Tensor,   # (B, T, 1)
        global_pred:     torch.Tensor,   # (B, 1)
        segment_targets: torch.Tensor,   # (B, T, 1)  per-segment duration (scaled)
        global_target:   torch.Tensor,   # (B, 1)     total trip duration (scaled)
        mask:            torch.Tensor,   # (B, T)     True = valid segment
        log_var_local:   torch.Tensor,   # scalar param
        log_var_global:  torch.Tensor,   # scalar param
    ):
        # Local loss — only over valid (non-padded) steps
        local_raw = self._base(local_preds, segment_targets)    # (B, T, 1)
        m = mask.unsqueeze(-1).float()
        local_loss = (local_raw * m).sum() / m.sum().clamp(min=1)

        # Global loss — one scalar per sample
        global_loss = self._base(global_pred, global_target).mean()

        # Uncertainty weighting
        prec_l = torch.exp(-log_var_local)
        prec_g = torch.exp(-log_var_global)
        total  = (prec_l * local_loss  + log_var_local
                + prec_g * global_loss + log_var_global)

        return total, {
            'total':        total.item(),
            'local':        local_loss.item(),
            'global':       global_loss.item(),
            'weight_local': prec_l.item(),
            'weight_global':prec_g.item(),
        }


# =============================================================================
# MAIN MODEL: MAGNN-LSTM-DUALTASK-MTL
# =============================================================================

class MAGNN_LSTM_DualTaskMTL(nn.Module):
    """
    Dual-Task MTL that learns FROM the residual LSTM.

    The residual model is used as a frozen per-segment encoder.
    Its spatial embedding (after the adaptive gate) becomes the
    input representation for the trip-level sequence encoder.

    LOCAL  task: each segment's duration within the trip.
    GLOBAL task: sum of all segment durations = total trip time.

    Parameters
    ----------
    residual_model : MAGNN_LSTM_Residual
        Pre-trained, already on the correct device.
        Will be frozen inside this model.
    spatial_dim : int
        Dimension of MAGNN spatial embeddings (= cfg.gat_hidden, typically 32).
    enc_hidden : int
        Hidden size of TripSequenceEncoder LSTM.
    local_hidden : int
        Hidden size of the local prediction head.
    global_hidden : int
        Hidden size of the global prediction head.
    dropout : float
    """

    def __init__(
        self,
        residual_model,                 # MAGNN_LSTM_Residual instance
        spatial_dim: int,
        enc_hidden:    int = 128,
        local_hidden:  int = 64,
        global_hidden: int = 128,
        dropout:       float = 0.2,
    ):
        super().__init__()

        # Frozen residual encoder — produces per-segment representations
        self.residual_model = residual_model
        for p in self.residual_model.parameters():
            p.requires_grad = False
        print("   MAGNN_LSTM_DualTaskMTL: residual encoder frozen")

        # The residual model exposes spatial_embeddings (spatial_dim,)
        # and alpha-gated correction (1,) — we concatenate them as the
        # per-segment representation fed into the trip encoder.
        # repr_dim = spatial_dim + 1   (spatial emb + alpha-weighted correction)
        repr_dim = spatial_dim + 1

        # Trip-level sequence encoder
        self.trip_encoder = TripSequenceEncoder(
            input_dim=repr_dim,
            hidden_dim=enc_hidden,
            dropout=dropout,
        )

        # Dual-task prediction heads
        self.mtl_head = DualTaskMTLHead(
            feature_dim=enc_hidden,
            local_hidden=local_hidden,
            global_hidden=global_hidden,
            dropout=dropout,
        )

        # Loss
        self.criterion = DualTaskMTLLoss()

        print(f"   TripSequenceEncoder input dim : {repr_dim} "
              f"(spatial={spatial_dim} + α·correction=1)")
        print(f"   TripSequenceEncoder hidden    : {enc_hidden}")

    # ------------------------------------------------------------------

    def _encode_segment(
        self,
        seg_indices:        torch.Tensor,   # (B,)
        temporal_features:  torch.Tensor,   # (B, 5)
        context_flags:      torch.Tensor,   # (B, 2)
        origin_operational: torch.Tensor,   # (B, 2)
        dest_operational:   torch.Tensor,   # (B, 2)
        origin_weather:     torch.Tensor,   # (B, 8)
        dest_weather:       torch.Tensor,   # (B, 8)
    ) -> torch.Tensor:
        """
        Run the frozen residual model for ONE time-step worth of segments.
        Returns the per-segment representation: (B, spatial_dim + 1)
        = [spatial_embedding  |  α * correction]
        """
        with torch.no_grad():
            baseline, correction, alpha, _ = self.residual_model(
                seg_indices,
                temporal_features,
                context_flags=context_flags,
                origin_operational=origin_operational,
                dest_operational=dest_operational,
                origin_weather=origin_weather,
                dest_weather=dest_weather,
                return_components=True,
            )
            # spatial_embeddings are inside residual_model; expose them
            spatial_emb = self.residual_model._get_spatial_embeddings(
                seg_indices)                                     # (B, spatial_dim)
            gated_corr  = alpha * correction                    # (B, 1)

        return torch.cat([spatial_emb, gated_corr], dim=1)      # (B, spatial_dim+1)

    # ------------------------------------------------------------------

    def forward(
        self,
        # All tensors are (B, T, *) — T = trip length (padded)
        seg_indices:        torch.Tensor,   # (B, T)
        temporal_features:  torch.Tensor,   # (B, T, 5)
        context_flags:      torch.Tensor,   # (B, T, 2)
        origin_operational: torch.Tensor,   # (B, T, 2)
        dest_operational:   torch.Tensor,   # (B, T, 2)
        origin_weather:     torch.Tensor,   # (B, T, 8)
        dest_weather:       torch.Tensor,   # (B, T, 8)
        mask:               torch.Tensor,   # (B, T)  True = valid segment
        lengths:            torch.Tensor,   # (B,)    actual trip lengths
        return_local:       bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        local_preds  : (B, T, 1)  — per-segment duration predictions
        global_pred  : (B, 1)    — total trip duration prediction
        """
        B, T = seg_indices.shape

        # ── Step 1: encode every segment with the frozen residual model ──
        segment_reprs = []
        for t in range(T):
            rep = self._encode_segment(
                seg_indices[:, t],
                temporal_features[:, t, :],
                context_flags[:, t, :],
                origin_operational[:, t, :],
                dest_operational[:, t, :],
                origin_weather[:, t, :],
                dest_weather[:, t, :],
            )                                                   # (B, repr_dim)
            segment_reprs.append(rep)

        trip_seq = torch.stack(segment_reprs, dim=1)           # (B, T, repr_dim)

        # ── Step 2: model trip-level temporal dependencies ──────────────
        enc_out = self.trip_encoder(trip_seq, lengths)          # (B, T, enc_hidden)

        # ── Step 3: dual-task predictions ───────────────────────────────
        local_preds, global_pred = self.mtl_head(enc_out, mask) # (B,T,1), (B,1)

        if return_local:
            return local_preds, global_pred
        return global_pred


# =============================================================================
# BACKWARD-COMPATIBLE STUBS (keep old imports working)
# =============================================================================

class MTLHead(nn.Module):
    """Deprecated — use DualTaskMTLHead."""

    def __init__(self, feature_dim, segment_hidden=64,
                 path_hidden=128, dropout=0.2):
        super().__init__()
        self.segment_predictor = nn.Sequential(
            nn.Linear(feature_dim, segment_hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(segment_hidden, 1))
        self.path_attention = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=4,
            dropout=dropout, batch_first=True)
        self.path_predictor = nn.Sequential(
            nn.Linear(feature_dim, path_hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(path_hidden, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x, mask=None, return_components=False):
        individual_preds = self.segment_predictor(x)
        attn_out, attn_w = self.path_attention(
            x, x, x,
            key_padding_mask=~mask if mask is not None else None)
        if mask is not None:
            m = mask.unsqueeze(-1).float()
            pooled = (attn_out * m).sum(1) / m.sum(1).clamp(min=1)
        else:
            pooled = attn_out.mean(1)
        collective = self.path_predictor(pooled)
        if return_components:
            return individual_preds, collective, attn_w
        return collective
>>>>>>> Stashed changes

        # ── Context vector = attention-weighted sum of hidden states ────────
        context = (attn_weights.unsqueeze(-1) * hidden).sum(dim=1)  # [batch, feature_dim]

        # ── Collective prediction from context ─────────────────────────────
        collective_pred = self.path_head(context)        # [batch, 1]

        if not return_components:
            return collective_pred

        return individual_preds, collective_pred, attn_weights


# =============================================================================
# MTL LOSS  (three-task, uncertainty-weighted)
# =============================================================================

class MTLLoss(nn.Module):
<<<<<<< Updated upstream
    """
    Three-task uncertainty-weighted MTL loss.
=======
    """Deprecated — use DualTaskMTLLoss."""
>>>>>>> Stashed changes

    Task A — Individual segment accuracy
             Base criterion (default SmoothL1) on per-step predictions vs
             the *scaled* total target (trip target used as proxy; replace
             with per-segment ground truth if available).

    Task B — Collective path accuracy
             Base criterion on the path-level prediction vs trip target.

    Task C — Sequential consistency  ← NEW, targets error accumulation
             Penalises |Σ(individual_preds) − trip_target| per sample.
             Forces the model to learn segment predictions that *sum* to the
             right total, directly preventing the compounding of per-segment
             errors into a large trip-level error.

    Weighting
    ──────────
    Each task has a learnable log-variance log_σ² (Kendall et al., 2018):

        L_total = Σ_k  [ L_k / (2 σ_k²) + log σ_k ]

    This replaces the hand-tuned α: hard tasks automatically get lower weight
    as the model increases their σ; easy tasks get amplified.

    The `lambda_weight` constructor argument shifts the prior towards
    the original MTL balance:
        lambda_weight = 0.5  →  equal individual / collective weighting
        lambda_weight = 0.3  →  30 % individual, 70 % collective (original)
    It initialises log_σ² values accordingly so training starts in a
    sensible region rather than flat (σ=1 for all tasks).

    Parameters
    ──────────
    lambda_weight : float  initial balance hint (used to set log_σ priors)
    criterion     : nn.Module  base loss function (default SmoothL1Loss)
    consistency_weight : float  weight for the sequential consistency term
                                (relative to the uncertainty-weighted terms)
    """

    def __init__(self,
                 lambda_weight: float = 0.3,
                 criterion: nn.Module = None,
                 consistency_weight: float = 0.5):
        super().__init__()
<<<<<<< Updated upstream

        self.base_criterion = criterion if criterion is not None else nn.SmoothL1Loss()
        self.consistency_weight = consistency_weight

        # ── Learnable log(σ²) for tasks A, B, C ───────────────────────────
        # Initialise from lambda_weight so the starting loss landscape
        # roughly matches the original hand-tuned balance.
        #
        # log σ²_A controls individual weight: higher σ_A → less weight on A
        # We want  λ ∝ 1/σ_A²  and  (1-λ) ∝ 1/σ_B²
        # ⇒  σ_A² = 1/λ,  σ_B² = 1/(1-λ)
        #
        # Clip λ away from 0/1 to avoid log(0)
        lam = max(0.05, min(0.95, lambda_weight))
        import math
        init_A = math.log(1.0 / lam)           # log σ²_A
        init_B = math.log(1.0 / (1.0 - lam))   # log σ²_B
        init_C = math.log(2.0)                  # consistency starts with moderate weight

        self.log_var_A = nn.Parameter(torch.tensor(init_A, dtype=torch.float32))
        self.log_var_B = nn.Parameter(torch.tensor(init_B, dtype=torch.float32))
        self.log_var_C = nn.Parameter(torch.tensor(init_C, dtype=torch.float32))

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _masked_loss(criterion, preds, target, mask):
        """
        Apply `criterion` only on valid (non-padding) positions.

        preds   : [batch, seq_len, 1]
        target  : [batch, 1]   (trip-level; broadcast across seq_len)
        mask    : [batch, seq_len] bool/float
        """
        if mask is None:
            # No sequence dimension — fall back to direct loss
            return criterion(preds.squeeze(-1), target.expand_as(preds.squeeze(-1)))

        float_mask = mask.float() if mask.dtype == torch.bool else mask  # [batch, seq_len]
        n_valid = float_mask.sum().clamp(min=1.0)

        # Expand target to [batch, seq_len, 1] and mask
        target_exp = target.unsqueeze(1).expand_as(preds)   # [batch, seq_len, 1]
        valid_mask3 = float_mask.unsqueeze(-1)               # [batch, seq_len, 1]

        # Compute element-wise loss, zero out padding, average over valid
        elem_loss = F.smooth_l1_loss(preds, target_exp, reduction='none')  # [b, s, 1]
        masked_loss = (elem_loss * valid_mask3).sum() / n_valid
        return masked_loss

    @staticmethod
    def _consistency_loss(individual_preds, collective_target, mask):
        """
        Sequential consistency:  Σ_t individual_preds_t  ≈  collective_target

        This is the KEY loss that fights error accumulation.  Each segment
        prediction must "know its share" of the total trip.

        individual_preds : [batch, seq_len, 1]
        collective_target: [batch, 1]
        mask             : [batch, seq_len]
        """
        float_mask = mask.float() if (mask is not None and mask.dtype == torch.bool) \
                     else (mask.float() if mask is not None else None)

        if float_mask is not None:
            # Zero out padding positions before summing
            valid_mask3 = float_mask.unsqueeze(-1)               # [batch, seq_len, 1]
            seg_sum = (individual_preds * valid_mask3).sum(dim=1) # [batch, 1]
        else:
            seg_sum = individual_preds.sum(dim=1)                 # [batch, 1]

        # L1 consistency: |Σ segments - trip target|
        consistency = F.smooth_l1_loss(seg_sum, collective_target)
        return consistency

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, individual_preds, collective_pred, trip_target, mask=None):
        """
        Args
        ────
        individual_preds : [batch, seq_len, 1]   per-segment predictions
        collective_pred  : [batch, 1]            path-level prediction
        trip_target      : [batch, 1]            ground-truth total trip time (scaled)
        mask             : [batch, seq_len]      1=valid, 0=padding

        Returns
        ───────
        total_loss : scalar tensor (with grad)
        loss_dict  : dict with float values for logging
        """
        # ── Task A: individual segment accuracy ────────────────────────────
        loss_A = self._masked_loss(self.base_criterion,
                                   individual_preds, trip_target, mask)

        # ── Task B: collective path accuracy ──────────────────────────────
        loss_B = self.base_criterion(collective_pred, trip_target)

        # ── Task C: sequential consistency ────────────────────────────────
        loss_C = self._consistency_loss(individual_preds, trip_target, mask)

        # ── Uncertainty weighting  (Kendall et al. 2018) ──────────────────
        # L_total = Σ_k [ L_k * exp(-log_var_k) + log_var_k ]
        # exp(-log_var_k) = 1/σ_k² ;  log_var_k = log(σ_k²) = log σ_k²
        #
        # We use log_var (= log σ²) as the learnable parameter.
        # The "+ log_var" regularisation prevents σ → ∞ (all tasks ignored).

        w_A = torch.exp(-self.log_var_A)
        w_B = torch.exp(-self.log_var_B)
        w_C = torch.exp(-self.log_var_C)

        total_loss = (
            w_A * loss_A + self.log_var_A +
            w_B * loss_B + self.log_var_B +
            self.consistency_weight * (w_C * loss_C + self.log_var_C)
        )

        loss_dict = {
            'total':       total_loss.item(),
            'individual':  loss_A.item(),
            'collective':  loss_B.item(),
            'consistency': loss_C.item(),
            # Effective task weights (for monitoring)
            'w_individual':  w_A.item(),
            'w_collective':  w_B.item(),
            'w_consistency': w_C.item(),
            # σ² values (should stay bounded; blowup = unstable training)
            'sigma2_A': torch.exp(self.log_var_A).item(),
            'sigma2_B': torch.exp(self.log_var_B).item(),
            'sigma2_C': torch.exp(self.log_var_C).item(),
        }

        return total_loss, loss_dict
=======
        self.lw = lambda_weight
        self.c  = criterion or nn.SmoothL1Loss(reduction='none')

    def forward(self, individual_preds, collective_pred, target, mask):
        il = self.c(individual_preds.squeeze(-1),
                    target.expand_as(individual_preds).squeeze(-1))
        if mask is not None:
            il = (il * mask.float()).sum() / mask.float().sum().clamp(min=1)
        else:
            il = il.mean()
        cl = self.c(collective_pred, target).mean()
        total = self.lw * il + (1 - self.lw) * cl
        return total, {'total': total.item(),
                       'individual': il.item(), 'collective': cl.item()}
>>>>>>> Stashed changes
