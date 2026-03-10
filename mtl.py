"""
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
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
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
    """
    Three-task uncertainty-weighted MTL loss.

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