from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftMaskGate(nn.Module):


    def __init__(self, feature_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(feature_dim, 1),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.gate[0].weight)
        nn.init.constant_(self.gate[0].bias, 2.0)

    def forward(self, x: torch.Tensor,
                hard_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        soft_weights = self.gate(x)
        if hard_mask is not None:
            soft_weights = soft_weights * hard_mask.unsqueeze(-1).float()
        return soft_weights


class TripSequenceEncoder(nn.Module):


    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 n_layers: int = 1, dropout: float = 0.3):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        nn.init.xavier_uniform_(self.input_proj[0].weight)
        nn.init.zeros_(self.input_proj[0].bias)

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
        )
        nn.init.orthogonal_(self.lstm.weight_hh_l0)
        nn.init.xavier_uniform_(self.lstm.weight_ih_l0)

        self.out_dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor,
                lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, _ = x.shape
        proj = self.input_proj(x.reshape(B * T, -1)).reshape(B, T, self.hidden_dim)

        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                proj, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out_packed, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(
                out_packed, batch_first=True, total_length=T)
        else:
            out, _ = self.lstm(proj)

        return self.norm(self.out_dropout(out) + proj)


class DualTaskMTLHead(nn.Module):

    def __init__(self, feature_dim: int,
                 local_hidden:  int   = 64,
                 global_hidden: int   = 64,
                 dropout:       float = 0.3,
                 mask_type:     str   = 'hard'):
        super().__init__()

        assert mask_type in ('hard', 'soft'), \
            f"mask_type must be 'hard' or 'soft', got '{mask_type}'"
        self.mask_type = mask_type

        self.local_head = nn.Sequential(
            nn.Linear(feature_dim, local_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(local_hidden, 1),
        )

        if mask_type == 'soft':
            self.soft_gate = SoftMaskGate(feature_dim)
            self.global_attention = nn.MultiheadAttention(
                embed_dim=feature_dim, num_heads=4,
                dropout=dropout, batch_first=True,
            )

        self.global_head = nn.Sequential(
            nn.Linear(feature_dim, global_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(global_hidden, 1),
        )

        self.log_var_local  = nn.Parameter(torch.zeros(1))
        self.log_var_global = nn.Parameter(torch.zeros(1))

        for seq in (self.local_head, self.global_head):
            for m in seq:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

        print(f"   DualTaskMTLHead  mask_type='{mask_type}'")

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:

        local_preds = self.local_head(x)

        if self.mask_type == 'hard':
            if mask is not None:
                m = mask.unsqueeze(-1).float()
                global_repr = (x * m).sum(1) / m.sum(1).clamp(min=1)
            else:
                global_repr = x.mean(1)

        else:
            soft_w  = self.soft_gate(x, mask)
            x_gated = x * soft_w
            attn_out, _ = self.global_attention(x_gated, x_gated, x_gated)
            weight_sum  = soft_w.sum(dim=1).clamp(min=1e-6)
            global_repr = (attn_out * soft_w).sum(dim=1) / weight_sum

        global_pred = self.global_head(global_repr)
        return local_preds, global_pred


class DualTaskMTLLoss(nn.Module):
    """
    Uncertainty-weighted MTL loss with additive consistency constraint.

    L = precision_L * L_local  + log_var_local
      + precision_G * L_global + log_var_global
      + lambda_cons * L_consistency

    L_consistency: sum(local_preds, valid steps) ≈ global_pred
      Enforces the additive trip-total structure.  Without this, the model
      can satisfy each head independently while their sums diverge.
    """

    def __init__(self, lambda_cons: float = 0.1):
        super().__init__()
        self._base       = nn.SmoothL1Loss(reduction='none')
        self.lambda_cons = lambda_cons

    def forward(
        self,
        local_preds:     torch.Tensor,
        global_pred:     torch.Tensor,
        segment_targets: torch.Tensor,
        global_target:   torch.Tensor,
        mask:            torch.Tensor,
        log_var_local:   torch.Tensor,
        log_var_global:  torch.Tensor,
    ):
        m          = mask.unsqueeze(-1).float()
        local_loss = (self._base(local_preds, segment_targets) * m).sum() / m.sum().clamp(min=1)
        global_loss = self._base(global_pred, global_target).mean()
        local_sum   = (local_preds * m).sum(dim=1)
        cons_loss   = F.smooth_l1_loss(local_sum, global_pred.detach())

        prec_l = torch.exp(-log_var_local)
        prec_g = torch.exp(-log_var_global)
        total  = (prec_l * local_loss  + log_var_local
                + prec_g * global_loss + log_var_global
                + self.lambda_cons * cons_loss)

        return total, {
            'total':         total.item(),
            'local':         local_loss.item(),
            'global':        global_loss.item(),
            'consistency':   cons_loss.item(),
            'weight_local':  prec_l.item(),
            'weight_global': prec_g.item(),
        }


class HierarchicalTripPredictor(nn.Module):

    def __init__(
        self,
        residual_model,
        spatial_dim:    int,
        lstm_hidden:    int   = 128,
        enc_hidden:     int   = 64,
        dropout:        float = 0.2,
        lambda_local:   float = 0.5,
        lambda_bias:    float = 0.05,
        lambda_day:     float = 0.2,
    ):
        super().__init__()
        self.lambda_local  = lambda_local
        self.lambda_bias   = lambda_bias
        self.lambda_day    = lambda_day
        self.spatial_dim   = spatial_dim
        self.lstm_hidden   = lstm_hidden

        self.residual_model = residual_model
        for p in self.residual_model.parameters():
            p.requires_grad = False
        print("   HierarchicalTripPredictor: residual encoder FROZEN")

        step_dim  = spatial_dim + 6
        self._step_dim = step_dim

        self.trip_encoder = TripSequenceEncoder(
            input_dim=step_dim,
            hidden_dim=enc_hidden,
            n_layers=1,
            dropout=dropout,
        )

        self.local_head = nn.Sequential(
            nn.Linear(enc_hidden, enc_hidden // 2), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(enc_hidden // 2, 1),
        )

        self.global_bias_head = nn.Sequential(
            nn.Linear(enc_hidden, enc_hidden // 2), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(enc_hidden // 2, 1),
        )
        nn.init.zeros_(self.global_bias_head[-1].weight)
        nn.init.zeros_(self.global_bias_head[-1].bias)

        self.day_head = nn.Sequential(
            nn.Linear(enc_hidden, enc_hidden // 2), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(enc_hidden // 2, 1),
        )
        nn.init.zeros_(self.day_head[-1].weight)
        nn.init.zeros_(self.day_head[-1].bias)

        for head in (self.local_head,):
            for layer in head:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
        for head in (self.global_bias_head, self.day_head):
            for layer in list(head)[:-1]:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

        n_enc  = sum(p.numel() for p in self.trip_encoder.parameters())
        n_loc  = sum(p.numel() for p in self.local_head.parameters())
        n_bias = sum(p.numel() for p in self.global_bias_head.parameters())
        n_day  = sum(p.numel() for p in self.day_head.parameters())
        print(f"   step_dim      : {step_dim}  "
              f"(spatial={spatial_dim} + seg_pred + cum_pred + pos_frac "
              f"+ cum_error + cum_prior_pred + cum_trip_error)")
        print(f"   TripEncoder   : {step_dim} → {enc_hidden}  ({n_enc:,} params)")
        print(f"   LocalHead     : {enc_hidden} → 1  ({n_loc:,} params)  [per-leg]")
        print(f"   GlobalBias    : {enc_hidden} → 1  ({n_bias:,} params)  [per-trip correction]")
        print(f"   DayHead       : {enc_hidden} → 1  ({n_day:,} params)  [full-day super-global]")
        print(f"   lambda_local={lambda_local}  lambda_bias={lambda_bias}  lambda_day={lambda_day}")

    def _encode_all_segments(
        self,
        seg_indices:        torch.Tensor,
        temporal_features:  torch.Tensor,
        context_flags:      torch.Tensor,
        origin_operational: torch.Tensor,
        dest_operational:   torch.Tensor,
        origin_weather:     torch.Tensor,
        dest_weather:       torch.Tensor,
        mask:               torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B, T   = seg_indices.shape
        device = seg_indices.device
        cum_err = torch.zeros(B, 1, device=device)

        with torch.no_grad():
            _, _, _, seg_preds = self.residual_model(
                seg_indices, temporal_features,
                context_flags=context_flags,
                origin_operational=origin_operational,
                dest_operational=dest_operational,
                origin_weather=origin_weather,
                dest_weather=dest_weather,
                return_components=True,
                cumulative_magnn_error=cum_err,
            )
            spatial_seq = self.residual_model._get_spatial_embeddings(seg_indices)

        return seg_preds, spatial_seq

    def forward(
        self,
        seg_indices:        torch.Tensor,
        temporal_features:  torch.Tensor,
        context_flags:      torch.Tensor,
        origin_operational: torch.Tensor,
        dest_operational:   torch.Tensor,
        origin_weather:     torch.Tensor,
        dest_weather:       torch.Tensor,
        mask:               torch.Tensor,
        lengths:            torch.Tensor,
        return_local:       bool = False,
        seg_targets:        Optional[torch.Tensor] = None,
        cum_prior_actual:   Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Returns: (global_pred, local_preds_or_None, day_pred)
          global_pred : (B, 1)  per-trip total prediction
          local_preds : (B, T, 1) or None
          day_pred    : (B, 1)  full-day super-global prediction
        """
        B, T   = seg_indices.shape
        device = seg_indices.device

        seg_preds, spatial_seq = self._encode_all_segments(
            seg_indices, temporal_features, context_flags,
            origin_operational, dest_operational,
            origin_weather, dest_weather, mask,
        )

        m_seq = mask.float().unsqueeze(-1)

        masked_preds = seg_preds * m_seq
        cumsum       = masked_preds.cumsum(dim=1)
        cum_pred = torch.cat([
            torch.zeros(B, 1, 1, device=device),
            cumsum[:, :-1, :]
        ], dim=1)

        if seg_targets is not None:
            masked_gt  = seg_targets * m_seq
            step_error = (seg_preds - masked_gt) * m_seq
            err_cs     = step_error.cumsum(dim=1)
            cum_error  = torch.cat([
                torch.zeros(B, 1, 1, device=device),
                err_cs[:, :-1, :]
            ], dim=1)
        else:
            cum_error = torch.zeros(B, T, 1, device=device)

        if cum_prior_actual is not None:
            cp = cum_prior_actual.unsqueeze(1).expand(B, T, 1)
        else:
            cp = torch.zeros(B, T, 1, device=device)

        cum_trip_error = torch.zeros(B, T, 1, device=device)

        t_idx    = torch.arange(T, device=device).float()
        len_f    = lengths.float().unsqueeze(1)
        pos_frac = (t_idx.unsqueeze(0) / len_f.clamp(min=1)).clamp(0, 1)

        trip_seq = torch.cat([
            spatial_seq,
            seg_preds,
            cum_pred,
            pos_frac.unsqueeze(-1),
            cum_error,
            cp,                        # (B, T, 1)  prior trips' actual time today
            cum_trip_error,            # (B, T, 1)  prior trips' prediction error
        ], dim=2)

        enc_out = self.trip_encoder(trip_seq, lengths)

        local_preds = self.local_head(enc_out)

        naive_sum  = (local_preds * m_seq).sum(dim=1)
        idx        = (lengths - 1).clamp(min=0).long()
        idx_exp    = idx.view(B, 1, 1).expand(B, 1, enc_out.size(2))
        last_h     = enc_out.gather(1, idx_exp).squeeze(1)
        trip_bias  = self.global_bias_head(last_h)
        global_pred = naive_sum + trip_bias

        day_pred = self.day_head(last_h)

        self._last_naive_sum  = naive_sum.detach()
        self._last_trip_bias  = trip_bias.detach()

        if return_local:
            return global_pred, local_preds, day_pred
        return global_pred, None, day_pred


MAGNN_LSTM_DualTaskMTL = HierarchicalTripPredictor


class MTLHead(nn.Module):


    def __init__(self, feature_dim, segment_hidden=64,
                 path_hidden=128, dropout=0.2, mask_type: str = 'hard'):
        super().__init__()
        assert mask_type in ('hard', 'soft')
        self.mask_type = mask_type

        self.segment_predictor = nn.Sequential(
            nn.Linear(feature_dim, segment_hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(segment_hidden, 1))
        self.path_attention = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=4,
            dropout=dropout, batch_first=True)
        self.path_predictor = nn.Sequential(
            nn.Linear(feature_dim, path_hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(path_hidden, 64),
            nn.ReLU(), nn.Linear(64, 1))

        if mask_type == 'soft':
            self.soft_gate = SoftMaskGate(feature_dim)

    def forward(self, x, mask=None, return_components=False):
        individual_preds = self.segment_predictor(x)

        if self.mask_type == 'hard':
            attn_out, attn_w = self.path_attention(
                x, x, x,
                key_padding_mask=~mask if mask is not None else None)
            if mask is not None:
                m = mask.unsqueeze(-1).float()
                pooled = (attn_out * m).sum(1) / m.sum(1).clamp(min=1)
            else:
                pooled = attn_out.mean(1)
        else:
            soft_w  = self.soft_gate(x, mask)
            x_gated = x * soft_w
            attn_out, attn_w = self.path_attention(x_gated, x_gated, x_gated)
            weight_sum = soft_w.sum(dim=1).clamp(min=1e-6)
            pooled = (attn_out * soft_w).sum(dim=1) / weight_sum

        collective = self.path_predictor(pooled)
        if return_components:
            return individual_preds, collective, attn_w
        return collective


class MTLLoss(nn.Module):

    def __init__(self, lambda_weight=0.5, criterion=None):
        super().__init__()
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
        return total, {
            'total': total.item(),
            'individual': il.item(),
            'collective': cl.item(),
        }
