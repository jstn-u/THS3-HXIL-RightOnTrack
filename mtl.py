"""
mtl.py - REDESIGNED Multi-Task Learning
========================================
✅ LOCAL + GLOBAL dual-task approach
   - Local: Segment-level predictions (fine-grained)
   - Global: Station-to-station predictions (coarse-grained)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DualTaskMTLHead(nn.Module):
    """
    Dual-Task MTL: Local segments + Global station-to-station
    """

    def __init__(self, feature_dim, local_hidden=64, global_hidden=128, dropout=0.2):
        super().__init__()

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

        # Task weighting (learnable)
        self.log_var_local = nn.Parameter(torch.zeros(1))
        self.log_var_global = nn.Parameter(torch.zeros(1))

    def forward(self, x, mask=None):
        """
        Args:
            x: [batch, seq_len, feature_dim] - sequence of segment embeddings
            mask: [batch, seq_len] - padding mask
        Returns:
            local_preds: [batch, seq_len, 1] - per-segment predictions
            global_pred: [batch, 1] - aggregated prediction
        """
        # Local task: Predict each segment independently
        local_preds = self.local_head(x)  # [batch, seq_len, 1]

        # Global task: Attend over sequence and predict aggregate
        attn_out, _ = self.global_attention(x, x, x, key_padding_mask=~mask if mask is not None else None)

        # Global pooling (mean over valid segments)
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            global_repr = (attn_out * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            global_repr = attn_out.mean(dim=1)

        global_pred = self.global_head(global_repr)  # [batch, 1]

        return local_preds, global_pred


class DualTaskMTLLoss(nn.Module):
    """
    Loss for dual-task MTL with uncertainty weighting
    """

    def __init__(self, criterion=None):
        super().__init__()
        self.criterion = criterion if criterion is not None else nn.SmoothL1Loss(reduction='none')

    def forward(self, local_preds, global_pred, segment_targets, global_target,
                mask, log_var_local, log_var_global):
        """
        Args:
            local_preds: [batch, seq_len, 1] - predicted segment durations
            global_pred: [batch, 1] - predicted total duration
            segment_targets: [batch, seq_len, 1] - true segment durations
            global_target: [batch, 1] - true total duration
            mask: [batch, seq_len] - valid segments
            log_var_local: learnable variance for local task
            log_var_global: learnable variance for global task
        """
        # Local loss (per-segment)
        local_loss = self.criterion(local_preds, segment_targets)
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            local_loss = (local_loss * mask_expanded).sum() / mask_expanded.sum().clamp(min=1)
        else:
            local_loss = local_loss.mean()

        # Global loss (aggregate)
        global_loss = self.criterion(global_pred, global_target).mean()

        # Uncertainty weighting (Kendall et al. 2018)
        precision_local = torch.exp(-log_var_local)
        precision_global = torch.exp(-log_var_global)

        total_loss = (
                precision_local * local_loss + log_var_local +
                precision_global * global_loss + log_var_global
        )

        return total_loss, {
            'total': total_loss.item(),
            'local': local_loss.item(),
            'global': global_loss.item(),
            'weight_local': precision_local.item(),
            'weight_global': precision_global.item()
        }


# Keep old MTLHead for backward compatibility
class MTLHead(nn.Module):
    """Original MTL head (deprecated - use DualTaskMTLHead)"""

    def __init__(self, feature_dim, segment_hidden=64, path_hidden=128, dropout=0.2):
        super().__init__()
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

    def forward(self, x, mask=None, return_components=False):
        individual_preds = self.segment_predictor(x)
        attn_out, attn_weights = self.path_attention(x, x, x,
                                                     key_padding_mask=~mask if mask is not None else None)
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            pooled = (attn_out * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            pooled = attn_out.mean(dim=1)
        collective_pred = self.path_predictor(pooled)

        if return_components:
            return individual_preds, collective_pred, attn_weights
        return collective_pred


class MTLLoss(nn.Module):
    """Original MTL loss (deprecated)"""

    def __init__(self, lambda_weight=0.5, criterion=None):
        super().__init__()
        self.lambda_weight = lambda_weight
        self.criterion = criterion if criterion is not None else nn.SmoothL1Loss(reduction='none')

    def forward(self, individual_preds, collective_pred, target, mask):
        individual_loss = self.criterion(individual_preds.squeeze(-1), target.expand_as(individual_preds).squeeze(-1))
        if mask is not None:
            individual_loss = (individual_loss * mask.float()).sum() / mask.float().sum().clamp(min=1)
        else:
            individual_loss = individual_loss.mean()
        collective_loss = self.criterion(collective_pred, target).mean()
        total_loss = self.lambda_weight * individual_loss + (1 - self.lambda_weight) * collective_loss
        return total_loss, {
            'total': total_loss.item(),
            'individual': individual_loss.item(),
            'collective': collective_loss.item()
        }