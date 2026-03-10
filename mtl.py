"""
mtl.py
======
Multi-Task Learning (MTL) components for travel time estimation.

Based on AttentionTTE approach:
- Task 1: Individual segment predictions (local accuracy)
- Task 2: Collective path prediction (global patterns)
- Weighted combination using L2-norm attention
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# SEGMENT-LEVEL PREDICTION (TASK 1)
# =============================================================================

class SegmentPredictor(nn.Module):
    """
    Predicts travel time for individual segments.
    Uses a two-layer FC network as per AttentionTTE.
    """

    def __init__(self, feature_dim, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, segment_features):
        """
        Args:
            segment_features: [batch, feature_dim] - features for one segment
        Returns:
            prediction: [batch, 1] - individual segment time prediction
        """
        return self.fc(segment_features)


# =============================================================================
# L2-NORM ATTENTION WEIGHTING
# =============================================================================

class L2NormAttention(nn.Module):
    """
    Calculates attention weights based on L2-norm (magnitude) of features.
    Formula: α_i = ||h_i|| / Σ||h_j||
    """

    def __init__(self):
        super().__init__()

    def forward(self, features):
        """
        Args:
            features: [batch, seq_len, feature_dim]
        Returns:
            weights: [batch, seq_len] - normalized attention weights
        """
        # Calculate L2 norm for each feature vector
        norms = torch.norm(features, p=2, dim=2)  # [batch, seq_len]

        # Normalize to get weights (avoid division by zero)
        weights = norms / (norms.sum(dim=1, keepdim=True) + 1e-8)

        return weights


# =============================================================================
# PATH-LEVEL PREDICTION (TASK 2)
# =============================================================================

class PathPredictor(nn.Module):
    """
    Predicts total travel time for the entire path using weighted features.
    Uses residual network as per AttentionTTE.
    """

    def __init__(self, feature_dim, hidden_dim=128, dropout=0.2):
        super().__init__()

        # Residual network with skip connections
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # Skip connection projection if dimensions don't match
        self.skip = nn.Linear(feature_dim, hidden_dim) if feature_dim != hidden_dim else None

    def forward(self, global_feature):
        """
        Args:
            global_feature: [batch, feature_dim] - weighted sum of segment features
        Returns:
            prediction: [batch, 1] - collective path time prediction
        """
        # First layer
        x = self.fc1(global_feature)
        x = self.relu(x)
        x = self.dropout(x)

        # Residual connection
        identity = self.skip(global_feature) if self.skip is not None else global_feature

        # Second layer with residual
        x2 = self.fc2(x)
        x2 = self.relu(x2 + identity)  # Residual connection
        x2 = self.dropout(x2)

        # Output layer
        out = self.fc3(x2)

        return out


# =============================================================================
# MULTI-TASK LEARNING HEAD
# =============================================================================

# Replace the MTLHead class (around line 113):

class MTLHead(nn.Module):
    """
    Complete Multi-Task Learning module combining:
    1. Individual segment predictions
    2. L2-norm attention weighting
    3. Collective path prediction
    """

    def __init__(self, feature_dim, segment_hidden=64, path_hidden=128, dropout=0.2):
        super().__init__()

        self.segment_predictor = SegmentPredictor(feature_dim, segment_hidden, dropout)
        self.l2_attention = L2NormAttention()
        self.path_predictor = PathPredictor(feature_dim, path_hidden, dropout)

    def forward(self, segment_features, mask=None, return_components=False):  # ✅ ADDED mask
        """
        Args:
            segment_features: [batch, seq_len, feature_dim] - features for all segments
            mask: [batch, seq_len] - binary mask (1 = valid, 0 = padding)
            return_components: if True, return individual predictions and weights

        Returns:
            If return_components=False:
                path_prediction: [batch, 1] - final collective prediction
            If return_components=True:
                (individual_preds, path_prediction, attention_weights)
        """
        batch_size, seq_len, feature_dim = segment_features.size()

        # Task 1: Individual segment predictions
        segments_flat = segment_features.view(-1, feature_dim)
        individual_preds_flat = self.segment_predictor(segments_flat)
        individual_preds = individual_preds_flat.view(batch_size, seq_len, 1)

        # Calculate L2-norm attention weights
        attention_weights = self.l2_attention(segment_features)  # [batch, seq_len]

        # ✅ FIXED: Apply mask to attention weights
        if mask is not None:
            attention_weights = attention_weights * mask.float()
            # Re-normalize after masking
            attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-8)

        # Task 2: Collective path prediction
        weighted_features = segment_features * attention_weights.unsqueeze(2)
        global_feature = weighted_features.sum(dim=1)

        path_prediction = self.path_predictor(global_feature)

        if return_components:
            return individual_preds, path_prediction, attention_weights
        else:
            return path_prediction



# =============================================================================
# MTL LOSS FUNCTION
# =============================================================================

class MTLLoss(nn.Module):
    """
    Combined loss for multi-task learning:
    Total_Loss = (1 - λ) * Loss_individual + λ * Loss_collective

    λ (lambda_weight) controls the balance:
    - λ = 0.0: Only individual predictions matter
    - λ = 0.5: Equal weight to both tasks
    - λ = 1.0: Only collective prediction matters
    """

    def __init__(self, lambda_weight=0.5, criterion=None):
        super().__init__()
        self.lambda_weight = lambda_weight
        self.criterion = criterion if criterion is not None else nn.SmoothL1Loss()

    def forward(self, individual_preds, path_pred, target, mask=None):  # ✅ ADDED mask parameter
        """
        Args:
            individual_preds: [batch, seq_len, 1] - predictions for each segment
            path_pred: [batch, 1] - prediction for entire path
            target: [batch, 1] - ground truth total travel time
            mask: [batch, seq_len] - binary mask (1 = valid, 0 = padding)

        Returns:
            total_loss: Combined MTL loss
            loss_dict: Dictionary with individual and collective losses (for logging)
        """
        # ✅ FIXED: Apply mask to individual predictions before summing
        if mask is not None:
            # Only sum valid (non-padded) segments
            masked_individual = individual_preds * mask.unsqueeze(2).float()
            individual_sum = masked_individual.sum(dim=1)
        else:
            # Fallback: sum all segments if no mask provided
            individual_sum = individual_preds.sum(dim=1)

        # Individual loss: compare sum of segment predictions to target
        individual_loss = self.criterion(individual_sum, target)

        # Collective loss: compare path prediction to target
        collective_loss = self.criterion(path_pred, target)

        # Combined loss
        total_loss = (1 - self.lambda_weight) * individual_loss + self.lambda_weight * collective_loss

        loss_dict = {
            'total': total_loss.item(),
            'individual': individual_loss.item(),
            'collective': collective_loss.item()
        }

        return total_loss, loss_dict

# =============================================================================
# HELPER: SEGMENT FEATURE EXTRACTION
# =============================================================================

def extract_segment_features(combined_features, seq_len=1):
    """
    Helper function to convert single-point features into sequence format for MTL.

    Args:
        combined_features: [batch, feature_dim] - concatenated spatial+operational+weather+temporal
        seq_len: int - number of segments (default 1 for current implementation)

    Returns:
        segment_features: [batch, seq_len, feature_dim]
    """
    # For now, with seq_len=1, we just add a dimension
    # Future: Can be extended to handle actual multi-segment sequences
    return combined_features.unsqueeze(1)  # [batch, 1, feature_dim]