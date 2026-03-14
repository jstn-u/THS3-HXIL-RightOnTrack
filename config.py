"""
config.py
=========
Configuration for MAGNN-LSTM transit prediction

✅ UPDATED: Learning rates optimized for residual learning
"""

import torch


class Config:
    # =============================================================================
    # PATHS
    # =============================================================================
    data_folder = './data'

    # =============================================================================
    # TRAINING HYPERPARAMETERS - ✅ FIXED FOR RESIDUAL LEARNING
    # =============================================================================

    # Sampling
    sample_fraction = 0.2 # Use all data (or 0.1 for 10% sample)

    # Training
    n_iterations = 5  # Number of complete training runs
    n_epochs = 50  # Epochs per iteration
    batch_size = 64  # Batch size

    # ✅ FIX: Separate learning rates for different models
    learning_rate = 0.001  # Base LR for MAGNN (default)
    lstm_learning_rate = 0.0005  # ✅ NEW: Dedicated LR for LSTM components
    residual_learning_rate = 0.0005  # ✅ NEW: For residual model specifically

    # ✅ FIX: Reduced weight decay for residual learning
    weight_decay = 1e-5  # Base weight decay for MAGNN
    lstm_weight_decay = 1e-6  # ✅ NEW: Lower for LSTM (prevents correction collapse)

    # Learning rate scheduling
    lr_scheduler_factor = 0.5
    lr_scheduler_patience = 5
    early_stopping_patience = 50

    # =============================================================================
    # MODEL ARCHITECTURE
    # =============================================================================

    # MAGNN
    n_heads = 3  # Number of GAT attention heads
    node_embed_dim = 32  # Node embedding dimension
    gat_hidden = 32  # GAT hidden dimension
    lstm_hidden = 64  # MAGNN LSTM hidden dimension
    historical_dim = 16  # Historical embedding dimension
    dropout = 0.3  # Dropout rate

    # MTL
    mtl_lambda = 0.5  # Initial balance for uncertainty weighting


    # =============================================================================
    # DEVICE
    # =============================================================================

    @property
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# GLOBAL SETTINGS
# =============================================================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def haversine_meters(lat1, lon1, lat2, lon2):
    """Calculate distance between two GPS coordinates in meters."""
    from math import radians, cos, sin, asin, sqrt

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371000  # Earth radius in meters
    return c * r