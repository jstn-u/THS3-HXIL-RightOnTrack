"""
config.py
=========
Shared configuration, constants, and utility functions used by all modules.

Imports this module instead of duplicating across files:
    from config import Config, DEVICE, print_section, haversine_meters
"""

import numpy as np
import pandas as pd
from math import radians, cos, sin, asin, sqrt
from sklearn.neighbors import BallTree
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings
import os
from datetime import datetime
import json
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — safe for all environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import contextily as ctx

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# CONFIGURATION - ADJUST THESE PARAMETERS
# =============================================================================

class Config:
    """
    Training Configuration - Adjust these parameters based on your needs

    QUICK START GUIDE:
    ==================

    For QUICK TESTING (single run):
    - Set n_iterations = 1
    - Set n_epochs = 10
    - Set sample_fraction = 0.1 (10% of data)

    For FULL TRAINING (production):
    - Set n_iterations = 50
    - Set n_epochs = 50
    - Set sample_fraction = 1.0 (100% of data)

    PARAMETER EXPLANATIONS:
    =======================
    """

    # =========================================================================
    # DATA PARAMETERS
    # =========================================================================
    data_folder = './data'

    sample_fraction = 0.01
    """
    SAMPLE_FRACTION: What percentage of data to use
    - 0.1 = 10% of trips (FAST - for testing/debugging)
    - 0.5 = 50% of trips (MEDIUM - for initial experiments)
    - 1.0 = 100% of trips (SLOW - for final production model)

    Example: If you have 1000 trips, 0.1 uses 100 random trips
    """

    # =========================================================================
    # ITERATION PARAMETERS
    # =========================================================================
    n_iterations = 1
    # For quick testing keep this at 1. Increase for production runs.
    """
    N_ITERATIONS: How many complete training runs to perform
    - 1 = Single run (RECOMMENDED for testing)
    - 5 = Multiple runs for averaging results
    - 50 = Full experiment with different random samples

    Why multiple iterations?
    - Each iteration uses different random trips (due to sampling)
    - Helps assess model stability and generalization
    - Final metrics can be averaged across all iterations

    Time estimate: Each iteration takes ~10-30 minutes depending on data size
    """

    # =========================================================================
    # TRAINING PARAMETERS
    # =========================================================================
    n_epochs = 50          # keep low for quick testing; raise to 50+ for production
    batch_size = 64        # larger batch = fewer iterations per epoch = faster
    learning_rate = 0.001
    dropout = 0.3
    weight_decay = 1e-5
    early_stopping_patience = 50
    lr_scheduler_patience = 3
    lr_scheduler_factor = 0.5

    # =========================================================================
    # MODEL ARCHITECTURE PARAMETERS  (smaller = faster, sufficient for testing)
    # =========================================================================
    n_heads = 3            # FIXED — one per adjacency matrix (geo, dist, social)
    node_embed_dim = 32    # node embedding size fed into GAT  (was 64)
    gat_hidden = 32        # GAT output per head               (was 64)
    lstm_hidden = 64       # LSTM hidden state size            (was 128)
    historical_dim = 16    # per-segment historical embedding  (was 32)


# Create config instance
config = Config()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================



# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def print_section(title):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)



def haversine_meters(lat1, lon1, lat2, lon2):
    """Calculate great circle distance in meters."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return c * 6371000


# =============================================================================
# DATA LOADING
# =============================================================================

