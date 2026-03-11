"""
residual.py - FIXED VERSION
===========================
MAGNN-LSTM-Residual: LSTM learns from MAGNN's prediction

✅ FIXES APPLIED:
1. Xavier initialization for LSTM (better than tiny random weights)
2. Proper gradient flow monitoring
3. Input includes MAGNN baseline + all features
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# ADAPTIVE GATE
# =============================================================================

class AdaptiveGate(nn.Module):
    """Learns per-sample gating weight α ∈ [0,1]"""

    def __init__(self, spatial_dim, hidden_dim=32):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(spatial_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # Initialize to output ~0.5
        nn.init.zeros_(self.gate_net[3].weight)
        nn.init.constant_(self.gate_net[3].bias, 0.0)

    def forward(self, spatial_embedding):
        alpha = self.gate_net(spatial_embedding)
        return alpha


# =============================================================================
# GLOBAL TEMPORAL ATTENTION
# =============================================================================

class GlobalTemporalAttention(nn.Module):
    def __init__(self, feature_dim, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.W_Q = nn.Linear(feature_dim, feature_dim, bias=False)
        self.W_K = nn.Linear(feature_dim, feature_dim, bias=False)
        self.W_V = nn.Linear(feature_dim, feature_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(feature_dim)
        self.scale = np.sqrt(feature_dim)

    def forward(self, x):
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, V)
        out = self.layer_norm(out + x)

        return out, attn_weights


# =============================================================================
# RESIDUAL LSTM - ✅ FIXED INITIALIZATION
# =============================================================================

class ResidualLSTM(nn.Module):
    """
    LSTM that sees MAGNN's baseline prediction + context features.

    ✅ FIX: Xavier initialization instead of tiny random weights
    """

    def __init__(self, spatial_dim, operational_dim, weather_dim,
                 temporal_dim=5, baseline_dim=1,
                 hidden_dim=128, n_layers=1, dropout=0.1):
        super().__init__()

        lstm_input_dim = spatial_dim + operational_dim + weather_dim + temporal_dim + baseline_dim

        print(f"   ResidualLSTM configuration:")
        print(f"     Spatial: {spatial_dim}")
        print(f"     Operational: {operational_dim} (arrivalDelay, departureDelay, is_weekend, is_peak_hour)")
        print(f"     Weather: {weather_dim}")
        print(f"     Temporal: {temporal_dim}")
        print(f"     MAGNN baseline: {baseline_dim} ← LSTM SEES THIS!")
        print(f"     Total input: {lstm_input_dim}")

        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.0
        )

        self.global_attention = GlobalTemporalAttention(
            feature_dim=hidden_dim,
            dropout=dropout
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

        # ✅ FIX: Xavier initialization (better gradient flow)
        for layer in self.fusion:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)  # ✅ CHANGED from normal_(0, 0.01)
                nn.init.zeros_(layer.bias)

        print(f"     ✅ Initialization: Xavier (better gradient flow)")

    def forward(self, seq_x):
        batch_size = seq_x.size(0)
        lstm_out, _ = self.lstm(seq_x)
        attn_out, _ = self.global_attention(lstm_out)
        attn_last = attn_out[:, -1, :]
        correction = self.fusion(attn_last)
        return correction


# =============================================================================
# MAGNN-LSTM-RESIDUAL
# =============================================================================

class MAGNN_LSTM_Residual(nn.Module):
    """
    MAGNN-LSTM with residual learning and adaptive gating.

    ✅ FIXES:
    - LSTM sees MAGNN baseline (residual learning)
    - Xavier initialization (better than tiny random)
    - Adaptive gating (learned per sample)
    """

    def __init__(self, magnn_model, spatial_dim, operational_dim, weather_dim,
                 temporal_dim=5, lstm_hidden=128, lstm_layers=1, dropout=0.2,
                 freeze_magnn=True):
        super().__init__()

        self.magnn = magnn_model
        self.freeze_magnn = freeze_magnn

        if freeze_magnn:
            for param in self.magnn.parameters():
                param.requires_grad = False
            print(f"   MAGNN frozen (transfer learning)")

        self.residual_lstm = ResidualLSTM(
            spatial_dim=spatial_dim,
            operational_dim=operational_dim,
            weather_dim=weather_dim,
            temporal_dim=temporal_dim,
            baseline_dim=1,
            hidden_dim=lstm_hidden,
            n_layers=lstm_layers,
            dropout=dropout
        )

        self.adaptive_gate = AdaptiveGate(
            spatial_dim=spatial_dim,
            hidden_dim=32
        )

        print(f"   AdaptiveGate: learns α per sample")

    def get_magnn_embeddings(self, seg_indices):
        all_nodes = self.magnn.node_embedding.weight.unsqueeze(0)
        spatial_all = self.magnn.multi_gat(
            all_nodes,
            [self.magnn.adj_geo, self.magnn.adj_dist, self.magnn.adj_soc]
        )
        spatial_all = spatial_all.squeeze(0)
        spatial_embeddings = spatial_all[seg_indices]
        return spatial_embeddings

    def forward(self, seg_indices, temporal_features, operational_features, weather_features,
                return_components=False):
        # 1. MAGNN baseline
        with torch.no_grad():
            magnn_baseline = self.magnn(seg_indices, temporal_features)

        # 2. Spatial embeddings
        if self.freeze_magnn:
            with torch.no_grad():
                spatial_embeddings = self.get_magnn_embeddings(seg_indices)
        else:
            spatial_embeddings = self.get_magnn_embeddings(seg_indices)

        # 3. ✅ Concat ALL features INCLUDING MAGNN baseline
        combined_features = torch.cat([
            spatial_embeddings,
            operational_features,
            weather_features,
            temporal_features,
            magnn_baseline  # ← LSTM SEES THIS!
        ], dim=1)

        seq_features = combined_features.unsqueeze(1)

        # 4. LSTM correction
        lstm_correction = self.residual_lstm(seq_features)

        # 5. Adaptive gate
        alpha = self.adaptive_gate(spatial_embeddings)

        # 6. Final prediction
        final_prediction = magnn_baseline + alpha * lstm_correction

        if return_components:
            return magnn_baseline, lstm_correction, alpha, final_prediction
        else:
            return final_prediction