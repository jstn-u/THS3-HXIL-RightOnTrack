"""
lstm.py
=======
LSTM with Global Temporal Attention for travel time prediction.
Combines MAGNN spatial embeddings with operational delay and weather features.

Integration approach:
- MAGNN embeddings (X): spatial graph features from GAT
- Operational features (O): arrivalDelay, departureDelay, congestionLevel
- Weather features (W): temperature, precipitation, wind, etc.
- Input to LSTM: X + O + W (concatenated)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# GLOBAL TEMPORAL ATTENTION
# =============================================================================

class GlobalTemporalAttention(nn.Module):
    """Transformer-based global temporal attention with residual connection."""

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
        """
        Args:
            x: [batch, seq_len, feature_dim]
        Returns:
            out: [batch, seq_len, feature_dim]
            attn_weights: [batch, seq_len, seq_len]
        """
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, V)
        out = self.layer_norm(out + x)  # Residual connection

        return out, attn_weights


# =============================================================================
# LSTM WITH GLOBAL TEMPORAL ATTENTION
# =============================================================================

class LSTMWithGlobalTemporalAttention(nn.Module):
    """
    LSTM + Global Temporal Attention for travel time prediction.

    Input: MAGNN embeddings + operational features + weather features
    Output: Travel time prediction
    """

    def __init__(self, spatial_dim, operational_dim, weather_dim,
                 temporal_dim=5,  # hour_sin, hour_cos, dow_sin, dow_cos, speed_scaled
                 hidden_dim=64, n_layers=2, dropout=0.2, out_dim=1):
        super().__init__()

        # LSTM input: spatial (from MAGNN) + operational + weather + temporal
        lstm_input_dim = spatial_dim + operational_dim + weather_dim + temporal_dim

        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0
        )

        self.global_attention = GlobalTemporalAttention(
            feature_dim=hidden_dim,
            dropout=dropout
        )

        # Fusion layer for final prediction
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, out_dim)
        )

    def forward(self, seq_x):
        """
        Args:
            seq_x: [batch, seq_len, feature_dim] - concatenated features
        Returns:
            out: [batch, 1] - travel time prediction
        """
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(seq_x)  # [batch, seq_len, hidden_dim]

        # Global temporal attention
        attn_out, attn_weights = self.global_attention(lstm_out)  # [batch, seq_len, hidden_dim]

        # Take last time step
        attn_last = attn_out[:, -1, :]  # [batch, hidden_dim]

        # Final prediction
        out = self.fusion(attn_last)  # [batch, 1]

        return out


# =============================================================================
# MAGNN-LSTM COMBINED MODEL
# =============================================================================

class MAGNN_LSTM(nn.Module):
    """
    Combined MAGNN + LSTM model.

    Pipeline:
    1. MAGNN produces spatial embeddings from graph structure
    2. Spatial embeddings + operational + weather + temporal → LSTM
    3. LSTM with attention → final prediction
    """

    def __init__(self, magnn_model, spatial_dim, operational_dim, weather_dim,
                 temporal_dim=5, lstm_hidden=64, lstm_layers=2, dropout=0.2,
                 freeze_magnn=False):
        """
        Args:
            magnn_model: Pre-trained MAGNN model
            spatial_dim: Dimension of MAGNN spatial embeddings (gat_hidden)
            operational_dim: Number of operational features
            weather_dim: Number of weather features
            temporal_dim: Number of temporal features (default 5)
            lstm_hidden: LSTM hidden dimension
            lstm_layers: Number of LSTM layers
            dropout: Dropout rate
            freeze_magnn: If True, freeze MAGNN weights during training
        """
        super().__init__()

        self.magnn = magnn_model
        self.freeze_magnn = freeze_magnn

        if freeze_magnn:
            for param in self.magnn.parameters():
                param.requires_grad = False

        self.lstm_model = LSTMWithGlobalTemporalAttention(
            spatial_dim=spatial_dim,
            operational_dim=operational_dim,
            weather_dim=weather_dim,
            temporal_dim=temporal_dim,
            hidden_dim=lstm_hidden,
            n_layers=lstm_layers,
            dropout=dropout,
            out_dim=1
        )

    def get_magnn_embeddings(self, seg_indices):
        """
        Extract spatial embeddings from MAGNN's GAT layer.

        Args:
            seg_indices: [batch] - segment indices
        Returns:
            spatial_embeddings: [batch, spatial_dim]
        """
        device = seg_indices.device

        # Run GAT over all nodes (this is already optimized in MAGNN)
        all_nodes = self.magnn.node_embedding.weight.unsqueeze(0)  # [1, num_nodes, node_embed_dim]

        # Get spatial embeddings from GAT
        spatial_all = self.magnn.multi_gat(
            all_nodes,
            [self.magnn.adj_geo, self.magnn.adj_dist, self.magnn.adj_soc]
        )  # [1, num_nodes, gat_hidden]

        spatial_all = spatial_all.squeeze(0)  # [num_nodes, gat_hidden]

        # Index the relevant embeddings for this batch
        spatial_embeddings = spatial_all[seg_indices]  # [batch, gat_hidden]

        return spatial_embeddings

    def forward(self, seg_indices, temporal_features, operational_features, weather_features):
        """
        Forward pass through combined model.

        Args:
            seg_indices: [batch] - segment type indices
            temporal_features: [batch, temporal_dim] - cyclical time + speed
            operational_features: [batch, operational_dim] - delays, congestion
            weather_features: [batch, weather_dim] - weather conditions
        Returns:
            predictions: [batch, 1]
        """
        # 1. Get MAGNN spatial embeddings
        if self.freeze_magnn:
            with torch.no_grad():
                spatial_embeddings = self.get_magnn_embeddings(seg_indices)
        else:
            spatial_embeddings = self.get_magnn_embeddings(seg_indices)

        # 2. Concatenate all features: X + O + W + T
        # Shape: [batch, spatial_dim + operational_dim + weather_dim + temporal_dim]
        combined_features = torch.cat([
            spatial_embeddings,      # X: MAGNN spatial
            operational_features,    # O: operational delays
            weather_features,        # W: weather
            temporal_features        # T: temporal (time + speed)
        ], dim=1)

        # 3. Add sequence dimension (seq_len=1 for now, expandable later)
        seq_features = combined_features.unsqueeze(1)  # [batch, 1, total_dim]

        # 4. LSTM with attention
        predictions = self.lstm_model(seq_features)  # [batch, 1]

        return predictions