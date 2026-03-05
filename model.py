"""
model.py - COMPLETE VERSION WITH MTL
All neural network components including Multi-Task Learning
"""

import numpy as np
import pandas as pd
import os
import json
import warnings
from datetime import datetime
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import RobustScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from config import Config, DEVICE, print_section, haversine_meters
from mtl import MTLHead, MTLLoss, extract_segment_features

warnings.filterwarnings('ignore')


# =============================================================================
# BASE DATASET
# =============================================================================

class SegmentDataset(Dataset):
    def __init__(self, segments_df, segment_types,
                 fit_scalers: bool = True,
                 target_scaler: RobustScaler = None,
                 speed_scaler: RobustScaler = None):
        self.segments_df = segments_df.copy()
        self.segment_types = list(segment_types)
        self.seg_to_idx = {seg: i for i, seg in enumerate(self.segment_types)}

        self.segments_df['seg_type_idx'] = self.segments_df['segment_id'].map(self.seg_to_idx)
        n_before = len(self.segments_df)
        self.segments_df = self.segments_df.dropna(subset=['seg_type_idx']).copy()
        n_dropped = n_before - len(self.segments_df)
        if n_dropped > 0:
            print(f"   ⚠️  Dropped {n_dropped:,} rows with unseen segment types")
        self.segments_df['seg_type_idx'] = self.segments_df['seg_type_idx'].astype(int)

        self.segments_df['hour_sin'] = np.sin(2 * np.pi * self.segments_df['hour'] / 24)
        self.segments_df['hour_cos'] = np.cos(2 * np.pi * self.segments_df['hour'] / 24)
        self.segments_df['dow_sin'] = np.sin(2 * np.pi * self.segments_df['day_of_week'] / 7)
        self.segments_df['dow_cos'] = np.cos(2 * np.pi * self.segments_df['day_of_week'] / 7)

        if fit_scalers:
            self.target_scaler = RobustScaler()
            self.segments_df['duration_scaled'] = self.target_scaler.fit_transform(
                self.segments_df[['duration_sec']]
            )
        else:
            if target_scaler is None:
                raise ValueError("target_scaler must be provided when fit_scalers=False")
            self.target_scaler = target_scaler
            self.segments_df['duration_scaled'] = self.target_scaler.transform(
                self.segments_df[['duration_sec']]
            )

        if 'speed_mps' in self.segments_df.columns:
            speed_vals = self.segments_df[['speed_mps']].copy()
            speed_vals = speed_vals.replace([np.inf, -np.inf], np.nan)
            if fit_scalers:
                self.speed_scaler = RobustScaler()
                speed_scaled = self.speed_scaler.fit_transform(
                    speed_vals.fillna(speed_vals.median())
                )
            else:
                if speed_scaler is None:
                    raise ValueError("speed_scaler must be provided when fit_scalers=False")
                self.speed_scaler = speed_scaler
                speed_scaled = self.speed_scaler.transform(
                    speed_vals.fillna(speed_vals.median())
                )
            speed_scaled = np.nan_to_num(speed_scaled, nan=0.0)
            self.segments_df['speed_scaled'] = speed_scaled.flatten()
        else:
            self.speed_scaler = RobustScaler() if fit_scalers else speed_scaler
            self.segments_df['speed_scaled'] = 0.0

        self.segments_df['seq_len'] = 1

    def __len__(self):
        return len(self.segments_df)

    def __getitem__(self, idx):
        row = self.segments_df.iloc[idx]
        seg_type_idx = int(row['seg_type_idx'])
        seq_len = int(row['seq_len'])

        temporal = torch.FloatTensor([
            float(row['hour_sin']),
            float(row['hour_cos']),
            float(row['dow_sin']),
            float(row['dow_cos']),
            float(row['speed_scaled']),
        ])

        target = torch.FloatTensor([float(row['duration_scaled'])])
        return seg_type_idx, temporal, target, seq_len


# =============================================================================
# ENHANCED DATASET
# =============================================================================

class EnhancedSegmentDataset(SegmentDataset):
    def __init__(self, segments_df, segment_types,
                 fit_scalers: bool = True,
                 target_scaler: RobustScaler = None,
                 speed_scaler: RobustScaler = None,
                 operational_scaler: RobustScaler = None,
                 weather_scaler: RobustScaler = None):
        super().__init__(segments_df, segment_types, fit_scalers, target_scaler, speed_scaler)

        operational_cols = ['arrivalDelay', 'departureDelay']

        for col in operational_cols:
            if col not in self.segments_df.columns:
                if col == 'congestion':
                    if 'distance_m' in self.segments_df.columns:
                        avg_speed_mps = 8.33
                        expected_duration = self.segments_df['distance_m'] / avg_speed_mps
                        expected_duration = expected_duration.replace(0, np.nan)
                        self.segments_df['congestion'] = (
                                self.segments_df['duration_sec'] / expected_duration
                        ).fillna(1.0).clip(0.1, 5.0)
                    else:
                        self.segments_df['congestion'] = 1.0
                else:
                    self.segments_df[col] = 0.0

        if fit_scalers:
            print(f"\n🔍 DEBUG: Operational features BEFORE RobustScaler:")
            for col in operational_cols:
                if col in self.segments_df.columns:
                    vals = self.segments_df[col].values
                    print(f"   {col}: min={vals.min():.4f}, max={vals.max():.4f}, "
                          f"mean={vals.mean():.4f}, std={vals.std():.4f}, unique={len(np.unique(vals))}")

        operational_data = self.segments_df[operational_cols].copy()
        operational_data = operational_data.replace([np.inf, -np.inf], np.nan)
        operational_data = operational_data.fillna(operational_data.median())

        if fit_scalers:
            self.operational_scaler = RobustScaler()
            operational_scaled = self.operational_scaler.fit_transform(operational_data)
        else:
            if operational_scaler is None:
                raise ValueError("operational_scaler must be provided")
            self.operational_scaler = operational_scaler
            operational_scaled = self.operational_scaler.transform(operational_data)

        if fit_scalers:
            print(f"\n🔍 DEBUG: Operational features AFTER RobustScaler:")
            for i, col in enumerate(operational_cols):
                vals = operational_scaled[:, i]
                print(f"   {col}_scaled: min={vals.min():.4f}, max={vals.max():.4f}, "
                      f"mean={vals.mean():.4f}, std={vals.std():.4f}, unique={len(np.unique(vals))}")

        for i, col in enumerate(operational_cols):
            self.segments_df[f'{col}_scaled'] = operational_scaled[:, i]

        weather_cols = ['temperature_2m', 'apparent_temperature',
                        'windspeed_10m', 'windgusts_10m', 'winddirection_10m']

        for col in weather_cols:
            if col not in self.segments_df.columns:
                defaults = {
                    'temperature_2m': 0.0,
                    'apparent_temperature': 0.0,
                    'windspeed_10m': 0.0,
                    'windgusts_10m': 0.0,
                    'winddirection_10m': 0.0
                }
                self.segments_df[col] = defaults.get(col, 0.0)

        if fit_scalers:
            print(f"\n🔍 DEBUG: Weather features (using RAW - already scaled in CSV):")
            for col in weather_cols:
                if col in self.segments_df.columns:
                    vals = self.segments_df[col].values
                    print(f"   {col}:")
                    print(f"      min={vals.min():.4f}, max={vals.max():.4f}, "
                          f"mean={vals.mean():.4f}, std={vals.std():.4f}")
                    print(f"      unique values: {len(np.unique(vals))}")

        weather_data = self.segments_df[weather_cols].copy()
        weather_data = weather_data.replace([np.inf, -np.inf], np.nan)
        weather_data = weather_data.fillna(0.0)

        if fit_scalers:
            self.weather_scaler = RobustScaler()
            self.weather_scaler.fit(weather_data)
            weather_scaled = weather_data.values
        else:
            if weather_scaler is None:
                raise ValueError("weather_scaler must be provided when fit_scalers=False")
            self.weather_scaler = weather_scaler
            weather_scaled = weather_data.values

        for i, col in enumerate(weather_cols):
            self.segments_df[f'{col}_scaled'] = weather_scaled[:, i]

        self.operational_cols_scaled = [f'{col}_scaled' for col in operational_cols]
        self.weather_cols_scaled = [f'{col}_scaled' for col in weather_cols]

    def __getitem__(self, idx):
        row = self.segments_df.iloc[idx]
        seg_type_idx = int(row['seg_type_idx'])
        seq_len = int(row['seq_len'])

        temporal = torch.FloatTensor([
            float(row['hour_sin']), float(row['hour_cos']),
            float(row['dow_sin']), float(row['dow_cos']),
            float(row['speed_scaled']),
        ])

        operational = torch.FloatTensor([float(row[col]) for col in self.operational_cols_scaled])
        weather = torch.FloatTensor([float(row[col]) for col in self.weather_cols_scaled])
        target = torch.FloatTensor([float(row['duration_scaled'])])

        return seg_type_idx, temporal, operational, weather, target, seq_len


# =============================================================================
# MODELS
# =============================================================================

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.3, alpha=0.2):
        super().__init__()
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = dropout

    def forward(self, h, adj):
        batch_size, num_nodes, _ = h.size()
        if not isinstance(adj, torch.Tensor):
            adj = torch.FloatTensor(adj)
        adj = adj.to(h.device)
        Wh = torch.matmul(h, self.W)
        Wh_i = Wh.unsqueeze(2).repeat(1, 1, num_nodes, 1)
        Wh_j = Wh.unsqueeze(1).repeat(1, num_nodes, 1, 1)
        a_input = torch.cat([Wh_i, Wh_j], dim=3)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj.unsqueeze(0) > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)
        return h_prime


class MultiRelationalGAT(nn.Module):
    def __init__(self, n_heads, in_features, out_per_head, dropout=0.3):
        super().__init__()
        self.gat_heads = nn.ModuleList([
            GraphAttentionLayer(in_features, out_per_head, dropout)
            for _ in range(n_heads)
        ])
        self.out_proj = nn.Linear(out_per_head * n_heads, out_per_head)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, adj_list):
        head_outputs = [gat(h, adj) for gat, adj in zip(self.gat_heads, adj_list)]
        h_concat = torch.cat(head_outputs, dim=2)
        h_out = self.out_proj(h_concat)
        h_out = F.elu(h_out)
        return self.dropout(h_out)


class HistoricalEmbedding(nn.Module):
    def __init__(self, num_segments, embed_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(num_segments, embed_dim)
        nn.init.normal_(self.embedding.weight, mean=0, std=0.1)

    def forward(self, segment_ids):
        return self.embedding(segment_ids)


class MAGTTE(nn.Module):
    def __init__(self, num_nodes, n_heads=3, node_embed_dim=32,
                 gat_hidden=32, lstm_hidden=64, historical_dim=16, dropout=0.3):
        super().__init__()
        self.node_embedding = nn.Embedding(num_nodes, node_embed_dim)
        nn.init.normal_(self.node_embedding.weight, mean=0, std=0.1)
        self.multi_gat = MultiRelationalGAT(n_heads, node_embed_dim, gat_hidden, dropout)
        self.historical_embed = HistoricalEmbedding(num_nodes, historical_dim)

        fusion_in = gat_hidden + historical_dim + 5
        fusion_out = max(fusion_in // 2, 16)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, fusion_out),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
            input_size=fusion_out,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            dropout=0.0
        )

        self.regression_head = nn.Sequential(
            nn.Linear(lstm_hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

        self.register_buffer('adj_geo', None)
        self.register_buffer('adj_dist', None)
        self.register_buffer('adj_soc', None)

    def set_adjacency_matrices(self, adj_geo, adj_dist, adj_soc):
        self.register_buffer('adj_geo', torch.FloatTensor(adj_geo))
        self.register_buffer('adj_dist', torch.FloatTensor(adj_dist))
        self.register_buffer('adj_soc', torch.FloatTensor(adj_soc))

    def forward(self, seg_indices, temporal_features):
        all_nodes = self.node_embedding.weight.unsqueeze(0)
        spatial_all = self.multi_gat(all_nodes, [self.adj_geo, self.adj_dist, self.adj_soc])
        spatial_all = spatial_all.squeeze(0)
        segment_spatial = spatial_all[seg_indices]
        segment_historical = self.historical_embed(seg_indices)
        combined = torch.cat([segment_spatial, segment_historical, temporal_features], dim=1)
        fused = self.fusion(combined)
        lstm_out, _ = self.lstm(fused.unsqueeze(1))
        lstm_out = lstm_out.squeeze(1)
        return self.regression_head(lstm_out)


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


class LSTMWithGlobalTemporalAttention(nn.Module):
    def __init__(self, spatial_dim, operational_dim, weather_dim,
                 temporal_dim=5, hidden_dim=128, n_layers=1, dropout=0.1, out_dim=1):
        super().__init__()
        lstm_input_dim = spatial_dim + operational_dim + weather_dim + temporal_dim
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.0
        )
        self.global_attention = GlobalTemporalAttention(feature_dim=hidden_dim, dropout=dropout)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, out_dim)
        )
        for layer in self.fusion:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, 0, 0.01)
                nn.init.zeros_(layer.bias)

    def forward(self, seq_x):
        batch_size = seq_x.size(0)
        seq_x = seq_x.reshape(batch_size, 1, -1)
        lstm_out, _ = self.lstm(seq_x)
        attn_out, _ = self.global_attention(lstm_out)
        attn_last = attn_out[:, -1, :]
        out = self.fusion(attn_last)
        return out


class MAGNN_LSTM(nn.Module):
    def __init__(self, magnn_model, spatial_dim, operational_dim, weather_dim,
                 temporal_dim=5, lstm_hidden=32, lstm_layers=1, dropout=0.4, freeze_magnn=True):
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
        all_nodes = self.magnn.node_embedding.weight.unsqueeze(0)
        spatial_all = self.magnn.multi_gat(
            all_nodes,
            [self.magnn.adj_geo, self.magnn.adj_dist, self.magnn.adj_soc]
        )
        spatial_all = spatial_all.squeeze(0)
        spatial_embeddings = spatial_all[seg_indices]
        return spatial_embeddings

    def forward(self, seg_indices, temporal_features, operational_features, weather_features):
        with torch.no_grad():
            magnn_baseline = self.magnn(seg_indices, temporal_features)
        if self.freeze_magnn:
            with torch.no_grad():
                spatial_embeddings = self.get_magnn_embeddings(seg_indices)
        else:
            spatial_embeddings = self.get_magnn_embeddings(seg_indices)
        combined_features = torch.cat([
            spatial_embeddings,
            operational_features,
            weather_features,
            temporal_features
        ], dim=1)
        seq_features = combined_features.unsqueeze(1)
        lstm_correction = self.lstm_model(seq_features)
        final_prediction = magnn_baseline + 0.5 * lstm_correction
        return final_prediction


# =============================================================================
# NEW: MAGNN-LSTM-MTL MODEL
# =============================================================================

class MAGNN_LSTM_MTL(nn.Module):
    """
    MAGNN-LSTM with Multi-Task Learning head.

    Architecture:
    1. MAGNN provides baseline prediction + spatial embeddings
    2. LSTM processes spatial + operational + weather + temporal features
    3. MTL head performs dual prediction:
       - Task 1: Individual segment predictions
       - Task 2: Collective path prediction (with L2-norm attention)
    4. Final output: Residual learning (MAGNN baseline + MTL correction)
    """

    def __init__(self, magnn_model, spatial_dim, operational_dim, weather_dim,
                 temporal_dim=5, lstm_hidden=32, lstm_layers=1, dropout=0.4,
                 mtl_segment_hidden=64, mtl_path_hidden=128,
                 freeze_magnn=True):
        super().__init__()

        self.magnn = magnn_model
        self.freeze_magnn = freeze_magnn

        if freeze_magnn:
            for param in self.magnn.parameters():
                param.requires_grad = False

        # LSTM for feature processing
        self.lstm_model = LSTMWithGlobalTemporalAttention(
            spatial_dim=spatial_dim,
            operational_dim=operational_dim,
            weather_dim=weather_dim,
            temporal_dim=temporal_dim,
            hidden_dim=lstm_hidden,
            n_layers=lstm_layers,
            dropout=dropout,
            out_dim=lstm_hidden  # Output features, not final prediction
        )

        # MTL head for dual-task prediction
        self.mtl_head = MTLHead(
            feature_dim=lstm_hidden,
            segment_hidden=mtl_segment_hidden,
            path_hidden=mtl_path_hidden,
            dropout=dropout
        )

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
                return_mtl_components=False):
        """
        Args:
            seg_indices: [batch] - segment indices
            temporal_features: [batch, 5]
            operational_features: [batch, operational_dim]
            weather_features: [batch, weather_dim]
            return_mtl_components: if True, return MTL individual predictions and attention

        Returns:
            If return_mtl_components=False:
                final_prediction: [batch, 1]
            If return_mtl_components=True:
                (final_prediction, individual_preds, path_pred, attention_weights)
        """
        # Get MAGNN baseline
        with torch.no_grad():
            magnn_baseline = self.magnn(seg_indices, temporal_features)

        # Get spatial embeddings from MAGNN
        if self.freeze_magnn:
            with torch.no_grad():
                spatial_embeddings = self.get_magnn_embeddings(seg_indices)
        else:
            spatial_embeddings = self.get_magnn_embeddings(seg_indices)

        # Concatenate all features
        combined_features = torch.cat([
            spatial_embeddings,
            operational_features,
            weather_features,
            temporal_features
        ], dim=1)

        # Process through LSTM to get rich features
        seq_features = combined_features.unsqueeze(1)
        lstm_features = self.lstm_model(seq_features)  # [batch, lstm_hidden]

        # Convert to sequence format for MTL
        segment_features = extract_segment_features(lstm_features, seq_len=1)  # [batch, 1, lstm_hidden]

        # MTL prediction
        if return_mtl_components:
            individual_preds, path_pred, attention_weights = self.mtl_head(
                segment_features, return_components=True
            )

            # Final prediction: MAGNN baseline + MTL correction
            final_prediction = magnn_baseline + 0.5 * path_pred

            return final_prediction, individual_preds, path_pred, attention_weights
        else:
            path_pred = self.mtl_head(segment_features, return_components=False)

            # Final prediction: MAGNN baseline + MTL correction
            final_prediction = magnn_baseline + 0.5 * path_pred

            return final_prediction


class SimpleMLP(nn.Module):
    def __init__(self, num_segments, embed_dim=32, dropout=0.3):
        super().__init__()
        self.seg_embed = nn.Embedding(num_segments, embed_dim)
        nn.init.normal_(self.seg_embed.weight, 0, 0.1)
        in_dim = embed_dim + 5
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, seg_indices, temporal):
        emb = self.seg_embed(seg_indices)
        x = torch.cat([emb, temporal], dim=1)
        return self.net(x)


# =============================================================================
# COLLATE FUNCTIONS
# =============================================================================

def masked_collate_fn(batch):
    seg_indices = torch.LongTensor([item[0] for item in batch])
    targets = torch.stack([item[2] for item in batch])
    lengths = torch.LongTensor([item[3] for item in batch])
    temporal_seqs = [item[1].unsqueeze(0) for item in batch]
    max_len = int(lengths.max().item())
    temporal_dim = temporal_seqs[0].size(-1)
    temporal_pad = torch.zeros(len(batch), max_len, temporal_dim)
    for i, (seq, slen) in enumerate(zip(temporal_seqs, lengths)):
        slen = int(slen.item())
        temporal_pad[i, :slen, :] = seq[:slen]
    mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)
    return seg_indices, temporal_pad, targets, lengths, mask


def enhanced_collate_fn(batch):
    seg_indices = torch.LongTensor([item[0] for item in batch])
    targets = torch.stack([item[4] for item in batch])
    lengths = torch.LongTensor([item[5] for item in batch])
    max_len = int(lengths.max().item())

    temporal_seqs = [item[1].unsqueeze(0) for item in batch]
    temporal_dim = temporal_seqs[0].size(-1)
    temporal_pad = torch.zeros(len(batch), max_len, temporal_dim)
    for i, (seq, slen) in enumerate(zip(temporal_seqs, lengths)):
        slen = int(slen.item())
        temporal_pad[i, :slen, :] = seq[:slen]

    operational_seqs = [item[2].unsqueeze(0) for item in batch]
    operational_dim = operational_seqs[0].size(-1)
    operational_pad = torch.zeros(len(batch), max_len, operational_dim)
    for i, (seq, slen) in enumerate(zip(operational_seqs, lengths)):
        slen = int(slen.item())
        operational_pad[i, :slen, :] = seq[:slen]

    weather_seqs = [item[3].unsqueeze(0) for item in batch]
    weather_dim = weather_seqs[0].size(-1)
    weather_pad = torch.zeros(len(batch), max_len, weather_dim)
    for i, (seq, slen) in enumerate(zip(weather_seqs, lengths)):
        slen = int(slen.item())
        weather_pad[i, :slen, :] = seq[:slen]

    mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)
    return seg_indices, temporal_pad, operational_pad, weather_pad, targets, lengths, mask


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

def _safe_batch(seg_idx, temporal, target):
    t_flat = temporal.reshape(temporal.size(0), -1)
    valid = ~torch.isnan(t_flat).any(dim=1)
    valid &= ~torch.isinf(t_flat).any(dim=1)
    valid &= ~torch.isnan(target).any(dim=1)
    if valid.sum() == 0:
        return None, None, None
    return seg_idx[valid], temporal[valid], target[valid]


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in dataloader:
        seg_idx, temporal_pad, target, lengths, mask = batch
        temporal = temporal_pad.squeeze(1)
        seg_idx, temporal, target = _safe_batch(
            seg_idx.to(device), temporal.to(device), target.to(device)
        )
        if seg_idx is None:
            continue
        predictions = model(seg_idx, temporal)
        loss = criterion(predictions, target)
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def evaluate(model, dataloader, criterion, device, scaler):
    model.eval()
    predictions_list = []
    targets_list = []
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in dataloader:
            seg_idx, temporal_pad, target, lengths, mask = batch
            temporal = temporal_pad.squeeze(1)
            seg_idx, temporal, target = _safe_batch(
                seg_idx.to(device), temporal.to(device), target.to(device)
            )
            if seg_idx is None:
                continue
            predictions = model(seg_idx, temporal)
            loss = criterion(predictions, target)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            total_loss += loss.item()
            n_batches += 1
            predictions_list.append(predictions.cpu().numpy())
            targets_list.append(target.cpu().numpy())
    if not predictions_list:
        return {'loss': float('nan'), 'r2': float('nan'),
                'rmse': float('nan'), 'mae': float('nan'),
                'mape': float('nan'), 'preds': [], 'actual': []}
    preds = np.concatenate(predictions_list)
    targets = np.concatenate(targets_list)
    preds_orig = scaler.inverse_transform(preds)
    targets_orig = scaler.inverse_transform(targets)
    r2 = r2_score(targets_orig, preds_orig)
    rmse = np.sqrt(mean_squared_error(targets_orig, preds_orig))
    mae = mean_absolute_error(targets_orig, preds_orig)
    mask = targets_orig.flatten() > 0
    mape = (np.mean(np.abs((targets_orig.flatten()[mask] - preds_orig.flatten()[mask]) /
                           targets_orig.flatten()[mask])) * 100 if mask.any() else float('nan'))
    return {
        'loss': total_loss / max(n_batches, 1),
        'r2': float(r2),
        'rmse': float(rmse),
        'mae': float(mae),
        'mape': float(mape),
        'preds': preds_orig.flatten().tolist(),
        'actual': targets_orig.flatten().tolist(),
    }


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_magtte(train_loader, val_loader, test_loader,
                 adj_geo, adj_dist, adj_soc,
                 segment_types, scaler,
                 output_folder, device, cfg):
    print_section("MAGTTE + GAT TRAINING")
    num_segments = len(segment_types)
    print(f"   Segments: {num_segments}, Epochs: {cfg.n_epochs}, Device: {device}")

    model = MAGTTE(num_segments, cfg.n_heads, cfg.node_embed_dim, cfg.gat_hidden,
                   cfg.lstm_hidden, cfg.historical_dim, cfg.dropout).to(device)
    model.set_adjacency_matrices(adj_geo, adj_dist, adj_soc)

    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=cfg.lr_scheduler_factor,
                                                     patience=cfg.lr_scheduler_patience)
    criterion = nn.SmoothL1Loss()
    best_val_loss = float('inf')
    patience_counter = 0
    best_ckpt = os.path.join(output_folder, 'magtte_best.pth')

    print()
    for epoch in range(1, cfg.n_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device, scaler)
        val_loss = val_metrics['loss']
        scheduler.step(val_loss if not np.isnan(val_loss) else best_val_loss)

        if epoch % max(1, cfg.n_epochs // 5) == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3}/{cfg.n_epochs}  train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  val_R²={val_metrics['r2']:.4f}")

        if not np.isnan(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_ckpt)
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stopping_patience:
                print(f"\n  ⏹️  Early stopping at epoch {epoch}")
                break

    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=device))

    print_section("MAGTTE — FINAL RESULTS")

    def _eval(loader, name):
        m = evaluate(model, loader, criterion, device, scaler)
        print(f"   {name:<6}  R²={m['r2']:.4f}  RMSE={m['rmse']:.2f}s  "
              f"MAE={m['mae']:.2f}s  MAPE={m['mape']:.2f}%")
        return m

    results = {'Train': _eval(train_loader, 'Train'), 'Val': _eval(val_loader, 'Val'),
               'Test': _eval(test_loader, 'Test')}

    test_res = results.get('Test', {})
    if test_res.get('preds'):
        print(f"\n{'Idx':>4}  {'Actual(s)':>10}  {'Pred(s)':>10}  {'Error(s)':>9}  {'Error%':>7}")
        print("  " + "-" * 48)
        for i in range(min(20, len(test_res['actual']))):
            a, p = test_res['actual'][i], test_res['preds'][i]
            err = p - a
            pct = (err / a * 100) if a > 0 else 0.0
            print(f"  {i:>3}  {a:>10.2f}  {p:>10.2f}  {err:>9.2f}  {pct:>6.2f}%")

    return results, model


def train_magnn_lstm(train_loader, val_loader, test_loader,
                     adj_geo, adj_dist, adj_soc,
                     segment_types, scaler,
                     output_folder, device, cfg,
                     pretrained_magnn_path=None,
                     freeze_magnn=True):
    print_section("MAGNN-LSTM TRAINING")
    num_segments = len(segment_types)

    magnn_base = MAGTTE(num_segments, cfg.n_heads, cfg.node_embed_dim, cfg.gat_hidden,
                        cfg.lstm_hidden, cfg.historical_dim, cfg.dropout).to(device)
    magnn_base.set_adjacency_matrices(adj_geo, adj_dist, adj_soc)

    if pretrained_magnn_path and os.path.exists(pretrained_magnn_path):
        magnn_base.load_state_dict(torch.load(pretrained_magnn_path, map_location=device))
        print(f"   ✓ Loaded pre-trained MAGNN")

    model = MAGNN_LSTM(magnn_base, cfg.gat_hidden, 2, 5, 5, 32, 1, 0.1, True).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Trainable params: {n_params:,}")

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=cfg.learning_rate * 0.1, weight_decay=cfg.weight_decay * 5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.SmoothL1Loss()
    best_val_loss = float('inf')
    patience_counter = 0
    best_ckpt = os.path.join(output_folder, 'magnn_lstm_best.pth')

    print()
    for epoch in range(1, cfg.n_epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            seg_idx, temporal, operational, weather, target, lengths, mask = batch
            seg_idx = seg_idx.to(device)
            temporal = temporal.squeeze(1).to(device)
            operational = operational.squeeze(1).to(device)
            weather = weather.squeeze(1).to(device)
            target = target.to(device)
            predictions = model(seg_idx, temporal, operational, weather)
            loss = criterion(predictions, target)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        train_loss /= max(n_batches, 1)

        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                seg_idx, temporal, operational, weather, target, lengths, mask = batch
                seg_idx = seg_idx.to(device)
                temporal = temporal.squeeze(1).to(device)
                operational = operational.squeeze(1).to(device)
                weather = weather.squeeze(1).to(device)
                target = target.to(device)
                predictions = model(seg_idx, temporal, operational, weather)
                loss = criterion(predictions, target)
                if not torch.isnan(loss) and not torch.isinf(loss):
                    val_loss += loss.item()
                    n_val += 1
                    val_preds.append(predictions.cpu().numpy())
                    val_targets.append(target.cpu().numpy())
        val_loss /= max(n_val, 1)
        scheduler.step(val_loss)

        if val_preds:
            vp = scaler.inverse_transform(np.concatenate(val_preds))
            vt = scaler.inverse_transform(np.concatenate(val_targets))
            val_r2 = r2_score(vt, vp)
        else:
            val_r2 = float('nan')

        if epoch % max(1, cfg.n_epochs // 5) == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3}/{cfg.n_epochs}  train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  val_R²={val_r2:.4f}")

        if not np.isnan(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_ckpt)
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stopping_patience:
                print(f"\n  ⏹️  Early stopping at epoch {epoch}")
                break

    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=device))

    print_section("MAGNN-LSTM — FINAL RESULTS")

    def _eval(loader, name):
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in loader:
                seg_idx, temporal, operational, weather, target, lengths, mask = batch
                seg_idx = seg_idx.to(device)
                temporal = temporal.squeeze(1).to(device)
                operational = operational.squeeze(1).to(device)
                weather = weather.squeeze(1).to(device)
                pred = model(seg_idx, temporal, operational, weather)
                preds.append(pred.cpu().numpy())
                targets.append(target.cpu().numpy())
        if not preds:
            return {}
        p = scaler.inverse_transform(np.concatenate(preds))
        t = scaler.inverse_transform(np.concatenate(targets))
        r2 = r2_score(t, p)
        rmse = np.sqrt(mean_squared_error(t, p))
        mae = mean_absolute_error(t, p)
        mask = t.flatten() > 0
        mape = np.mean(np.abs((t.flatten()[mask] - p.flatten()[mask]) /
                              t.flatten()[mask])) * 100 if mask.any() else float('nan')
        print(f"   {name:<6}  R²={r2:.4f}  RMSE={rmse:.2f}s  MAE={mae:.2f}s  MAPE={mape:.2f}%")
        return {'r2': r2, 'rmse': rmse, 'mae': mae, 'mape': mape,
                'preds': p.flatten().tolist(), 'actual': t.flatten().tolist()}

    results = {'Train': _eval(train_loader, 'Train'), 'Val': _eval(val_loader, 'Val'),
               'Test': _eval(test_loader, 'Test')}

    test_res = results.get('Test', {})
    if test_res.get('preds'):
        print(f"\n{'Idx':>4}  {'Actual(s)':>10}  {'Pred(s)':>10}  {'Error(s)':>9}  {'Error%':>7}")
        print("  " + "-" * 48)
        for i in range(min(20, len(test_res['actual']))):
            a, p = test_res['actual'][i], test_res['preds'][i]
            err = p - a
            epct = (err / a * 100) if a > 0 else 0.0
            print(f"  {i:>3}  {a:>10.2f}  {p:>10.2f}  {err:>9.2f}  {epct:>6.2f}%")

    return results, model


# =============================================================================
# NEW: MAGNN-LSTM-MTL TRAINING FUNCTION
# =============================================================================

def train_magnn_lstm_mtl(train_loader, val_loader, test_loader,
                         adj_geo, adj_dist, adj_soc,
                         segment_types, scaler,
                         output_folder, device, cfg,
                         pretrained_magnn_path=None,
                         freeze_magnn=True):
    """Train MAGNN-LSTM-MTL model with multi-task learning."""
    print_section("MAGNN-LSTM-MTL TRAINING (Multi-Task Learning)")
    num_segments = len(segment_types)

    # Load pre-trained MAGNN
    magnn_base = MAGTTE(num_segments, cfg.n_heads, cfg.node_embed_dim, cfg.gat_hidden,
                        cfg.lstm_hidden, cfg.historical_dim, cfg.dropout).to(device)
    magnn_base.set_adjacency_matrices(adj_geo, adj_dist, adj_soc)

    if pretrained_magnn_path and os.path.exists(pretrained_magnn_path):
        magnn_base.load_state_dict(torch.load(pretrained_magnn_path, map_location=device))
        print(f"   ✓ Loaded pre-trained MAGNN from {pretrained_magnn_path}")

    # Create MTL model
    model = MAGNN_LSTM_MTL(
        magnn_base,
        cfg.gat_hidden,
        2,  # operational_dim
        5,  # weather_dim
        5,  # temporal_dim
        32,  # lstm_hidden
        1,  # lstm_layers
        0.1,  # dropout
        cfg.mtl_segment_hidden,
        cfg.mtl_path_hidden,
        freeze_magnn
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Trainable params: {n_params:,}")
    print(f"   MTL lambda: {cfg.mtl_lambda} (individual vs collective task balance)")

    # Optimizer and scheduler
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.learning_rate * 0.1,
        weight_decay=cfg.weight_decay * 5
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # MTL loss with lambda weighting
    mtl_criterion = MTLLoss(lambda_weight=cfg.mtl_lambda, criterion=nn.SmoothL1Loss())

    best_val_loss = float('inf')
    patience_counter = 0
    best_ckpt = os.path.join(output_folder, 'magnn_lstm_mtl_best.pth')

    print()
    for epoch in range(1, cfg.n_epochs + 1):
        model.train()
        train_loss = 0.0
        train_individual_loss = 0.0
        train_collective_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            seg_idx, temporal, operational, weather, target, lengths, mask = batch
            seg_idx = seg_idx.to(device)
            temporal = temporal.squeeze(1).to(device)
            operational = operational.squeeze(1).to(device)
            weather = weather.squeeze(1).to(device)
            target = target.to(device)

            # Forward pass with MTL components
            final_pred, individual_preds, path_pred, attention_weights = model(
                seg_idx, temporal, operational, weather, return_mtl_components=True
            )

            # Calculate MTL loss
            loss, loss_dict = mtl_criterion(individual_preds, path_pred, target)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss_dict['total']
            train_individual_loss += loss_dict['individual']
            train_collective_loss += loss_dict['collective']
            n_batches += 1

        train_loss /= max(n_batches, 1)
        train_individual_loss /= max(n_batches, 1)
        train_collective_loss /= max(n_batches, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        val_individual_loss = 0.0
        val_collective_loss = 0.0
        val_preds, val_targets = [], []
        n_val = 0

        with torch.no_grad():
            for batch in val_loader:
                seg_idx, temporal, operational, weather, target, lengths, mask = batch
                seg_idx = seg_idx.to(device)
                temporal = temporal.squeeze(1).to(device)
                operational = operational.squeeze(1).to(device)
                weather = weather.squeeze(1).to(device)
                target = target.to(device)

                final_pred, individual_preds, path_pred, attention_weights = model(
                    seg_idx, temporal, operational, weather, return_mtl_components=True
                )

                loss, loss_dict = mtl_criterion(individual_preds, path_pred, target)

                if not torch.isnan(loss) and not torch.isinf(loss):
                    val_loss += loss_dict['total']
                    val_individual_loss += loss_dict['individual']
                    val_collective_loss += loss_dict['collective']
                    n_val += 1
                    val_preds.append(final_pred.cpu().numpy())
                    val_targets.append(target.cpu().numpy())

        val_loss /= max(n_val, 1)
        val_individual_loss /= max(n_val, 1)
        val_collective_loss /= max(n_val, 1)
        scheduler.step(val_loss)

        if val_preds:
            vp = scaler.inverse_transform(np.concatenate(val_preds))
            vt = scaler.inverse_transform(np.concatenate(val_targets))
            val_r2 = r2_score(vt, vp)
        else:
            val_r2 = float('nan')

        if epoch % max(1, cfg.n_epochs // 5) == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3}/{cfg.n_epochs}  "
                  f"train_loss={train_loss:.4f} (ind:{train_individual_loss:.4f} col:{train_collective_loss:.4f})  "
                  f"val_loss={val_loss:.4f}  val_R²={val_r2:.4f}")

        if not np.isnan(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_ckpt)
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stopping_patience:
                print(f"\n  ⏹️  Early stopping at epoch {epoch}")
                break

    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=device))

    print_section("MAGNN-LSTM-MTL — FINAL RESULTS")

    def _eval(loader, name):
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in loader:
                seg_idx, temporal, operational, weather, target, lengths, mask = batch
                seg_idx = seg_idx.to(device)
                temporal = temporal.squeeze(1).to(device)
                operational = operational.squeeze(1).to(device)
                weather = weather.squeeze(1).to(device)
                pred = model(seg_idx, temporal, operational, weather, return_mtl_components=False)
                preds.append(pred.cpu().numpy())
                targets.append(target.cpu().numpy())
        if not preds:
            return {}
        p = scaler.inverse_transform(np.concatenate(preds))
        t = scaler.inverse_transform(np.concatenate(targets))
        r2 = r2_score(t, p)
        rmse = np.sqrt(mean_squared_error(t, p))
        mae = mean_absolute_error(t, p)
        mask = t.flatten() > 0
        mape = np.mean(np.abs((t.flatten()[mask] - p.flatten()[mask]) /
                              t.flatten()[mask])) * 100 if mask.any() else float('nan')
        print(f"   {name:<6}  R²={r2:.4f}  RMSE={rmse:.2f}s  MAE={mae:.2f}s  MAPE={mape:.2f}%")
        return {'r2': r2, 'rmse': rmse, 'mae': mae, 'mape': mape,
                'preds': p.flatten().tolist(), 'actual': t.flatten().tolist()}

    results = {'Train': _eval(train_loader, 'Train'),
               'Val': _eval(val_loader, 'Val'),
               'Test': _eval(test_loader, 'Test')}

    test_res = results.get('Test', {})
    if test_res.get('preds'):
        print(f"\n{'Idx':>4}  {'Actual(s)':>10}  {'Pred(s)':>10}  {'Error(s)':>9}  {'Error%':>7}")
        print("  " + "-" * 48)
        for i in range(min(20, len(test_res['actual']))):
            a, p = test_res['actual'][i], test_res['preds'][i]
            err = p - a
            epct = (err / a * 100) if a > 0 else 0.0
            print(f"  {i:>3}  {a:>10.2f}  {p:>10.2f}  {err:>9.2f}  {epct:>6.2f}%")

    return results, model


def train_simple(train_loader, val_loader, test_loader, segment_types, scaler,
                 output_folder, device, n_epochs=50, lr=0.001, dropout=0.3):
    print_section("SIMPLE MLP TRAINING")
    num_segments = len(segment_types)
    model = SimpleMLP(num_segments, embed_dim=32, dropout=dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.SmoothL1Loss()
    best_val_loss = float('inf')

    print(f"   Segments: {num_segments}, Epochs: {n_epochs}, Device: {device}\n")

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            seg_idx, temporal_pad, target, lengths, mask = batch
            temporal = temporal_pad.squeeze(1)
            pred = model(seg_idx.to(device), temporal.to(device))
            loss = criterion(pred, target.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                seg_idx, temporal_pad, target, lengths, mask = batch
                temporal = temporal_pad.squeeze(1)
                pred = model(seg_idx.to(device), temporal.to(device))
                val_loss += criterion(pred, target.to(device)).item()
        val_loss /= len(val_loader)

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:>3}/{n_epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(output_folder, 'simple_best.pth'))

    return {}, model