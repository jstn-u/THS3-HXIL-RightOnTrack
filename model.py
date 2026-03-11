"""
model.py - UPDATED WITH BUG FIXES (NO MTL)
===========================================
✅ Fixed weather scaling bug (was using raw values)
✅ Updated operational features: 4 features (arrivalDelay, departureDelay, is_weekend, is_peak_hour)
✅ Updated weather features: 8 features (all weather columns)
✅ All datasets updated to handle binary flags
✅ MTL components removed for cleaner codebase

All neural network components for MAGNN and MAGNN-LSTM-Residual
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
    """✅ FIXED: Smart weather scaling + don't scale binary flags"""

    def __init__(self, segments_df, segment_types,
                 fit_scalers: bool = True,
                 target_scaler: RobustScaler = None,
                 speed_scaler: RobustScaler = None,
                 operational_scaler: RobustScaler = None,
                 weather_scaler: RobustScaler = None):
        super().__init__(segments_df, segment_types, fit_scalers, target_scaler, speed_scaler)

        # ✅ FIX: Separate continuous and binary operational features
        continuous_operational_cols = ['arrivalDelay', 'departureDelay']
        binary_flag_cols = ['is_weekend', 'is_peak_hour']

        for col in continuous_operational_cols + binary_flag_cols:
            if col not in self.segments_df.columns:
                self.segments_df[col] = 0.0

        # Scale ONLY continuous operational features
        continuous_data = self.segments_df[continuous_operational_cols].copy()
        continuous_data = continuous_data.replace([np.inf, -np.inf], np.nan)
        continuous_data = continuous_data.fillna(continuous_data.median())

        if fit_scalers:
            self.operational_scaler = RobustScaler()
            continuous_scaled = self.operational_scaler.fit_transform(continuous_data)

            print(f"\n✅ Operational features (mixed scaling):")
            print(f"   Continuous (RobustScaler): {continuous_operational_cols}")
            for i, col in enumerate(continuous_operational_cols):
                vals = continuous_scaled[:, i]
                print(f"     {col}: min={vals.min():.2f}, max={vals.max():.2f}, std={vals.std():.2f}")
        else:
            if operational_scaler is None:
                raise ValueError("operational_scaler must be provided")
            self.operational_scaler = operational_scaler
            continuous_scaled = self.operational_scaler.transform(continuous_data)

        # Store scaled continuous features
        for i, col in enumerate(continuous_operational_cols):
            self.segments_df[f'{col}_scaled'] = continuous_scaled[:, i]

        # ✅ FIX: Keep binary flags as-is (0 or 1, don't scale!)
        if fit_scalers:
            print(f"   Binary (no scaling): {binary_flag_cols}")

        for col in binary_flag_cols:
            self.segments_df[f'{col}_scaled'] = self.segments_df[col].values
            if fit_scalers:
                vals = self.segments_df[col].values
                unique = np.unique(vals)
                counts = np.bincount(vals.astype(int))
                print(f"     {col}: values={unique.tolist()}, dist={counts.tolist()}")

        self.operational_cols_scaled = [f'{col}_scaled' for col in continuous_operational_cols + binary_flag_cols]

        # ✅ FIX: Smart weather scaling (detect if already normalized)
        weather_cols = ['temperature_2m', 'apparent_temperature', 'precipitation',
                        'rain', 'snowfall', 'windspeed_10m', 'windgusts_10m',
                        'winddirection_10m']

        for col in weather_cols:
            if col not in self.segments_df.columns:
                self.segments_df[col] = 0.0

        weather_data = self.segments_df[weather_cols].copy()
        weather_data = weather_data.replace([np.inf, -np.inf], np.nan)
        weather_data = weather_data.fillna(0.0)

        if fit_scalers:
            # Check if weather is already normalized (z-scores)
            weather_mean = weather_data.mean().mean()
            weather_std = weather_data.std().mean()

            print(f"\n🔍 Weather feature check:")
            print(f"   Overall mean: {weather_mean:.4f}")
            print(f"   Overall std: {weather_std:.4f}")

            # If mean ≈ 0 and std ≈ 1, likely already normalized
            if abs(weather_mean) < 0.5 and 0.5 < weather_std < 1.5:
                print(f"   ✅ Weather appears already normalized (z-scores)")
                print(f"   Skipping RobustScaler to avoid double-scaling")
                self.weather_scaler = None
                weather_scaled = weather_data.values
            else:
                print(f"   🔧 Applying RobustScaler to weather")
                self.weather_scaler = RobustScaler()
                weather_scaled = self.weather_scaler.fit_transform(weather_data)
        else:
            # ✅ FIX: Set self.weather_scaler BEFORE using it
            self.weather_scaler = weather_scaler

            if self.weather_scaler is not None:
                weather_scaled = self.weather_scaler.transform(weather_data)
            else:
                # Weather was not scaled in training (already normalized)
                weather_scaled = weather_data.values

        for i, col in enumerate(weather_cols):
            self.segments_df[f'{col}_scaled'] = weather_scaled[:, i]

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