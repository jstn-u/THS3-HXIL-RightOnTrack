"""
model.py
========
All neural network components: GAT layers, MAGTTE model, training loops,
dataset class, and SimpleMLP fallback — shared across all clustering methods.

Public API:
    SegmentDataset
    GraphAttentionLayer
    MultiRelationalGAT
    HistoricalEmbedding
    MAGTTE
    masked_collate_fn
    train_magtte(train_loader, val_loader, test_loader, segments_df,
                 adj_geo, adj_dist, adj_soc, cluster_centers, config)
    SimpleMLP
    train_simple(train_loader, val_loader, test_loader, config)
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
# DATASET
# =============================================================================

class SegmentDataset(Dataset):
    """
    Dataset for segment-level predictions.

    Data-pipeline improvements applied here:
    ──────────────────────────────────────────────────────────────────────────
    1. CYCLICAL TIME ENCODING
       hour  → (sin, cos) with period 24  so hour 23 is adjacent to hour 0
       dow   → (sin, cos) with period  7
       These replace raw integer hour / day_of_week features entirely.

    2. ROBUST SCALING  (RobustScaler replaces StandardScaler)
       Fitted on the training set ONLY; reused on val/test via
       `fit_scalers=False` + passing the fitted scaler objects in.
       RobustScaler uses the median and IQR, so extreme delay outliers
       do not skew the scaling of normal observations.

    3. LSTM SEQUENCE-MASKING SUPPORT
       Every sample carries a `seq_len` field (always 1 for the current
       single-step LSTM, but the DataLoader collate function uses it to
       build a boolean padding mask when sequences are padded to a common
       length in a future multi-step extension).

    4. NaN-SAFE FEATURE FILL
       Any residual NaN in scaled features is filled with 0.0 AFTER
       scaling so that an "unknown" value maps to the median, not to the
       mean — this is the correct semantic for RobustScaler-centered data.
    ──────────────────────────────────────────────────────────────────────────
    """

    def __init__(self, segments_df, segment_types,
                 fit_scalers: bool = True,
                 target_scaler: RobustScaler = None,
                 speed_scaler:  RobustScaler = None):
        """
        Args:
            segments_df    : DataFrame of segments (train / val / test).
            segment_types  : Ordered array/list of all segment IDs from
                             the TRAINING set (defines the embedding index).
            fit_scalers    : True  → fit new RobustScalers (use for training set)
                             False → reuse `target_scaler` and `speed_scaler`
                                     (use for val and test sets)
            target_scaler  : Pre-fitted RobustScaler for duration_sec.
                             Ignored when fit_scalers=True.
            speed_scaler   : Pre-fitted RobustScaler for speed_mps.
                             Ignored when fit_scalers=True.
        """
        self.segments_df   = segments_df.copy()
        self.segment_types = list(segment_types)
        self.seg_to_idx    = {seg: i for i, seg in enumerate(self.segment_types)}

        # ── Map segment id → embedding index ─────────────────────────────
        self.segments_df['seg_type_idx'] = (
            self.segments_df['segment_id'].map(self.seg_to_idx)
        )
        n_before = len(self.segments_df)
        self.segments_df = self.segments_df.dropna(subset=['seg_type_idx']).copy()
        n_dropped = n_before - len(self.segments_df)
        if n_dropped > 0:
            print(f"   ⚠️  Dropped {n_dropped:,} rows with unseen segment types")
        self.segments_df['seg_type_idx'] = self.segments_df['seg_type_idx'].astype(int)

        # ── 1. CYCLICAL TIME ENCODING ─────────────────────────────────────
        # Period-24 for hour, period-7 for day-of-week
        self.segments_df['hour_sin'] = np.sin(2 * np.pi * self.segments_df['hour'] / 24)
        self.segments_df['hour_cos'] = np.cos(2 * np.pi * self.segments_df['hour'] / 24)
        self.segments_df['dow_sin']  = np.sin(2 * np.pi * self.segments_df['day_of_week'] / 7)
        self.segments_df['dow_cos']  = np.cos(2 * np.pi * self.segments_df['day_of_week'] / 7)

        # ── 2. ROBUST SCALING ─────────────────────────────────────────────
        # Target: duration_sec
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

        # Speed feature: also robustly scaled (outlier delays inflate raw speed)
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
            # NaN-safe fill after scaling (maps unknown → median = 0 in robust scale)
            speed_scaled = np.nan_to_num(speed_scaled, nan=0.0)
            self.segments_df['speed_scaled'] = speed_scaled.flatten()
        else:
            self.speed_scaler = RobustScaler() if fit_scalers else speed_scaler
            self.segments_df['speed_scaled'] = 0.0

        # ── 3. SEQUENCE LENGTH (masking support) ─────────────────────────
        # Currently every sample is a single-step sequence (seq_len = 1).
        # The collate function in the DataLoader can use this field to build
        # a packed/padded batch with a boolean mask when seq_len varies.
        self.segments_df['seq_len'] = 1

    def __len__(self):
        return len(self.segments_df)

    def __getitem__(self, idx):
        row = self.segments_df.iloc[idx]

        seg_type_idx = int(row['seg_type_idx'])
        seq_len      = int(row['seq_len'])

        # Temporal features: 4 cyclical components + robustly-scaled speed
        temporal = torch.FloatTensor([
            float(row['hour_sin']),
            float(row['hour_cos']),
            float(row['dow_sin']),
            float(row['dow_cos']),
            float(row['speed_scaled']),   # ← extra robust-scaled speed signal
        ])

        target = torch.FloatTensor([float(row['duration_scaled'])])

        return seg_type_idx, temporal, target, seq_len


# =============================================================================
# MAGNN MODEL
# =============================================================================


# =============================================================================
# GRAPH ATTENTION NETWORK
# =============================================================================

class GraphAttentionLayer(nn.Module):
    """Graph Attention Layer with LeakyReLU."""

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
    """Multi-head GAT for 3 adjacency views."""

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
    """Segment-specific historical embeddings."""

    def __init__(self, num_segments, embed_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(num_segments, embed_dim)
        nn.init.normal_(self.embedding.weight, mean=0, std=0.1)

    def forward(self, segment_ids):
        return self.embedding(segment_ids)



class MAGTTE(nn.Module):
    """
    Multi-Attention Graph Neural Network for Travel Time Estimation.

    Speed optimisation: GAT runs once over the full node table per forward()
    call, then we index the relevant row for each sample in the batch.
    This is O(N²) once per call instead of O(N² × batch_size).
    """

    def __init__(self, num_nodes, n_heads=3, node_embed_dim=32,
                 gat_hidden=32, lstm_hidden=64, historical_dim=16, dropout=0.3):
        super().__init__()

        # Learnable node (segment) embeddings — input to GAT
        self.node_embedding = nn.Embedding(num_nodes, node_embed_dim)
        nn.init.normal_(self.node_embedding.weight, mean=0, std=0.1)

        # Multi-relational GAT: one head per adjacency matrix
        self.multi_gat = MultiRelationalGAT(n_heads, node_embed_dim, gat_hidden, dropout)

        # Per-segment historical embedding (captures segment-specific patterns)
        self.historical_embed = HistoricalEmbedding(num_nodes, historical_dim)

        # Fusion: spatial (gat_hidden) + historical + temporal (5 features)
        fusion_in  = gat_hidden + historical_dim + 5  # 4 cyclical + speed_scaled
        fusion_out = max(fusion_in // 2, 16)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, fusion_out),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Single-step LSTM — one segment per prediction
        self.lstm = nn.LSTM(
            input_size=fusion_out,
            hidden_size=lstm_hidden,
            num_layers=1,          # 1 layer keeps it fast
            batch_first=True,
            dropout=0.0            # dropout only meaningful with >1 layer
        )

        self.regression_head = nn.Sequential(
            nn.Linear(lstm_hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

        # Adjacency matrices stored as buffers (moved to device automatically)
        self.register_buffer('adj_geo',  None)
        self.register_buffer('adj_dist', None)
        self.register_buffer('adj_soc',  None)

    def set_adjacency_matrices(self, adj_geo, adj_dist, adj_soc):
        """Register the three adjacency matrices as model buffers."""
        self.register_buffer('adj_geo',  torch.FloatTensor(adj_geo))
        self.register_buffer('adj_dist', torch.FloatTensor(adj_dist))
        self.register_buffer('adj_soc',  torch.FloatTensor(adj_soc))

    def forward(self, seg_indices, temporal_features):
        """
        seg_indices     : (B,)  — integer index of segment type for each sample
        temporal_features: (B,4) — [hour_sin, hour_cos, dow_sin, dow_cos]
        """
        device = seg_indices.device

        # ---- 1. GAT: run ONCE over the full node table ----
        # Shape: (1, num_nodes, node_embed_dim)
        all_nodes = self.node_embedding.weight.unsqueeze(0)

        # spatial_all: (1, num_nodes, gat_hidden)
        spatial_all = self.multi_gat(
            all_nodes,
            [self.adj_geo, self.adj_dist, self.adj_soc]
        )
        # Remove batch dim → (num_nodes, gat_hidden)
        spatial_all = spatial_all.squeeze(0)

        # ---- 2. Index the relevant row for each sample ----
        # seg_indices: (B,) → segment_spatial: (B, gat_hidden)
        segment_spatial   = spatial_all[seg_indices]
        segment_historical = self.historical_embed(seg_indices)  # (B, historical_dim)

        # ---- 3. Fuse spatial + historical + temporal ----
        combined = torch.cat([segment_spatial, segment_historical,
                               temporal_features], dim=1)   # (B, fusion_in)
        fused = self.fusion(combined)                        # (B, fusion_out)

        # ---- 4. LSTM temporal step ----
        lstm_out, _ = self.lstm(fused.unsqueeze(1))          # (B, 1, lstm_hidden)
        lstm_out = lstm_out.squeeze(1)                       # (B, lstm_hidden)

        # ---- 5. Regression head ----
        return self.regression_head(lstm_out)                # (B, 1)


# =============================================================================
# MASKED COLLATE  (LSTM sequence masking support)
# =============================================================================


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

def masked_collate_fn(batch):
    """
    Custom collate function for DataLoader.

    Each item from SegmentDataset is a 4-tuple:
        (seg_type_idx, temporal, target, seq_len)

    For the current single-step LSTM seq_len is always 1, so no actual
    padding occurs.  When sequences of variable length are introduced
    (multi-step LSTM), this collate function:
      1. Pads temporal tensors to the longest sequence in the batch.
      2. Returns a boolean `padding_mask` of shape (B, max_len) where
         True = real token, False = padded position.

    The forward pass can multiply the LSTM output by this mask before
    computing the loss, so padded positions contribute zero gradient.

    Returns:
        seg_indices  : LongTensor  (B,)
        temporal_pad : FloatTensor (B, max_len, temporal_dim)  — padded
        targets      : FloatTensor (B, 1)
        lengths      : LongTensor  (B,)
        mask         : BoolTensor  (B, max_len) — True = valid position
    """
    seg_indices = torch.LongTensor([item[0] for item in batch])
    targets     = torch.stack([item[2] for item in batch])
    lengths     = torch.LongTensor([item[3] for item in batch])

    # Temporal tensors — each is shape (temporal_dim,) for seq_len=1
    # Reshape to (seq_len, temporal_dim) then pad along seq dimension
    temporal_seqs = [item[1].unsqueeze(0) for item in batch]   # list of (1, D)
    max_len       = int(lengths.max().item())
    temporal_dim  = temporal_seqs[0].size(-1)

    # Pre-allocate padded tensor (filled with 0 = masked/padding value)
    temporal_pad = torch.zeros(len(batch), max_len, temporal_dim)
    for i, (seq, slen) in enumerate(zip(temporal_seqs, lengths)):
        slen = int(slen.item())
        temporal_pad[i, :slen, :] = seq[:slen]

    # mask: True where data is real
    mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)  # (B, max_len)

    return seg_indices, temporal_pad, targets, lengths, mask


# =============================================================================
# TRAINING  — NaN-safe helpers + full MAGTTE training loop
# =============================================================================


def _safe_batch(seg_idx, temporal, target):
    """
    Drop any samples within a batch that contain NaN/Inf values.
    Returns None if the batch becomes empty after filtering.

    Handles both the old 3-tuple API and the new masked-collate API.
    temporal may be (B, D) or (B, seq_len, D) — we flatten to 2-D for NaN check.
    """
    t_flat = temporal.reshape(temporal.size(0), -1)
    valid  = ~torch.isnan(t_flat).any(dim=1)
    valid &= ~torch.isinf(t_flat).any(dim=1)
    valid &= ~torch.isnan(target).any(dim=1)
    if valid.sum() == 0:
        return None, None, None
    return seg_idx[valid], temporal[valid], target[valid]



def train_epoch(model, dataloader, optimizer, criterion, device):
    """
    Train one epoch — skips any NaN-contaminated batches.

    DataLoader now uses masked_collate_fn which returns a 5-tuple:
        (seg_idx, temporal_pad, target, lengths, mask)
    We squeeze the seq_len dimension (always 1 for now) before forwarding.
    """
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for batch in dataloader:
        # Unpack 5-tuple from masked_collate_fn
        seg_idx, temporal_pad, target, lengths, mask = batch
        # temporal_pad is (B, seq_len, D) — squeeze seq_len for single-step
        temporal = temporal_pad.squeeze(1)   # → (B, D)

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
        n_batches  += 1

    return total_loss / max(n_batches, 1)



def evaluate(model, dataloader, criterion, device, scaler):
    """
    Evaluate model — NaN-safe.

    Unpacks 5-tuple from masked_collate_fn:
        (seg_idx, temporal_pad, target, lengths, mask)
    """
    model.eval()
    predictions_list = []
    targets_list     = []
    total_loss       = 0.0
    n_batches        = 0

    with torch.no_grad():
        for batch in dataloader:
            seg_idx, temporal_pad, target, lengths, mask = batch
            temporal = temporal_pad.squeeze(1)   # (B, D)

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
            n_batches  += 1
            predictions_list.append(predictions.cpu().numpy())
            targets_list.append(target.cpu().numpy())

    if not predictions_list:
        return {'loss': float('nan'), 'r2': float('nan'),
                'rmse': float('nan'), 'mae': float('nan'),
                'mape': float('nan'), 'preds': [], 'actual': []}

    preds   = np.concatenate(predictions_list)
    targets = np.concatenate(targets_list)

    preds_orig   = scaler.inverse_transform(preds)
    targets_orig = scaler.inverse_transform(targets)

    r2   = r2_score(targets_orig, preds_orig)
    rmse = np.sqrt(mean_squared_error(targets_orig, preds_orig))
    mae  = mean_absolute_error(targets_orig, preds_orig)
    mask = targets_orig.flatten() > 0
    mape = (np.mean(np.abs((targets_orig.flatten()[mask]
                            - preds_orig.flatten()[mask])
                           / targets_orig.flatten()[mask])) * 100
            if mask.any() else float('nan'))

    return {
        'loss':   total_loss / max(n_batches, 1),
        'r2':     float(r2),
        'rmse':   float(rmse),
        'mae':    float(mae),
        'mape':   float(mape),
        'preds':  preds_orig.flatten().tolist(),
        'actual': targets_orig.flatten().tolist(),
    }



def train_magtte(train_loader, val_loader, test_loader,
                 adj_geo, adj_dist, adj_soc,
                 segment_types, scaler,
                 output_folder, device, cfg):
    """
    Full MAGTTE + GAT + LSTM training pipeline.

    Uses the real graph-attention model with all three adjacency matrices.
    GAT is precomputed once per forward() call (not once per batch) so it
    is fast enough for quick testing with small epoch counts.
    """
    print_section("MAGTTE + GAT TRAINING  (real model)")

    num_segments = len(segment_types)
    print(f"   Segments       : {num_segments}")
    print(f"   Node embed dim : {cfg.node_embed_dim}")
    print(f"   GAT hidden     : {cfg.gat_hidden}  (per head)")
    print(f"   LSTM hidden    : {cfg.lstm_hidden}")
    print(f"   Historical dim : {cfg.historical_dim}")
    print(f"   Epochs         : {cfg.n_epochs}")
    print(f"   Batch size     : {cfg.batch_size}")
    print(f"   Device         : {device}")

    model = MAGTTE(
        num_nodes      = num_segments,
        n_heads        = cfg.n_heads,
        node_embed_dim = cfg.node_embed_dim,
        gat_hidden     = cfg.gat_hidden,
        lstm_hidden    = cfg.lstm_hidden,
        historical_dim = cfg.historical_dim,
        dropout        = cfg.dropout,
    ).to(device)

    model.set_adjacency_matrices(adj_geo, adj_dist, adj_soc)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Trainable params: {n_params:,}")

    optimizer = optim.Adam(model.parameters(),
                           lr=cfg.learning_rate,
                           weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=cfg.lr_scheduler_factor,
        patience=cfg.lr_scheduler_patience
    )
    criterion = nn.SmoothL1Loss()

    best_val_loss     = float('inf')
    patience_counter  = 0
    best_ckpt         = os.path.join(output_folder, 'magtte_best.pth')

    print()
    for epoch in range(1, cfg.n_epochs + 1):
        train_loss  = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device, scaler)
        val_loss    = val_metrics['loss']

        scheduler.step(val_loss if not np.isnan(val_loss) else best_val_loss)

        if epoch % max(1, cfg.n_epochs // 5) == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3}/{cfg.n_epochs}  "
                  f"train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  "
                  f"val_R²={val_metrics['r2']:.4f}  "
                  f"val_MAPE={val_metrics['mape']:.2f}%")

        if not np.isnan(val_loss) and val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_ckpt)
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stopping_patience:
                print(f"\n  ⏹️  Early stopping at epoch {epoch}")
                break

    # Load best checkpoint
    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=device))
        print(f"\n  ✓ Loaded best checkpoint (val_loss={best_val_loss:.4f})")

    # ---- Final evaluation ----
    print_section("MAGTTE — FINAL RESULTS")

    def _eval(loader, name):
        m = evaluate(model, loader, criterion, device, scaler)
        print(f"   {name:<6}  R²={m['r2']:.4f}  RMSE={m['rmse']:.2f}s  "
              f"MAE={m['mae']:.2f}s  MAPE={m['mape']:.2f}%")
        return m

    results = {
        'Train': _eval(train_loader, 'Train'),
        'Val':   _eval(val_loader,   'Val'),
        'Test':  _eval(test_loader,  'Test'),
    }

    # Sample predictions table
    test_res = results.get('Test', {})
    if test_res.get('preds'):
        print(f"\n{'Idx':>4}  {'Actual(s)':>10}  {'Pred(s)':>10}  "
              f"{'Error(s)':>9}  {'Error%':>7}")
        print("  " + "-" * 48)
        for i in range(min(20, len(test_res['actual']))):
            a   = test_res['actual'][i]
            p   = test_res['preds'][i]
            err = p - a
            pct = (err / a * 100) if a > 0 else 0.0
            print(f"  {i:>3}  {a:>10.2f}  {p:>10.2f}  {err:>9.2f}  {pct:>6.2f}%")

    return results, model


# =============================================================================
# KNOWN STOPS EXTRACTION
# =============================================================================

# Global cache: { physical_station_name: (lat, lon) }
# Populated by get_known_stops() so the adjacency builder can access
# station coordinates without needing the raw dataframe passed through.
_known_stops_cache = {}



# =============================================================================
# SIMPLE MLP FALLBACK
# =============================================================================

class SimpleMLP(nn.Module):
    """
    Straightforward MLP for segment travel-time prediction.
    Inputs per sample:
      - segment embedding  (num_segments-dim one-hot looked up via nn.Embedding)
      - temporal features  (hour_sin, hour_cos, dow_sin, dow_cos, speed_scaled)

    This is intentionally simple so training never hangs:
    no graph operations, no LSTM unrolling.
    """

    def __init__(self, num_segments, embed_dim=32, dropout=0.3):
        super().__init__()
        self.seg_embed = nn.Embedding(num_segments, embed_dim)
        nn.init.normal_(self.seg_embed.weight, 0, 0.1)

        in_dim = embed_dim + 5   # 4 cyclical + 1 speed_scaled
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
        emb = self.seg_embed(seg_indices)           # (B, embed_dim)
        x   = torch.cat([emb, temporal], dim=1)    # (B, embed_dim + 5)
        return self.net(x)                          # (B, 1)


# =============================================================================
# SIMPLE TRAINING LOOP
# =============================================================================


def train_simple(train_loader, val_loader, test_loader,
                 segment_types, scaler,
                 output_folder, device,
                 n_epochs=50, lr=0.001, dropout=0.3):
    """
    Simple MLP training loop — replaces the full MAGNN/LSTM pipeline
    when you just want to see outputs quickly without the model hanging.

    Kept separate from train_epoch / evaluate so the original MAGNN
    training functions are untouched.
    """
    print_section("SIMPLE MLP TRAINING  (lightweight fallback)")

    num_segments = len(segment_types)
    model = SimpleMLP(num_segments, embed_dim=32, dropout=dropout).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=False
    )
    criterion = nn.SmoothL1Loss()

    best_val_loss = float('inf')
    patience_counter = 0
    PATIENCE = 15

    print(f"   Segments : {num_segments}")
    print(f"   Epochs   : {n_epochs}")
    print(f"   LR       : {lr}")
    print(f"   Device   : {device}")
    print()

    for epoch in range(1, n_epochs + 1):
        # ---- train ----
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            seg_idx, temporal_pad, target, lengths, mask = batch
            temporal = temporal_pad.squeeze(1)   # (B, D)
            seg_idx  = seg_idx.to(device)
            temporal = temporal.to(device)
            target   = target.to(device)

            pred = model(seg_idx, temporal)
            loss = criterion(pred, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= max(len(train_loader), 1)

        # ---- validate ----
        model.eval()
        val_loss = 0.0
        val_preds, val_tgts = [], []
        with torch.no_grad():
            for batch in val_loader:
                seg_idx, temporal_pad, target, lengths, mask = batch
                temporal = temporal_pad.squeeze(1)   # (B, D)
                seg_idx  = seg_idx.to(device)
                temporal = temporal.to(device)
                target   = target.to(device)
                pred = model(seg_idx, temporal)
                val_loss += criterion(pred, target).item()
                val_preds.append(pred.cpu().numpy())
                val_tgts.append(target.cpu().numpy())

        val_loss /= max(len(val_loader), 1)
        scheduler.step(val_loss)

        # quick R² on val
        if val_preds:
            vp = scaler.inverse_transform(np.concatenate(val_preds))
            vt = scaler.inverse_transform(np.concatenate(val_tgts))
            val_r2 = r2_score(vt, vp)
        else:
            val_r2 = float('nan')

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3}/{n_epochs}  "
                  f"train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  "
                  f"val_R²={val_r2:.4f}")

        # early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(),
                       os.path.join(output_folder, 'simple_best_model.pth'))
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n  ⏹️  Early stopping at epoch {epoch}")
                break

    # ---- final evaluation on best checkpoint ----
    model.load_state_dict(torch.load(
        os.path.join(output_folder, 'simple_best_model.pth'),
        map_location=device
    ))
    model.eval()

    def _eval_split(loader, name):
        preds, tgts = [], []
        with torch.no_grad():
            for batch in loader:
                seg_idx, temporal_pad, target, lengths, mask = batch
                temporal = temporal_pad.squeeze(1)   # (B, D)
                pred = model(seg_idx.to(device), temporal.to(device))
                preds.append(pred.cpu().numpy())
                tgts.append(target.cpu().numpy())
        if not preds:
            return {}
        p = scaler.inverse_transform(np.concatenate(preds))
        t = scaler.inverse_transform(np.concatenate(tgts))
        r2   = r2_score(t, p)
        rmse = np.sqrt(mean_squared_error(t, p))
        mae  = mean_absolute_error(t, p)
        mask = t > 0
        mape = np.mean(np.abs((t[mask] - p[mask]) / t[mask])) * 100 if mask.any() else float('nan')
        print(f"   {name:<6}  R²={r2:.4f}  RMSE={rmse:.2f}s  "
              f"MAE={mae:.2f}s  MAPE={mape:.2f}%")
        return {'r2': r2, 'rmse': rmse, 'mae': mae, 'mape': mape,
                'preds': p.tolist(), 'actual': t.tolist()}

    print_section("SIMPLE MLP — FINAL RESULTS")
    results = {
        'Train': _eval_split(train_loader, 'Train'),
        'Val':   _eval_split(val_loader,   'Val'),
        'Test':  _eval_split(test_loader,  'Test'),
    }

    # sample predictions table
    if results.get('Test') and results['Test'].get('preds'):
        print(f"\n{'Idx':>4}  {'Actual(s)':>10}  {'Pred(s)':>10}  "
              f"{'Error(s)':>9}  {'Error%':>7}")
        print("  " + "-" * 48)
        for i in range(min(20, len(results['Test']['actual']))):
            a = results['Test']['actual'][i]
            p = results['Test']['preds'][i]
            err = p - a
            epct = (err / a * 100) if a > 0 else 0.0
            print(f"  {i:>3}  {a:>10.2f}  {p:>10.2f}  {err:>9.2f}  {epct:>6.2f}%")

    return results, model


# =============================================================================
# MAIN EXECUTION
# =============================================================================

