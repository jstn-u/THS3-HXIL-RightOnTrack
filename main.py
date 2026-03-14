"""
main.py
=======
Entry point for MAGNN-LSTM transit travel time prediction.

✅ THREE MODEL COMPARISON:
   1. MAGNN (baseline)
<<<<<<< Updated upstream
   2. MAGNN-LSTM-Residual (fixed version with residual learning + adaptive gating)
   3. MAGNN-LSTM-MTL (multi-segment paths with MTL)

Usage:
    python main.py                  # Train MAGNN only (baseline)
    python main.py --compare-all    # Compare all three models
    python main.py --ablation       # Feature ablation studies

HOW TO SWITCH CLUSTERING METHOD
--------------------------------
Change only the single import line below:

    from cluster_hdbscan import event_driven_clustering_fixed   # default
    from cluster_dbscan  import event_driven_clustering_fixed
    from cluster_knn     import event_driven_clustering_fixed
    from cluster_gmm     import event_driven_clustering_fixed
    from cluster_kmeans  import event_driven_clustering_fixed
=======
   2. MAGNN-LSTM-Residual (station-aware residual learning + adaptive gating)
   3. MAGNN-LSTM-DualTaskMTL (Local segments + Global stations)

✅ STATION-AWARE RESIDUAL CHANGES:
   - GPS lookup indexes raw pings so origin/dest features map to nearest GPS point
   - SegmentStationMapper maps cluster indices → named light-rail station labels
   - EnhancedSegmentDataset.__getitem__ returns 9-item tuples (split features)
   - enhanced_collate_fn handles the new tuple layout
   - train_magnn_lstm_residual unpacks 10-item batches and calls new forward sig

Usage:
    python main.py                      # defaults (sample from config, hdbscan)
    python main.py 0.1                  # 10% sample, hdbscan
    python main.py 1.0 knn             # 100% sample, KNN clustering
    python main.py 0.5 dbscan          # 50% sample, DBSCAN clustering
    python main.py --compare-all        # Compare all three models
    python main.py --ablation           # Feature ablation studies
    python main.py 0.1 knn --compare-all

Supported clustering methods: hdbscan, dbscan, knn, gmm, kmeans
>>>>>>> Stashed changes
"""

# =============================================================================
# CLUSTERING METHOD — change this ONE line to swap methods
# =============================================================================
from cluster_hdbscan import event_driven_clustering_fixed  # ← swap here

<<<<<<< Updated upstream
ALGORITHM_NAME = event_driven_clustering_fixed.__module__.replace('cluster_', '').upper()
=======
VALID_METHODS = {'hdbscan', 'dbscan', 'knn', 'gmm', 'kmeans'}

_sample_fraction_cli = None
_cluster_method      = 'hdbscan'
_mode                = None

for _arg in sys.argv[1:]:
    if _arg in ('--compare-all', '--ablation'):
        _mode = _arg
    elif _arg.lower() in VALID_METHODS:
        _cluster_method = _arg.lower()
    else:
        try:
            _sample_fraction_cli = float(_arg)
        except ValueError:
            print(f"⚠️  Unknown argument '{_arg}' — ignored.")
            print(f"   Valid methods: {', '.join(sorted(VALID_METHODS))}")
            print(f"   Valid flags:   --compare-all, --ablation")
            print(f"   Or pass a number for sample fraction (e.g. 0.1)")

_cluster_module = importlib.import_module(f'cluster_{_cluster_method}')
event_driven_clustering_fixed = _cluster_module.event_driven_clustering_fixed

ALGORITHM_NAME = _cluster_method.upper()
>>>>>>> Stashed changes

# =============================================================================
# SHARED MODULES
# =============================================================================
from config import Config, DEVICE, print_section, haversine_meters
from data_loader import (load_data_fixed, load_train_test_val_data_fixed,
                         get_known_stops)
from segments import (build_segments_fixed, build_adjacency_matrices_fixed,
                      aggregate_segments_into_paths)
from visualizations import (plot_clusters, plot_segments,
                            plot_segment_statistics)
from model import (SegmentDataset, masked_collate_fn,
                   train_magtte, SimpleMLP, train_simple,
                   EnhancedSegmentDataset, enhanced_collate_fn,
                   PathDataset, path_collate_fn,
                   TripDataset, trip_collate_fn,
                   train_magnn_lstm_mtl,
<<<<<<< Updated upstream
=======
                   train_magnn_lstm_dualtask_mtl,
>>>>>>> Stashed changes
                   MAGTTE)
from residual import (MAGNN_LSTM_Residual,
                      NearestGPSFeatureLookup,
                      SegmentStationMapper,
                      split_features_for_segment)

import numpy as np
import pandas as pd
import os
import json
import warnings
from datetime import datetime
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import torch

warnings.filterwarnings('ignore')


# =============================================================================
# HELPERS — attach GPS lookup + station mapper to an EnhancedSegmentDataset
# =============================================================================

def _attach_gps_lookup(dataset: EnhancedSegmentDataset,
                       raw_df,
                       clusters,
                       known_stops: dict):
    """
    Build (or reuse) a NearestGPSFeatureLookup and SegmentStationMapper and
    attach them to `dataset` so that __getitem__ can call split_features_for_segment
    with GPS-level origin/dest feature vectors.

    Safe to call on val/test datasets with the same lookup built from train_df.
    """
    if not hasattr(dataset, 'gps_lookup') or dataset.gps_lookup is None:
        dataset.gps_lookup     = NearestGPSFeatureLookup(raw_df)
        dataset.station_mapper = SegmentStationMapper(
            np.asarray(clusters), known_stops=known_stops
        )
    return dataset


def _make_enhanced_loaders(train_segments, val_segments, test_segments,
                            segment_types, config,
                            train_df, clusters, known_stops,
                            collate_fn=None):
    """
    Build train/val/test EnhancedSegmentDatasets with GPS lookup attached,
    then wrap in DataLoaders.  Returns (train_loader, val_loader, test_loader,
    train_dataset).
    """
    if collate_fn is None:
        collate_fn = enhanced_collate_fn

    train_ds = EnhancedSegmentDataset(train_segments, segment_types,
                                      fit_scalers=True)
    val_ds   = EnhancedSegmentDataset(val_segments, segment_types,
                                      fit_scalers=False,
                                      target_scaler=train_ds.target_scaler,
                                      speed_scaler=train_ds.speed_scaler,
                                      operational_scaler=train_ds.operational_scaler,
                                      weather_scaler=train_ds.weather_scaler)
    test_ds  = EnhancedSegmentDataset(test_segments, segment_types,
                                      fit_scalers=False,
                                      target_scaler=train_ds.target_scaler,
                                      speed_scaler=train_ds.speed_scaler,
                                      operational_scaler=train_ds.operational_scaler,
                                      weather_scaler=train_ds.weather_scaler)

    # Attach GPS lookup to all three datasets (val/test reuse train lookup)
    _attach_gps_lookup(train_ds, train_df, clusters, known_stops)
    val_ds.gps_lookup      = train_ds.gps_lookup
    val_ds.station_mapper  = train_ds.station_mapper
    test_ds.gps_lookup     = train_ds.gps_lookup
    test_ds.station_mapper = train_ds.station_mapper

    train_loader = DataLoader(train_ds, batch_size=config.batch_size,
                              shuffle=True,  num_workers=0, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=config.batch_size,
                              shuffle=False, num_workers=0, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=config.batch_size,
                              shuffle=False, num_workers=0, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader, train_ds


def _make_trip_loaders(train_segments, val_segments, test_segments,
                       segment_types, config,
                       train_df, clusters, known_stops,
                       max_trip_length: int = 15):
    """
    Build train/val/test TripDatasets (trip_id + same-day grouping),
    attach GPS lookup, and return DataLoaders using trip_collate_fn.

    The global target for each trip = sum of its segment durations.
    Scalers are fit on training data only and shared with val/test.

    Returns (train_loader, val_loader, test_loader, train_dataset)
    """
    # Build GPS lookup once from train_df
    gps_lookup     = NearestGPSFeatureLookup(train_df)
    station_mapper = SegmentStationMapper(
        np.asarray(clusters), known_stops=known_stops)

    train_ds = TripDataset(train_segments, segment_types,
                           max_trip_length=max_trip_length,
                           fit_scalers=True)
    val_ds   = TripDataset(val_segments, segment_types,
                           max_trip_length=max_trip_length,
                           fit_scalers=False,
                           target_scaler=train_ds.target_scaler,
                           speed_scaler=train_ds.speed_scaler,
                           operational_scaler=train_ds.operational_scaler,
                           weather_scaler=train_ds.weather_scaler)
    test_ds  = TripDataset(test_segments, segment_types,
                           max_trip_length=max_trip_length,
                           fit_scalers=False,
                           target_scaler=train_ds.target_scaler,
                           speed_scaler=train_ds.speed_scaler,
                           operational_scaler=train_ds.operational_scaler,
                           weather_scaler=train_ds.weather_scaler)

    # Attach GPS lookup to all three datasets (val/test reuse train lookup)
    for ds in (train_ds, val_ds, test_ds):
        ds.gps_lookup     = gps_lookup
        ds.station_mapper = station_mapper

    # Trip batch size: trips are longer sequences, use smaller batches
    trip_batch = max(4, config.batch_size // 4)

    train_loader = DataLoader(train_ds, batch_size=trip_batch,
                              shuffle=True,  num_workers=0,
                              collate_fn=trip_collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=trip_batch,
                              shuffle=False, num_workers=0,
                              collate_fn=trip_collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=trip_batch,
                              shuffle=False, num_workers=0,
                              collate_fn=trip_collate_fn)

    return train_loader, val_loader, test_loader, train_ds





# =============================================================================
# RESIDUAL BATCH UNPACKING HELPER
# =============================================================================

def _unpack_residual_batch(batch, device):
    """
    Unpack the 10-item tuple produced by the updated enhanced_collate_fn
    and move every tensor to `device`.

    Batch layout (from enhanced_collate_fn in model.py):
        [0]  seg_indices        LongTensor  (B,)
        [1]  temporal_pad       FloatTensor (B, 1, 5)
        [2]  context_pad        FloatTensor (B, 1, 2)   is_weekend, is_peak_hour
        [3]  origin_op_pad      FloatTensor (B, 1, 2)   arrivalDelay, departureDelay @ origin
        [4]  dest_op_pad        FloatTensor (B, 1, 2)   same @ dest
        [5]  origin_weather_pad FloatTensor (B, 1, 8)   8 weather vars @ origin GPS point
        [6]  dest_weather_pad   FloatTensor (B, 1, 8)   same @ dest
        [7]  targets            FloatTensor (B, 1)
        [8]  lengths            LongTensor  (B,)
        [9]  mask               BoolTensor  (B, 1)
    """
    (seg_idx, temporal, context_flags,
     origin_op, dest_op,
     origin_weather, dest_weather,
     target, lengths, mask) = batch

    seg_idx        = seg_idx.to(device)
    temporal       = temporal.squeeze(1).to(device)        # (B, 5)
    context_flags  = context_flags.squeeze(1).to(device)   # (B, 2)
    origin_op      = origin_op.squeeze(1).to(device)       # (B, 2)
    dest_op        = dest_op.squeeze(1).to(device)         # (B, 2)
    origin_weather = origin_weather.squeeze(1).to(device)  # (B, 8)
    dest_weather   = dest_weather.squeeze(1).to(device)    # (B, 8)
    target         = target.to(device)

    return (seg_idx, temporal, context_flags,
            origin_op, dest_op,
            origin_weather, dest_weather,
            target)


# =============================================================================
# RESIDUAL MODEL TRAINING FUNCTION  (✅ STATION-AWARE)
# =============================================================================

def train_magnn_lstm_residual(train_loader, val_loader, test_loader,
                              adj_geo, adj_dist, adj_soc,
                              segment_types, scaler,
                              output_folder, device, cfg,
                              clusters=None,
                              known_stops=None,
                              pretrained_magnn_path=None,
                              freeze_magnn=True):
    """
    Train MAGNN-LSTM-Residual (station-aware version).

    The model now receives 7 separate input tensors per sample:
        temporal, context_flags,
        origin_operational, dest_operational,
        origin_weather, dest_weather
    instead of the old merged operational + weather blobs.

    Parameters
    ----------
    clusters    : np.ndarray | None   cluster centroids (lat, lon) — used to
                                       build SegmentStationMapper for logging
    known_stops : dict | None         station_name → (lat, lon)
    """
    print_section("MAGNN-LSTM-RESIDUAL TRAINING  (Station-Aware)")
    num_segments = len(segment_types)

    # ------------------------------------------------------------------
    # Build MAGNN base
    # ------------------------------------------------------------------
    magnn_base = MAGTTE(num_segments,
                        cfg.n_heads, cfg.node_embed_dim, cfg.gat_hidden,
                        cfg.lstm_hidden, cfg.historical_dim,
                        cfg.dropout).to(device)
    magnn_base.set_adjacency_matrices(adj_geo, adj_dist, adj_soc)

    if pretrained_magnn_path and os.path.exists(pretrained_magnn_path):
        magnn_base.load_state_dict(
            torch.load(pretrained_magnn_path, map_location=device))
        print(f"   ✓ Loaded pre-trained MAGNN from {pretrained_magnn_path}")

    # ------------------------------------------------------------------
    # Station mapper (for interpretable logging)
    # ------------------------------------------------------------------
    station_mapper = None
    if clusters is not None:
        station_mapper = SegmentStationMapper(
            np.asarray(clusters), known_stops=known_stops or {}
        )

    # ------------------------------------------------------------------
    # Build residual model
    # ------------------------------------------------------------------
    model = MAGNN_LSTM_Residual(
        magnn_base,
        spatial_dim=cfg.gat_hidden,
        station_mapper=station_mapper,
        freeze_magnn=freeze_magnn,
        lstm_hidden=128,
        lstm_layers=1,
        dropout=0.2,
        temporal_dim=5,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Trainable params: {n_params:,}")

    learning_rate = (cfg.residual_learning_rate
                     if hasattr(cfg, 'residual_learning_rate') else 5e-4)
    weight_decay  = (cfg.lstm_weight_decay
                     if hasattr(cfg, 'lstm_weight_decay') else 1e-6)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate, weight_decay=weight_decay
    )
    print(f"   Learning rate: {learning_rate}   Weight decay: {weight_decay}")

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.SmoothL1Loss()

    best_val_loss   = float('inf')
    patience_counter = 0
    best_ckpt = os.path.join(output_folder, 'magnn_lstm_residual_best.pth')

    history = {'train_loss': [], 'val_loss': [], 'val_r2': [],
               'alpha': [], 'correction': []}

    print()
    for epoch in range(1, cfg.n_epochs + 1):
        # ------ TRAIN ------
        model.train()
        train_loss = alpha_sum = correction_sum = 0.0
        n_batches = 0

        for batch in train_loader:
            (seg_idx, temporal, context_flags,
             origin_op, dest_op,
             origin_weather, dest_weather,
             target) = _unpack_residual_batch(batch, device)

            baseline, correction, alpha, predictions = model(
                seg_idx, temporal,
                context_flags=context_flags,
                origin_operational=origin_op,
                dest_operational=dest_op,
                origin_weather=origin_weather,
                dest_weather=dest_weather,
                return_components=True,
            )

            loss = criterion(predictions, target)
            if torch.isnan(loss) or torch.isinf(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss    += loss.item()
            alpha_sum     += alpha.mean().item()
            correction_sum += correction.abs().mean().item()
            n_batches     += 1

        train_loss   /= max(n_batches, 1)
        avg_alpha     = alpha_sum     / max(n_batches, 1)
        avg_correction = correction_sum / max(n_batches, 1)

        # ------ VALIDATE ------
        model.eval()
        val_loss = val_alpha_sum = val_correction_sum = 0.0
        val_preds, val_targets = [], []
        n_val = 0

        with torch.no_grad():
            for batch in val_loader:
                (seg_idx, temporal, context_flags,
                 origin_op, dest_op,
                 origin_weather, dest_weather,
                 target) = _unpack_residual_batch(batch, device)

                baseline, correction, alpha, predictions = model(
                    seg_idx, temporal,
                    context_flags=context_flags,
                    origin_operational=origin_op,
                    dest_operational=dest_op,
                    origin_weather=origin_weather,
                    dest_weather=dest_weather,
                    return_components=True,
                )

                loss = criterion(predictions, target)
                if not torch.isnan(loss) and not torch.isinf(loss):
                    val_loss          += loss.item()
                    val_alpha_sum     += alpha.mean().item()
                    val_correction_sum += correction.abs().mean().item()
                    n_val             += 1
                    val_preds.append(predictions.cpu().numpy())
                    val_targets.append(target.cpu().numpy())

        val_loss /= max(n_val, 1)
        val_alpha  = val_alpha_sum / max(n_val, 1)
        scheduler.step(val_loss)

        if val_preds:
            vp = scaler.inverse_transform(np.concatenate(val_preds))
            vt = scaler.inverse_transform(np.concatenate(val_targets))
            val_r2 = r2_score(vt, vp)
        else:
            val_r2 = float('nan')

        correction_seconds = (avg_correction * scaler.scale_[0]
                              if hasattr(scaler, 'scale_') else avg_correction)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_r2'].append(val_r2)
        history['alpha'].append(avg_alpha)
        history['correction'].append(correction_seconds)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3}/{cfg.n_epochs}  "
                  f"loss={train_loss:.4f}  val_R²={val_r2:.4f}  "
                  f"α={avg_alpha:.3f}  corr={correction_seconds:.2f}s")

        if not np.isnan(val_loss) and val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_ckpt)
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stopping_patience:
                print(f"\n  ⏹️  Early stopping at epoch {epoch}")
                break

    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=device))

    # ------------------------------------------------------------------
    # FINAL EVALUATION
    # ------------------------------------------------------------------
    print_section("MAGNN-LSTM-RESIDUAL — FINAL RESULTS")

    def _eval(loader, name):
        model.eval()
        preds, targets, alphas, corrections = [], [], [], []

        with torch.no_grad():
            for batch in loader:
                (seg_idx, temporal, context_flags,
                 origin_op, dest_op,
                 origin_weather, dest_weather,
                 target) = _unpack_residual_batch(batch, device)

                baseline, correction, alpha, pred = model(
                    seg_idx, temporal,
                    context_flags=context_flags,
                    origin_operational=origin_op,
                    dest_operational=dest_op,
                    origin_weather=origin_weather,
                    dest_weather=dest_weather,
                    return_components=True,
                )

                preds.append(pred.cpu().numpy())
                targets.append(target.cpu().numpy())
                alphas.append(alpha.cpu().numpy())
                corrections.append(correction.cpu().numpy())

        if not preds:
            return {}

        p = scaler.inverse_transform(np.concatenate(preds))
        t = scaler.inverse_transform(np.concatenate(targets))
        a = np.concatenate(alphas).mean()
        c_raw = np.concatenate(corrections)
        c_inv = scaler.inverse_transform(c_raw)
        c_mean, c_std = c_inv.mean(), c_inv.std()

        r2   = r2_score(t, p)
        rmse = np.sqrt(mean_squared_error(t, p))
        mae  = mean_absolute_error(t, p)
        mask_v = t.flatten() > 0
        mape = (np.mean(np.abs((t.flatten()[mask_v] - p.flatten()[mask_v]) /
                               t.flatten()[mask_v])) * 100
                if mask_v.any() else float('nan'))

        print(f"   {name:<6}  R²={r2:.4f}  RMSE={rmse:.2f}s  "
              f"MAE={mae:.2f}s  MAPE={mape:.2f}%")
        print(f"           α={a:.3f}  corr={c_mean:.2f}±{c_std:.2f}s")

        return {'r2': r2, 'rmse': rmse, 'mae': mae, 'mape': mape,
                'preds': p.flatten().tolist(), 'actual': t.flatten().tolist(),
                'alpha': float(a),
                'correction_mean': float(c_mean),
                'correction_std':  float(c_std)}

    results = {
        'Train': _eval(train_loader, 'Train'),
        'Val':   _eval(val_loader,   'Val'),
        'Test':  _eval(test_loader,  'Test'),
    }

    test_res = results.get('Test', {})
    if test_res.get('preds'):
        print(f"\n{'Idx':>4}  {'Actual(s)':>10}  {'Pred(s)':>10}  "
              f"{'Error(s)':>9}  {'Error%':>7}")
        print("  " + "-" * 48)
        for i in range(min(20, len(test_res['actual']))):
            a, p_val = test_res['actual'][i], test_res['preds'][i]
            err = p_val - a
            epct = (err / a * 100) if a > 0 else 0.0
            print(f"  {i:>3}  {a:>10.2f}  {p_val:>10.2f}  "
                  f"{err:>9.2f}  {epct:>6.2f}%")

    return results, model


# =============================================================================
# MAIN PIPELINE (BASELINE MAGNN ONLY)
# =============================================================================

def main():
    """Main training loop — baseline MAGNN only."""

    config = Config()

    print_section("MAGNN TRAINING CONFIGURATION")
    print(f"  Device: {DEVICE}")
    print(f"  Algorithm: {ALGORITHM_NAME}")
    print(f"  Data sampling: {config.sample_fraction * 100}%")
    print(f"  Iterations: {config.n_iterations}")
    print(f"  Epochs per iteration: {config.n_epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print("=" * 80)

    for iteration in range(1, config.n_iterations + 1):
        print_section(f"ITERATION {iteration}/{config.n_iterations}")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_folder = f"outputs/{ALGORITHM_NAME}_{timestamp}_i{iteration}"
        os.makedirs(output_folder, exist_ok=True)
        print(f"📁 Output: {output_folder}")

        try:
            # ------------------------------------------------------------------
            # 1. Load data
            # ------------------------------------------------------------------
            train_df, test_df, val_df = load_train_test_val_data_fixed(
                data_folder=config.data_folder,
                sample_fraction=config.sample_fraction
            )

            if len(train_df) == 0:
                print("❌ No training data")
                continue

            known_stops_train = get_known_stops(train_df)
            known_stops_test  = get_known_stops(test_df)
            known_stops_val   = get_known_stops(val_df)
            known_stops = {**known_stops_test, **known_stops_val,
                           **known_stops_train}
            print(f"   Known stops: {len(known_stops)} "
                  f"(train={len(known_stops_train)}, "
                  f"test={len(known_stops_test)}, "
                  f"val={len(known_stops_val)})")

            # ------------------------------------------------------------------
            # 2. Clustering
            # ------------------------------------------------------------------
            clusters, station_cluster_ids = event_driven_clustering_fixed(
                train_df, known_stops=known_stops
            )
            if len(clusters) == 0:
                print("❌ No clusters")
                continue

            # ------------------------------------------------------------------
            # 3. Build segments
            # ------------------------------------------------------------------
            train_segments = build_segments_fixed(train_df, clusters)
            if len(train_segments) == 0:
                print("❌ No segments")
                continue

            test_segments = build_segments_fixed(test_df, clusters)
            val_segments  = build_segments_fixed(val_df,  clusters)

            # ------------------------------------------------------------------
            # 4. Plots
            # ------------------------------------------------------------------
            print_section("GENERATING VISUALISATIONS")
            plot_clusters(
                clusters, {},
                algorithm_name=ALGORITHM_NAME,
                save_path=os.path.join(
                    output_folder, f'{ALGORITHM_NAME.lower()}-clusters.png'))
            plot_segments(
                train_segments, clusters, max_segments=100,
                algorithm_name=ALGORITHM_NAME,
                save_path=os.path.join(
                    output_folder, f'{ALGORITHM_NAME.lower()}-segments.png'))
            plot_segment_statistics(
                train_segments,
                algorithm_name=ALGORITHM_NAME,
                save_path=os.path.join(
                    output_folder,
                    f'{ALGORITHM_NAME.lower()}-segment_stats.png'))

            # ------------------------------------------------------------------
            # 5. Adjacency matrices
            # ------------------------------------------------------------------
            adj_geo, adj_dist, adj_soc, segment_types = \
                build_adjacency_matrices_fixed(
                    train_segments, clusters, known_stops=known_stops)

            if adj_geo is None:
                print("❌ Adjacency failed")
                continue

            # ------------------------------------------------------------------
            # 6. Datasets & loaders
            # ------------------------------------------------------------------
            train_dataset = SegmentDataset(
                train_segments, segment_types, fit_scalers=True)
            val_dataset = SegmentDataset(
                val_segments, segment_types, fit_scalers=False,
                target_scaler=train_dataset.target_scaler,
                speed_scaler=train_dataset.speed_scaler)
            test_dataset = SegmentDataset(
                test_segments, segment_types, fit_scalers=False,
                target_scaler=train_dataset.target_scaler,
                speed_scaler=train_dataset.speed_scaler)

            train_loader = DataLoader(
                train_dataset, batch_size=config.batch_size,
                shuffle=True, num_workers=0, collate_fn=masked_collate_fn)
            val_loader = DataLoader(
                val_dataset, batch_size=config.batch_size,
                shuffle=False, num_workers=0, collate_fn=masked_collate_fn)
            test_loader = DataLoader(
                test_dataset, batch_size=config.batch_size,
                shuffle=False, num_workers=0, collate_fn=masked_collate_fn)

            print(f"\n📊 Data Summary:")
            print(f"   Segment types: {len(segment_types)}")
            print(f"   Train: {len(train_dataset):,}  "
                  f"Val: {len(val_dataset):,}  "
                  f"Test: {len(test_dataset):,}")

            if len(train_loader) == 0:
                print("❌ Training loader empty — skipping")
                continue

            # ------------------------------------------------------------------
            # 7. MAGTTE + GAT training
            # ------------------------------------------------------------------
            results, _ = train_magtte(
                train_loader, val_loader, test_loader,
                adj_geo, adj_dist, adj_soc,
                segment_types, train_dataset.target_scaler,
                output_folder, DEVICE, config
            )

            # ------------------------------------------------------------------
            # 8. Save metrics JSON
            # ------------------------------------------------------------------
            DATA_DRIVEN = {'knn', 'hdbscan', 'dbscan'}
            metrics_config = {
                'n_epochs':       config.n_epochs,
                'batch_size':     config.batch_size,
                'learning_rate':  config.learning_rate,
                'sample_fraction': config.sample_fraction,
                'n_clusters':     len(clusters),
                'cluster_count_method': (
                    'data-driven' if _cluster_method in DATA_DRIVEN
                    else 'requested'),
                'n_segment_types': len(segment_types),
                'node_embed_dim': config.node_embed_dim,
                'gat_hidden':     config.gat_hidden,
                'lstm_hidden':    config.lstm_hidden,
                'historical_dim': config.historical_dim,
            }

            metrics_out = {
                'graph_method': ALGORITHM_NAME.lower(),
                'config': metrics_config,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            }

            for ds_name, res in [('Train', results.get('Train')),
                                  ('Val',   results.get('Val')),
                                  ('Test',  results.get('Test'))]:
                if res:
                    metrics_out[ds_name] = {
                        'R2':   f"{res['r2']:.4f}",
                        'RMSE': f"{res['rmse']:.2f} seconds",
                        'MAE':  f"{res['mae']:.2f} seconds",
                        'MAPE': f"{res['mape']:.2f}%",
                    }

            json_path = os.path.join(output_folder, 'metrics.json')
            with open(json_path, 'w') as f:
                json.dump(metrics_out, f, indent=2)
            print(f"\n✓ Metrics saved → {json_path}")
            print(f"✅ Iteration {iteration} complete")

        except Exception as e:
            print(f"\n❌ Error in iteration {iteration}: {e}")
            import traceback
            traceback.print_exc()
            continue


# =============================================================================
# THREE-WAY COMPARISON TABLE HELPER
# =============================================================================

def print_three_way_comparison_table(magnn_results, residual_results,
                                     mtl_results, split_name):
    magnn_m    = magnn_results.get(split_name, {})
    residual_m = residual_results.get(split_name, {})
    mtl_m      = mtl_results.get(split_name, {})

    print(f"\n{'=' * 120}")
    print(f"{split_name.upper()} SET - THREE-WAY COMPARISON")
    print(f"{'=' * 120}")
<<<<<<< Updated upstream
    print(f"{'Metric':<10} {'MAGNN':<20} {'Residual':<20} {'MTL':<20} {'Best Model':<20} {'Improvement':<15}")
=======
    print(f"{'Metric':<10} {'MAGNN':<20} {'Residual':<20} "
          f"{'DualTaskMTL':<20} {'Best Model':<20} {'Improvement':<15}")
>>>>>>> Stashed changes
    print(f"{'-' * 120}")

    for display_name, metric_key, higher_better in [
        ('R²',   'r2',   True),
        ('RMSE', 'rmse', False),
        ('MAE',  'mae',  False),
        ('MAPE', 'mape', False),
    ]:
        mv  = magnn_m.get(metric_key, float('nan'))
        rv  = residual_m.get(metric_key, float('nan'))
        tv  = mtl_m.get(metric_key, float('nan'))

        def _fmt(v):
            if metric_key in ('rmse', 'mae'): return f"{v:.2f}s"
            if metric_key == 'mape':          return f"{v:.2f}%"
            return f"{v:.4f}"

<<<<<<< Updated upstream
        if metric_key in ['rmse', 'mae']:
            magnn_str = f"{magnn_val:.2f}s"
            residual_str = f"{residual_val:.2f}s"
            mtl_str = f"{mtl_val:.2f}s"
        elif metric_key == 'mape':
            magnn_str = f"{magnn_val:.2f}%"
            residual_str = f"{residual_val:.2f}%"
            mtl_str = f"{mtl_val:.2f}%"
        else:
            magnn_str = f"{magnn_val:.4f}"
            residual_str = f"{residual_val:.4f}"
            mtl_str = f"{mtl_val:.4f}"

        if not (np.isnan(magnn_val) or np.isnan(residual_val) or np.isnan(mtl_val)):
            if higher_is_better:
                best_val = max(magnn_val, residual_val, mtl_val)
                if best_val == residual_val:
                    best_model = "Residual ✓"
                    improvement = ((residual_val - magnn_val) / abs(magnn_val) * 100)
                elif best_val == mtl_val:
                    best_model = "MTL ✓"
                    improvement = ((mtl_val - magnn_val) / abs(magnn_val) * 100)
                else:
                    best_model = "MAGNN (baseline)"
                    improvement = 0.0
            else:
                best_val = min(magnn_val, residual_val, mtl_val)
                if best_val == residual_val:
                    best_model = "Residual ✓"
                    improvement = ((magnn_val - residual_val) / abs(magnn_val) * 100)
                elif best_val == mtl_val:
                    best_model = "MTL ✓"
                    improvement = ((magnn_val - mtl_val) / abs(magnn_val) * 100)
                else:
                    best_model = "MAGNN (baseline)"
                    improvement = 0.0
=======
        if not any(np.isnan(x) for x in [mv, rv, tv]):
            if higher_better:
                best = max(mv, rv, tv)
            else:
                best = min(mv, rv, tv)
>>>>>>> Stashed changes

            if best == rv:
                best_model = "Residual ✓"
                improvement = (abs(rv - mv) / abs(mv) * 100) if mv else 0
            elif best == tv:
                best_model = "DualTaskMTL ✓"
                improvement = (abs(tv - mv) / abs(mv) * 100) if mv else 0
            else:
                best_model = "MAGNN (baseline)"
                improvement = 0.0

            imp_str = (f"+{improvement:.2f}%" if improvement > 0
                       else f"{improvement:.2f}%")
        else:
            best_model = "N/A"
            imp_str    = "N/A"

        print(f"{display_name:<10} {_fmt(mv):<20} {_fmt(rv):<20} "
              f"{_fmt(tv):<20} {best_model:<20} {imp_str:<15}")

    print(f"{'=' * 120}\n")


# =============================================================================
# THREE-WAY COMPARISON MODE
# =============================================================================

def main_with_three_way_comparison():
    """Three-way comparison: MAGNN vs MAGNN-LSTM-Residual vs MAGNN-LSTM-MTL"""

    config = Config()

    print_section("🔬 THREE-WAY COMPARISON MODE")
    print(f"  Device: {DEVICE}   Algorithm: {ALGORITHM_NAME}")
    print(f"\n  Models:")
    print(f"    1. MAGNN (baseline)")
<<<<<<< Updated upstream
    print(f"    2. MAGNN-LSTM-Residual (✅ fixed + residual learning + adaptive gate)")
    print(f"    3. MAGNN-LSTM-MTL (multi-segment paths)")
    print(f"\n  Residual model features:")
    print(f"    ✅ Weather features properly scaled")
    print(f"    ✅ LSTM sees MAGNN prediction (residual learning)")
    print(f"    ✅ Adaptive gating (learns α per sample)")
    print(f"    ✅ Binary flags: is_weekend, is_peak_hour")
=======
    print(f"    2. MAGNN-LSTM-Residual (station-aware residual + adaptive gate)")
    print(f"    3. MAGNN-LSTM-DualTaskMTL (Local + Global tasks)")
>>>>>>> Stashed changes
    print("=" * 80)

    all_magnn_results    = []
    all_residual_results = []
    all_mtl_results      = []

    for iteration in range(1, config.n_iterations + 1):
        print_section(f"🔄 ITERATION {iteration}/{config.n_iterations}")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_folder = (f"outputs/{ALGORITHM_NAME}_{timestamp}"
                         f"_i{iteration}_three_way")
        os.makedirs(output_folder, exist_ok=True)

        try:
<<<<<<< Updated upstream
            print("\n[1/11] Loading data...")
=======
            # ── 1. Data ───────────────────────────────────────────────
            print("\n[1/10] Loading data...")
>>>>>>> Stashed changes
            train_df, test_df, val_df = load_train_test_val_data_fixed(
                data_folder=config.data_folder,
                sample_fraction=config.sample_fraction
            )
            known_stops_train = get_known_stops(train_df)
            known_stops_test  = get_known_stops(test_df)
            known_stops_val   = get_known_stops(val_df)
            known_stops = {**known_stops_test, **known_stops_val,
                           **known_stops_train}

<<<<<<< Updated upstream
            print("[2/11] Clustering stops...")
=======
            # ── 2. Clustering ─────────────────────────────────────────
            print("[2/10] Clustering stops...")
>>>>>>> Stashed changes
            clusters, station_cluster_ids = event_driven_clustering_fixed(
                train_df, known_stops=known_stops)

<<<<<<< Updated upstream
            print("[3/11] Building segments...")
=======
            # ── 3. Segments ───────────────────────────────────────────
            print("[3/10] Building segments...")
>>>>>>> Stashed changes
            train_segments = build_segments_fixed(train_df, clusters)
            test_segments  = build_segments_fixed(test_df,  clusters)
            val_segments   = build_segments_fixed(val_df,   clusters)

<<<<<<< Updated upstream
            print("[4/11] Building adjacency matrices...")
            adj_geo, adj_dist, adj_soc, segment_types = build_adjacency_matrices_fixed(
                train_segments, clusters, known_stops=known_stops
            )

            print("[5/11] Preparing MAGNN datasets...")
            train_dataset_magnn = SegmentDataset(train_segments, segment_types, fit_scalers=True)
            val_dataset_magnn = SegmentDataset(val_segments, segment_types, fit_scalers=False,
                                               target_scaler=train_dataset_magnn.target_scaler,
                                               speed_scaler=train_dataset_magnn.speed_scaler)
            test_dataset_magnn = SegmentDataset(test_segments, segment_types, fit_scalers=False,
                                                target_scaler=train_dataset_magnn.target_scaler,
                                                speed_scaler=train_dataset_magnn.speed_scaler)

            train_loader_magnn = DataLoader(train_dataset_magnn, batch_size=config.batch_size,
                                            shuffle=True, num_workers=0, collate_fn=masked_collate_fn)
            val_loader_magnn = DataLoader(val_dataset_magnn, batch_size=config.batch_size,
                                          shuffle=False, num_workers=0, collate_fn=masked_collate_fn)
            test_loader_magnn = DataLoader(test_dataset_magnn, batch_size=config.batch_size,
                                           shuffle=False, num_workers=0, collate_fn=masked_collate_fn)

            print("[6/11] Preparing Residual datasets...")
            train_dataset_residual = EnhancedSegmentDataset(train_segments, segment_types, fit_scalers=True)
            val_dataset_residual = EnhancedSegmentDataset(val_segments, segment_types, fit_scalers=False,
                                                          target_scaler=train_dataset_residual.target_scaler,
                                                          speed_scaler=train_dataset_residual.speed_scaler,
                                                          operational_scaler=train_dataset_residual.operational_scaler,
                                                          weather_scaler=train_dataset_residual.weather_scaler)
            test_dataset_residual = EnhancedSegmentDataset(test_segments, segment_types, fit_scalers=False,
                                                           target_scaler=train_dataset_residual.target_scaler,
                                                           speed_scaler=train_dataset_residual.speed_scaler,
                                                           operational_scaler=train_dataset_residual.operational_scaler,
                                                           weather_scaler=train_dataset_residual.weather_scaler)

            train_loader_residual = DataLoader(train_dataset_residual, batch_size=config.batch_size,
                                               shuffle=True, num_workers=0, collate_fn=enhanced_collate_fn)
            val_loader_residual = DataLoader(val_dataset_residual, batch_size=config.batch_size,
                                             shuffle=False, num_workers=0, collate_fn=enhanced_collate_fn)
            test_loader_residual = DataLoader(test_dataset_residual, batch_size=config.batch_size,
                                              shuffle=False, num_workers=0, collate_fn=enhanced_collate_fn)

            print("[7/11] Aggregating segments into paths for MTL...")
            train_paths = aggregate_segments_into_paths(train_segments, max_path_length=10)
            val_paths = aggregate_segments_into_paths(val_segments, max_path_length=10)
            test_paths = aggregate_segments_into_paths(test_segments, max_path_length=10)

            train_dataset_mtl = PathDataset(train_paths, segment_types, max_path_length=10, fit_scalers=True)
            val_dataset_mtl = PathDataset(val_paths, segment_types, max_path_length=10, fit_scalers=False,
                                          target_scaler=train_dataset_mtl.target_scaler,
                                          speed_scaler=train_dataset_mtl.speed_scaler,
                                          operational_scaler=train_dataset_mtl.operational_scaler,
                                          weather_scaler=train_dataset_mtl.weather_scaler)
            test_dataset_mtl = PathDataset(test_paths, segment_types, max_path_length=10, fit_scalers=False,
                                           target_scaler=train_dataset_mtl.target_scaler,
                                           speed_scaler=train_dataset_mtl.speed_scaler,
                                           operational_scaler=train_dataset_mtl.operational_scaler,
                                           weather_scaler=train_dataset_mtl.weather_scaler)

            train_loader_mtl = DataLoader(train_dataset_mtl, batch_size=config.batch_size,
                                          shuffle=True, num_workers=0, collate_fn=path_collate_fn)
            val_loader_mtl = DataLoader(val_dataset_mtl, batch_size=config.batch_size,
                                        shuffle=False, num_workers=0, collate_fn=path_collate_fn)
            test_loader_mtl = DataLoader(test_dataset_mtl, batch_size=config.batch_size,
                                         shuffle=False, num_workers=0, collate_fn=path_collate_fn)

            print("\n[8/11] Training MAGNN (baseline)...")
=======
            # ── 3b. Visualisations ────────────────────────────────────
            print("[3b/10] Generating visualisations...")
            plot_clusters(
                clusters, {}, algorithm_name=ALGORITHM_NAME,
                save_path=os.path.join(
                    output_folder,
                    f'{ALGORITHM_NAME.lower()}-clusters.png'))
            plot_segments(
                train_segments, clusters, max_segments=100,
                algorithm_name=ALGORITHM_NAME,
                save_path=os.path.join(
                    output_folder,
                    f'{ALGORITHM_NAME.lower()}-segments.png'))
            plot_segment_statistics(
                train_segments, algorithm_name=ALGORITHM_NAME,
                save_path=os.path.join(
                    output_folder,
                    f'{ALGORITHM_NAME.lower()}-segment_stats.png'))

            # ── 4. Adjacency ──────────────────────────────────────────
            print("[4/10] Building adjacency matrices...")
            adj_geo, adj_dist, adj_soc, segment_types = \
                build_adjacency_matrices_fixed(
                    train_segments, clusters, known_stops=known_stops)

            # ── 5. MAGNN datasets ─────────────────────────────────────
            print("[5/10] Preparing MAGNN datasets...")
            train_ds_magnn = SegmentDataset(
                train_segments, segment_types, fit_scalers=True)
            val_ds_magnn = SegmentDataset(
                val_segments, segment_types, fit_scalers=False,
                target_scaler=train_ds_magnn.target_scaler,
                speed_scaler=train_ds_magnn.speed_scaler)
            test_ds_magnn = SegmentDataset(
                test_segments, segment_types, fit_scalers=False,
                target_scaler=train_ds_magnn.target_scaler,
                speed_scaler=train_ds_magnn.speed_scaler)

            train_loader_magnn = DataLoader(
                train_ds_magnn, batch_size=config.batch_size,
                shuffle=True, num_workers=0, collate_fn=masked_collate_fn)
            val_loader_magnn = DataLoader(
                val_ds_magnn, batch_size=config.batch_size,
                shuffle=False, num_workers=0, collate_fn=masked_collate_fn)
            test_loader_magnn = DataLoader(
                test_ds_magnn, batch_size=config.batch_size,
                shuffle=False, num_workers=0, collate_fn=masked_collate_fn)

            # ── 6. Enhanced datasets (Residual) ──────────────────────
            print("[6/10] Preparing Enhanced datasets "
                  "(station-aware GPS lookup)...")
            (train_loader_enh, val_loader_enh, test_loader_enh,
             train_ds_enh) = _make_enhanced_loaders(
                train_segments, val_segments, test_segments,
                segment_types, config,
                train_df, clusters, known_stops
            )

            # ── 6b. Trip datasets (DualTaskMTL) ──────────────────────
            print("[6b/10] Preparing Trip datasets "
                  "(trip_id + same-day grouping for MTL)...")
            (train_loader_trip, val_loader_trip, test_loader_trip,
             train_ds_trip) = _make_trip_loaders(
                train_segments, val_segments, test_segments,
                segment_types, config,
                train_df, clusters, known_stops,
                max_trip_length=15,
            )

            print(f"\n📊 Dataset Summary:")
            print(f"   Segment train: {len(train_ds_enh):,}  "
                  f"val: {len(val_loader_enh.dataset):,}  "
                  f"test: {len(test_loader_enh.dataset):,}")
            print(f"   Trip    train: {len(train_ds_trip):,} trips  "
                  f"val: {len(val_loader_trip.dataset):,}  "
                  f"test: {len(test_loader_trip.dataset):,}")
            if train_ds_enh.station_mapper is not None:
                sm = train_ds_enh.station_mapper
                n_matched = len(sm.cluster_to_station)
                print(f"   Clusters matched to named stations: "
                      f"{n_matched}/{sm.n_clusters}")

            # ── 7. Train MAGNN ────────────────────────────────────────
            print("\n[7/10] Training MAGNN (baseline)...")
>>>>>>> Stashed changes
            print("-" * 80)
            magnn_results, magnn_model = train_magtte(
                train_loader_magnn, val_loader_magnn, test_loader_magnn,
                adj_geo, adj_dist, adj_soc,
                segment_types, train_ds_magnn.target_scaler,
                output_folder, DEVICE, config
            )
            all_magnn_results.append(magnn_results)
            magnn_checkpoint = os.path.join(output_folder, 'magtte_best.pth')

<<<<<<< Updated upstream
            print("\n[9/11] Training MAGNN-LSTM-Residual...")
            print("-" * 80)
            residual_results, residual_model = train_magnn_lstm_residual(
                train_loader_residual, val_loader_residual, test_loader_residual,
                adj_geo, adj_dist, adj_soc,
                segment_types, train_dataset_residual.target_scaler,
=======
            # ── 8. Train Residual ─────────────────────────────────────
            print("\n[8/10] Training MAGNN-LSTM-Residual (station-aware)...")
            print("-" * 80)
            residual_results, residual_model = train_magnn_lstm_residual(
                train_loader_enh, val_loader_enh, test_loader_enh,
                adj_geo, adj_dist, adj_soc,
                segment_types, train_ds_enh.target_scaler,
>>>>>>> Stashed changes
                output_folder, DEVICE, config,
                clusters=clusters,
                known_stops=known_stops,
                pretrained_magnn_path=magnn_checkpoint,
                freeze_magnn=True,
            )
            all_residual_results.append(residual_results)

<<<<<<< Updated upstream
            print("\n[10/11] Training MAGNN-LSTM-MTL...")
            print("-" * 80)
            mtl_results, mtl_model = train_magnn_lstm_mtl(
                train_loader_mtl, val_loader_mtl, test_loader_mtl,
                adj_geo, adj_dist, adj_soc,
                segment_types, train_dataset_mtl.target_scaler,
=======
            # ── 9. Train DualTaskMTL ──────────────────────────────────
            # Uses the trained residual model as its frozen per-segment
            # encoder, and trip-level sequences as training data.
            print("\n[9/10] Training MAGNN-LSTM-DualTaskMTL "
                  "(trip-aware, residual encoder)...")
            print("-" * 80)
            mtl_results, mtl_model = train_magnn_lstm_dualtask_mtl(
                train_loader_trip, val_loader_trip, test_loader_trip,
                adj_geo, adj_dist, adj_soc,
                segment_types, train_ds_trip.target_scaler,
>>>>>>> Stashed changes
                output_folder, DEVICE, config,
                pretrained_residual_model=residual_model,
            )
            all_mtl_results.append(mtl_results)

<<<<<<< Updated upstream
            print("\n[11/11] Generating comparison report...")
            print_section(f"📊 ITERATION {iteration} - THREE-WAY COMPARISON")

=======
            # ── 10. Report ────────────────────────────────────────────
            print("\n[10/10] Generating comparison report...")
            print_section(
                f"📊 ITERATION {iteration} — THREE-WAY COMPARISON")
>>>>>>> Stashed changes
            for split in ['Train', 'Val', 'Test']:
                print_three_way_comparison_table(
                    magnn_results, residual_results, mtl_results, split)

            comparison_json = {
                'iteration': iteration,
                'algorithm': ALGORITHM_NAME,
                'config': {
                    'n_epochs':        config.n_epochs,
                    'batch_size':      config.batch_size,
                    'learning_rate':   config.learning_rate,
                    'sample_fraction': config.sample_fraction,
                    'mtl_lambda': config.mtl_lambda,
                },
                'magnn': {
                    sp: {k: float(v) for k, v in
                         magnn_results.get(sp, {}).items()
                         if k in ('r2', 'rmse', 'mae', 'mape')}
                    for sp in ('Train', 'Val', 'Test')
                },
                'residual': {
                    sp: {k: float(v) for k, v in
                         residual_results.get(sp, {}).items()
                         if k in ('r2', 'rmse', 'mae', 'mape', 'alpha')}
                    for sp in ('Train', 'Val', 'Test')
                },
<<<<<<< Updated upstream
                'mtl': {
                    split: {k: float(v) if isinstance(v, (int, float, np.number)) else str(v)
                            for k, v in mtl_results.get(split, {}).items()
                            if k in ['r2', 'rmse', 'mae', 'mape']}
                    for split in ['Train', 'Val', 'Test']
=======
                'dualtask_mtl': {
                    sp: {k: float(v) for k, v in
                         mtl_results.get(sp, {}).items()
                         if k in ('r2', 'rmse', 'mae', 'mape')}
                    for sp in ('Train', 'Val', 'Test')
>>>>>>> Stashed changes
                },
            }

            json_path = os.path.join(output_folder,
                                     'three_way_comparison.json')
            with open(json_path, 'w') as f:
                json.dump(comparison_json, f, indent=2)
            print(f"✓ Comparison saved → {json_path}\n")

        except Exception as e:
            print(f"\n❌ Error in iteration {iteration}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # ── Final summary ─────────────────────────────────────────────────────
    if all_magnn_results and all_residual_results and all_mtl_results:
        print_section("🏆 FINAL SUMMARY — AVERAGED ACROSS ALL ITERATIONS")

        for split in ['Train', 'Val', 'Test']:
            print(f"\n{'=' * 120}")
            print(f"{split.upper()} SET — "
                  f"AVERAGE OVER {len(all_magnn_results)} ITERATION(S)")
            print(f"{'=' * 120}")
<<<<<<< Updated upstream
            print(f"{'Metric':<10} {'MAGNN':<20} {'Residual':<20} {'MTL':<20} {'Best':<15}")
=======
            print(f"{'Metric':<10} {'MAGNN':<20} {'Residual':<20} "
                  f"{'DualTaskMTL':<20} {'Best':<15}")
>>>>>>> Stashed changes
            print(f"{'-' * 120}")

            for display_name, metric_key, higher_better in [
                ('R²',   'r2',   True),
                ('RMSE', 'rmse', False),
                ('MAE',  'mae',  False),
                ('MAPE', 'mape', False),
            ]:
                mv_vals = [r.get(split, {}).get(metric_key, float('nan'))
                           for r in all_magnn_results]
                rv_vals = [r.get(split, {}).get(metric_key, float('nan'))
                           for r in all_residual_results]
                tv_vals = [r.get(split, {}).get(metric_key, float('nan'))
                           for r in all_mtl_results]

                ma, ms = np.nanmean(mv_vals), np.nanstd(mv_vals)
                ra, rs = np.nanmean(rv_vals), np.nanstd(rv_vals)
                ta, ts = np.nanmean(tv_vals), np.nanstd(tv_vals)

                def _fmts(avg, std):
                    if metric_key in ('rmse', 'mae'):
                        return f"{avg:.2f}±{std:.2f}s"
                    if metric_key == 'mape':
                        return f"{avg:.2f}±{std:.2f}%"
                    return f"{avg:.4f}±{std:.4f}"

<<<<<<< Updated upstream
                magnn_std = np.nanstd(magnn_vals)
                residual_std = np.nanstd(residual_vals)
                mtl_std = np.nanstd(mtl_vals)

                if metric_key in ['rmse', 'mae']:
                    magnn_str = f"{magnn_avg:.2f}±{magnn_std:.2f}s"
                    residual_str = f"{residual_avg:.2f}±{residual_std:.2f}s"
                    mtl_str = f"{mtl_avg:.2f}±{mtl_std:.2f}s"
                elif metric_key == 'mape':
                    magnn_str = f"{magnn_avg:.2f}±{magnn_std:.2f}%"
                    residual_str = f"{residual_avg:.2f}±{residual_std:.2f}%"
                    mtl_str = f"{mtl_avg:.2f}±{mtl_std:.2f}%"
                else:
                    magnn_str = f"{magnn_avg:.4f}±{magnn_std:.4f}"
                    residual_str = f"{residual_avg:.4f}±{residual_std:.4f}"
                    mtl_str = f"{mtl_avg:.4f}��{mtl_std:.4f}"

                if not any(np.isnan(v) for v in [magnn_avg, residual_avg, mtl_avg]):
                    if higher_is_better:
                        best_val = max(magnn_avg, residual_avg, mtl_avg)
                        if best_val == residual_avg:
                            best_model = "Residual ✓"
                        elif best_val == mtl_avg:
                            best_model = "MTL ✓"
                        else:
                            best_model = "MAGNN"
                    else:
                        best_val = min(magnn_avg, residual_avg, mtl_avg)
                        if best_val == residual_avg:
                            best_model = "Residual ✓"
                        elif best_val == mtl_avg:
                            best_model = "MTL ✓"
                        else:
                            best_model = "MAGNN"
=======
                if not any(np.isnan(x) for x in [ma, ra, ta]):
                    best = (max if higher_better else min)(ma, ra, ta)
                    best_model = ("Residual ✓" if best == ra
                                  else "DualTaskMTL ✓" if best == ta
                                  else "MAGNN")
>>>>>>> Stashed changes
                else:
                    best_model = "N/A"

                print(f"{display_name:<10} {_fmts(ma,ms):<20} "
                      f"{_fmts(ra,rs):<20} {_fmts(ta,ts):<20} "
                      f"{best_model:<15}")

            print(f"{'=' * 120}\n")

<<<<<<< Updated upstream
        print_section("✅ THREE-WAY COMPARISON COMPLETE")
        print(f"  Total iterations: {len(all_magnn_results)}")
        print(f"\n  Model Details:")
        print(f"    MAGNN: Graph attention + LSTM baseline")
        print(f"    MAGNN-LSTM-Residual: ✅ Fixed + residual learning + adaptive gate")
        print(f"    MAGNN-LSTM-MTL: Multi-segment paths with MTL")
        print(f"\n  Results saved in: outputs/{ALGORITHM_NAME}_*_three_way/")
        print("=" * 80)
=======
    print_section("✅ THREE-WAY COMPARISON COMPLETE")
    print(f"  Results saved in: outputs/{ALGORITHM_NAME}_*_three_way/")
    print("=" * 80)
>>>>>>> Stashed changes


# =============================================================================
# ABLATION STUDY MODE
# =============================================================================

def print_ablation_comparison_table(full_results, no_op_results,
                                    no_weather_results, baseline_results,
                                    split_name):
    """Print formatted ablation study comparison table."""

    full_m       = full_results.get(split_name, {})
    no_op_m      = no_op_results.get(split_name, {})
    no_weather_m = no_weather_results.get(split_name, {})
    baseline_m   = baseline_results.get(split_name, {})

    print(f"\n{'=' * 130}")
    print(f"{split_name.upper()} SET — ABLATION STUDY COMPARISON")
    print(f"{'=' * 130}")
    print(f"{'Metric':<10} {'Full':<20} {'No Operational':<20} "
          f"{'No Weather':<20} {'Baseline':<20} {'Best':<15}")
    print(f"{'-' * 130}")

    for display_name, metric_key, higher_better in [
        ('R²',   'r2',   True),
        ('RMSE', 'rmse', False),
        ('MAE',  'mae',  False),
        ('MAPE', 'mape', False),
    ]:
        fv  = full_m.get(metric_key, float('nan'))
        ov  = no_op_m.get(metric_key, float('nan'))
        wv  = no_weather_m.get(metric_key, float('nan'))
        bv  = baseline_m.get(metric_key, float('nan'))

        def _fmt(v):
            if metric_key in ('rmse', 'mae'): return f"{v:.2f}s"
            if metric_key == 'mape':          return f"{v:.2f}%"
            return f"{v:.4f}"

        if not any(np.isnan(x) for x in [fv, ov, wv, bv]):
            best = (max if higher_better else min)(fv, ov, wv, bv)
            best_model = ("Full ✓"       if best == fv
                          else "No Op ✓"      if best == ov
                          else "No Weather ✓" if best == wv
                          else "Baseline ✓")
        else:
            best_model = "N/A"

        print(f"{display_name:<10} {_fmt(fv):<20} {_fmt(ov):<20} "
              f"{_fmt(wv):<20} {_fmt(bv):<20} {best_model:<15}")

    print(f"{'=' * 130}\n")


def main_with_ablation_study():
<<<<<<< Updated upstream
    """Ablation study: Feature importance analysis."""
=======
    """Ablation study: feature importance using MAGNN-LSTM-Residual."""
>>>>>>> Stashed changes

    config = Config()

    print_section("🔬 ABLATION STUDY MODE")
<<<<<<< Updated upstream
    print(f"  Device: {DEVICE}")
    print(f"  Algorithm: {ALGORITHM_NAME}")
    print(f"  Data sampling: {config.sample_fraction * 100}%")
    print(f"\n  Models to compare:")
    print(f"    1. MAGNN-LSTM-MTL (Full) - All features")
    print(f"    2. MAGNN-LSTM-MTL (No Operational) - Remove delay features")
    print(f"    3. MAGNN-LSTM-MTL (No Weather) - Remove weather features")
    print(f"    4. MAGNN-LSTM-MTL (Baseline) - Only spatial + temporal")
    print("=" * 80)

    # ... (rest of ablation code stays the same)
    print("Ablation study not modified - keeping original implementation")
=======
    print(f"  Device: {DEVICE}   Algorithm: {ALGORITHM_NAME}")
    print(f"\n  Variants:")
    print(f"    1. Full            — All features")
    print(f"    2. No Operational  — Remove delay features")
    print(f"    3. No Weather      — Remove weather features")
    print(f"    4. Baseline        — Spatial + temporal only")
    print("=" * 80)

    all_full_results          = []
    all_no_operational_results = []
    all_no_weather_results    = []
    all_baseline_results      = []

    WEATHER_COLS = ['temperature_2m', 'apparent_temperature', 'precipitation',
                    'rain', 'snowfall', 'windspeed_10m', 'windgusts_10m',
                    'winddirection_10m']

    for iteration in range(1, config.n_iterations + 1):
        print_section(f"🔄 ITERATION {iteration}/{config.n_iterations}")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_folder = (f"outputs/{ALGORITHM_NAME}_{timestamp}"
                         f"_i{iteration}_ablation")
        os.makedirs(output_folder, exist_ok=True)

        try:
            # ── 1. Data ───────────────────────────────────────────────
            print("\n[1/9] Loading data...")
            train_df, test_df, val_df = load_train_test_val_data_fixed(
                data_folder=config.data_folder,
                sample_fraction=config.sample_fraction
            )
            known_stops_train = get_known_stops(train_df)
            known_stops_test  = get_known_stops(test_df)
            known_stops_val   = get_known_stops(val_df)
            known_stops = {**known_stops_test, **known_stops_val,
                           **known_stops_train}
            print(f"   Known stops: {len(known_stops)}")

            # ── 2. Clustering ─────────────────────────────────────────
            print("[2/9] Clustering stops...")
            clusters, _ = event_driven_clustering_fixed(
                train_df, known_stops=known_stops)

            # ── 3. Segments ───────────────────────────────────────────
            print("[3/9] Building segments...")
            train_segments = build_segments_fixed(train_df, clusters)
            test_segments  = build_segments_fixed(test_df,  clusters)
            val_segments   = build_segments_fixed(val_df,   clusters)

            # ── 4. Adjacency ──────────────────────────────────────────
            print("[4/9] Building adjacency matrices...")
            adj_geo, adj_dist, adj_soc, segment_types = \
                build_adjacency_matrices_fixed(
                    train_segments, clusters, known_stops=known_stops)

            # ── 5. Train base MAGNN ───────────────────────────────────
            print("\n[5/9] Training base MAGNN for transfer learning...")
            print("-" * 80)
            train_ds_magnn = SegmentDataset(
                train_segments, segment_types, fit_scalers=True)
            val_ds_magnn = SegmentDataset(
                val_segments, segment_types, fit_scalers=False,
                target_scaler=train_ds_magnn.target_scaler,
                speed_scaler=train_ds_magnn.speed_scaler)
            test_ds_magnn = SegmentDataset(
                test_segments, segment_types, fit_scalers=False,
                target_scaler=train_ds_magnn.target_scaler,
                speed_scaler=train_ds_magnn.speed_scaler)

            tl_m = DataLoader(train_ds_magnn, batch_size=config.batch_size,
                              shuffle=True,  num_workers=0,
                              collate_fn=masked_collate_fn)
            vl_m = DataLoader(val_ds_magnn, batch_size=config.batch_size,
                              shuffle=False, num_workers=0,
                              collate_fn=masked_collate_fn)
            tstl_m = DataLoader(test_ds_magnn, batch_size=config.batch_size,
                                shuffle=False, num_workers=0,
                                collate_fn=masked_collate_fn)

            _, magnn_model = train_magtte(
                tl_m, vl_m, tstl_m,
                adj_geo, adj_dist, adj_soc,
                segment_types, train_ds_magnn.target_scaler,
                output_folder, DEVICE, config
            )
            magnn_checkpoint = os.path.join(output_folder, 'magtte_best.pth')

            # ----------------------------------------------------------
            # Helper: build enhanced loaders from (optionally zeroed) segs
            # ----------------------------------------------------------
            def _ablation_loaders(tr_seg, va_seg, te_seg):
                return _make_enhanced_loaders(
                    tr_seg, va_seg, te_seg,
                    segment_types, config,
                    train_df, clusters, known_stops
                )

            def _run_residual(tl, vl, tstl, ds_train, tag):
                return train_magnn_lstm_residual(
                    tl, vl, tstl,
                    adj_geo, adj_dist, adj_soc,
                    segment_types, ds_train.target_scaler,
                    output_folder, DEVICE, config,
                    clusters=clusters,
                    known_stops=known_stops,
                    pretrained_magnn_path=magnn_checkpoint,
                    freeze_magnn=True,
                )

            # ── 6. FULL ───────────────────────────────────────────────
            print("\n[6/9] Training Residual — FULL (all features)...")
            print("-" * 80)
            tl, vl, tstl, ds = _ablation_loaders(
                train_segments, val_segments, test_segments)
            full_results, _ = _run_residual(tl, vl, tstl, ds, "full")
            all_full_results.append(full_results)
            print("   ✓ Full model trained")

            # ── 7. NO OPERATIONAL ─────────────────────────────────────
            print("\n[7/9] Training Residual — NO OPERATIONAL...")
            print("-" * 80)
            tr_no_op = train_segments.copy()
            va_no_op = val_segments.copy()
            te_no_op = test_segments.copy()
            for df in (tr_no_op, va_no_op, te_no_op):
                df['arrivalDelay']   = 0.0
                df['departureDelay'] = 0.0
                df['is_weekend']     = 0.0
                df['is_peak_hour']   = 0.0

            tl, vl, tstl, ds = _ablation_loaders(tr_no_op, va_no_op, te_no_op)
            no_op_results, _ = _run_residual(tl, vl, tstl, ds, "no_op")
            all_no_operational_results.append(no_op_results)
            print("   ✓ No-operational model trained")

            # ── 8. NO WEATHER ─────────────────────────────────────────
            print("\n[8/9] Training Residual — NO WEATHER...")
            print("-" * 80)
            tr_no_w = train_segments.copy()
            va_no_w = val_segments.copy()
            te_no_w = test_segments.copy()
            for df in (tr_no_w, va_no_w, te_no_w):
                for col in WEATHER_COLS:
                    if col in df.columns:
                        df[col] = 0.0

            tl, vl, tstl, ds = _ablation_loaders(tr_no_w, va_no_w, te_no_w)
            no_weather_results, _ = _run_residual(tl, vl, tstl, ds,
                                                  "no_weather")
            all_no_weather_results.append(no_weather_results)
            print("   ✓ No-weather model trained")

            # ── 9. BASELINE ───────────────────────────────────────────
            print("\n[9/9] Training Residual — BASELINE "
                  "(spatial + temporal only)...")
            print("-" * 80)
            tr_bl = train_segments.copy()
            va_bl = val_segments.copy()
            te_bl = test_segments.copy()
            for df in (tr_bl, va_bl, te_bl):
                df['arrivalDelay']   = 0.0
                df['departureDelay'] = 0.0
                df['is_weekend']     = 0.0
                df['is_peak_hour']   = 0.0
                for col in WEATHER_COLS:
                    if col in df.columns:
                        df[col] = 0.0

            tl, vl, tstl, ds = _ablation_loaders(tr_bl, va_bl, te_bl)
            baseline_results, _ = _run_residual(tl, vl, tstl, ds, "baseline")
            all_baseline_results.append(baseline_results)
            print("   ✓ Baseline model trained")

            # ── Comparison table ──────────────────────────────────────
            print_section(
                f"📊 ITERATION {iteration} — ABLATION STUDY RESULTS")
            for split in ('Train', 'Val', 'Test'):
                print_ablation_comparison_table(
                    full_results, no_op_results,
                    no_weather_results, baseline_results, split)

            # Save JSON
            ablation_json = {
                'iteration': iteration,
                'algorithm': ALGORITHM_NAME,
                'model':     'MAGNN-LSTM-Residual (station-aware)',
                'config': {
                    'n_epochs':        config.n_epochs,
                    'batch_size':      config.batch_size,
                    'learning_rate':   config.learning_rate,
                    'sample_fraction': config.sample_fraction,
                },
            }
            _keep = ('r2', 'rmse', 'mae', 'mape', 'alpha',
                     'correction_mean', 'correction_std')
            for key, res in [('full',           full_results),
                              ('no_operational', no_op_results),
                              ('no_weather',     no_weather_results),
                              ('baseline',       baseline_results)]:
                ablation_json[key] = {
                    sp: {k: float(v) for k, v in res.get(sp, {}).items()
                         if k in _keep}
                    for sp in ('Train', 'Val', 'Test')
                }

            json_path = os.path.join(output_folder, 'ablation_study.json')
            with open(json_path, 'w') as f:
                json.dump(ablation_json, f, indent=2)
            print(f"✓ Ablation results saved → {json_path}\n")

        except Exception as e:
            print(f"\n❌ Error in iteration {iteration}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # ── Final summary ─────────────────────────────────────────────────────
    if (all_full_results and all_no_operational_results
            and all_no_weather_results and all_baseline_results):

        print_section("🏆 ABLATION STUDY — FINAL SUMMARY")

        for split in ('Train', 'Val', 'Test'):
            print(f"\n{'=' * 130}")
            print(f"{split.upper()} — "
                  f"AVERAGE OVER {len(all_full_results)} ITERATION(S)")
            print(f"{'=' * 130}")
            print(f"{'Metric':<10} {'Full':<20} {'No Operational':<20} "
                  f"{'No Weather':<20} {'Baseline':<20} {'Best':<15}")
            print(f"{'-' * 130}")

            for display_name, metric_key, higher_better in [
                ('R²',   'r2',   True),
                ('RMSE', 'rmse', False),
                ('MAE',  'mae',  False),
                ('MAPE', 'mape', False),
            ]:
                fa = np.nanmean([r.get(split, {}).get(metric_key, float('nan'))
                                 for r in all_full_results])
                oa = np.nanmean([r.get(split, {}).get(metric_key, float('nan'))
                                 for r in all_no_operational_results])
                wa = np.nanmean([r.get(split, {}).get(metric_key, float('nan'))
                                 for r in all_no_weather_results])
                ba = np.nanmean([r.get(split, {}).get(metric_key, float('nan'))
                                 for r in all_baseline_results])
                fs = np.nanstd([r.get(split, {}).get(metric_key, float('nan'))
                                for r in all_full_results])
                os_ = np.nanstd([r.get(split, {}).get(metric_key, float('nan'))
                                 for r in all_no_operational_results])
                ws = np.nanstd([r.get(split, {}).get(metric_key, float('nan'))
                                for r in all_no_weather_results])
                bs = np.nanstd([r.get(split, {}).get(metric_key, float('nan'))
                                for r in all_baseline_results])

                def _fmts(avg, std):
                    if metric_key in ('rmse', 'mae'):
                        return f"{avg:.2f}±{std:.2f}s"
                    if metric_key == 'mape':
                        return f"{avg:.2f}±{std:.2f}%"
                    return f"{avg:.4f}±{std:.4f}"

                if not any(np.isnan(x) for x in [fa, oa, wa, ba]):
                    best = (max if higher_better else min)(fa, oa, wa, ba)
                    best_model = ("Full ✓"       if best == fa
                                  else "No Op ✓"      if best == oa
                                  else "No Weather ✓" if best == wa
                                  else "Baseline ✓")
                else:
                    best_model = "N/A"

                print(f"{display_name:<10} {_fmts(fa,fs):<20} "
                      f"{_fmts(oa,os_):<20} {_fmts(wa,ws):<20} "
                      f"{_fmts(ba,bs):<20} {best_model:<15}")

            print(f"{'=' * 130}\n")

        # Feature importance analysis
        print_section("🔍 FEATURE IMPORTANCE ANALYSIS")
        for split in ('Test',):
            full_r2     = np.nanmean([r.get(split, {}).get('r2', float('nan'))
                                      for r in all_full_results])
            no_op_r2    = np.nanmean([r.get(split, {}).get('r2', float('nan'))
                                      for r in all_no_operational_results])
            no_w_r2     = np.nanmean([r.get(split, {}).get('r2', float('nan'))
                                      for r in all_no_weather_results])
            baseline_r2 = np.nanmean([r.get(split, {}).get('r2', float('nan'))
                                      for r in all_baseline_results])

            op_contribution = full_r2 - no_op_r2
            w_contribution  = full_r2 - no_w_r2
            combined        = full_r2 - baseline_r2

            print(f"\n{split.upper()} SET — Feature Contribution Analysis")
            print(f"  Full Model (All features):          {full_r2:.4f}")
            print(f"  No Operational (Weather only):      {no_op_r2:.4f}")
            print(f"  No Weather (Operational only):      {no_w_r2:.4f}")
            print(f"  Baseline (Spatial + Temporal only): {baseline_r2:.4f}")
            print(f"\n  ΔR² Contributions:")
            print(f"    Operational features:    {op_contribution:+.4f}")
            print(f"    Weather features:        {w_contribution:+.4f}")
            print(f"    Combined (Op + Weather): {combined:+.4f}")

            if op_contribution > w_contribution:
                print(f"\n  💡 Operational features contribute MORE "
                      f"(Δ={op_contribution - w_contribution:+.4f})")
            elif w_contribution > op_contribution:
                print(f"\n  💡 Weather features contribute MORE "
                      f"(Δ={w_contribution - op_contribution:+.4f})")
            else:
                print(f"\n  💡 Both feature groups contribute equally")

            # Alpha analysis
            fa_alpha = np.nanmean([r.get(split, {}).get('alpha', float('nan'))
                                   for r in all_full_results])
            oa_alpha = np.nanmean([r.get(split, {}).get('alpha', float('nan'))
                                   for r in all_no_operational_results])
            wa_alpha = np.nanmean([r.get(split, {}).get('alpha', float('nan'))
                                   for r in all_no_weather_results])
            ba_alpha = np.nanmean([r.get(split, {}).get('alpha', float('nan'))
                                   for r in all_baseline_results])
            fs_alpha = np.nanstd([r.get(split, {}).get('alpha', float('nan'))
                                  for r in all_full_results])
            os_alpha = np.nanstd([r.get(split, {}).get('alpha', float('nan'))
                                  for r in all_no_operational_results])
            ws_alpha = np.nanstd([r.get(split, {}).get('alpha', float('nan'))
                                  for r in all_no_weather_results])
            bs_alpha = np.nanstd([r.get(split, {}).get('alpha', float('nan'))
                                  for r in all_baseline_results])

            print(f"\n  Adaptive Gating (α) Analysis:")
            print(f"    Full:           α = {fa_alpha:.3f} ± {fs_alpha:.3f}")
            print(f"    No Operational: α = {oa_alpha:.3f} ± {os_alpha:.3f}")
            print(f"    No Weather:     α = {wa_alpha:.3f} ± {ws_alpha:.3f}")
            print(f"    Baseline:       α = {ba_alpha:.3f} ± {bs_alpha:.3f}")

    print_section("✅ ABLATION STUDY COMPLETE")
    print(f"  Total iterations: {len(all_full_results)}")
    print(f"  Results saved in: outputs/{ALGORITHM_NAME}_*_ablation/")
    print("=" * 80)
>>>>>>> Stashed changes


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == '__main__':
<<<<<<< Updated upstream
    import sys
=======
    if _sample_fraction_cli is not None:
        Config.sample_fraction = _sample_fraction_cli
>>>>>>> Stashed changes

    if len(sys.argv) > 1:
        if sys.argv[1] == '--compare-all':
            main_with_three_way_comparison()
        elif sys.argv[1] == '--ablation':
            main_with_ablation_study()
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print("\nUsage:")
            print("  python main.py                # Train MAGNN only")
            print("  python main.py --compare-all  # Compare MAGNN vs Residual vs MTL")
            print("  python main.py --ablation     # Feature ablation study")
    else:
        main()