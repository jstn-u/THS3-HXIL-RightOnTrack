"""
main.py
=======
Entry point for MAGNN-LSTM transit travel time prediction.

✅ THREE MODEL COMPARISON:
   1. MAGNN (baseline)
   2. MAGNN-LSTM-Residual (fixed version with residual learning + adaptive gating)
   3. MAGNN-LSTM-DualTaskMTL (✅ NEW: Local segments + Global stations)

Usage:
    python main.py                      # defaults (sample from config, hdbscan)
    python main.py 0.1                  # 10% sample, hdbscan
    python main.py 1.0 knn             # 100% sample, KNN clustering
    python main.py 0.5 dbscan          # 50% sample, DBSCAN clustering
    python main.py --compare-all        # Compare all three models
    python main.py --ablation           # Feature ablation studies
    python main.py 0.1 knn --compare-all  # combine all options

Supported clustering methods: hdbscan, dbscan, knn, gmm, kmeans
Both sample_fraction and method are optional and can appear in any order.
"""

# =============================================================================
# DYNAMIC CLUSTERING — resolved at runtime from CLI args
# =============================================================================
import sys
import importlib

VALID_METHODS = {'hdbscan', 'dbscan', 'knn', 'gmm', 'kmeans'}

# --- Parse CLI args (sample fraction, clustering method, mode flag) ---
_sample_fraction_cli = None          # None → use Config default
_cluster_method = 'hdbscan'          # default clustering method
_mode = None                         # None → baseline main()

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

# Dynamically import the chosen clustering module
_cluster_module = importlib.import_module(f'cluster_{_cluster_method}')
event_driven_clustering_fixed = _cluster_module.event_driven_clustering_fixed

ALGORITHM_NAME = _cluster_method.upper()

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
                   train_magnn_lstm_mtl,
                   train_magnn_lstm_dualtask_mtl,  # ✅ NEW
                   MAGTTE)
from residual import MAGNN_LSTM_Residual

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
# MAIN PIPELINE (BASELINE MAGNN ONLY)
# =============================================================================

def main():
    """Main training loop - baseline MAGNN only."""

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
            known_stops_test = get_known_stops(test_df)
            known_stops_val = get_known_stops(val_df)
            known_stops = {**known_stops_test, **known_stops_val, **known_stops_train}
            print(f"   Known stops found in data: {len(known_stops)} "
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
            val_segments = build_segments_fixed(val_df, clusters)

            # ------------------------------------------------------------------
            # 4. PLOTS
            # ------------------------------------------------------------------
            print_section("GENERATING VISUALISATIONS")

            plot_clusters(clusters, {},
                          algorithm_name=ALGORITHM_NAME,
                          save_path=os.path.join(output_folder, f'{ALGORITHM_NAME.lower()}-clusters.png'))
            plot_segments(train_segments, clusters, max_segments=100,
                          algorithm_name=ALGORITHM_NAME,
                          save_path=os.path.join(output_folder, f'{ALGORITHM_NAME.lower()}-segments.png'))
            plot_segment_statistics(train_segments,
                                    algorithm_name=ALGORITHM_NAME,
                                    save_path=os.path.join(output_folder,
                                                           f'{ALGORITHM_NAME.lower()}-segment_stats.png'))

            # ------------------------------------------------------------------
            # 5. Adjacency matrices
            # ------------------------------------------------------------------
            adj_geo, adj_dist, adj_soc, segment_types = build_adjacency_matrices_fixed(
                train_segments, clusters,
                known_stops=known_stops
            )

            if adj_geo is None:
                print("❌ Adjacency failed")
                continue

            # ------------------------------------------------------------------
            # 6. Datasets & data loaders
            # ------------------------------------------------------------------
            train_dataset = SegmentDataset(
                train_segments, segment_types,
                fit_scalers=True
            )
            val_dataset = SegmentDataset(
                val_segments, segment_types,
                fit_scalers=False,
                target_scaler=train_dataset.target_scaler,
                speed_scaler=train_dataset.speed_scaler,
            )
            test_dataset = SegmentDataset(
                test_segments, segment_types,
                fit_scalers=False,
                target_scaler=train_dataset.target_scaler,
                speed_scaler=train_dataset.speed_scaler,
            )

            train_loader = DataLoader(
                train_dataset, batch_size=config.batch_size,
                shuffle=True, num_workers=0,
                collate_fn=masked_collate_fn
            )
            val_loader = DataLoader(
                val_dataset, batch_size=config.batch_size,
                shuffle=False, num_workers=0,
                collate_fn=masked_collate_fn
            )
            test_loader = DataLoader(
                test_dataset, batch_size=config.batch_size,
                shuffle=False, num_workers=0,
                collate_fn=masked_collate_fn
            )

            print(f"\n📊 Data Summary:")
            print(f"   Segment types (unique): {len(segment_types)}")
            print(f"   Training samples      : {len(train_dataset):,}")
            print(f"   Validation samples    : {len(val_dataset):,}")
            print(f"   Test samples          : {len(test_dataset):,}")
            print(f"   Training batches      : {len(train_loader)}")
            print(f"   Scaler type           : RobustScaler (median+IQR, outlier-resistant)")

            if len(train_loader) == 0:
                print("❌ Training loader is empty — skipping")
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
            metrics_out = {
                'graph_method': ALGORITHM_NAME.lower(),
                'config': {
                    'n_epochs': config.n_epochs,
                    'batch_size': config.batch_size,
                    'learning_rate': config.learning_rate,
                    'sample_fraction': config.sample_fraction,
                    'n_clusters': len(clusters),
                    'n_segment_types': len(segment_types),
                    'node_embed_dim': config.node_embed_dim,
                    'gat_hidden': config.gat_hidden,
                    'lstm_hidden': config.lstm_hidden,
                    'historical_dim': config.historical_dim,
                },
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            }

            for dataset_name, res in [('Train', results.get('Train')),
                                      ('Val', results.get('Val')),
                                      ('Test', results.get('Test'))]:
                if res:
                    metrics_out[dataset_name] = {
                        'R2': f"{res['r2']:.4f}",
                        'RMSE': f"{res['rmse']:.2f} seconds",
                        'MAE': f"{res['mae']:.2f} seconds",
                        'MAPE': f"{res['mape']:.2f}%",
                    }

            json_path = os.path.join(output_folder, 'metrics.json')
            with open(json_path, 'w') as f:
                json.dump(metrics_out, f, indent=2)
            print(f"\n✓ Metrics saved → {json_path}")

            print(f"\n✅ Iteration {iteration} complete")
            print(f"   Output files saved to: {output_folder}/")

        except Exception as e:
            print(f"\n❌ Error in iteration {iteration}: {e}")
            import traceback
            traceback.print_exc()
            continue


# =============================================================================
# RESIDUAL MODEL TRAINING FUNCTION
# =============================================================================

def train_magnn_lstm_residual(train_loader, val_loader, test_loader,
                              adj_geo, adj_dist, adj_soc,
                              segment_types, scaler,
                              output_folder, device, cfg,
                              pretrained_magnn_path=None,
                              freeze_magnn=True):
    """Train MAGNN-LSTM-Residual with optimized hyperparameters."""

    print_section("MAGNN-LSTM-RESIDUAL TRAINING")
    num_segments = len(segment_types)

    magnn_base = MAGTTE(num_segments, cfg.n_heads, cfg.node_embed_dim, cfg.gat_hidden,
                        cfg.lstm_hidden, cfg.historical_dim, cfg.dropout).to(device)
    magnn_base.set_adjacency_matrices(adj_geo, adj_dist, adj_soc)

    if pretrained_magnn_path and os.path.exists(pretrained_magnn_path):
        magnn_base.load_state_dict(torch.load(pretrained_magnn_path, map_location=device))
        print(f"   ✓ Loaded pre-trained MAGNN")

    model = MAGNN_LSTM_Residual(
        magnn_base,
        cfg.gat_hidden,
        4,  # operational_dim
        8,  # weather_dim
        5,  # temporal_dim
        128,  # lstm_hidden
        1,  # lstm_layers
        0.2,  # dropout
        freeze_magnn
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Trainable params: {n_params:,}")

    learning_rate = cfg.residual_learning_rate if hasattr(cfg, 'residual_learning_rate') else 0.0005
    weight_decay = cfg.lstm_weight_decay if hasattr(cfg, 'lstm_weight_decay') else 1e-6

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    print(f"   ✅ Optimization settings:")
    print(f"      Learning rate: {learning_rate}")
    print(f"      Weight decay: {weight_decay}")

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    criterion = nn.SmoothL1Loss()

    best_val_loss = float('inf')
    patience_counter = 0
    best_ckpt = os.path.join(output_folder, 'magnn_lstm_residual_best.pth')

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_r2': [],
        'alpha': [],
        'correction': []
    }

    print()
    for epoch in range(1, cfg.n_epochs + 1):
        model.train()
        train_loss = 0.0
        alpha_sum = 0.0
        correction_sum = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            seg_idx, temporal, operational, weather, target, lengths, mask = batch
            seg_idx = seg_idx.to(device)
            temporal = temporal.squeeze(1).to(device)
            operational = operational.squeeze(1).to(device)
            weather = weather.squeeze(1).to(device)
            target = target.to(device)

            baseline, correction, alpha, predictions = model(
                seg_idx, temporal, operational, weather, return_components=True
            )

            loss = criterion(predictions, target)
            if torch.isnan(loss) or torch.isinf(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            alpha_sum += alpha.mean().item()
            correction_sum += correction.abs().mean().item()
            n_batches += 1

        train_loss /= max(n_batches, 1)
        avg_alpha = alpha_sum / max(n_batches, 1)
        avg_correction = correction_sum / max(n_batches, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        val_alpha_sum = 0.0
        val_correction_sum = 0.0
        n_val = 0

        with torch.no_grad():
            for batch in val_loader:
                seg_idx, temporal, operational, weather, target, lengths, mask = batch
                seg_idx = seg_idx.to(device)
                temporal = temporal.squeeze(1).to(device)
                operational = operational.squeeze(1).to(device)
                weather = weather.squeeze(1).to(device)
                target = target.to(device)

                baseline, correction, alpha, predictions = model(
                    seg_idx, temporal, operational, weather, return_components=True
                )

                loss = criterion(predictions, target)
                if not torch.isnan(loss) and not torch.isinf(loss):
                    val_loss += loss.item()
                    n_val += 1
                    val_preds.append(predictions.cpu().numpy())
                    val_targets.append(target.cpu().numpy())
                    val_alpha_sum += alpha.mean().item()
                    val_correction_sum += correction.abs().mean().item()

        val_loss /= max(n_val, 1)
        val_alpha = val_alpha_sum / max(n_val, 1)
        val_correction = val_correction_sum / max(n_val, 1)
        scheduler.step(val_loss)

        if val_preds:
            vp = scaler.inverse_transform(np.concatenate(val_preds))
            vt = scaler.inverse_transform(np.concatenate(val_targets))
            val_r2 = r2_score(vt, vp)
        else:
            val_r2 = float('nan')

        correction_seconds = avg_correction * scaler.scale_[0] if hasattr(scaler, 'scale_') else avg_correction
        val_correction_seconds = val_correction * scaler.scale_[0] if hasattr(scaler, 'scale_') else val_correction

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_r2'].append(val_r2)
        history['alpha'].append(avg_alpha)
        history['correction'].append(correction_seconds)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3}/{cfg.n_epochs}  "
                  f"loss={train_loss:.4f}  "
                  f"val_R²={val_r2:.4f}  "
                  f"α={avg_alpha:.3f}  "
                  f"corr={correction_seconds:.2f}s")

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

    print_section("MAGNN-LSTM-RESIDUAL — FINAL RESULTS")

    def _eval(loader, name):
        model.eval()
        preds, targets = [], []
        alphas = []
        corrections = []

        with torch.no_grad():
            for batch in loader:
                seg_idx, temporal, operational, weather, target, lengths, mask = batch
                seg_idx = seg_idx.to(device)
                temporal = temporal.squeeze(1).to(device)
                operational = operational.squeeze(1).to(device)
                weather = weather.squeeze(1).to(device)

                baseline, correction, alpha, pred = model(
                    seg_idx, temporal, operational, weather, return_components=True
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
        c = np.concatenate(corrections)
        c_mean = scaler.inverse_transform(c).mean()
        c_std = scaler.inverse_transform(c).std()

        r2 = r2_score(t, p)
        rmse = np.sqrt(mean_squared_error(t, p))
        mae = mean_absolute_error(t, p)
        mask_valid = t.flatten() > 0
        mape = np.mean(np.abs((t.flatten()[mask_valid] - p.flatten()[mask_valid]) /
                              t.flatten()[mask_valid])) * 100 if mask_valid.any() else float('nan')

        print(f"   {name:<6}  R²={r2:.4f}  RMSE={rmse:.2f}s  MAE={mae:.2f}s  MAPE={mape:.2f}%")
        print(f"           α={a:.3f}  corr={c_mean:.2f}±{c_std:.2f}s")

        return {'r2': r2, 'rmse': rmse, 'mae': mae, 'mape': mape,
                'preds': p.flatten().tolist(), 'actual': t.flatten().tolist(),
                'alpha': float(a), 'correction_mean': float(c_mean),
                'correction_std': float(c_std)}

    results = {'Train': _eval(train_loader, 'Train'),
               'Val': _eval(val_loader, 'Val'),
               'Test': _eval(test_loader, 'Test')}

    return results, model


# =============================================================================
# THREE-WAY COMPARISON HELPER
# =============================================================================

def print_three_way_comparison_table(magnn_results, residual_results, mtl_results, split_name):
    """Print formatted three-way comparison table."""

    magnn_m = magnn_results.get(split_name, {})
    residual_m = residual_results.get(split_name, {})
    mtl_m = mtl_results.get(split_name, {})

    print(f"\n{'=' * 120}")
    print(f"{split_name.upper()} SET - THREE-WAY COMPARISON")
    print(f"{'=' * 120}")
    print(f"{'Metric':<10} {'MAGNN':<20} {'Residual':<20} {'DualTaskMTL':<20} {'Best Model':<20} {'Improvement':<15}")
    print(f"{'-' * 120}")

    metrics_info = [
        ('R²', 'r2', True),
        ('RMSE', 'rmse', False),
        ('MAE', 'mae', False),
        ('MAPE', 'mape', False),
    ]

    for display_name, metric_key, higher_is_better in metrics_info:
        magnn_val = magnn_m.get(metric_key, float('nan'))
        residual_val = residual_m.get(metric_key, float('nan'))
        mtl_val = mtl_m.get(metric_key, float('nan'))

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
                    best_model = "DualTaskMTL ✓"
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
                    best_model = "DualTaskMTL ✓"
                    improvement = ((magnn_val - mtl_val) / abs(magnn_val) * 100)
                else:
                    best_model = "MAGNN (baseline)"
                    improvement = 0.0

            improvement_str = f"+{improvement:.2f}%" if improvement > 0 else f"{improvement:.2f}%"
        else:
            best_model = "N/A"
            improvement_str = "N/A"

        print(
            f"{display_name:<10} {magnn_str:<20} {residual_str:<20} {mtl_str:<20} {best_model:<20} {improvement_str:<15}")

    print(f"{'=' * 120}\n")


# =============================================================================
# THREE-WAY COMPARISON MODE (✅ UPDATED WITH DUALTASK MTL)
# =============================================================================

def main_with_three_way_comparison():
    """Three-way comparison: MAGNN vs MAGNN-LSTM-Residual vs MAGNN-LSTM-DualTaskMTL"""

    config = Config()

    print_section("🔬 THREE-WAY COMPARISON MODE")
    print(f"  Device: {DEVICE}")
    print(f"  Algorithm: {ALGORITHM_NAME}")
    print(f"\n  Models to compare:")
    print(f"    1. MAGNN (baseline)")
    print(f"    2. MAGNN-LSTM-Residual (residual learning + adaptive gate)")
    print(f"    3. MAGNN-LSTM-DualTaskMTL (✅ NEW: Local + Global tasks)")
    print(f"\n  DualTaskMTL features:")
    print(f"    ✅ Uses segment-level data (45K+ samples, not 2K paths)")
    print(f"    ✅ Local task: Predict individual segment durations")
    print(f"    ✅ Global task: Predict total journey time")
    print(f"    ✅ Uncertainty weighting: Auto-balance both tasks")
    print("=" * 80)

    all_magnn_results = []
    all_residual_results = []
    all_mtl_results = []

    for iteration in range(1, config.n_iterations + 1):
        print_section(f"🔄 ITERATION {iteration}/{config.n_iterations}")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_folder = f"outputs/{ALGORITHM_NAME}_{timestamp}_i{iteration}_three_way"
        os.makedirs(output_folder, exist_ok=True)

        try:
            print("\n[1/10] Loading data...")
            train_df, test_df, val_df = load_train_test_val_data_fixed(
                data_folder=config.data_folder,
                sample_fraction=config.sample_fraction
            )

            known_stops_train = get_known_stops(train_df)
            known_stops_test = get_known_stops(test_df)
            known_stops_val = get_known_stops(val_df)
            known_stops = {**known_stops_test, **known_stops_val, **known_stops_train}

            print("[2/10] Clustering stops...")
            clusters, station_cluster_ids = event_driven_clustering_fixed(
                train_df, known_stops=known_stops
            )

            print("[3/10] Building segments...")
            train_segments = build_segments_fixed(train_df, clusters)
            test_segments = build_segments_fixed(test_df, clusters)
            val_segments = build_segments_fixed(val_df, clusters)

            # ------------------------------------------------------------------
            # [3b/10] Generate cluster / segment visualisations
            #         (same helpers as normal main() — no pipeline changes)
            # ------------------------------------------------------------------
            print("[3b/10] Generating visualisations...")
            plot_clusters(clusters, {},
                          algorithm_name=ALGORITHM_NAME,
                          save_path=os.path.join(output_folder,
                                                  f'{ALGORITHM_NAME.lower()}-clusters.png'))
            plot_segments(train_segments, clusters, max_segments=100,
                          algorithm_name=ALGORITHM_NAME,
                          save_path=os.path.join(output_folder,
                                                  f'{ALGORITHM_NAME.lower()}-segments.png'))
            plot_segment_statistics(train_segments,
                                    algorithm_name=ALGORITHM_NAME,
                                    save_path=os.path.join(output_folder,
                                                           f'{ALGORITHM_NAME.lower()}-segment_stats.png'))

            print("[4/10] Building adjacency matrices...")
            adj_geo, adj_dist, adj_soc, segment_types = build_adjacency_matrices_fixed(
                train_segments, clusters, known_stops=known_stops
            )

            print("[5/10] Preparing MAGNN datasets...")
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

            print("[6/10] Preparing Enhanced datasets (for Residual & DualTaskMTL)...")
            train_dataset_enhanced = EnhancedSegmentDataset(train_segments, segment_types, fit_scalers=True)
            val_dataset_enhanced = EnhancedSegmentDataset(val_segments, segment_types, fit_scalers=False,
                                                          target_scaler=train_dataset_enhanced.target_scaler,
                                                          speed_scaler=train_dataset_enhanced.speed_scaler,
                                                          operational_scaler=train_dataset_enhanced.operational_scaler,
                                                          weather_scaler=train_dataset_enhanced.weather_scaler)
            test_dataset_enhanced = EnhancedSegmentDataset(test_segments, segment_types, fit_scalers=False,
                                                           target_scaler=train_dataset_enhanced.target_scaler,
                                                           speed_scaler=train_dataset_enhanced.speed_scaler,
                                                           operational_scaler=train_dataset_enhanced.operational_scaler,
                                                           weather_scaler=train_dataset_enhanced.weather_scaler)

            train_loader_enhanced = DataLoader(train_dataset_enhanced, batch_size=config.batch_size,
                                               shuffle=True, num_workers=0, collate_fn=enhanced_collate_fn)
            val_loader_enhanced = DataLoader(val_dataset_enhanced, batch_size=config.batch_size,
                                             shuffle=False, num_workers=0, collate_fn=enhanced_collate_fn)
            test_loader_enhanced = DataLoader(test_dataset_enhanced, batch_size=config.batch_size,
                                              shuffle=False, num_workers=0, collate_fn=enhanced_collate_fn)

            print(f"\n📊 Dataset Summary:")
            print(f"   Training segments: {len(train_dataset_enhanced):,}")
            print(f"   Validation segments: {len(val_dataset_enhanced):,}")
            print(f"   Test segments: {len(test_dataset_enhanced):,}")

            print("\n[7/10] Training MAGNN (baseline)...")
            print("-" * 80)
            magnn_results, magnn_model = train_magtte(
                train_loader_magnn, val_loader_magnn, test_loader_magnn,
                adj_geo, adj_dist, adj_soc,
                segment_types, train_dataset_magnn.target_scaler,
                output_folder, DEVICE, config
            )
            all_magnn_results.append(magnn_results)
            magnn_checkpoint = os.path.join(output_folder, 'magtte_best.pth')

            print("\n[8/10] Training MAGNN-LSTM-Residual...")
            print("-" * 80)
            residual_results, residual_model = train_magnn_lstm_residual(
                train_loader_enhanced, val_loader_enhanced, test_loader_enhanced,
                adj_geo, adj_dist, adj_soc,
                segment_types, train_dataset_enhanced.target_scaler,
                output_folder, DEVICE, config,
                pretrained_magnn_path=magnn_checkpoint,
                freeze_magnn=True
            )
            all_residual_results.append(residual_results)

            # ✅ NEW: DualTaskMTL using SEGMENT data (not paths!)
            print("\n[9/10] Training MAGNN-LSTM-DualTaskMTL (Local + Global)...")
            print("-" * 80)
            mtl_results, mtl_model = train_magnn_lstm_dualtask_mtl(
                train_loader_enhanced, val_loader_enhanced, test_loader_enhanced,
                adj_geo, adj_dist, adj_soc,
                segment_types, train_dataset_enhanced.target_scaler,
                output_folder, DEVICE, config,
                pretrained_magnn_path=magnn_checkpoint,
                freeze_magnn=True
            )
            all_mtl_results.append(mtl_results)

            print("\n[10/10] Generating comparison report...")
            print_section(f"📊 ITERATION {iteration} - THREE-WAY COMPARISON")

            for split in ['Train', 'Val', 'Test']:
                print_three_way_comparison_table(magnn_results, residual_results, mtl_results, split)

            comparison_json = {
                'iteration': iteration,
                'algorithm': ALGORITHM_NAME,
                'config': {
                    'n_epochs': config.n_epochs,
                    'batch_size': config.batch_size,
                    'learning_rate': config.learning_rate,
                    'sample_fraction': config.sample_fraction,
                },
                'magnn': {
                    split: {k: float(v) if isinstance(v, (int, float, np.number)) else str(v)
                            for k, v in magnn_results.get(split, {}).items()
                            if k in ['r2', 'rmse', 'mae', 'mape']}
                    for split in ['Train', 'Val', 'Test']
                },
                'residual': {
                    split: {k: float(v) if isinstance(v, (int, float, np.number)) else str(v)
                            for k, v in residual_results.get(split, {}).items()
                            if k in ['r2', 'rmse', 'mae', 'mape', 'alpha']}
                    for split in ['Train', 'Val', 'Test']
                },
                'dualtask_mtl': {
                    split: {k: float(v) if isinstance(v, (int, float, np.number)) else str(v)
                            for k, v in mtl_results.get(split, {}).items()
                            if k in ['r2', 'rmse', 'mae', 'mape']}
                    for split in ['Train', 'Val', 'Test']
                },
            }

            json_path = os.path.join(output_folder, 'three_way_comparison.json')
            with open(json_path, 'w') as f:
                json.dump(comparison_json, f, indent=2)
            print(f"✓ Comparison saved → {json_path}\n")

        except Exception as e:
            print(f"\n❌ Error in iteration {iteration}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Final summary
    if all_magnn_results and all_residual_results and all_mtl_results:
        print_section("🏆 FINAL SUMMARY - AVERAGED ACROSS ALL ITERATIONS")

        for split in ['Train', 'Val', 'Test']:
            print(f"\n{'=' * 120}")
            print(f"{split.upper()} SET - AVERAGE OVER {len(all_magnn_results)} ITERATION(S)")
            print(f"{'=' * 120}")
            print(f"{'Metric':<10} {'MAGNN':<20} {'Residual':<20} {'DualTaskMTL':<20} {'Best':<15}")
            print(f"{'-' * 120}")

            metrics_info = [
                ('R²', 'r2', True),
                ('RMSE', 'rmse', False),
                ('MAE', 'mae', False),
                ('MAPE', 'mape', False),
            ]

            for display_name, metric_key, higher_is_better in metrics_info:
                magnn_vals = [r.get(split, {}).get(metric_key, float('nan')) for r in all_magnn_results]
                residual_vals = [r.get(split, {}).get(metric_key, float('nan')) for r in all_residual_results]
                mtl_vals = [r.get(split, {}).get(metric_key, float('nan')) for r in all_mtl_results]

                magnn_avg = np.nanmean(magnn_vals)
                residual_avg = np.nanmean(residual_vals)
                mtl_avg = np.nanmean(mtl_vals)

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
                    mtl_str = f"{mtl_avg:.4f}±{mtl_std:.4f}"

                if not any(np.isnan(v) for v in [magnn_avg, residual_avg, mtl_avg]):
                    if higher_is_better:
                        best_val = max(magnn_avg, residual_avg, mtl_avg)
                        if best_val == residual_avg:
                            best_model = "Residual ✓"
                        elif best_val == mtl_avg:
                            best_model = "DualTaskMTL ✓"
                        else:
                            best_model = "MAGNN"
                    else:
                        best_val = min(magnn_avg, residual_avg, mtl_avg)
                        if best_val == residual_avg:
                            best_model = "Residual ✓"
                        elif best_val == mtl_avg:
                            best_model = "DualTaskMTL ✓"
                        else:
                            best_model = "MAGNN"
                else:
                    best_model = "N/A"

                print(f"{display_name:<10} {magnn_str:<20} {residual_str:<20} {mtl_str:<20} {best_model:<15}")

            print(f"{'=' * 120}\n")

        print_section("✅ THREE-WAY COMPARISON COMPLETE")
        print(f"  Total iterations: {len(all_magnn_results)}")
        print(f"\n  Model Details:")
        print(f"    MAGNN: Graph attention + LSTM baseline")
        print(f"    MAGNN-LSTM-Residual: Residual learning + adaptive gate")
        print(f"    MAGNN-LSTM-DualTaskMTL: Local segments + Global stations (NEW!)")
        print(f"\n  Results saved in: outputs/{ALGORITHM_NAME}_*_three_way/")
        print("=" * 80)


# =============================================================================
# ABLATION STUDY MODE
# =============================================================================

def main_with_ablation_study():
    """Ablation study: Feature importance analysis using MAGNN-LSTM-Residual."""

    config = Config()

    print_section("🔬 ABLATION STUDY MODE")
    print(f"  Device: {DEVICE}")
    print(f"  Algorithm: {ALGORITHM_NAME}")
    print(f"  Data sampling: {config.sample_fraction * 100}%")
    print(f"\n  Models to compare:")
    print(f"    1. MAGNN-LSTM-Residual (Full) - All features")
    print(f"    2. MAGNN-LSTM-Residual (No Operational) - Remove delay features")
    print(f"    3. MAGNN-LSTM-Residual (No Weather) - Remove weather features")
    print(f"    4. MAGNN-LSTM-Residual (Baseline) - Only spatial + temporal")
    print("=" * 80)

    all_full_results = []
    all_no_operational_results = []
    all_no_weather_results = []
    all_baseline_results = []

    for iteration in range(1, config.n_iterations + 1):
        print_section(f"🔄 ITERATION {iteration}/{config.n_iterations}")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_folder = f"outputs/{ALGORITHM_NAME}_{timestamp}_i{iteration}_ablation"
        os.makedirs(output_folder, exist_ok=True)

        try:
            print("\n[1/9] Loading data...")
            train_df, test_df, val_df = load_train_test_val_data_fixed(
                data_folder=config.data_folder,
                sample_fraction=config.sample_fraction
            )

            known_stops_train = get_known_stops(train_df)
            known_stops_test = get_known_stops(test_df)
            known_stops_val = get_known_stops(val_df)
            known_stops = {**known_stops_test, **known_stops_val, **known_stops_train}
            print(f"   Known stops: {len(known_stops)}")

            print("[2/9] Clustering stops...")
            clusters, station_cluster_ids = event_driven_clustering_fixed(
                train_df, known_stops=known_stops
            )

            print("[3/9] Building segments...")
            train_segments = build_segments_fixed(train_df, clusters)
            test_segments = build_segments_fixed(test_df, clusters)
            val_segments = build_segments_fixed(val_df, clusters)

            print("[4/9] Building adjacency matrices...")
            adj_geo, adj_dist, adj_soc, segment_types = build_adjacency_matrices_fixed(
                train_segments, clusters, known_stops=known_stops
            )

            # Train base MAGNN for transfer learning (used by all variants)
            print("\n[5/9] Training base MAGNN for transfer learning...")
            print("-" * 80)

            train_dataset_magnn = SegmentDataset(train_segments, segment_types, fit_scalers=True)
            val_dataset_magnn = SegmentDataset(
                val_segments, segment_types, fit_scalers=False,
                target_scaler=train_dataset_magnn.target_scaler,
                speed_scaler=train_dataset_magnn.speed_scaler
            )
            test_dataset_magnn = SegmentDataset(
                test_segments, segment_types, fit_scalers=False,
                target_scaler=train_dataset_magnn.target_scaler,
                speed_scaler=train_dataset_magnn.speed_scaler
            )

            train_loader_magnn = DataLoader(
                train_dataset_magnn, batch_size=config.batch_size,
                shuffle=True, num_workers=0, collate_fn=masked_collate_fn
            )
            val_loader_magnn = DataLoader(
                val_dataset_magnn, batch_size=config.batch_size,
                shuffle=False, num_workers=0, collate_fn=masked_collate_fn
            )
            test_loader_magnn = DataLoader(
                test_dataset_magnn, batch_size=config.batch_size,
                shuffle=False, num_workers=0, collate_fn=masked_collate_fn
            )

            _, magnn_model = train_magtte(
                train_loader_magnn, val_loader_magnn, test_loader_magnn,
                adj_geo, adj_dist, adj_soc,
                segment_types, train_dataset_magnn.target_scaler,
                output_folder, DEVICE, config
            )
            magnn_checkpoint = os.path.join(output_folder, 'magtte_best.pth')

            # ================================================================
            # VARIANT 1: FULL MODEL (All features)
            # ================================================================
            print("\n[6/9] Training MAGNN-LSTM-Residual (FULL - All features)...")
            print("-" * 80)

            train_dataset_full = EnhancedSegmentDataset(
                train_segments, segment_types, fit_scalers=True
            )
            val_dataset_full = EnhancedSegmentDataset(
                val_segments, segment_types, fit_scalers=False,
                target_scaler=train_dataset_full.target_scaler,
                speed_scaler=train_dataset_full.speed_scaler,
                operational_scaler=train_dataset_full.operational_scaler,
                weather_scaler=train_dataset_full.weather_scaler
            )
            test_dataset_full = EnhancedSegmentDataset(
                test_segments, segment_types, fit_scalers=False,
                target_scaler=train_dataset_full.target_scaler,
                speed_scaler=train_dataset_full.speed_scaler,
                operational_scaler=train_dataset_full.operational_scaler,
                weather_scaler=train_dataset_full.weather_scaler
            )

            train_loader_full = DataLoader(
                train_dataset_full, batch_size=config.batch_size,
                shuffle=True, num_workers=0, collate_fn=enhanced_collate_fn
            )
            val_loader_full = DataLoader(
                val_dataset_full, batch_size=config.batch_size,
                shuffle=False, num_workers=0, collate_fn=enhanced_collate_fn
            )
            test_loader_full = DataLoader(
                test_dataset_full, batch_size=config.batch_size,
                shuffle=False, num_workers=0, collate_fn=enhanced_collate_fn
            )

            full_results, full_model = train_magnn_lstm_residual(
                train_loader_full, val_loader_full, test_loader_full,
                adj_geo, adj_dist, adj_soc,
                segment_types, train_dataset_full.target_scaler,
                output_folder, DEVICE, config,
                pretrained_magnn_path=magnn_checkpoint,
                freeze_magnn=True
            )
            all_full_results.append(full_results)
            print(f"   ✓ Full model trained")

            # ================================================================
            # VARIANT 2: NO OPERATIONAL (Remove delay features)
            # ================================================================
            print("\n[7/9] Training MAGNN-LSTM-Residual (NO OPERATIONAL)...")
            print("-" * 80)

            # Create datasets with operational features zeroed out
            train_segments_no_op = train_segments.copy()
            val_segments_no_op = val_segments.copy()
            test_segments_no_op = test_segments.copy()

            # Zero out operational features
            for df in [train_segments_no_op, val_segments_no_op, test_segments_no_op]:
                df['arrivalDelay'] = 0.0
                df['departureDelay'] = 0.0
                df['is_weekend'] = 0.0
                df['is_peak_hour'] = 0.0

            train_dataset_no_op = EnhancedSegmentDataset(
                train_segments_no_op, segment_types, fit_scalers=True
            )
            val_dataset_no_op = EnhancedSegmentDataset(
                val_segments_no_op, segment_types, fit_scalers=False,
                target_scaler=train_dataset_no_op.target_scaler,
                speed_scaler=train_dataset_no_op.speed_scaler,
                operational_scaler=train_dataset_no_op.operational_scaler,
                weather_scaler=train_dataset_no_op.weather_scaler
            )
            test_dataset_no_op = EnhancedSegmentDataset(
                test_segments_no_op, segment_types, fit_scalers=False,
                target_scaler=train_dataset_no_op.target_scaler,
                speed_scaler=train_dataset_no_op.speed_scaler,
                operational_scaler=train_dataset_no_op.operational_scaler,
                weather_scaler=train_dataset_no_op.weather_scaler
            )

            train_loader_no_op = DataLoader(
                train_dataset_no_op, batch_size=config.batch_size,
                shuffle=True, num_workers=0, collate_fn=enhanced_collate_fn
            )
            val_loader_no_op = DataLoader(
                val_dataset_no_op, batch_size=config.batch_size,
                shuffle=False, num_workers=0, collate_fn=enhanced_collate_fn
            )
            test_loader_no_op = DataLoader(
                test_dataset_no_op, batch_size=config.batch_size,
                shuffle=False, num_workers=0, collate_fn=enhanced_collate_fn
            )

            no_op_results, no_op_model = train_magnn_lstm_residual(
                train_loader_no_op, val_loader_no_op, test_loader_no_op,
                adj_geo, adj_dist, adj_soc,
                segment_types, train_dataset_no_op.target_scaler,
                output_folder, DEVICE, config,
                pretrained_magnn_path=magnn_checkpoint,
                freeze_magnn=True
            )
            all_no_operational_results.append(no_op_results)
            print(f"   ✓ No operational model trained")

            # ================================================================
            # VARIANT 3: NO WEATHER (Remove weather features)
            # ================================================================
            print("\n[8/9] Training MAGNN-LSTM-Residual (NO WEATHER)...")
            print("-" * 80)

            # Create datasets with weather features zeroed out
            train_segments_no_weather = train_segments.copy()
            val_segments_no_weather = val_segments.copy()
            test_segments_no_weather = test_segments.copy()

            # Zero out weather features
            weather_cols = ['temperature_2m', 'apparent_temperature', 'precipitation',
                           'rain', 'snowfall', 'windspeed_10m', 'windgusts_10m',
                           'winddirection_10m']
            for df in [train_segments_no_weather, val_segments_no_weather, test_segments_no_weather]:
                for col in weather_cols:
                    if col in df.columns:
                        df[col] = 0.0

            train_dataset_no_weather = EnhancedSegmentDataset(
                train_segments_no_weather, segment_types, fit_scalers=True
            )
            val_dataset_no_weather = EnhancedSegmentDataset(
                val_segments_no_weather, segment_types, fit_scalers=False,
                target_scaler=train_dataset_no_weather.target_scaler,
                speed_scaler=train_dataset_no_weather.speed_scaler,
                operational_scaler=train_dataset_no_weather.operational_scaler,
                weather_scaler=train_dataset_no_weather.weather_scaler
            )
            test_dataset_no_weather = EnhancedSegmentDataset(
                test_segments_no_weather, segment_types, fit_scalers=False,
                target_scaler=train_dataset_no_weather.target_scaler,
                speed_scaler=train_dataset_no_weather.speed_scaler,
                operational_scaler=train_dataset_no_weather.operational_scaler,
                weather_scaler=train_dataset_no_weather.weather_scaler
            )

            train_loader_no_weather = DataLoader(
                train_dataset_no_weather, batch_size=config.batch_size,
                shuffle=True, num_workers=0, collate_fn=enhanced_collate_fn
            )
            val_loader_no_weather = DataLoader(
                val_dataset_no_weather, batch_size=config.batch_size,
                shuffle=False, num_workers=0, collate_fn=enhanced_collate_fn
            )
            test_loader_no_weather = DataLoader(
                test_dataset_no_weather, batch_size=config.batch_size,
                shuffle=False, num_workers=0, collate_fn=enhanced_collate_fn
            )

            no_weather_results, no_weather_model = train_magnn_lstm_residual(
                train_loader_no_weather, val_loader_no_weather, test_loader_no_weather,
                adj_geo, adj_dist, adj_soc,
                segment_types, train_dataset_no_weather.target_scaler,
                output_folder, DEVICE, config,
                pretrained_magnn_path=magnn_checkpoint,
                freeze_magnn=True
            )
            all_no_weather_results.append(no_weather_results)
            print(f"   ✓ No weather model trained")

            # ================================================================
            # VARIANT 4: BASELINE (Only spatial + temporal)
            # ================================================================
            print("\n[9/9] Training MAGNN-LSTM-Residual (BASELINE - spatial + temporal only)...")
            print("-" * 80)

            # Create datasets with both operational and weather features zeroed out
            train_segments_baseline = train_segments.copy()
            val_segments_baseline = val_segments.copy()
            test_segments_baseline = test_segments.copy()

            # Zero out ALL optional features
            for df in [train_segments_baseline, val_segments_baseline, test_segments_baseline]:
                # Operational
                df['arrivalDelay'] = 0.0
                df['departureDelay'] = 0.0
                df['is_weekend'] = 0.0
                df['is_peak_hour'] = 0.0
                # Weather
                for col in weather_cols:
                    if col in df.columns:
                        df[col] = 0.0

            train_dataset_baseline = EnhancedSegmentDataset(
                train_segments_baseline, segment_types, fit_scalers=True
            )
            val_dataset_baseline = EnhancedSegmentDataset(
                val_segments_baseline, segment_types, fit_scalers=False,
                target_scaler=train_dataset_baseline.target_scaler,
                speed_scaler=train_dataset_baseline.speed_scaler,
                operational_scaler=train_dataset_baseline.operational_scaler,
                weather_scaler=train_dataset_baseline.weather_scaler
            )
            test_dataset_baseline = EnhancedSegmentDataset(
                test_segments_baseline, segment_types, fit_scalers=False,
                target_scaler=train_dataset_baseline.target_scaler,
                speed_scaler=train_dataset_baseline.speed_scaler,
                operational_scaler=train_dataset_baseline.operational_scaler,
                weather_scaler=train_dataset_baseline.weather_scaler
            )

            train_loader_baseline = DataLoader(
                train_dataset_baseline, batch_size=config.batch_size,
                shuffle=True, num_workers=0, collate_fn=enhanced_collate_fn
            )
            val_loader_baseline = DataLoader(
                val_dataset_baseline, batch_size=config.batch_size,
                shuffle=False, num_workers=0, collate_fn=enhanced_collate_fn
            )
            test_loader_baseline = DataLoader(
                test_dataset_baseline, batch_size=config.batch_size,
                shuffle=False, num_workers=0, collate_fn=enhanced_collate_fn
            )

            baseline_results, baseline_model = train_magnn_lstm_residual(
                train_loader_baseline, val_loader_baseline, test_loader_baseline,
                adj_geo, adj_dist, adj_soc,
                segment_types, train_dataset_baseline.target_scaler,
                output_folder, DEVICE, config,
                pretrained_magnn_path=magnn_checkpoint,
                freeze_magnn=True
            )
            all_baseline_results.append(baseline_results)
            print(f"   ✓ Baseline model trained")

            # ================================================================
            # COMPARISON TABLE
            # ================================================================
            print_section(f"📊 ITERATION {iteration} - ABLATION STUDY RESULTS")

            for split in ['Train', 'Val', 'Test']:
                print_ablation_comparison_table(
                    full_results, no_op_results, no_weather_results, baseline_results, split
                )

            # Save JSON
            ablation_json = {
                'iteration': iteration,
                'algorithm': ALGORITHM_NAME,
                'model': 'MAGNN-LSTM-Residual',
                'config': {
                    'n_epochs': config.n_epochs,
                    'batch_size': config.batch_size,
                    'learning_rate': config.learning_rate,
                    'sample_fraction': config.sample_fraction,
                },
                'full': {
                    split: {k: float(v) if isinstance(v, (int, float, np.number)) else str(v)
                            for k, v in full_results.get(split, {}).items()
                            if k in ['r2', 'rmse', 'mae', 'mape', 'alpha', 'correction_mean', 'correction_std']}
                    for split in ['Train', 'Val', 'Test']
                },
                'no_operational': {
                    split: {k: float(v) if isinstance(v, (int, float, np.number)) else str(v)
                            for k, v in no_op_results.get(split, {}).items()
                            if k in ['r2', 'rmse', 'mae', 'mape', 'alpha', 'correction_mean', 'correction_std']}
                    for split in ['Train', 'Val', 'Test']
                },
                'no_weather': {
                    split: {k: float(v) if isinstance(v, (int, float, np.number)) else str(v)
                            for k, v in no_weather_results.get(split, {}).items()
                            if k in ['r2', 'rmse', 'mae', 'mape', 'alpha', 'correction_mean', 'correction_std']}
                    for split in ['Train', 'Val', 'Test']
                },
                'baseline': {
                    split: {k: float(v) if isinstance(v, (int, float, np.number)) else str(v)
                            for k, v in baseline_results.get(split, {}).items()
                            if k in ['r2', 'rmse', 'mae', 'mape', 'alpha', 'correction_mean', 'correction_std']}
                    for split in ['Train', 'Val', 'Test']
                },
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

    # ====================================================================
    # FINAL SUMMARY
    # ====================================================================
    if all_full_results and all_no_operational_results and all_no_weather_results and all_baseline_results:
        print_section("🏆 ABLATION STUDY - FINAL SUMMARY")

        for split in ['Train', 'Val', 'Test']:
            print(f"\n{'=' * 130}")
            print(f"{split.upper()} SET - AVERAGE OVER {len(all_full_results)} ITERATION(S)")
            print(f"{'=' * 130}")
            print(f"{'Metric':<10} {'Full':<20} {'No Operational':<20} {'No Weather':<20} {'Baseline':<20} {'Best':<15}")
            print(f"{'-' * 130}")

            metrics_info = [
                ('R²', 'r2', True),
                ('RMSE', 'rmse', False),
                ('MAE', 'mae', False),
                ('MAPE', 'mape', False),
            ]

            for display_name, metric_key, higher_is_better in metrics_info:
                full_vals = [r.get(split, {}).get(metric_key, float('nan')) for r in all_full_results]
                no_op_vals = [r.get(split, {}).get(metric_key, float('nan')) for r in all_no_operational_results]
                no_weather_vals = [r.get(split, {}).get(metric_key, float('nan')) for r in all_no_weather_results]
                baseline_vals = [r.get(split, {}).get(metric_key, float('nan')) for r in all_baseline_results]

                full_avg = np.nanmean(full_vals)
                no_op_avg = np.nanmean(no_op_vals)
                no_weather_avg = np.nanmean(no_weather_vals)
                baseline_avg = np.nanmean(baseline_vals)

                full_std = np.nanstd(full_vals)
                no_op_std = np.nanstd(no_op_vals)
                no_weather_std = np.nanstd(no_weather_vals)
                baseline_std = np.nanstd(baseline_vals)

                if metric_key in ['rmse', 'mae']:
                    full_str = f"{full_avg:.2f}±{full_std:.2f}s"
                    no_op_str = f"{no_op_avg:.2f}±{no_op_std:.2f}s"
                    no_weather_str = f"{no_weather_avg:.2f}±{no_weather_std:.2f}s"
                    baseline_str = f"{baseline_avg:.2f}±{baseline_std:.2f}s"
                elif metric_key == 'mape':
                    full_str = f"{full_avg:.2f}±{full_std:.2f}%"
                    no_op_str = f"{no_op_avg:.2f}±{no_op_std:.2f}%"
                    no_weather_str = f"{no_weather_avg:.2f}±{no_weather_std:.2f}%"
                    baseline_str = f"{baseline_avg:.2f}±{baseline_std:.2f}%"
                else:
                    full_str = f"{full_avg:.4f}±{full_std:.4f}"
                    no_op_str = f"{no_op_avg:.4f}±{no_op_std:.4f}"
                    no_weather_str = f"{no_weather_avg:.4f}±{no_weather_std:.4f}"
                    baseline_str = f"{baseline_avg:.4f}±{baseline_std:.4f}"

                if not any(np.isnan(v) for v in [full_avg, no_op_avg, no_weather_avg, baseline_avg]):
                    if higher_is_better:
                        best_val = max(full_avg, no_op_avg, no_weather_avg, baseline_avg)
                        if best_val == full_avg:
                            best_model = "Full ✓"
                        elif best_val == no_op_avg:
                            best_model = "No Op ✓"
                        elif best_val == no_weather_avg:
                            best_model = "No Weather ✓"
                        else:
                            best_model = "Baseline ✓"
                    else:
                        best_val = min(full_avg, no_op_avg, no_weather_avg, baseline_avg)
                        if best_val == full_avg:
                            best_model = "Full ✓"
                        elif best_val == no_op_avg:
                            best_model = "No Op ✓"
                        elif best_val == no_weather_avg:
                            best_model = "No Weather ✓"
                        else:
                            best_model = "Baseline ✓"
                else:
                    best_model = "N/A"

                print(f"{display_name:<10} {full_str:<20} {no_op_str:<20} {no_weather_str:<20} {baseline_str:<20} {best_model:<15}")

            print(f"{'=' * 130}\n")

        # Feature importance analysis
        print_section("🔍 FEATURE IMPORTANCE ANALYSIS")

        for split in ['Test']:  # Focus on test set
            print(f"\n{split.upper()} SET - Feature Contribution Analysis")
            print(f"{'-' * 80}")

            full_vals = [r.get(split, {}).get('r2', float('nan')) for r in all_full_results]
            no_op_vals = [r.get(split, {}).get('r2', float('nan')) for r in all_no_operational_results]
            no_weather_vals = [r.get(split, {}).get('r2', float('nan')) for r in all_no_weather_results]
            baseline_vals = [r.get(split, {}).get('r2', float('nan')) for r in all_baseline_results]

            full_r2 = np.nanmean(full_vals)
            no_op_r2 = np.nanmean(no_op_vals)
            no_weather_r2 = np.nanmean(no_weather_vals)
            baseline_r2 = np.nanmean(baseline_vals)

            operational_contribution = full_r2 - no_op_r2
            weather_contribution = full_r2 - no_weather_r2
            combined_contribution = full_r2 - baseline_r2

            print(f"\nR² Score Comparison:")
            print(f"  Full Model (All features):          {full_r2:.4f}")
            print(f"  No Operational (Weather only):      {no_op_r2:.4f}")
            print(f"  No Weather (Operational only):      {no_weather_r2:.4f}")
            print(f"  Baseline (Spatial + Temporal only): {baseline_r2:.4f}")

            print(f"\nFeature Contributions (ΔR²):")
            print(f"  Operational features:    {operational_contribution:+.4f}")
            print(f"  Weather features:        {weather_contribution:+.4f}")
            print(f"  Combined (Op + Weather): {combined_contribution:+.4f}")

            if operational_contribution > weather_contribution:
                print(f"\n💡 Operational features contribute MORE to prediction accuracy")
                print(f"   (Δ = {operational_contribution - weather_contribution:+.4f} R² improvement)")
            elif weather_contribution > operational_contribution:
                print(f"\n💡 Weather features contribute MORE to prediction accuracy")
                print(f"   (Δ = {weather_contribution - operational_contribution:+.4f} R² improvement)")
            else:
                print(f"\n💡 Both feature groups contribute equally")

            # Alpha analysis
            full_alphas = [r.get(split, {}).get('alpha', float('nan')) for r in all_full_results]
            no_op_alphas = [r.get(split, {}).get('alpha', float('nan')) for r in all_no_operational_results]
            no_weather_alphas = [r.get(split, {}).get('alpha', float('nan')) for r in all_no_weather_results]
            baseline_alphas = [r.get(split, {}).get('alpha', float('nan')) for r in all_baseline_results]

            print(f"\nAdaptive Gating (α) Analysis:")
            print(f"  Full:           α = {np.nanmean(full_alphas):.3f} ± {np.nanstd(full_alphas):.3f}")
            print(f"  No Operational: α = {np.nanmean(no_op_alphas):.3f} ± {np.nanstd(no_op_alphas):.3f}")
            print(f"  No Weather:     α = {np.nanmean(no_weather_alphas):.3f} ± {np.nanstd(no_weather_alphas):.3f}")
            print(f"  Baseline:       α = {np.nanmean(baseline_alphas):.3f} ± {np.nanstd(baseline_alphas):.3f}")

        print_section("✅ ABLATION STUDY COMPLETE")
        print(f"  Total iterations: {len(all_full_results)}")
        print(f"  Model: MAGNN-LSTM-Residual")
        print(f"\n  Variants tested:")
        print(f"    MAGNN-LSTM-Residual (Full): All features")
        print(f"    MAGNN-LSTM-Residual (No Operational): Weather features only")
        print(f"    MAGNN-LSTM-Residual (No Weather): Operational features only")
        print(f"    MAGNN-LSTM-Residual (Baseline): Spatial + temporal features only")
        print(f"\n  Results saved in: outputs/{ALGORITHM_NAME}_*_ablation/")
        print("=" * 80)


def print_ablation_comparison_table(full_results, no_op_results, no_weather_results, baseline_results, split_name):
    """Print formatted ablation study comparison table."""

    full_m = full_results.get(split_name, {})
    no_op_m = no_op_results.get(split_name, {})
    no_weather_m = no_weather_results.get(split_name, {})
    baseline_m = baseline_results.get(split_name, {})

    print(f"\n{'=' * 130}")
    print(f"{split_name.upper()} SET - ABLATION STUDY COMPARISON")
    print(f"{'=' * 130}")
    print(f"{'Metric':<10} {'Full':<20} {'No Operational':<20} {'No Weather':<20} {'Baseline':<20} {'Best':<15}")
    print(f"{'-' * 130}")

    metrics_info = [
        ('R²', 'r2', True),
        ('RMSE', 'rmse', False),
        ('MAE', 'mae', False),
        ('MAPE', 'mape', False),
    ]

    for display_name, metric_key, higher_is_better in metrics_info:
        full_val = full_m.get(metric_key, float('nan'))
        no_op_val = no_op_m.get(metric_key, float('nan'))
        no_weather_val = no_weather_m.get(metric_key, float('nan'))
        baseline_val = baseline_m.get(metric_key, float('nan'))

        if metric_key in ['rmse', 'mae']:
            full_str = f"{full_val:.2f}s"
            no_op_str = f"{no_op_val:.2f}s"
            no_weather_str = f"{no_weather_val:.2f}s"
            baseline_str = f"{baseline_val:.2f}s"
        elif metric_key == 'mape':
            full_str = f"{full_val:.2f}%"
            no_op_str = f"{no_op_val:.2f}%"
            no_weather_str = f"{no_weather_val:.2f}%"
            baseline_str = f"{baseline_val:.2f}%"
        else:
            full_str = f"{full_val:.4f}"
            no_op_str = f"{no_op_val:.4f}"
            no_weather_str = f"{no_weather_val:.4f}"
            baseline_str = f"{baseline_val:.4f}"

        if not any(np.isnan(v) for v in [full_val, no_op_val, no_weather_val, baseline_val]):
            if higher_is_better:
                best_val = max(full_val, no_op_val, no_weather_val, baseline_val)
                if best_val == full_val:
                    best_model = "Full ✓"
                elif best_val == no_op_val:
                    best_model = "No Op ✓"
                elif best_val == no_weather_val:
                    best_model = "No Weather ✓"
                else:
                    best_model = "Baseline ✓"
            else:
                best_val = min(full_val, no_op_val, no_weather_val, baseline_val)
                if best_val == full_val:
                    best_model = "Full ✓"
                elif best_val == no_op_val:
                    best_model = "No Op ✓"
                elif best_val == no_weather_val:
                    best_model = "No Weather ✓"
                else:
                    best_model = "Baseline ✓"
        else:
            best_model = "N/A"

        print(f"{display_name:<10} {full_str:<20} {no_op_str:<20} {no_weather_str:<20} {baseline_str:<20} {best_model:<15}")

    print(f"{'=' * 130}\n")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    # Apply CLI sample fraction override (if provided)
    if _sample_fraction_cli is not None:
        Config.sample_fraction = _sample_fraction_cli

    print(f"🔧 Clustering method : {ALGORITHM_NAME}")
    print(f"🔧 Sample fraction   : {Config.sample_fraction}")

    if _mode == '--compare-all':
        main_with_three_way_comparison()
    elif _mode == '--ablation':
        main_with_ablation_study()
    else:
        main()