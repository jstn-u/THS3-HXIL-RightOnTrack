"""
main.py
=======
Entry point for MAGNN-LSTM transit travel time prediction.

✅ TWO MODEL COMPARISON:
   1. MAGNN (baseline)
   2. MAGNN-LSTM-Residual (fixed version with residual learning + adaptive gating)

Usage:
    python main.py                  # Train MAGNN only (baseline)
    python main.py --compare        # Compare MAGNN vs MAGNN-LSTM-Residual

HOW TO SWITCH CLUSTERING METHOD
--------------------------------
Change only the single import line below:

    from cluster_hdbscan import event_driven_clustering_fixed   # default
    from cluster_dbscan  import event_driven_clustering_fixed
    from cluster_knn     import event_driven_clustering_fixed
    from cluster_gmm     import event_driven_clustering_fixed
    from cluster_kmeans  import event_driven_clustering_fixed
"""

# =============================================================================
# CLUSTERING METHOD — change this ONE line to swap methods
# =============================================================================
from cluster_hdbscan import event_driven_clustering_fixed  # ← swap here

ALGORITHM_NAME = event_driven_clustering_fixed.__module__.replace('cluster_', '').upper()

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
# TWO-WAY COMPARISON HELPER
# =============================================================================

def print_comparison_table(magnn_results, residual_results, split_name):
    """Print formatted two-way comparison table."""

    magnn_m = magnn_results.get(split_name, {})
    residual_m = residual_results.get(split_name, {})

    print(f"\n{'=' * 100}")
    print(f"{split_name.upper()} SET - COMPARISON")
    print(f"{'=' * 100}")
    print(f"{'Metric':<10} {'MAGNN':<20} {'Residual':<20} {'Improvement':<20} {'Winner':<15}")
    print(f"{'-' * 100}")

    metrics_info = [
        ('R²', 'r2', True),
        ('RMSE', 'rmse', False),
        ('MAE', 'mae', False),
        ('MAPE', 'mape', False),
    ]

    for display_name, metric_key, higher_is_better in metrics_info:
        magnn_val = magnn_m.get(metric_key, float('nan'))
        residual_val = residual_m.get(metric_key, float('nan'))

        if metric_key in ['rmse', 'mae']:
            magnn_str = f"{magnn_val:.2f}s"
            residual_str = f"{residual_val:.2f}s"
        elif metric_key == 'mape':
            magnn_str = f"{magnn_val:.2f}%"
            residual_str = f"{residual_val:.2f}%"
        else:
            magnn_str = f"{magnn_val:.4f}"
            residual_str = f"{residual_val:.4f}"

        if not (np.isnan(magnn_val) or np.isnan(residual_val)):
            if higher_is_better:
                improvement = ((residual_val - magnn_val) / abs(magnn_val) * 100)
                winner = "Residual ✓" if residual_val > magnn_val else "MAGNN"
            else:
                improvement = ((magnn_val - residual_val) / abs(magnn_val) * 100)
                winner = "Residual ✓" if residual_val < magnn_val else "MAGNN"

            improvement_str = f"+{improvement:.2f}%" if improvement > 0 else f"{improvement:.2f}%"
        else:
            improvement_str = "N/A"
            winner = "N/A"

        print(f"{display_name:<10} {magnn_str:<20} {residual_str:<20} {improvement_str:<20} {winner:<15}")

    print(f"{'=' * 100}\n")


# =============================================================================
# TWO-WAY COMPARISON MODE
# =============================================================================

def main_with_comparison():
    """Two-way comparison: MAGNN vs MAGNN-LSTM-Residual"""

    config = Config()

    print_section("🔬 TWO-WAY COMPARISON MODE")
    print(f"  Device: {DEVICE}")
    print(f"  Algorithm: {ALGORITHM_NAME}")
    print(f"\n  Models to compare:")
    print(f"    1. MAGNN (baseline)")
    print(f"    2. MAGNN-LSTM-Residual (✅ fixed + residual learning + adaptive gate)")
    print(f"\n  Residual model features:")
    print(f"    ✅ Weather features properly scaled")
    print(f"    ✅ LSTM sees MAGNN prediction (residual learning)")
    print(f"    ✅ Adaptive gating (learns α per sample)")
    print(f"    ✅ Binary flags: is_weekend, is_peak_hour")
    print("=" * 80)

    all_magnn_results = []
    all_residual_results = []

    for iteration in range(1, config.n_iterations + 1):
        print_section(f"🔄 ITERATION {iteration}/{config.n_iterations}")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_folder = f"outputs/{ALGORITHM_NAME}_{timestamp}_i{iteration}_comparison"
        os.makedirs(output_folder, exist_ok=True)

        try:
            print("\n[1/8] Loading data...")
            train_df, test_df, val_df = load_train_test_val_data_fixed(
                data_folder=config.data_folder,
                sample_fraction=config.sample_fraction
            )

            known_stops_train = get_known_stops(train_df)
            known_stops_test = get_known_stops(test_df)
            known_stops_val = get_known_stops(val_df)
            known_stops = {**known_stops_test, **known_stops_val, **known_stops_train}

            print("[2/8] Clustering stops...")
            clusters, station_cluster_ids = event_driven_clustering_fixed(
                train_df, known_stops=known_stops
            )

            print("[3/8] Building segments...")
            train_segments = build_segments_fixed(train_df, clusters)
            test_segments = build_segments_fixed(test_df, clusters)
            val_segments = build_segments_fixed(val_df, clusters)

            print("[4/8] Building adjacency matrices...")
            adj_geo, adj_dist, adj_soc, segment_types = build_adjacency_matrices_fixed(
                train_segments, clusters, known_stops=known_stops
            )

            print("[5/8] Preparing MAGNN datasets...")
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

            print("[6/8] Preparing Residual datasets...")
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

            print("\n[7/8] Training MAGNN (baseline)...")
            print("-" * 80)
            magnn_results, magnn_model = train_magtte(
                train_loader_magnn, val_loader_magnn, test_loader_magnn,
                adj_geo, adj_dist, adj_soc,
                segment_types, train_dataset_magnn.target_scaler,
                output_folder, DEVICE, config
            )
            all_magnn_results.append(magnn_results)
            magnn_checkpoint = os.path.join(output_folder, 'magtte_best.pth')

            print("\n[8/8] Training MAGNN-LSTM-Residual...")
            print("-" * 80)
            residual_results, residual_model = train_magnn_lstm_residual(
                train_loader_residual, val_loader_residual, test_loader_residual,
                adj_geo, adj_dist, adj_soc,
                segment_types, train_dataset_residual.target_scaler,
                output_folder, DEVICE, config,
                pretrained_magnn_path=magnn_checkpoint,
                freeze_magnn=True
            )
            all_residual_results.append(residual_results)

            print_section(f"📊 ITERATION {iteration} - COMPARISON")

            for split in ['Train', 'Val', 'Test']:
                print_comparison_table(magnn_results, residual_results, split)

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
            }

            json_path = os.path.join(output_folder, 'comparison.json')
            with open(json_path, 'w') as f:
                json.dump(comparison_json, f, indent=2)
            print(f"✓ Comparison saved → {json_path}\n")

        except Exception as e:
            print(f"\n❌ Error in iteration {iteration}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Final summary
    if all_magnn_results and all_residual_results:
        print_section("🏆 FINAL SUMMARY - AVERAGED ACROSS ALL ITERATIONS")

        for split in ['Train', 'Val', 'Test']:
            print(f"\n{'=' * 100}")
            print(f"{split.upper()} SET - AVERAGE OVER {len(all_magnn_results)} ITERATION(S)")
            print(f"{'=' * 100}")
            print(f"{'Metric':<10} {'MAGNN':<25} {'Residual':<25} {'Best':<15}")
            print(f"{'-' * 100}")

            metrics_info = [
                ('R²', 'r2', True),
                ('RMSE', 'rmse', False),
                ('MAE', 'mae', False),
                ('MAPE', 'mape', False),
            ]

            for display_name, metric_key, higher_is_better in metrics_info:
                magnn_vals = [r.get(split, {}).get(metric_key, float('nan')) for r in all_magnn_results]
                residual_vals = [r.get(split, {}).get(metric_key, float('nan')) for r in all_residual_results]

                magnn_avg = np.nanmean(magnn_vals)
                residual_avg = np.nanmean(residual_vals)

                magnn_std = np.nanstd(magnn_vals)
                residual_std = np.nanstd(residual_vals)

                if metric_key in ['rmse', 'mae']:
                    magnn_str = f"{magnn_avg:.2f}±{magnn_std:.2f}s"
                    residual_str = f"{residual_avg:.2f}±{residual_std:.2f}s"
                elif metric_key == 'mape':
                    magnn_str = f"{magnn_avg:.2f}±{magnn_std:.2f}%"
                    residual_str = f"{residual_avg:.2f}±{residual_std:.2f}%"
                else:
                    magnn_str = f"{magnn_avg:.4f}±{magnn_std:.4f}"
                    residual_str = f"{residual_avg:.4f}±{residual_std:.4f}"

                if not any(np.isnan(v) for v in [magnn_avg, residual_avg]):
                    if higher_is_better:
                        best_model = "Residual ✓" if residual_avg > magnn_avg else "MAGNN"
                    else:
                        best_model = "Residual ✓" if residual_avg < magnn_avg else "MAGNN"
                else:
                    best_model = "N/A"

                print(f"{display_name:<10} {magnn_str:<25} {residual_str:<25} {best_model:<15}")

            print(f"{'=' * 100}\n")

        print_section("✅ COMPARISON COMPLETE")
        print(f"  Total iterations: {len(all_magnn_results)}")
        print(f"\n  Model Details:")
        print(f"    MAGNN: Graph attention + LSTM baseline")
        print(f"    MAGNN-LSTM-Residual: ✅ Fixed + residual learning + adaptive gate")
        print(f"\n  Results saved in: outputs/{ALGORITHM_NAME}_*_comparison/")
        print("=" * 80)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == '--compare':
            main_with_comparison()
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print("\nUsage:")
            print("  python main.py            # Train MAGNN only")
            print("  python main.py --compare  # Compare MAGNN vs Residual")
    else:
        main()