"""
main.py
=======
Entry point for MAGNN-LSTM transit travel time prediction.

Usage:
    python main.py                  # Train MAGNN only (baseline)
    python main.py --compare        # Compare MAGNN vs MAGNN-LSTM
    python main.py --compare-all    # Compare MAGNN vs MAGNN-LSTM vs MAGNN-LSTM-MTL
    python main.py --ablation       # for feature ablation studies

HOW TO SWITCH CLUSTERING METHOD
--------------------------------
Change only the single import line below:

    from cluster_hdbscan import event_driven_clustering_fixed   # default
    from cluster_dbscan  import event_driven_clustering_fixed
    from cluster_knn     import event_driven_clustering_fixed
    from cluster_gmm     import event_driven_clustering_fixed
    from cluster_kmeans  import event_driven_clustering_fixed

Everything else — data loading, segment building, model training,
visualisations, metrics — is identical regardless of method.
"""

# =============================================================================
# CLUSTERING METHOD — change this ONE line to swap methods
# =============================================================================
from cluster_hdbscan import event_driven_clustering_fixed   # ← swap here

# Derived automatically from the module name — never needs manual editing.
# cluster_hdbscan → HDBSCAN,  cluster_knn → KNN,  cluster_gmm → GMM, etc.
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
                   PathDataset, path_collate_fn,
                   train_magnn_lstm, train_magnn_lstm_mtl)

import numpy as np
import pandas as pd
import os
import json
import warnings
from datetime import datetime
from torch.utils.data import DataLoader

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

            # Extract known named stops from ALL splits
            known_stops_train = get_known_stops(train_df)
            known_stops_test = get_known_stops(test_df)
            known_stops_val = get_known_stops(val_df)
            # Merge: test → val → train (train wins on conflict)
            known_stops = {**known_stops_test, **known_stops_val, **known_stops_train}
            print(f"   Known stops found in data: {len(known_stops)} "
                  f"(train={len(known_stops_train)}, "
                  f"test={len(known_stops_test)}, "
                  f"val={len(known_stops_val)})")

            # ------------------------------------------------------------------
            # 2. Clustering — stations always injected
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
            # 4. PLOTS — generated before training
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
# COMPARISON HELPER FUNCTIONS
# =============================================================================

def print_comparison_table(magnn_results, lstm_results, split_name):
    """Print a formatted comparison table for a specific data split."""

    magnn_m = magnn_results.get(split_name, {})
    lstm_m = lstm_results.get(split_name, {})

    print(f"\n{'=' * 90}")
    print(f"{split_name.upper()} SET COMPARISON")
    print(f"{'=' * 90}")
    print(f"{'Metric':<10} {'MAGNN':<15} {'MAGNN-LSTM':<15} {'Improvement':<15} {'Winner':<10}")
    print(f"{'-' * 90}")

    metrics_info = [
        ('R²', 'r2', True),
        ('RMSE', 'rmse', False),
        ('MAE', 'mae', False),
        ('MAPE', 'mape', False),
    ]

    for display_name, metric_key, higher_is_better in metrics_info:
        magnn_val = magnn_m.get(metric_key, float('nan'))
        lstm_val = lstm_m.get(metric_key, float('nan'))

        if not np.isnan(magnn_val) and not np.isnan(lstm_val) and magnn_val != 0:
            if higher_is_better:
                improvement = ((lstm_val - magnn_val) / abs(magnn_val) * 100)
                winner = "MAGNN-LSTM ✓" if lstm_val > magnn_val else "MAGNN"
            else:
                improvement = ((magnn_val - lstm_val) / abs(magnn_val) * 100)
                winner = "MAGNN-LSTM ✓" if lstm_val < magnn_val else "MAGNN"

            improvement_str = f"+{improvement:.2f}%" if improvement > 0 else f"{improvement:.2f}%"
        else:
            improvement_str = "N/A"
            winner = "N/A"

        if metric_key in ['rmse', 'mae']:
            magnn_str = f"{magnn_val:.2f}s"
            lstm_str = f"{lstm_val:.2f}s"
        elif metric_key == 'mape':
            magnn_str = f"{magnn_val:.2f}%"
            lstm_str = f"{lstm_val:.2f}%"
        else:
            magnn_str = f"{magnn_val:.4f}"
            lstm_str = f"{lstm_val:.4f}"

        print(f"{display_name:<10} {magnn_str:<15} {lstm_str:<15} {improvement_str:<15} {winner:<10}")

    print(f"{'=' * 90}\n")


def print_three_way_comparison_table(magnn_results, lstm_results, mtl_results, split_name):
    """Print a formatted three-way comparison table."""

    magnn_m = magnn_results.get(split_name, {})
    lstm_m = lstm_results.get(split_name, {})
    mtl_m = mtl_results.get(split_name, {})

    print(f"\n{'=' * 120}")
    print(f"{split_name.upper()} SET - THREE-WAY COMPARISON")
    print(f"{'=' * 120}")
    print(
        f"{'Metric':<10} {'MAGNN':<15} {'MAGNN-LSTM':<15} {'MAGNN-LSTM-MTL':<18} {'Best Model':<20} {'Improvement':<15}")
    print(f"{'-' * 120}")

    metrics_info = [
        ('R²', 'r2', True),
        ('RMSE', 'rmse', False),
        ('MAE', 'mae', False),
        ('MAPE', 'mape', False),
    ]

    for display_name, metric_key, higher_is_better in metrics_info:
        magnn_val = magnn_m.get(metric_key, float('nan'))
        lstm_val = lstm_m.get(metric_key, float('nan'))
        mtl_val = mtl_m.get(metric_key, float('nan'))

        # Format values
        if metric_key in ['rmse', 'mae']:
            magnn_str = f"{magnn_val:.2f}s"
            lstm_str = f"{lstm_val:.2f}s"
            mtl_str = f"{mtl_val:.2f}s"
        elif metric_key == 'mape':
            magnn_str = f"{magnn_val:.2f}%"
            lstm_str = f"{lstm_val:.2f}%"
            mtl_str = f"{mtl_val:.2f}%"
        else:
            magnn_str = f"{magnn_val:.4f}"
            lstm_str = f"{lstm_val:.4f}"
            mtl_str = f"{mtl_val:.4f}"

        # Determine best model and improvement
        if not (np.isnan(magnn_val) or np.isnan(lstm_val) or np.isnan(mtl_val)):
            if higher_is_better:
                best_val = max(magnn_val, lstm_val, mtl_val)
                if best_val == mtl_val:
                    best_model = "MAGNN-LSTM-MTL ✓"
                    improvement = ((mtl_val - magnn_val) / abs(magnn_val) * 100)
                elif best_val == lstm_val:
                    best_model = "MAGNN-LSTM ✓"
                    improvement = ((lstm_val - magnn_val) / abs(magnn_val) * 100)
                else:
                    best_model = "MAGNN (baseline)"
                    improvement = 0.0
            else:
                best_val = min(magnn_val, lstm_val, mtl_val)
                if best_val == mtl_val:
                    best_model = "MAGNN-LSTM-MTL ✓"
                    improvement = ((magnn_val - mtl_val) / abs(magnn_val) * 100)
                elif best_val == lstm_val:
                    best_model = "MAGNN-LSTM ✓"
                    improvement = ((magnn_val - lstm_val) / abs(magnn_val) * 100)
                else:
                    best_model = "MAGNN (baseline)"
                    improvement = 0.0

            improvement_str = f"+{improvement:.2f}%" if improvement > 0 else f"{improvement:.2f}%"
        else:
            best_model = "N/A"
            improvement_str = "N/A"

        print(
            f"{display_name:<10} {magnn_str:<15} {lstm_str:<15} {mtl_str:<18} {best_model:<20} {improvement_str:<15}")

    print(f"{'=' * 120}\n")


# =============================================================================
# THREE-WAY COMPARISON MODE (WITH MTL)
# =============================================================================

def main_with_three_way_comparison():
    """Main training loop with MAGNN vs MAGNN-LSTM vs MAGNN-LSTM-MTL comparison."""

    config = Config()

    print_section("🔬 THREE-WAY COMPARISON MODE")
    print(f"  Device: {DEVICE}")
    print(f"  Algorithm: {ALGORITHM_NAME}")
    print(f"  Data sampling: {config.sample_fraction * 100}%")
    print(f"  Iterations: {config.n_iterations}")
    print(f"  Epochs per iteration: {config.n_epochs}")
    print(f"\n  Models to compare:")
    print(f"    1. MAGNN (baseline) - Graph attention + LSTM")
    print(f"    2. MAGNN-LSTM - Add operational + weather features (single segments)")
    print(f"    3. MAGNN-LSTM-MTL - Multi-segment paths with MTL")
    print(f"       • Aggregates consecutive segments by trip_id into paths")
    print(f"       • Task 1: Individual segment predictions (local accuracy)")
    print(f"       • Task 2: Collective path predictions (global patterns)")
    print(f"       • L2-norm attention weighting")
    print(f"       • MTL lambda = {config.mtl_lambda}")
    print("=" * 80)

    all_magnn_results = []
    all_lstm_results = []
    all_mtl_results = []

    for iteration in range(1, config.n_iterations + 1):
        print_section(f"🔄 ITERATION {iteration}/{config.n_iterations}")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_folder = f"outputs/{ALGORITHM_NAME}_{timestamp}_i{iteration}_three_way"
        os.makedirs(output_folder, exist_ok=True)
        print(f"📁 Output: {output_folder}")

        try:
            print("\n[1/12] Loading data...")
            train_df, test_df, val_df = load_train_test_val_data_fixed(
                data_folder=config.data_folder,
                sample_fraction=config.sample_fraction
            )

            if len(train_df) == 0:
                print("❌ No training data")
                continue

            # Extract known stops from ALL splits
            known_stops_train = get_known_stops(train_df)
            known_stops_test = get_known_stops(test_df)
            known_stops_val = get_known_stops(val_df)
            known_stops = {**known_stops_test, **known_stops_val, **known_stops_train}
            print(f"      ✓ Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
            print(f"      ✓ Known stops: {len(known_stops)}")

            print("[2/12] Clustering stops...")
            clusters, station_cluster_ids = event_driven_clustering_fixed(
                train_df, known_stops=known_stops
            )
            if len(clusters) == 0:
                print("❌ No clusters")
                continue
            print(f"      ✓ {len(clusters)} clusters created")

            print("[3/12] Building segments...")
            train_segments = build_segments_fixed(train_df, clusters)
            test_segments = build_segments_fixed(test_df, clusters)
            val_segments = build_segments_fixed(val_df, clusters)

            if len(train_segments) == 0:
                print("❌ No segments")
                continue
            print(f"      ✓ {len(train_segments):,} training segments")

            print("[4/12] Building adjacency matrices...")
            adj_geo, adj_dist, adj_soc, segment_types = build_adjacency_matrices_fixed(
                train_segments, clusters,
                known_stops=known_stops
            )

            if adj_geo is None:
                print("❌ Adjacency failed")
                continue
            print(f"      ✓ 3 adjacency matrices (geo, dist, social)")

            print("[5/12] Preparing MAGNN datasets...")

            train_dataset_magnn = SegmentDataset(
                train_segments, segment_types, fit_scalers=True
            )
            val_dataset_magnn = SegmentDataset(
                val_segments, segment_types, fit_scalers=False,
                target_scaler=train_dataset_magnn.target_scaler,
                speed_scaler=train_dataset_magnn.speed_scaler,
            )
            test_dataset_magnn = SegmentDataset(
                test_segments, segment_types, fit_scalers=False,
                target_scaler=train_dataset_magnn.target_scaler,
                speed_scaler=train_dataset_magnn.speed_scaler,
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

            print(f"      ✓ MAGNN: {len(train_dataset_magnn):,} samples")

            print("[6/12] Preparing MAGNN-LSTM datasets (single segments)...")

            train_dataset_lstm = EnhancedSegmentDataset(
                train_segments, segment_types, fit_scalers=True
            )

            val_dataset_lstm = EnhancedSegmentDataset(
                val_segments, segment_types, fit_scalers=False,
                target_scaler=train_dataset_lstm.target_scaler,
                speed_scaler=train_dataset_lstm.speed_scaler,
                operational_scaler=train_dataset_lstm.operational_scaler,
                weather_scaler=train_dataset_lstm.weather_scaler,
            )
            test_dataset_lstm = EnhancedSegmentDataset(
                test_segments, segment_types, fit_scalers=False,
                target_scaler=train_dataset_lstm.target_scaler,
                speed_scaler=train_dataset_lstm.speed_scaler,
                operational_scaler=train_dataset_lstm.operational_scaler,
                weather_scaler=train_dataset_lstm.weather_scaler,
            )

            train_loader_lstm = DataLoader(
                train_dataset_lstm, batch_size=config.batch_size,
                shuffle=True, num_workers=0, collate_fn=enhanced_collate_fn
            )
            val_loader_lstm = DataLoader(
                val_dataset_lstm, batch_size=config.batch_size,
                shuffle=False, num_workers=0, collate_fn=enhanced_collate_fn
            )
            test_loader_lstm = DataLoader(
                test_dataset_lstm, batch_size=config.batch_size,
                shuffle=False, num_workers=0, collate_fn=enhanced_collate_fn
            )

            print(f"      ✓ MAGNN-LSTM: {len(train_dataset_lstm):,} samples (single segments)")

            # NEW: Aggregate paths for MTL
            print("[7/12] Aggregating segments into multi-segment paths for MTL...")
            train_paths = aggregate_segments_into_paths(train_segments, max_path_length=10)
            val_paths = aggregate_segments_into_paths(val_segments, max_path_length=10)
            test_paths = aggregate_segments_into_paths(test_segments, max_path_length=10)

            if len(train_paths) == 0:
                print("❌ No paths created - skipping MTL")
                continue

            train_dataset_mtl = PathDataset(
                train_paths, segment_types, max_path_length=10, fit_scalers=True
            )
            val_dataset_mtl = PathDataset(
                val_paths, segment_types, max_path_length=10, fit_scalers=False,
                target_scaler=train_dataset_mtl.target_scaler,
                speed_scaler=train_dataset_mtl.speed_scaler,
                operational_scaler=train_dataset_mtl.operational_scaler,
                weather_scaler=train_dataset_mtl.weather_scaler,
            )
            test_dataset_mtl = PathDataset(
                test_paths, segment_types, max_path_length=10, fit_scalers=False,
                target_scaler=train_dataset_mtl.target_scaler,
                speed_scaler=train_dataset_mtl.speed_scaler,
                operational_scaler=train_dataset_mtl.operational_scaler,
                weather_scaler=train_dataset_mtl.weather_scaler,
            )

            train_loader_mtl = DataLoader(
                train_dataset_mtl, batch_size=config.batch_size,
                shuffle=True, num_workers=0, collate_fn=path_collate_fn
            )
            val_loader_mtl = DataLoader(
                val_dataset_mtl, batch_size=config.batch_size,
                shuffle=False, num_workers=0, collate_fn=path_collate_fn
            )
            test_loader_mtl = DataLoader(
                test_dataset_mtl, batch_size=config.batch_size,
                shuffle=False, num_workers=0, collate_fn=path_collate_fn
            )

            print(f"      ✓ MTL: {len(train_dataset_mtl):,} paths (multi-segment)")

            print("\n[8/12] Training MAGNN (baseline)...")
            print("-" * 80)

            magnn_results, magnn_model = train_magtte(
                train_loader_magnn, val_loader_magnn, test_loader_magnn,
                adj_geo, adj_dist, adj_soc,
                segment_types, train_dataset_magnn.target_scaler,
                output_folder, DEVICE, config
            )

            all_magnn_results.append(magnn_results)

            print("\n[9/12] Training MAGNN-LSTM (single segments)...")
            print("-" * 80)

            magnn_checkpoint = os.path.join(output_folder, 'magtte_best.pth')

            lstm_results, lstm_model = train_magnn_lstm(
                train_loader_lstm, val_loader_lstm, test_loader_lstm,
                adj_geo, adj_dist, adj_soc,
                segment_types, train_dataset_lstm.target_scaler,
                output_folder, DEVICE, config,
                pretrained_magnn_path=magnn_checkpoint,
                freeze_magnn=True
            )

            all_lstm_results.append(lstm_results)

            print("\n[10/12] Training MAGNN-LSTM-MTL (multi-segment paths)...")
            print("-" * 80)

            mtl_results, mtl_model = train_magnn_lstm_mtl(
                train_loader_mtl, val_loader_mtl, test_loader_mtl,
                adj_geo, adj_dist, adj_soc,
                segment_types, train_dataset_mtl.target_scaler,
                output_folder, DEVICE, config,
                pretrained_magnn_path=magnn_checkpoint,
                freeze_magnn=True
            )

            all_mtl_results.append(mtl_results)

            print("\n[11/12] Generating comparison report...")
            print_section(f"📊 ITERATION {iteration} - THREE-WAY COMPARISON")

            for split in ['Train', 'Val', 'Test']:
                print_three_way_comparison_table(magnn_results, lstm_results, mtl_results, split)

            print("[12/12] Saving comparison metrics...")

            comparison_data = []
            for split in ['Train', 'Val', 'Test']:
                magnn_m = magnn_results.get(split, {})
                lstm_m = lstm_results.get(split, {})
                mtl_m = mtl_results.get(split, {})

                for metric in ['r2', 'rmse', 'mae', 'mape']:
                    magnn_val = magnn_m.get(metric, float('nan'))
                    lstm_val = lstm_m.get(metric, float('nan'))
                    mtl_val = mtl_m.get(metric, float('nan'))

                    if metric == 'r2':
                        lstm_imp = ((lstm_val - magnn_val) / abs(magnn_val) * 100) if magnn_val != 0 else 0
                        mtl_imp = ((mtl_val - magnn_val) / abs(magnn_val) * 100) if magnn_val != 0 else 0
                    else:
                        lstm_imp = ((magnn_val - lstm_val) / abs(magnn_val) * 100) if magnn_val != 0 else 0
                        mtl_imp = ((magnn_val - mtl_val) / abs(magnn_val) * 100) if magnn_val != 0 else 0

                    comparison_data.append({
                        'iteration': iteration,
                        'split': split,
                        'metric': metric,
                        'magnn': float(magnn_val) if not np.isnan(magnn_val) else None,
                        'magnn_lstm': float(lstm_val) if not np.isnan(lstm_val) else None,
                        'magnn_lstm_mtl': float(mtl_val) if not np.isnan(mtl_val) else None,
                        'lstm_improvement_pct': float(lstm_imp) if not np.isnan(lstm_imp) else None,
                        'mtl_improvement_pct': float(mtl_imp) if not np.isnan(mtl_imp) else None,
                    })

            comparison_json = {
                'iteration': iteration,
                'algorithm': ALGORITHM_NAME,
                'config': {
                    'n_epochs': config.n_epochs,
                    'batch_size': config.batch_size,
                    'learning_rate': config.learning_rate,
                    'sample_fraction': config.sample_fraction,
                    'mtl_lambda': config.mtl_lambda,
                    'max_path_length': 10,
                },
                'data_summary': {
                    'magnn_samples': len(train_dataset_magnn),
                    'lstm_samples': len(train_dataset_lstm),
                    'mtl_paths': len(train_dataset_mtl),
                },
                'magnn': {
                    split: {k: float(v) if isinstance(v, (int, float, np.number)) else str(v)
                            for k, v in magnn_results.get(split, {}).items()
                            if k in ['r2', 'rmse', 'mae', 'mape']}
                    for split in ['Train', 'Val', 'Test']
                },
                'magnn_lstm': {
                    split: {k: float(v) if isinstance(v, (int, float, np.number)) else str(v)
                            for k, v in lstm_results.get(split, {}).items()
                            if k in ['r2', 'rmse', 'mae', 'mape']}
                    for split in ['Train', 'Val', 'Test']
                },
                'magnn_lstm_mtl': {
                    split: {k: float(v) if isinstance(v, (int, float, np.number)) else str(v)
                            for k, v in mtl_results.get(split, {}).items()
                            if k in ['r2', 'rmse', 'mae', 'mape']}
                    for split in ['Train', 'Val', 'Test']
                },
                'comparison': comparison_data
            }

            json_path = os.path.join(output_folder, 'three_way_comparison.json')
            with open(json_path, 'w') as f:
                json.dump(comparison_json, f, indent=2)
            print(f"✓ Comparison metrics saved → {json_path}\n")

        except Exception as e:
            print(f"\n❌ Error in iteration {iteration}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Final summary
    if all_magnn_results and all_lstm_results and all_mtl_results:
        print_section("🏆 FINAL SUMMARY - AVERAGED ACROSS ALL ITERATIONS")

        for split in ['Train', 'Val', 'Test']:
            print(f"\n{'=' * 120}")
            print(f"{split.upper()} SET - AVERAGE OVER {len(all_magnn_results)} ITERATION(S)")
            print(f"{'=' * 120}")
            print(f"{'Metric':<10} {'MAGNN':<20} {'MAGNN-LSTM':<20} {'MAGNN-LSTM-MTL':<20} {'Best Model':<15}")
            print(f"{'-' * 120}")

            metrics_info = [
                ('R²', 'r2', True),
                ('RMSE', 'rmse', False),
                ('MAE', 'mae', False),
                ('MAPE', 'mape', False),
            ]

            for display_name, metric_key, higher_is_better in metrics_info:
                magnn_vals = [r.get(split, {}).get(metric_key, float('nan'))
                              for r in all_magnn_results]
                lstm_vals = [r.get(split, {}).get(metric_key, float('nan'))
                             for r in all_lstm_results]
                mtl_vals = [r.get(split, {}).get(metric_key, float('nan'))
                            for r in all_mtl_results]

                magnn_avg = np.nanmean(magnn_vals)
                lstm_avg = np.nanmean(lstm_vals)
                mtl_avg = np.nanmean(mtl_vals)
                magnn_std = np.nanstd(magnn_vals)
                lstm_std = np.nanstd(lstm_vals)
                mtl_std = np.nanstd(mtl_vals)

                if metric_key in ['rmse', 'mae']:
                    magnn_str = f"{magnn_avg:.2f}±{magnn_std:.2f}s"
                    lstm_str = f"{lstm_avg:.2f}±{lstm_std:.2f}s"
                    mtl_str = f"{mtl_avg:.2f}±{mtl_std:.2f}s"
                elif metric_key == 'mape':
                    magnn_str = f"{magnn_avg:.2f}±{magnn_std:.2f}%"
                    lstm_str = f"{lstm_avg:.2f}±{lstm_std:.2f}%"
                    mtl_str = f"{mtl_avg:.2f}±{mtl_std:.2f}%"
                else:
                    magnn_str = f"{magnn_avg:.4f}±{magnn_std:.4f}"
                    lstm_str = f"{lstm_avg:.4f}±{lstm_std:.4f}"
                    mtl_str = f"{mtl_avg:.4f}±{mtl_std:.4f}"

                # Determine best
                if not (np.isnan(magnn_avg) or np.isnan(lstm_avg) or np.isnan(mtl_avg)):
                    if higher_is_better:
                        best_val = max(magnn_avg, lstm_avg, mtl_avg)
                        if best_val == mtl_avg:
                            best_model = "MTL ✓"
                        elif best_val == lstm_avg:
                            best_model = "LSTM ✓"
                        else:
                            best_model = "MAGNN"
                    else:
                        best_val = min(magnn_avg, lstm_avg, mtl_avg)
                        if best_val == mtl_avg:
                            best_model = "MTL ✓"
                        elif best_val == lstm_avg:
                            best_model = "LSTM ✓"
                        else:
                            best_model = "MAGNN"
                else:
                    best_model = "N/A"

                print(f"{display_name:<10} {magnn_str:<20} {lstm_str:<20} {mtl_str:<20} {best_model:<15}")

            print(f"{'=' * 120}\n")

        print_section("✅ THREE-WAY COMPARISON COMPLETE")
        print(f"  Total iterations: {len(all_magnn_results)}")
        print(f"  All models trained on identical data splits")
        print(f"\n  Model Details:")
        print(f"    MAGNN: Graph attention + LSTM baseline (single segments)")
        print(f"    MAGNN-LSTM: + operational + weather features (single segments)")
        print(f"    MAGNN-LSTM-MTL: Multi-segment paths with MTL (λ={config.mtl_lambda})")
        print(f"      • Paths aggregated by trip_id")
        print(f"      • Task 1: Individual segment predictions")
        print(f"      • Task 2: Collective path predictions")
        print(f"      • L2-norm attention weighting")
        print(f"\n  Results saved in: outputs/{ALGORITHM_NAME}_*_three_way/")
        print("=" * 80)


# =============================================================================
# TWO-WAY COMPARISON MODE (LEGACY)
# =============================================================================

def main_with_lstm_comparison():
    """Main training loop with MAGNN vs MAGNN-LSTM comparison."""

    config = Config()

    print_section("🔬 MAGNN vs MAGNN-LSTM COMPARISON MODE")
    print(f"  Device: {DEVICE}")
    print(f"  Algorithm: {ALGORITHM_NAME}")
    print(f"  Data sampling: {config.sample_fraction * 100}%")
    print(f"  Iterations: {config.n_iterations}")
    print(f"  Epochs per iteration: {config.n_epochs}")
    print(f"\n  Models to compare:")
    print(f"    1. MAGNN (baseline)")
    print(f"    2. MAGNN-LSTM")
    print("=" * 80)

    all_magnn_results = []
    all_lstm_results = []

    for iteration in range(1, config.n_iterations + 1):
        print_section(f"🔄 ITERATION {iteration}/{config.n_iterations}")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_folder = f"outputs/{ALGORITHM_NAME}_{timestamp}_i{iteration}_comparison"
        os.makedirs(output_folder, exist_ok=True)
        print(f"📁 Output: {output_folder}")

        try:
            print("\n[1/9] Loading data...")
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
            print(f"      ✓ Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

            print("[2/9] Clustering stops...")
            clusters, station_cluster_ids = event_driven_clustering_fixed(
                train_df, known_stops=known_stops
            )
            if len(clusters) == 0:
                print("❌ No clusters")
                continue
            print(f"      ✓ {len(clusters)} clusters created")

            print("[3/9] Building segments...")
            train_segments = build_segments_fixed(train_df, clusters)
            test_segments = build_segments_fixed(test_df, clusters)
            val_segments = build_segments_fixed(val_df, clusters)

            if len(train_segments) == 0:
                print("❌ No segments")
                continue
            print(f"      ✓ {len(train_segments):,} training segments")

            print("[4/9] Building adjacency matrices...")
            adj_geo, adj_dist, adj_soc, segment_types = build_adjacency_matrices_fixed(
                train_segments, clusters,
                known_stops=known_stops
            )

            if adj_geo is None:
                print("❌ Adjacency failed")
                continue
            print(f"      ✓ 3 adjacency matrices (geo, dist, social)")

            print("[5/9] Preparing MAGNN datasets...")

            train_dataset_magnn = SegmentDataset(
                train_segments, segment_types, fit_scalers=True
            )
            val_dataset_magnn = SegmentDataset(
                val_segments, segment_types, fit_scalers=False,
                target_scaler=train_dataset_magnn.target_scaler,
                speed_scaler=train_dataset_magnn.speed_scaler,
            )
            test_dataset_magnn = SegmentDataset(
                test_segments, segment_types, fit_scalers=False,
                target_scaler=train_dataset_magnn.target_scaler,
                speed_scaler=train_dataset_magnn.speed_scaler,
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

            print(f"      ✓ MAGNN: {len(train_dataset_magnn):,} samples")

            print("[6/9] Preparing MAGNN-LSTM datasets...")

            train_dataset_lstm = EnhancedSegmentDataset(
                train_segments, segment_types, fit_scalers=True
            )

            val_dataset_lstm = EnhancedSegmentDataset(
                val_segments, segment_types, fit_scalers=False,
                target_scaler=train_dataset_lstm.target_scaler,
                speed_scaler=train_dataset_lstm.speed_scaler,
                operational_scaler=train_dataset_lstm.operational_scaler,
                weather_scaler=train_dataset_lstm.weather_scaler,
            )
            test_dataset_lstm = EnhancedSegmentDataset(
                test_segments, segment_types, fit_scalers=False,
                target_scaler=train_dataset_lstm.target_scaler,
                speed_scaler=train_dataset_lstm.speed_scaler,
                operational_scaler=train_dataset_lstm.operational_scaler,
                weather_scaler=train_dataset_lstm.weather_scaler,
            )

            train_loader_lstm = DataLoader(
                train_dataset_lstm, batch_size=config.batch_size,
                shuffle=True, num_workers=0, collate_fn=enhanced_collate_fn
            )
            val_loader_lstm = DataLoader(
                val_dataset_lstm, batch_size=config.batch_size,
                shuffle=False, num_workers=0, collate_fn=enhanced_collate_fn
            )
            test_loader_lstm = DataLoader(
                test_dataset_lstm, batch_size=config.batch_size,
                shuffle=False, num_workers=0, collate_fn=enhanced_collate_fn
            )

            print(f"      ✓ LSTM: {len(train_dataset_lstm):,} samples")

            print("\n[7/9] Training MAGNN (baseline)...")
            print("-" * 80)

            magnn_results, magnn_model = train_magtte(
                train_loader_magnn, val_loader_magnn, test_loader_magnn,
                adj_geo, adj_dist, adj_soc,
                segment_types, train_dataset_magnn.target_scaler,
                output_folder, DEVICE, config
            )

            all_magnn_results.append(magnn_results)

            print("\n[8/9] Training MAGNN-LSTM...")
            print("-" * 80)

            magnn_checkpoint = os.path.join(output_folder, 'magtte_best.pth')

            lstm_results, lstm_model = train_magnn_lstm(
                train_loader_lstm, val_loader_lstm, test_loader_lstm,
                adj_geo, adj_dist, adj_soc,
                segment_types, train_dataset_lstm.target_scaler,
                output_folder, DEVICE, config,
                pretrained_magnn_path=magnn_checkpoint,
                freeze_magnn=True
            )

            all_lstm_results.append(lstm_results)

            print("\n[9/9] Generating comparison report...")
            print_section(f"📊 ITERATION {iteration} - DETAILED COMPARISON")

            for split in ['Train', 'Val', 'Test']:
                print_comparison_table(magnn_results, lstm_results, split)

        except Exception as e:
            print(f"\n❌ Error in iteration {iteration}: {e}")
            import traceback
            traceback.print_exc()
            continue


# =============================================================================
# ABLATION STUDY MODE
# =============================================================================

def main_with_ablation_study():
    """Ablation study: Feature importance analysis."""

    config = Config()

    print_section("🔬 ABLATION STUDY MODE")
    print(f"  Device: {DEVICE}")
    print(f"  Algorithm: {ALGORITHM_NAME}")
    print(f"  Data sampling: {config.sample_fraction * 100}%")
    print(f"\n  Models to compare:")
    print(f"    1. MAGNN-LSTM-MTL (Full) - All features")
    print(f"    2. MAGNN-LSTM-MTL (No Operational) - Remove delay features")
    print(f"    3. MAGNN-LSTM-MTL (No Weather) - Remove weather features")
    print(f"    4. MAGNN-LSTM-MTL (Baseline) - Only spatial + temporal")
    print("=" * 80)

    all_full_results = []
    all_no_op_results = []
    all_no_weather_results = []
    all_baseline_results = []

    for iteration in range(1, config.n_iterations + 1):
        print_section(f"🔄 ITERATION {iteration}/{config.n_iterations}")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_folder = f"outputs/{ALGORITHM_NAME}_{timestamp}_i{iteration}_ablation"
        os.makedirs(output_folder, exist_ok=True)
        print(f"📁 Output: {output_folder}")

        try:
            # Load data
            print("\n[1/8] Loading data...")
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
            print(f"      ✓ Known stops: {len(known_stops)}")

            print("[2/8] Clustering stops...")
            clusters, station_cluster_ids = event_driven_clustering_fixed(
                train_df, known_stops=known_stops
            )
            if len(clusters) == 0:
                print("❌ No clusters")
                continue
            print(f"      ✓ {len(clusters)} clusters created")

            print("[3/8] Building segments...")
            train_segments = build_segments_fixed(train_df, clusters)
            test_segments = build_segments_fixed(test_df, clusters)
            val_segments = build_segments_fixed(val_df, clusters)

            if len(train_segments) == 0:
                print("❌ No segments")
                continue

            print("[4/8] Building adjacency matrices...")
            adj_geo, adj_dist, adj_soc, segment_types = build_adjacency_matrices_fixed(
                train_segments, clusters, known_stops=known_stops
            )

            if adj_geo is None:
                print("❌ Adjacency failed")
                continue

            print("[5/8] Aggregating paths...")
            train_paths = aggregate_segments_into_paths(train_segments, max_path_length=10)
            val_paths = aggregate_segments_into_paths(val_segments, max_path_length=10)
            test_paths = aggregate_segments_into_paths(test_segments, max_path_length=10)

            if len(train_paths) == 0:
                print("❌ No paths created")
                continue

            # Train baseline MAGNN first (needed for all variants)
            print("\n[6/8] Training baseline MAGNN (for transfer learning)...")
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

            magnn_results, magnn_model = train_magtte(
                train_loader_magnn, val_loader_magnn, test_loader_magnn,
                adj_geo, adj_dist, adj_soc,
                segment_types, train_dataset_magnn.target_scaler,
                output_folder, DEVICE, config
            )

            magnn_checkpoint = os.path.join(output_folder, 'magtte_best.pth')

            print("\n[7/8] Training ablation variants...")

            ablation_configs = [
                ("Full", True, True, "All features"),
                ("No Operational", False, True, "Spatial + Temporal + Weather"),
                ("No Weather", True, False, "Spatial + Temporal + Operational"),
                ("Baseline", False, False, "Spatial + Temporal only"),
            ]

            all_results = []

            for variant_name, use_op, use_weather, desc in ablation_configs:
                print(f"\n   Training: {variant_name} ({desc})")

                # Create datasets
                train_data = PathDataset(
                    train_paths, segment_types, max_path_length=10, fit_scalers=True,
                    use_operational=use_op, use_weather=use_weather
                )
                val_data = PathDataset(
                    val_paths, segment_types, max_path_length=10, fit_scalers=False,
                    target_scaler=train_data.target_scaler,
                    speed_scaler=train_data.speed_scaler,
                    operational_scaler=train_data.operational_scaler,
                    weather_scaler=train_data.weather_scaler,
                    use_operational=use_op, use_weather=use_weather
                )
                test_data = PathDataset(
                    test_paths, segment_types, max_path_length=10, fit_scalers=False,
                    target_scaler=train_data.target_scaler,
                    speed_scaler=train_data.speed_scaler,
                    operational_scaler=train_data.operational_scaler,
                    weather_scaler=train_data.weather_scaler,
                    use_operational=use_op, use_weather=use_weather
                )

                train_loader = DataLoader(train_data, batch_size=config.batch_size,
                                          shuffle=True, num_workers=0, collate_fn=path_collate_fn)
                val_loader = DataLoader(val_data, batch_size=config.batch_size,
                                        shuffle=False, num_workers=0, collate_fn=path_collate_fn)
                test_loader = DataLoader(test_data, batch_size=config.batch_size,
                                         shuffle=False, num_workers=0, collate_fn=path_collate_fn)

                # Train
                results, model = train_magnn_lstm_mtl(
                    train_loader, val_loader, test_loader,
                    adj_geo, adj_dist, adj_soc,
                    segment_types, train_data.target_scaler,
                    output_folder, DEVICE, config,
                    pretrained_magnn_path=magnn_checkpoint,
                    freeze_magnn=True
                )

                all_results.append((variant_name, results))

            print("\n[8/8] Generating ablation comparison...")
            print_section(f"📊 ITERATION {iteration} - ABLATION STUDY")

            # Print comparison table
            for split in ['Train', 'Val', 'Test']:
                print(f"\n{'=' * 140}")
                print(f"{split.upper()} SET - ABLATION STUDY")
                print(f"{'=' * 140}")
                print(
                    f"{'Metric':<10} {'Full':<15} {'No Operational':<18} {'No Weather':<18} {'Baseline':<18} {'Best':<15}")
                print(f"{'-' * 140}")

                metrics_info = [
                    ('R²', 'r2', True),
                    ('RMSE', 'rmse', False),
                    ('MAE', 'mae', False),
                    ('MAPE', 'mape', False),
                ]

                for display_name, metric_key, higher_is_better in metrics_info:
                    values = []
                    for variant_name, results in all_results:
                        val = results.get(split, {}).get(metric_key, float('nan'))
                        values.append((variant_name, val))

                    # Format values
                    formatted = []
                    for name, val in values:
                        if metric_key in ['rmse', 'mae']:
                            formatted.append(f"{val:.2f}s")
                        elif metric_key == 'mape':
                            formatted.append(f"{val:.2f}%")
                        else:
                            formatted.append(f"{val:.4f}")

                    # Determine best
                    numeric_vals = [v for _, v in values if not np.isnan(v)]
                    if numeric_vals:
                        if higher_is_better:
                            best_val = max(numeric_vals)
                        else:
                            best_val = min(numeric_vals)

                        best_name = [name for name, v in values if v == best_val][0]
                        best_str = f"{best_name} ✓"
                    else:
                        best_str = "N/A"

                    print(
                        f"{display_name:<10} {formatted[0]:<15} {formatted[1]:<18} {formatted[2]:<18} {formatted[3]:<18} {best_str:<15}")

                print(f"{'=' * 140}\n")

            # Store results
            all_full_results.append(all_results[0][1])
            all_no_op_results.append(all_results[1][1])
            all_no_weather_results.append(all_results[2][1])
            all_baseline_results.append(all_results[3][1])

            # Save JSON
            ablation_json = {
                'iteration': iteration,
                'algorithm': ALGORITHM_NAME,
                'config': {
                    'n_epochs': config.n_epochs,
                    'batch_size': config.batch_size,
                    'mtl_lambda': config.mtl_lambda,
                },
                'variants': {
                    name: {
                        split: {k: float(v) if isinstance(v, (int, float, np.number)) else str(v)
                                for k, v in results.get(split, {}).items()
                                if k in ['r2', 'rmse', 'mae', 'mape']}
                        for split in ['Train', 'Val', 'Test']
                    }
                    for name, results in all_results
                }
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

    print_section("✅ ABLATION STUDY COMPLETE")
    print(f"  Total iterations: {len(all_full_results)}")
    print(f"\n  Feature Importance Analysis:")
    print(f"    • Compare 'Full' vs 'No Operational' → Operational delay contribution")
    print(f"    • Compare 'Full' vs 'No Weather' → Weather feature contribution")
    print(f"    • Compare 'Full' vs 'Baseline' → Total feature contribution")
    print(f"\n  Results saved in: outputs/{ALGORITHM_NAME}_*_ablation/")
    print("=" * 80)

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == '--compare-all' or sys.argv[1] == '--mtl':
            main_with_three_way_comparison()
        elif sys.argv[1] == '--compare':
            main_with_lstm_comparison()
        elif sys.argv[1] == '--ablation':  # ADD THIS
            main_with_ablation_study()
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print("\nUsage:")
            print("  python main.py                # Train MAGNN only")
            print("  python main.py --compare      # Compare MAGNN vs MAGNN-LSTM")
            print("  python main.py --compare-all  # Compare all three models")
            print("  python main.py --ablation     # Feature ablation study")  # ADD THIS
    else:
        main()