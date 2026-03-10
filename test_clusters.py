"""
test_clusters.py
================
Sweep different cluster sizes for GMM and K-Means to find the best
configuration for MAGNN-LSTM transit travel-time prediction.

USAGE
-----
    python test_clusters.py                  # defaults from config
    python test_clusters.py 0.1              # 10% sample fraction
    python test_clusters.py 0.15             # 15% sample fraction

Tested cluster sizes: 500, 750, 1000, 1250, 1500
Algorithms tested:    gmm, kmeans

After the sweep completes, results are saved to:
    outputs/test_clusters_run<N>/cluster_metrics.csv
    outputs/test_clusters_run<N>/cluster_metrics.json
    outputs/test_clusters_run<N>/GMM_500/   (per-experiment files)
    outputs/test_clusters_run<N>/KMEANS_1000/ ...
"""

import importlib
import numpy as np
import os
import sys
import csv
import json
import warnings
from datetime import datetime
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore')

from config import Config, DEVICE, print_section
from data_loader import load_train_test_val_data_fixed, get_known_stops
from segments import build_segments_fixed, build_adjacency_matrices_fixed
from visualizations import plot_clusters, plot_segments, plot_segment_statistics
from model import (SegmentDataset, masked_collate_fn, train_magtte)

# ============================================================================
# SWEEP SETTINGS
# ============================================================================
CLUSTER_SIZES = [500, 750, 1000, 1250, 1500]
ALGORITHMS = ['gmm', 'kmeans']


def run_single_experiment(method, n_clusters, config, run_folder):
    """
    Run one full pipeline: cluster → segment → train → evaluate.

    Returns a dict with metrics, or None if the run failed.
    """
    module = importlib.import_module(f'cluster_{method}')
    event_driven_clustering_fixed = module.event_driven_clustering_fixed
    algo_name = method.upper()

    output_folder = os.path.join(run_folder, f"{algo_name}_{n_clusters}")
    os.makedirs(output_folder, exist_ok=True)

    print_section(f"EXPERIMENT: {algo_name}  |  n_clusters = {n_clusters}")
    print(f"  📁 Output: {output_folder}")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    train_df, test_df, val_df = load_train_test_val_data_fixed(
        data_folder=config.data_folder,
        sample_fraction=config.sample_fraction,
    )
    if len(train_df) == 0:
        print("❌ No training data")
        return None

    known_stops_train = get_known_stops(train_df)
    known_stops_test  = get_known_stops(test_df)
    known_stops_val   = get_known_stops(val_df)
    known_stops = {**known_stops_test, **known_stops_val, **known_stops_train}

    # ------------------------------------------------------------------
    # 2. Clustering — pass the n_clusters size being tested
    # ------------------------------------------------------------------
    clusters, station_cluster_ids = event_driven_clustering_fixed(
        train_df, known_stops=known_stops, n_clusters=n_clusters,
    )
    if len(clusters) == 0:
        print("❌ No clusters produced")
        return None

    # ------------------------------------------------------------------
    # 3. Build segments
    # ------------------------------------------------------------------
    train_segments = build_segments_fixed(train_df, clusters)
    if len(train_segments) == 0:
        print("❌ No segments")
        return None

    test_segments = build_segments_fixed(test_df, clusters)
    val_segments  = build_segments_fixed(val_df, clusters)

    # ------------------------------------------------------------------
    # 4. Visualisations
    # ------------------------------------------------------------------
    plot_clusters(clusters, {},
                  algorithm_name=algo_name,
                  save_path=os.path.join(output_folder,
                                         f'{algo_name.lower()}-clusters.png'))
    plot_segments(train_segments, clusters, max_segments=100,
                  algorithm_name=algo_name,
                  save_path=os.path.join(output_folder,
                                         f'{algo_name.lower()}-segments.png'))
    plot_segment_statistics(train_segments,
                            algorithm_name=algo_name,
                            save_path=os.path.join(output_folder,
                                                    f'{algo_name.lower()}-segment_stats.png'))

    # ------------------------------------------------------------------
    # 5. Adjacency matrices
    # ------------------------------------------------------------------
    adj_geo, adj_dist, adj_soc, segment_types = build_adjacency_matrices_fixed(
        train_segments, clusters, known_stops=known_stops,
    )
    if adj_geo is None:
        print("❌ Adjacency failed")
        return None

    # ------------------------------------------------------------------
    # 6. Datasets & data-loaders
    # ------------------------------------------------------------------
    train_dataset = SegmentDataset(train_segments, segment_types,
                                   fit_scalers=True)
    val_dataset = SegmentDataset(val_segments, segment_types,
                                  fit_scalers=False,
                                  target_scaler=train_dataset.target_scaler,
                                  speed_scaler=train_dataset.speed_scaler)
    test_dataset = SegmentDataset(test_segments, segment_types,
                                   fit_scalers=False,
                                   target_scaler=train_dataset.target_scaler,
                                   speed_scaler=train_dataset.speed_scaler)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, num_workers=0,
                              collate_fn=masked_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                            shuffle=False, num_workers=0,
                            collate_fn=masked_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                             shuffle=False, num_workers=0,
                             collate_fn=masked_collate_fn)

    print(f"\n📊  Data — segments: {len(segment_types)}  |  "
          f"train: {len(train_dataset):,}  |  "
          f"val: {len(val_dataset):,}  |  "
          f"test: {len(test_dataset):,}")

    if len(train_loader) == 0:
        print("❌ Training loader is empty")
        return None

    # ------------------------------------------------------------------
    # 7. MAGTTE training
    # ------------------------------------------------------------------
    results, _ = train_magtte(
        train_loader, val_loader, test_loader,
        adj_geo, adj_dist, adj_soc,
        segment_types, train_dataset.target_scaler,
        output_folder, DEVICE, config,
    )

    # ------------------------------------------------------------------
    # 8. Save metrics.json (per-run, same format as main.py)
    # ------------------------------------------------------------------
    metrics_out = {
        'graph_method': method,
        'config': {
            'n_epochs':        config.n_epochs,
            'batch_size':      config.batch_size,
            'learning_rate':   config.learning_rate,
            'sample_fraction': config.sample_fraction,
            'n_clusters':      len(clusters),
            'requested_clusters': n_clusters,
            'n_segment_types': len(segment_types),
            'node_embed_dim':  config.node_embed_dim,
            'gat_hidden':      config.gat_hidden,
            'lstm_hidden':     config.lstm_hidden,
            'historical_dim':  config.historical_dim,
        },
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    for split_name in ['Train', 'Val', 'Test']:
        res = results.get(split_name)
        if res:
            metrics_out[split_name] = {
                'R2':   f"{res['r2']:.4f}",
                'RMSE': f"{res['rmse']:.2f} seconds",
                'MAE':  f"{res['mae']:.2f} seconds",
                'MAPE': f"{res['mape']:.2f}%",
            }

    json_path = os.path.join(output_folder, 'metrics.json')
    with open(json_path, 'w') as f:
        json.dump(metrics_out, f, indent=2)
    print(f"\n✓ Metrics saved → {json_path}")

    # ------------------------------------------------------------------
    # 9. Collect row for aggregate CSV
    # ------------------------------------------------------------------
    row = {
        'algorithm':      algo_name,
        'cluster_count':  n_clusters,
        'actual_clusters': len(clusters),
        'n_segment_types': len(segment_types),
        'train_samples':  len(train_dataset),
        'val_samples':    len(val_dataset),
        'test_samples':   len(test_dataset),
        'output_folder':  output_folder,
    }

    for split_name in ['Train', 'Val', 'Test']:
        res = results.get(split_name)
        if res:
            row[f'{split_name}_R2']   = round(res['r2'],   4)
            row[f'{split_name}_RMSE'] = round(res['rmse'], 2)
            row[f'{split_name}_MAE']  = round(res['mae'],  2)
            row[f'{split_name}_MAPE'] = round(res['mape'], 2)
        else:
            row[f'{split_name}_R2']   = None
            row[f'{split_name}_RMSE'] = None
            row[f'{split_name}_MAE']  = None
            row[f'{split_name}_MAPE'] = None

    return row


# ============================================================================
# MAIN SWEEP
# ============================================================================

def _next_run_number():
    """Find the next available test_clusters_run<N> number inside outputs/."""
    os.makedirs('outputs', exist_ok=True)
    existing = [
        d for d in os.listdir('outputs')
        if d.startswith('test_clusters_run') and os.path.isdir(os.path.join('outputs', d))
    ]
    nums = []
    for d in existing:
        try:
            nums.append(int(d.replace('test_clusters_run', '')))
        except ValueError:
            pass
    return max(nums, default=0) + 1


def main():
    config = Config()

    # Optional CLI arg: sample fraction
    for arg in sys.argv[1:]:
        try:
            config.sample_fraction = float(arg)
        except ValueError:
            print(f"⚠️  Unknown argument '{arg}' — ignored")

    # Create the parent folder for this sweep
    run_num = _next_run_number()
    run_folder = f"outputs/test_clusters_run{run_num}"
    os.makedirs(run_folder, exist_ok=True)

    output_csv  = os.path.join(run_folder, 'cluster_metrics.csv')
    output_json = os.path.join(run_folder, 'cluster_metrics.json')

    print_section("CLUSTER SIZE SWEEP")
    print(f"  Run folder      : {run_folder}")
    print(f"  Device          : {DEVICE}")
    print(f"  Sample fraction : {config.sample_fraction * 100:.1f}%")
    print(f"  Algorithms      : {', '.join(a.upper() for a in ALGORITHMS)}")
    print(f"  Cluster sizes   : {CLUSTER_SIZES}")
    print(f"  Total runs      : {len(ALGORITHMS) * len(CLUSTER_SIZES)}")
    print("=" * 80)

    all_rows = []
    run_number = 0
    total_runs = len(ALGORITHMS) * len(CLUSTER_SIZES)

    for method in ALGORITHMS:
        for n_clusters in CLUSTER_SIZES:
            run_number += 1
            print(f"\n{'#' * 80}")
            print(f"# RUN {run_number}/{total_runs}  —  "
                  f"{method.upper()} with {n_clusters} clusters")
            print(f"{'#' * 80}")

            try:
                row = run_single_experiment(method, n_clusters, config, run_folder)
                if row is not None:
                    all_rows.append(row)
                else:
                    print(f"⚠️  Run failed (returned None)")
            except Exception as e:
                print(f"❌ Exception during {method.upper()} k={n_clusters}: {e}")
                import traceback
                traceback.print_exc()

    # ======================================================================
    # SAVE RESULTS
    # ======================================================================
    if not all_rows:
        print("\n❌ No successful runs — nothing to save.")
        return

    # CSV column order
    columns = [
        'algorithm', 'cluster_count', 'actual_clusters', 'n_segment_types',
        'train_samples', 'val_samples', 'test_samples',
        'Train_R2', 'Train_RMSE', 'Train_MAE', 'Train_MAPE',
        'Val_R2',   'Val_RMSE',   'Val_MAE',   'Val_MAPE',
        'Test_R2',  'Test_RMSE',  'Test_MAE',  'Test_MAPE',
        'output_folder',
    ]

    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(all_rows)

    # ------------------------------------------------------------------
    # Pretty-print the results table
    # ------------------------------------------------------------------
    print_section("CLUSTER SWEEP RESULTS")

    header = f"{'Algorithm':<10} {'Clusters':>8} {'Train R²':>9} {'Val R²':>9} " \
             f"{'Test R²':>9} {'Test RMSE':>10} {'Test MAE':>9} {'Test MAPE':>10}"
    print(header)
    print("-" * len(header))

    for row in all_rows:
        tr2  = f"{row['Train_R2']:.4f}"  if row['Train_R2'] is not None else '   N/A'
        vr2  = f"{row['Val_R2']:.4f}"    if row['Val_R2']   is not None else '   N/A'
        ter2 = f"{row['Test_R2']:.4f}"   if row['Test_R2']  is not None else '   N/A'
        rmse = f"{row['Test_RMSE']:.2f}" if row['Test_RMSE'] is not None else '   N/A'
        mae  = f"{row['Test_MAE']:.2f}"  if row['Test_MAE']  is not None else '   N/A'
        mape = f"{row['Test_MAPE']:.2f}%" if row['Test_MAPE'] is not None else '    N/A'
        print(f"{row['algorithm']:<10} {row['cluster_count']:>8} {tr2:>9} "
              f"{vr2:>9} {ter2:>9} {rmse:>10} {mae:>9} {mape:>10}")

    # ------------------------------------------------------------------
    # Determine the best cluster count per algorithm
    # ------------------------------------------------------------------
    print_section("BEST CLUSTER SIZE PER ALGORITHM")

    # We rank by Test R² (higher is better). If R² is tied, lower RMSE wins.
    best_per_algo = {}
    for row in all_rows:
        algo = row['algorithm']
        if row['Test_R2'] is None:
            continue
        if algo not in best_per_algo:
            best_per_algo[algo] = row
        else:
            current_best = best_per_algo[algo]
            # Higher R² is better; tie-break on lower RMSE
            if (row['Test_R2'] > current_best['Test_R2']) or \
               (row['Test_R2'] == current_best['Test_R2'] and
                row['Test_RMSE'] < current_best['Test_RMSE']):
                best_per_algo[algo] = row

    for algo, best in best_per_algo.items():
        print(f"\n  🏆  {algo}  →  best cluster count = {best['cluster_count']}")
        print(f"       Test R²   = {best['Test_R2']:.4f}")
        print(f"       Test RMSE = {best['Test_RMSE']:.2f} seconds")
        print(f"       Test MAE  = {best['Test_MAE']:.2f} seconds")
        print(f"       Test MAPE = {best['Test_MAPE']:.2f}%")

    # ------------------------------------------------------------------
    # Save best-per-algorithm summary to JSON
    # ------------------------------------------------------------------
    best_json = {
        'title': 'BEST CLUSTER SIZE PER ALGORITHM',
        'ranking_criteria': 'Test R² (higher is better), tie-break on Test RMSE (lower is better)',
        'cluster_sizes_tested': CLUSTER_SIZES,
        'algorithms_tested': [a.upper() for a in ALGORITHMS],
        'best_per_algorithm': {},
    }

    for algo, best in best_per_algo.items():
        best_json['best_per_algorithm'][algo] = {
            'best_cluster_count': best['cluster_count'],
            'Test_R2':   best['Test_R2'],
            'Test_RMSE': best['Test_RMSE'],
            'Test_MAE':  best['Test_MAE'],
            'Test_MAPE': best['Test_MAPE'],
            'Train_R2':  best['Train_R2'],
            'Val_R2':    best['Val_R2'],
            'output_folder': best['output_folder'],
        }

    with open(output_json, 'w') as f:
        json.dump(best_json, f, indent=2)

    print(f"\n✅ CSV results saved to : {output_csv}")
    print(f"✅ Best metrics saved to: {output_json}")


if __name__ == '__main__':
    main()
