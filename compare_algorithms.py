"""
compare_algorithms.py
=====================
Compare inference timing across all clustering algorithms for MAGNN-LSTM-MTL.

USAGE
-----
    python compare_algorithms.py                  # defaults from config
    python compare_algorithms.py 0.1              # 10% sample fraction

Algorithms tested: kmeans, knn, dbscan, hdbscan, gmm

Results saved to:
    outputs/algorithm_comparison_<timestamp>/comparison_results.csv
    outputs/algorithm_comparison_<timestamp>/comparison_results.json
    outputs/algorithm_comparison_<timestamp>/<ALGORITHM>/  (per-algorithm files)
"""

import importlib
import os
import sys
import csv
import json
import time
import warnings
from datetime import datetime
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore')

from config import Config, DEVICE, print_section
from data_loader import load_train_test_val_data_fixed, get_known_stops
from segments import build_segments_fixed, build_adjacency_matrices_fixed
from visualizations import plot_clusters, plot_segments, plot_segment_statistics
from model import (SegmentDataset, masked_collate_fn, train_magtte)

# All clustering algorithms to compare
ALGORITHMS = ['kmeans', 'knn', 'dbscan', 'hdbscan', 'gmm']


def run_algorithm_experiment(method, config, run_folder):
    """Run a single algorithm experiment and return timing metrics."""
    
    pipeline_start = time.time()
    
    module = importlib.import_module(f'cluster_{method}')
    event_driven_clustering_fixed = module.event_driven_clustering_fixed
    algo_name = method.upper()
    
    output_folder = os.path.join(run_folder, algo_name)
    os.makedirs(output_folder, exist_ok=True)
    
    print_section(f"ALGORITHM: {algo_name}")
    print(f"  📁 Output: {output_folder}")
    
    # ==================== DATA LOADING ====================
    data_load_start = time.time()
    train_df, test_df, val_df = load_train_test_val_data_fixed(
        data_folder=config.data_folder,
        sample_fraction=config.sample_fraction,
    )
    data_load_time = time.time() - data_load_start
    
    if len(train_df) == 0:
        print("❌ No training data")
        return None
    
    known_stops_train = get_known_stops(train_df)
    known_stops_test = get_known_stops(test_df)
    known_stops_val = get_known_stops(val_df)
    known_stops = {**known_stops_test, **known_stops_val, **known_stops_train}
    
    # ==================== CLUSTERING TIME ====================
    clustering_start = time.time()
    clusters, station_cluster_ids = event_driven_clustering_fixed(
        train_df, known_stops=known_stops,
    )
    clustering_time = time.time() - clustering_start
    
    if len(clusters) == 0:
        print("❌ No clusters produced")
        return None
    
    print(f"  ✓ Clusters created: {len(clusters)} (clustering took {clustering_time*1000:.2f} ms)")
    
    # ==================== SEGMENT BUILDING ====================
    segment_start = time.time()
    train_segments = build_segments_fixed(train_df, clusters)
    if len(train_segments) == 0:
        print("❌ No segments")
        return None
    
    test_segments = build_segments_fixed(test_df, clusters)
    val_segments = build_segments_fixed(val_df, clusters)
    segment_time = time.time() - segment_start
    
    # Save visualizations
    plot_clusters(clusters, {},
                  algorithm_name=algo_name,
                  save_path=os.path.join(output_folder, f'{algo_name.lower()}-clusters.png'))
    plot_segments(train_segments, clusters, max_segments=100,
                  algorithm_name=algo_name,
                  save_path=os.path.join(output_folder, f'{algo_name.lower()}-segments.png'))
    plot_segment_statistics(train_segments,
                            algorithm_name=algo_name,
                            save_path=os.path.join(output_folder, f'{algo_name.lower()}-segment_stats.png'))
    
    # ==================== ADJACENCY MATRICES ====================
    adj_start = time.time()
    adj_geo, adj_dist, adj_soc, segment_types = build_adjacency_matrices_fixed(
        train_segments, clusters, known_stops=known_stops,
    )
    adj_time = time.time() - adj_start
    
    if adj_geo is None:
        print("❌ Adjacency failed")
        return None
    
    # ==================== DATASET CREATION ====================
    dataset_start = time.time()
    train_dataset = SegmentDataset(train_segments, segment_types, fit_scalers=True)
    val_dataset = SegmentDataset(val_segments, segment_types,
                                  fit_scalers=False,
                                  target_scaler=train_dataset.target_scaler,
                                  speed_scaler=train_dataset.speed_scaler)
    test_dataset = SegmentDataset(test_segments, segment_types,
                                   fit_scalers=False,
                                   target_scaler=train_dataset.target_scaler,
                                   speed_scaler=train_dataset.speed_scaler)
    dataset_time = time.time() - dataset_start
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, num_workers=0,
                              collate_fn=masked_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                            shuffle=False, num_workers=0,
                            collate_fn=masked_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                             shuffle=False, num_workers=0,
                             collate_fn=masked_collate_fn)
    
    print(f"\n  Data — segments: {len(segment_types)}  |  "
          f"train: {len(train_dataset):,}  |  "
          f"val: {len(val_dataset):,}  |  "
          f"test: {len(test_dataset):,}")
    
    if len(train_loader) == 0:
        print("❌ Training loader is empty")
        return None
    
    # ==================== TRAINING ====================
    training_start = time.time()
    results, _ = train_magtte(
        train_loader, val_loader, test_loader,
        adj_geo, adj_dist, adj_soc,
        segment_types, train_dataset.target_scaler,
        output_folder, DEVICE, config,
    )
    training_time = time.time() - training_start
    
    total_pipeline_time = time.time() - pipeline_start
    
    # Extract inference timing from test results
    test_res = results.get('Test', {})
    inference_timing = test_res.get('inference_timing', {})
    
    # ==================== PRINT TIMING METRICS ====================
    print(f"\n{'='*60}")
    print(f"TIMING METRICS — {algo_name}")
    print(f"{'='*60}")
    print(f"  Data Loading:              {data_load_time*1000:.2f} ms")
    print(f"  Clustering Time:           {clustering_time*1000:.2f} ms")
    print(f"  Segment Building:          {segment_time*1000:.2f} ms")
    print(f"  Adjacency Matrices:        {adj_time*1000:.2f} ms")
    print(f"  Dataset Creation:          {dataset_time*1000:.2f} ms")
    print(f"  Training Time:             {training_time:.2f} s")
    print(f"  ---")
    print(f"  Avg Model Forward:         {inference_timing.get('avg_model_forward_ms', 0):.4f} ms")
    print(f"  Total Inference Time:      {inference_timing.get('total_inference_time_s', 0):.4f} s")
    print(f"  Throughput:                {inference_timing.get('throughput_samples_per_sec', 0):.2f} samples/sec")
    print(f"  Avg Latency per Sample:    {inference_timing.get('avg_latency_per_sample_ms', 0):.4f} ms")
    print(f"  ---")
    print(f"  Total Pipeline Time:       {total_pipeline_time:.2f} s ({total_pipeline_time*1000:.2f} ms)")
    print(f"{'='*60}")
    
    # Build result dictionary
    result = {
        'algorithm': algo_name,
        'n_clusters': len(clusters),
        'n_segment_types': len(segment_types),
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'test_samples': len(test_dataset),
        
        # Timing metrics (KEY COMPARISON METRICS)
        'clustering_time_ms': round(clustering_time * 1000, 4),
        'avg_model_forward_ms': round(inference_timing.get('avg_model_forward_ms', 0), 4),
        'total_inference_time_s': round(inference_timing.get('total_inference_time_s', 0), 4),
        'throughput_samples_per_sec': round(inference_timing.get('throughput_samples_per_sec', 0), 2),
        'avg_latency_per_sample_ms': round(inference_timing.get('avg_latency_per_sample_ms', 0), 4),
        'total_pipeline_time_ms': round(total_pipeline_time * 1000, 4),
        
        # Additional timing breakdown
        'data_load_time_ms': round(data_load_time * 1000, 4),
        'segment_building_time_ms': round(segment_time * 1000, 4),
        'adjacency_time_ms': round(adj_time * 1000, 4),
        'dataset_creation_time_ms': round(dataset_time * 1000, 4),
        'training_time_s': round(training_time, 4),
        
        # Model performance metrics
        'test_r2': round(test_res.get('r2', 0), 4),
        'test_rmse': round(test_res.get('rmse', 0), 2),
        'test_mae': round(test_res.get('mae', 0), 2),
        'test_mape': round(test_res.get('mape', 0), 2),
        
        'output_folder': output_folder,
    }
    
    # Save individual algorithm metrics
    json_path = os.path.join(output_folder, 'metrics.json')
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n✓ Metrics saved → {json_path}")
    
    return result


def print_comparison_table(all_results):
    """Print a formatted comparison table of all algorithms."""
    
    print("\n")
    print("=" * 100)
    print("ALGORITHM COMPARISON — INFERENCE TIMING METRICS")
    print("=" * 100)
    
    # Header
    print(f"\n{'Algorithm':<12} | {'Clustering':<12} | {'Avg Forward':<12} | {'Total Inf.':<12} | {'Throughput':<14} | {'Latency':<12} | {'Pipeline':<12}")
    print(f"{'':12} | {'(ms)':<12} | {'(ms)':<12} | {'(s)':<12} | {'(samples/s)':<14} | {'(ms)':<12} | {'(ms)':<12}")
    print("-" * 100)
    
    for r in all_results:
        print(f"{r['algorithm']:<12} | "
              f"{r['clustering_time_ms']:>10.2f}   | "
              f"{r['avg_model_forward_ms']:>10.4f}   | "
              f"{r['total_inference_time_s']:>10.4f}   | "
              f"{r['throughput_samples_per_sec']:>12.2f}   | "
              f"{r['avg_latency_per_sample_ms']:>10.4f}   | "
              f"{r['total_pipeline_time_ms']:>10.2f}")
    
    print("-" * 100)
    
    # Find best for each metric
    print("\n📊 BEST PERFORMERS:")
    
    # Clustering time (lower is better)
    best_cluster = min(all_results, key=lambda x: x['clustering_time_ms'])
    print(f"  Fastest Clustering:        {best_cluster['algorithm']} ({best_cluster['clustering_time_ms']:.2f} ms)")
    
    # Avg forward time (lower is better)
    best_forward = min(all_results, key=lambda x: x['avg_model_forward_ms'])
    print(f"  Fastest Model Forward:     {best_forward['algorithm']} ({best_forward['avg_model_forward_ms']:.4f} ms)")
    
    # Total inference (lower is better)
    best_inference = min(all_results, key=lambda x: x['total_inference_time_s'])
    print(f"  Fastest Total Inference:   {best_inference['algorithm']} ({best_inference['total_inference_time_s']:.4f} s)")
    
    # Throughput (higher is better)
    best_throughput = max(all_results, key=lambda x: x['throughput_samples_per_sec'])
    print(f"  Highest Throughput:        {best_throughput['algorithm']} ({best_throughput['throughput_samples_per_sec']:.2f} samples/sec)")
    
    # Latency (lower is better)
    best_latency = min(all_results, key=lambda x: x['avg_latency_per_sample_ms'])
    print(f"  Lowest Latency:            {best_latency['algorithm']} ({best_latency['avg_latency_per_sample_ms']:.4f} ms)")
    
    # Pipeline (lower is better)
    best_pipeline = min(all_results, key=lambda x: x['total_pipeline_time_ms'])
    print(f"  Fastest Pipeline:          {best_pipeline['algorithm']} ({best_pipeline['total_pipeline_time_ms']:.2f} ms)")
    
    # Best accuracy (higher R2 is better)
    best_accuracy = max(all_results, key=lambda x: x['test_r2'])
    print(f"  Best Accuracy (R²):        {best_accuracy['algorithm']} (R²={best_accuracy['test_r2']:.4f})")
    
    print("=" * 100)


def main():
    # Parse command line arguments
    sample_fraction = None
    
    for arg in sys.argv[1:]:
        try:
            val = float(arg)
            if val <= 1.0:
                sample_fraction = val
        except ValueError:
            print(f"⚠️  Unknown argument '{arg}' — ignored.")
    
    config = Config()
    if sample_fraction is not None:
        config.sample_fraction = sample_fraction
    
    # Create output folder
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_folder = f"outputs/algorithm_comparison_{timestamp}"
    os.makedirs(run_folder, exist_ok=True)
    
    print_section("ALGORITHM COMPARISON — MAGNN-LSTM-MTL")
    print(f"  Device: {DEVICE}")
    print(f"  Data sampling: {config.sample_fraction * 100}%")
    print(f"  Epochs: {config.n_epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Algorithms: {', '.join([a.upper() for a in ALGORITHMS])}")
    print(f"  📁 Output: {run_folder}")
    print("=" * 80)
    
    all_results = []
    
    for algorithm in ALGORITHMS:
        print(f"\n{'#' * 80}")
        print(f"# Running {algorithm.upper()}")
        print(f"{'#' * 80}\n")
        
        result = run_algorithm_experiment(algorithm, config, run_folder)
        
        if result:
            all_results.append(result)
        else:
            print(f"⚠️  {algorithm.upper()} experiment failed, skipping...")
    
    if not all_results:
        print("\n❌ All experiments failed!")
        return
    
    # Print comparison table
    print_comparison_table(all_results)
    
    # Save combined results to CSV
    csv_path = os.path.join(run_folder, 'comparison_results.csv')
    fieldnames = all_results[0].keys()
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\n✓ CSV saved → {csv_path}")
    
    # Save combined results to JSON
    json_path = os.path.join(run_folder, 'comparison_results.json')
    with open(json_path, 'w') as f:
        json.dump({
            'config': {
                'sample_fraction': config.sample_fraction,
                'n_epochs': config.n_epochs,
                'batch_size': config.batch_size,
                'learning_rate': config.learning_rate,
                'device': str(DEVICE),
            },
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'algorithms_tested': ALGORITHMS,
            'results': all_results,
        }, f, indent=2)
    print(f"✓ JSON saved → {json_path}")
    
    print(f"\n🎉 Comparison complete! Results in: {run_folder}")


if __name__ == '__main__':
    main()