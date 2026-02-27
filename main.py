"""
main.py
=======
Entry point for MAGNN-LSTM transit travel time prediction.

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
from cluster_knn     import event_driven_clustering_fixed   # ← swap here

# Derived automatically from the module name — never needs manual editing.
# cluster_hdbscan → HDBSCAN,  cluster_knn → KNN,  cluster_gmm → GMM, etc.
ALGORITHM_NAME = event_driven_clustering_fixed.__module__.replace('cluster_', '').upper()

# =============================================================================
# SHARED MODULES
# =============================================================================
from config import Config, DEVICE, print_section, haversine_meters
from data_loader import (load_data_fixed, load_train_test_val_data_fixed,
                          get_known_stops)
from segments import build_segments_fixed, build_adjacency_matrices_fixed
from visualizations import (plot_clusters, plot_segments,
                              plot_segment_statistics)
from model import (SegmentDataset, masked_collate_fn,
                   train_magtte, SimpleMLP, train_simple)

import numpy as np
import pandas as pd
import os
import json
import warnings
from datetime import datetime
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore')

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """Main training loop."""

    config = Config()

    print_section("MAGNN-LSTM TRAINING CONFIGURATION")
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
        output_folder = f"outputs/magnn_run_{timestamp}_iter{iteration}"
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

            # Extract known named stops BEFORE clustering so we can mark them
            known_stops = get_known_stops(train_df)
            print(f"   Known stops found in data: {len(known_stops)}")

            # ------------------------------------------------------------------
            # 2. Clustering  — stations always injected
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

            test_segments  = build_segments_fixed(test_df,  clusters)
            val_segments   = build_segments_fixed(val_df,   clusters)

            # ------------------------------------------------------------------
            # 4. PLOTS  — generated before training so you always see them
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
                                    save_path=os.path.join(output_folder, f'{ALGORITHM_NAME.lower()}-segment_stats.png'))

            # ------------------------------------------------------------------
            # 5. Adjacency matrices
            # ------------------------------------------------------------------
            adj_geo, adj_dist, adj_soc, segment_types = build_adjacency_matrices_fixed(
                train_segments, clusters
            )

            if adj_geo is None:
                print("❌ Adjacency failed")
                continue

            # ------------------------------------------------------------------
            # 6. Datasets & data loaders
            # ------------------------------------------------------------------
            # Train dataset: fit the RobustScalers here
            train_dataset = SegmentDataset(
                train_segments, segment_types,
                fit_scalers=True
            )
            # Val / Test datasets: reuse the fitted scalers from training
            # (prevents future data influencing the scaling transform)
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

            # All loaders use masked_collate_fn so batches carry the
            # padding mask for LSTM sequence masking support
            train_loader = DataLoader(
                train_dataset, batch_size=config.batch_size,
                shuffle=True,  num_workers=0,
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
            # 7. MAGTTE + GAT training  (real model, NaN-safe)
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
                    'n_epochs':        config.n_epochs,
                    'batch_size':      config.batch_size,
                    'learning_rate':   config.learning_rate,
                    'sample_fraction': config.sample_fraction,
                    'n_clusters':      len(clusters),
                    'n_segment_types': len(segment_types),
                    'node_embed_dim':  config.node_embed_dim,
                    'gat_hidden':      config.gat_hidden,
                    'lstm_hidden':     config.lstm_hidden,
                    'historical_dim':  config.historical_dim,
                },
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            }

            for dataset_name, res in [('Train', results.get('Train')),
                                       ('Val',   results.get('Val')),
                                       ('Test',  results.get('Test'))]:
                if res:
                    metrics_out[dataset_name] = {
                        'R2':   f"{res['r2']:.4f}",
                        'RMSE': f"{res['rmse']:.2f} seconds",
                        'MAE':  f"{res['mae']:.2f} seconds",
                        'MAPE': f"{res['mape']:.2f}%",
                    }

            json_path = os.path.join(output_folder, 'metrics.json')
            with open(json_path, 'w') as f:
                json.dump(metrics_out, f, indent=2)
            print(f"\n✓ Metrics saved → {json_path}")

            print(f"\n✅ Iteration {iteration} complete")
            print(f"   Output files saved to: {output_folder}/")
            print(f"     - {ALGORITHM_NAME.lower()}-clusters.png      (cluster locations)")
            print(f"     - {ALGORITHM_NAME.lower()}-segments.png      (segment connections)")
            print(f"     - {ALGORITHM_NAME.lower()}-segment_stats.png (segment statistics)")
            print(f"     - metrics.json              (evaluation metrics)")
            print(f"     - magtte_best.pth           (saved model weights)")

        except Exception as e:
            print(f"\n❌ Error in iteration {iteration}: {e}")
            import traceback
            traceback.print_exc()
            continue


if __name__ == '__main__':
    main()