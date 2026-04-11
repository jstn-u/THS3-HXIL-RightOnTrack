

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import BallTree
import warnings

warnings.filterwarnings('ignore')

from config import print_section, haversine_meters

def simple_clustering(df, n_clusters=50, speed_threshold=2.0):
    print_section("STEP 2: CLUSTERING STOPS/STATIONS (GMM)")

    if df.empty:
        print("✗ Empty DataFrame provided")
        return np.empty((0, 2)), {}

    required_cols = ['latitude', 'longitude']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"✗ Missing required columns: {missing_cols}")
        return np.empty((0, 2)), {}

    if 'speed_mps' in df.columns:
        stops = df[df['speed_mps'] < speed_threshold]
        print(f"✓ Found {len(stops):,} low-speed points (< {speed_threshold} m/s)")
    else:
        stops = df
        print(f"⚠️  No 'speed_mps' column - using all {len(stops):,} points")

    coords = stops[['latitude', 'longitude']].dropna()

    if coords.empty:
        print("✗ No valid coordinates found")
        return np.empty((0, 2)), {}

    # Filter invalid coordinates (0,0 is "null island" - indicates invalid GPS)
    coords = coords[(coords['latitude'] != 0) & (coords['longitude'] != 0)]

    lat_mean, lat_std = coords['latitude'].mean(), coords['latitude'].std()
    lon_mean, lon_std = coords['longitude'].mean(), coords['longitude'].std()

    if lat_std > 0 and lon_std > 0:
        coords = coords[
            (np.abs(coords['latitude'] - lat_mean) < 3 * lat_std) &
            (np.abs(coords['longitude'] - lon_mean) < 3 * lon_std)
        ]

    if len(coords) < n_clusters:
        print(f"✗ Too few valid coordinates ({len(coords)}) for {n_clusters} clusters")
        return np.empty((0, 2)), {}

    print(f"✓ Using {len(coords):,} valid coordinates for clustering")

    coord_values = coords[['latitude', 'longitude']].values
    n_points = len(coord_values)
    actual_n_clusters = min(n_clusters, n_points // 2)
    if actual_n_clusters != n_clusters:
        print(f"⚠️  Adjusted n_clusters from {n_clusters} to {actual_n_clusters} (data size constraint)")

    print(f"\n{'='*60}")
    print(f"GMM PARAMETERS")
    print(f"{'='*60}")
    print(f"  n_components: {actual_n_clusters}")
    print(f"  covariance_type: 'diag' (diagonal — sufficient for lat/lon, 2-3x faster than 'full')")
    print(f"  max_iter: 200")
    print(f"  n_init: 3 (reduced from 5 — kmeans init gives stable seeds)")
    print(f"  tol: 1e-3 (convergence tolerance)")
    print(f"  init_params: 'kmeans' (initialize with k-means)")
    print(f"{'='*60}")
    print(f"\n✓ Running GMM clustering...")

    gmm = GaussianMixture(
        n_components=actual_n_clusters,
        covariance_type='diag',
        max_iter=200,
        n_init=3,
        tol=1e-3,
        init_params='kmeans',
        random_state=42
    )

    cluster_labels = gmm.fit_predict(coord_values)
    cluster_centers = gmm.means_

    print(f"\n  GMM Results:")
    print(f"    Converged: {gmm.converged_}")
    print(f"    Iterations: {gmm.n_iter_}")
    print(f"    Log-likelihood: {gmm.lower_bound_:.4f}")

    weights = gmm.weights_
    print(f"    Component weights: min={weights.min():.4f}, max={weights.max():.4f}")

    aic = gmm.aic(coord_values)
    bic = gmm.bic(coord_values)
    print(f"    AIC: {aic:.2f}")
    print(f"    BIC: {bic:.2f}")

    n_components_actual = len(cluster_centers)
    sizes = np.bincount(cluster_labels, minlength=n_components_actual)

    spreads = np.sqrt(gmm.covariances_.sum(axis=1))

    cluster_info = {
        i: {
            'center':     (float(cluster_centers[i, 0]), float(cluster_centers[i, 1])),
            'size':       int(sizes[i]),
            'method':     'gmm',
            'weight':     float(weights[i]),
            'covariance': gmm.covariances_[i].tolist(),
            'spread':     float(spreads[i])
        }
        for i in range(n_components_actual)
    }

    if not gmm.converged_:
        print("\n" + "=" * 60)
        print("⚠️  GMM WARNING: Did not converge")
        print("=" * 60)
        print("  The EM algorithm did not converge within max_iter iterations.")
        print("  Results may be suboptimal.")
        print("\n  Suggested fixes:")
        print("    - Increase max_iter (current: 200)")
        print("    - Decrease n_clusters")
        print("    - Try different covariance_type")
        print("=" * 60)

    print(f"\n{'='*60}")
    print(f"GMM CLUSTERING SUMMARY")
    print(f"{'='*60}")
    print(f"  Algorithm: Gaussian Mixture Model (EM)")
    print(f"  Parameters:")
    print(f"    - n_components: {actual_n_clusters}")
    print(f"    - covariance_type: diag")
    print(f"    - n_init: 3")
    print(f"  Input points: {n_points:,}")
    print(f"  Output clusters: {len(cluster_centers)}")
    print(f"  Converged: {gmm.converged_}")
    print(f"  Final log-likelihood: {gmm.lower_bound_:.4f}")

    print(f"\n  Cluster size statistics:")
    print(f"    Min size: {sizes.min()} points")
    print(f"    Max size: {sizes.max()} points")
    print(f"    Mean size: {sizes.mean():.1f} points")
    print(f"    Median size: {np.median(sizes):.1f} points")
    print(f"    Total points clustered: {sizes.sum():,}")

    print(f"\n  Component weight statistics:")
    print(f"    Min weight: {weights.min():.4f}")
    print(f"    Max weight: {weights.max():.4f}")
    print(f"    Mean weight: {weights.mean():.4f}")

    print(f"\n  Cluster spread statistics (sqrt of summed variances):")
    print(f"    Min spread: {spreads.min():.6f}")
    print(f"    Max spread: {spreads.max():.6f}")
    print(f"    Mean spread: {spreads.mean():.6f}")

    print(f"\n  First 10 cluster centers:")
    print(f"  {'-'*50}")
    for i in range(min(10, len(cluster_centers))):
        lat, lon = cluster_centers[i]
        size = cluster_info[i]['size']
        weight = cluster_info[i]['weight']
        spread = cluster_info[i]['spread']
        print(f"    Cluster {i:3d}: Lat={lat:.6f}, Lon={lon:.6f}, "
              f"Size={size}, Weight={weight:.4f}, Spread={spread:.6f}")

    return cluster_centers, cluster_info



def event_driven_clustering_fixed(df, known_stops=None, n_clusters=35):
    print_section("EVENT-DRIVEN CLUSTERING  (GMM adapter)")

    if known_stops:
        print(f"   Note: known_stops ({len(known_stops)} entries) passed but not "
              f"used by GMM — clusters are data-driven via EM.")

    cluster_centers, cluster_info = simple_clustering(df, n_clusters=n_clusters)
    station_cluster_ids = set()

    print(f"   GMM clusters produced : {len(cluster_centers)}")
    print(f"   Station cluster IDs   : {station_cluster_ids} (none labelled by GMM)")

    return cluster_centers, station_cluster_ids
