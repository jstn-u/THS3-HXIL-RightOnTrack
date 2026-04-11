import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree
import warnings

warnings.filterwarnings('ignore')

from config import print_section, haversine_meters

def simple_clustering(df, n_clusters=50, speed_threshold=2.0):
    print_section("STEP 2: CLUSTERING STOPS/STATIONS (K-MEANS)")

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
    print(f"K-MEANS PARAMETERS")
    print(f"{'='*60}")
    print(f"  n_clusters: {actual_n_clusters}")
    print(f"  init: 'k-means++' (smart initialization)")
    print(f"  max_iter: 300")
    print(f"  n_init: 3 (reduced from 10 — k-means++ gives stable seeds)")
    print(f"  algorithm: 'lloyd' (classic k-means)")
    print(f"{'='*60}")
    print(f"\n✓ Running K-Means clustering...")

    kmeans = KMeans(
        n_clusters=actual_n_clusters,
        init='k-means++',
        max_iter=300,
        n_init=3,
        algorithm='lloyd',
        random_state=42
    )

    cluster_labels = kmeans.fit_predict(coord_values)
    cluster_centers = kmeans.cluster_centers_

    print(f"\n  K-Means Results:")
    print(f"    Converged: {kmeans.n_iter_ < 300}")
    print(f"    Iterations: {kmeans.n_iter_}")
    print(f"    Inertia (within-cluster variance): {kmeans.inertia_:.4f}")

    all_distances = np.sqrt(
        (coord_values[:, 0] - cluster_centers[cluster_labels, 0]) ** 2 +
        (coord_values[:, 1] - cluster_centers[cluster_labels, 1]) ** 2
    )

    n_clusters_actual = len(cluster_centers)
    sizes     = np.bincount(cluster_labels, minlength=n_clusters_actual)
    dist_sums = np.bincount(cluster_labels, weights=all_distances, minlength=n_clusters_actual)
    avg_dists = np.where(sizes > 0, dist_sums / sizes, 0.0)

    sort_idx = np.argsort(cluster_labels, kind='stable')
    sorted_dists = all_distances[sort_idx]
    boundaries = np.searchsorted(cluster_labels[sort_idx], np.arange(n_clusters_actual))
    radii = np.maximum.reduceat(sorted_dists, boundaries)
    radii = np.where(sizes > 0, radii, 0.0)

    cluster_info = {
        i: {
            'center': (float(cluster_centers[i, 0]), float(cluster_centers[i, 1])),
            'size': int(sizes[i]),
            'method': 'kmeans',
            'radius': float(radii[i]),
            'avg_distance': float(avg_dists[i])
        }
        for i in range(n_clusters_actual)
    }

    print(f"\n{'='*60}")
    print(f"K-MEANS CLUSTERING SUMMARY")
    print(f"{'='*60}")
    print(f"  Algorithm: K-Means (Lloyd's algorithm)")
    print(f"  Parameters:")
    print(f"    - n_clusters: {actual_n_clusters}")
    print(f"    - init: k-means++")
    print(f"    - n_init: 3")
    print(f"  Input points: {n_points:,}")
    print(f"  Output clusters: {len(cluster_centers)}")
    print(f"  Final inertia: {kmeans.inertia_:.4f}")
    print(f"  Iterations to converge: {kmeans.n_iter_}")

    print(f"\n  Cluster size statistics:")
    print(f"    Min size: {sizes.min()} points")
    print(f"    Max size: {sizes.max()} points")
    print(f"    Mean size: {sizes.mean():.1f} points")
    print(f"    Median size: {np.median(sizes):.1f} points")
    print(f"    Total points clustered: {sizes.sum():,}")

    print(f"\n  Cluster radius statistics:")
    print(f"    Min radius: {radii.min():.6f}")
    print(f"    Max radius: {radii.max():.6f}")
    print(f"    Mean radius: {radii.mean():.6f}")

    print(f"\n  Average distance to centroid:")
    print(f"    Min avg distance: {avg_dists.min():.6f}")
    print(f"    Max avg distance: {avg_dists.max():.6f}")
    print(f"    Mean avg distance: {avg_dists.mean():.6f}")

    print(f"\n  First 10 cluster centers:")
    print(f"  {'-'*50}")
    for i in range(min(10, len(cluster_centers))):
        lat, lon = cluster_centers[i]
        size = cluster_info[i]['size']
        radius = cluster_info[i]['radius']
        print(f"    Cluster {i:3d}: Lat={lat:.6f}, Lon={lon:.6f}, "
              f"Size={size}, Radius={radius:.6f}")

    return cluster_centers, cluster_info

def event_driven_clustering_fixed(df, known_stops=None, n_clusters=40):
    print_section("EVENT-DRIVEN CLUSTERING  (K-Means adapter)")

    if known_stops:
        print(f"   Note: known_stops ({len(known_stops)} entries) received "
              f"but not used — K-Means is fully data-driven.")

    cluster_centers, cluster_info = simple_clustering(df, n_clusters=n_clusters)
    station_cluster_ids = set()

    print(f"   K-Means clusters produced : {len(cluster_centers)}")
    print(f"   Station cluster IDs       : {station_cluster_ids} (none labelled by K-Means)")

    return cluster_centers, station_cluster_ids