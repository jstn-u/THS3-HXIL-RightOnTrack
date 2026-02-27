"""
cluster_kmeans.py
=================
K-Means clustering with k-means++ initialisation.

Public API (identical across all cluster_*.py modules):
    event_driven_clustering_fixed(df, known_stops=None, n_clusters=50)
        -> (cluster_centers: np.ndarray shape (N,2),
            station_cluster_ids: set of int)

To switch clustering method in main.py, change only:
    from cluster_kmeans import event_driven_clustering_fixed
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree
import warnings

warnings.filterwarnings('ignore')

from config import print_section, haversine_meters

# =============================================================================
# K-MEANS CLUSTERING
# =============================================================================

def simple_clustering(df, n_clusters=50, speed_threshold=2.0):
    """
    Cluster transit stops using K-Means algorithm.

    K-Means partitions data into K clusters by minimizing within-cluster
    variance (inertia). Each cluster is represented by its centroid (mean
    of all points in the cluster).

    K-Means Algorithm Overview:
        K-Means is a centroid-based clustering algorithm that iteratively
        refines cluster assignments to minimize the sum of squared distances
        from points to their assigned cluster centers.

        1. Initialization: Select K initial centers using k-means++
        2. Assignment (E-step): Assign each point to nearest center
        3. Update (M-step): Recalculate centers as mean of assigned points
        4. Repeat steps 2-3 until convergence (centers stabilize)

        Each cluster is defined by:
        - Centroid: Mean position of all points in the cluster
        - Inertia: Sum of squared distances to centroid (measures compactness)

    Algorithm Steps:
        1. Filter low-speed points (potential stops where vehicles pause)
        2. Remove outliers and invalid coordinates
        3. Fit K-Means with n_clusters = K
        4. Extract cluster centers as centroids
        5. Assign each point to nearest centroid (hard assignment)
        6. Calculate cluster metadata and statistics

    Args:
        df (pd.DataFrame): Transit GPS data with required columns:
            - 'latitude': GPS latitude coordinate (float)
            - 'longitude': GPS longitude coordinate (float)
            - 'speed_mps' (optional): Speed in meters per second (float)
        n_clusters (int): Number of clusters (K) to create. Default is 50.
            Unlike DBSCAN, this must be specified upfront.
        speed_threshold (float): Maximum speed (m/s) to consider a point as a
            potential stop. Points with speed >= threshold are excluded.
            Default is 2.0 m/s (~7.2 km/h, typical walking speed).

    Returns:
        tuple: A tuple containing:
            - cluster_centers (np.ndarray): Array of shape (n_clusters, 2) with
              [latitude, longitude] for each cluster centroid.
            - cluster_info (dict): Dictionary mapping cluster index to metadata:
              {
                  'center': (lat, lon),       # Cluster centroid
                  'size': int,                # Number of points assigned
                  'method': 'kmeans',         # Clustering method identifier
                  'radius': float,            # Max distance from center
                  'avg_distance': float       # Mean distance from center
              }

    Raises:
        ValueError: If K-Means clustering fails to find valid clusters.
            This ensures no silent fallback to different algorithms.

    Example:
        >>> clusters, info = simple_clustering(train_df, n_clusters=50)
        >>> print(f"Found {len(clusters)} clusters")
        Found 50 clusters
        >>> print(f"Cluster 0: center={info[0]['center']}, size={info[0]['size']}")
        Cluster 0: center=(40.7128, -74.0060), size=156

    Notes:
        - Uses k-means++ initialization for better convergence
        - Multiple initializations (n_init=3) — sufficient with k-means++ seeding
        - Assumes spherical clusters (equal variance in all directions)
        - Hard assignments: each point belongs to exactly one cluster

    Parameter Tuning Guide:
        n_clusters (K):
            - Too few: Clusters too large, lose stop granularity
            - Too many: Overfitting, some clusters have few points
            - Typical: 30-100 for urban transit networks

        n_init:
            - Higher values = better chance of finding global optimum
            - Trade-off with computation time
            - Recommended: 10-20

        max_iter:
            - Usually converges before 300 iterations
            - Increase if convergence warnings appear

    References:
        - Lloyd, S. (1982). Least squares quantization in PCM. IEEE Transactions
          on Information Theory, 28(2), 129-137.
        - Arthur, D. & Vassilvitskii, S. (2007). k-means++: The advantages of
          careful seeding. SODA '07 Proceedings, pp. 1027-1035.
    """
    print_section("STEP 2: CLUSTERING STOPS/STATIONS (K-MEANS)")

    # =========================================================================
    # STEP 1: Input Validation
    # =========================================================================
    if df.empty:
        print("✗ Empty DataFrame provided")
        return np.empty((0, 2)), {}

    required_cols = ['latitude', 'longitude']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"✗ Missing required columns: {missing_cols}")
        return np.empty((0, 2)), {}

    # =========================================================================
    # STEP 2: Filter Low-Speed Points (Potential Stops)
    # =========================================================================
    # Rationale: Low-speed points indicate where vehicles stop or slow down,
    # which typically corresponds to stations, stops, or traffic signals.
    # Default threshold of 2.0 m/s (~7.2 km/h) captures most stop events.
    if 'speed_mps' in df.columns:
        stops = df[df['speed_mps'] < speed_threshold]
        print(f"✓ Found {len(stops):,} low-speed points (< {speed_threshold} m/s)")
    else:
        stops = df
        print(f"⚠️  No 'speed_mps' column - using all {len(stops):,} points")

    # =========================================================================
    # STEP 3: Extract and Validate Coordinates
    # =========================================================================
    coords = stops[['latitude', 'longitude']].dropna()

    if coords.empty:
        print("✗ No valid coordinates found")
        return np.empty((0, 2)), {}

    # Filter invalid coordinates (0,0 is "null island" - indicates invalid GPS)
    coords = coords[(coords['latitude'] != 0) & (coords['longitude'] != 0)]

    # =========================================================================
    # STEP 4: Remove Statistical Outliers
    # =========================================================================
    # Rationale: GPS errors can create points far from the actual route.
    # Using 3 standard deviations captures 99.7% of valid data points.
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

    # =========================================================================
    # STEP 5: Prepare Data and Adjust Parameters
    # =========================================================================
    coord_values = coords[['latitude', 'longitude']].values
    n_points = len(coord_values)

    # Adjust n_clusters if necessary (can't have more clusters than points/2)
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

    # =========================================================================
    # STEP 6: Fit K-Means Model
    # =========================================================================
    print(f"\n✓ Running K-Means clustering...")

    kmeans = KMeans(
        n_clusters=actual_n_clusters,
        init='k-means++',       # Smart initialization for better convergence
        max_iter=300,           # Maximum iterations per run
        n_init=3,               # 3 restarts — sufficient with k-means++ seeding
        algorithm='lloyd',      # Classic K-Means algorithm
        random_state=42         # Reproducibility
    )

    # Fit the model and get cluster assignments
    cluster_labels = kmeans.fit_predict(coord_values)

    # Extract cluster centers (centroids)
    cluster_centers = kmeans.cluster_centers_

    # =========================================================================
    # STEP 7: Analyze K-Means Results
    # =========================================================================
    print(f"\n  K-Means Results:")
    print(f"    Converged: {kmeans.n_iter_ < 300}")
    print(f"    Iterations: {kmeans.n_iter_}")
    print(f"    Inertia (within-cluster variance): {kmeans.inertia_:.4f}")

    # =========================================================================
    # STEP 8: Build Cluster Information Dictionary (VECTORIZED)
    # =========================================================================
    # --- Vectorized distances: each point vs its own assigned centroid ---
    all_distances = np.sqrt(
        (coord_values[:, 0] - cluster_centers[cluster_labels, 0]) ** 2 +
        (coord_values[:, 1] - cluster_centers[cluster_labels, 1]) ** 2
    )

    # --- Aggregate counts and mean distances with bincount ---
    n_clusters_actual = len(cluster_centers)
    sizes     = np.bincount(cluster_labels, minlength=n_clusters_actual)
    dist_sums = np.bincount(cluster_labels, weights=all_distances, minlength=n_clusters_actual)
    avg_dists = np.where(sizes > 0, dist_sums / sizes, 0.0)

    # --- Radius (max distance per cluster) ---
    sort_idx = np.argsort(cluster_labels, kind='stable')
    sorted_dists = all_distances[sort_idx]
    boundaries = np.searchsorted(cluster_labels[sort_idx], np.arange(n_clusters_actual))
    radii = np.maximum.reduceat(sorted_dists, boundaries)
    radii = np.where(sizes > 0, radii, 0.0)

    # --- Build cluster_info dict ---
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

    # =========================================================================
    # STEP 9: Validate Clustering Results
    # =========================================================================
    if len(cluster_centers) == 0:
        print("\n" + "=" * 60)
        print("✗ K-MEANS CLUSTERING FAILED")
        print("=" * 60)
        print("  No valid clusters were found by the K-Means algorithm.")
        print("\n  Possible causes:")
        print("    1. n_clusters is too high for the data size")
        print("    2. Data has insufficient variance")
        print("    3. All points are identical or nearly identical")
        print("\n  Suggested fixes:")
        print(f"    - Decrease n_clusters (current: {actual_n_clusters})")
        print("    - Increase sample_fraction to include more data")
        print("    - Check data quality")
        print("=" * 60)
        raise ValueError("K-Means clustering failed: No valid clusters found. Cannot proceed.")

    # =========================================================================
    # STEP 10: Print Summary Statistics
    # =========================================================================
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



def event_driven_clustering_fixed(df, known_stops=None, n_clusters=50):
    """
    Adapter: replaces HDBSCAN-based clustering with K-Means-based clustering
    via `simple_clustering`.

    Returns the same two-value tuple expected by the rest of the pipeline:
        cluster_centers     : np.ndarray  shape (N, 2)  — [lat, lon] per cluster
        station_cluster_ids : set of int  — always empty; K-Means does not
                              distinguish station vs delay clusters.

    The `known_stops` argument is accepted for API compatibility but is not
    used by K-Means — the algorithm discovers cluster locations by minimising
    within-cluster variance directly from the GPS data.
    """
    print_section("EVENT-DRIVEN CLUSTERING  (K-Means adapter)")

    if known_stops:
        print(f"   Note: known_stops ({len(known_stops)} entries) received "
              f"but not used — K-Means is fully data-driven.")

    cluster_centers, cluster_info = simple_clustering(df, n_clusters=n_clusters)

    # Empty set: K-Means does not pre-label any cluster as a named station.
    # Downstream MAGNN code that checks station_cluster_ids still runs
    # correctly — it simply won't highlight any cluster as a known station.
    station_cluster_ids = set()

    print(f"   K-Means clusters produced : {len(cluster_centers)}")
    print(f"   Station cluster IDs       : {station_cluster_ids} (none labelled by K-Means)")

    return cluster_centers, station_cluster_ids


# =============================================================================
# SEGMENT BUILDING - CORRECTED VERSION
# =============================================================================

