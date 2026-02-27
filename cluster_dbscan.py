"""
dbscan.py
=================
DBSCAN-based clustering with adaptive eps via binary search.

Public API (identical across all cluster_*.py modules):
    event_driven_clustering_fixed(df, known_stops=None)
        -> (cluster_centers: np.ndarray shape (N,2),
            station_cluster_ids: set of int)

To switch clustering method in main.py, change only:
    from cluster_dbscan import event_driven_clustering_fixed
"""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import BallTree, NearestNeighbors as _NN
import warnings

warnings.filterwarnings('ignore')

from config import print_section, haversine_meters

# =============================================================================
# DBSCAN CLUSTERING
# =============================================================================

def simple_clustering(df, n_clusters=50, speed_threshold=2.0):
    """
    Cluster transit stops using DBSCAN (Density-Based Spatial Clustering).

    This function identifies potential transit stops by filtering low-speed GPS
    points, then applies DBSCAN clustering to discover natural groupings based
    on spatial density. Unlike K-means or KNN-based methods, DBSCAN does not
    require specifying the number of clusters beforehand.

    DBSCAN Algorithm Overview:
        DBSCAN groups points based on local density. It defines clusters as
        dense regions separated by sparser regions. The algorithm uses two
        key parameters:

        - eps (ε): Maximum radius to search for neighbors. Points within this
          distance are considered neighbors.
        - min_samples: Minimum number of points required to form a dense region
          (core point).

        Point Classification:
        - Core Point: Has ≥ min_samples neighbors within eps radius
        - Border Point: Within eps of a core point, but not core itself
        - Noise Point: Neither core nor border (labeled as -1)

    Algorithm Steps:
        1. Filter low-speed points (potential stops where vehicles pause)
        2. Remove outliers and invalid coordinates
        3. Convert coordinates to radians for Haversine distance
        4. Apply DBSCAN with Haversine metric via BallTree
        5. Extract cluster centers as centroids of each cluster
        6. Filter out noise points (label = -1)
        7. Calculate cluster metadata and statistics

    Args:
        df (pd.DataFrame): Transit GPS data with required columns:
            - 'latitude': GPS latitude coordinate (float)
            - 'longitude': GPS longitude coordinate (float)
            - 'speed_mps' (optional): Speed in meters per second (float)
        n_clusters (int): Target number of clusters (used to auto-tune eps).
            Default is 50. Note: DBSCAN may produce more or fewer clusters
            depending on data density.
        speed_threshold (float): Maximum speed (m/s) to consider a point as a
            potential stop. Points with speed >= threshold are excluded.
            Default is 2.0 m/s (~7.2 km/h, typical walking speed).

    Returns:
        tuple: A tuple containing:
            - cluster_centers (np.ndarray): Array of shape (n_clusters, 2) with
              [latitude, longitude] for each cluster center.
            - cluster_info (dict): Dictionary mapping cluster index to metadata:
              {
                  'center': (lat, lon),       # Cluster centroid
                  'size': int,                # Number of points in cluster
                  'method': 'dbscan',         # Clustering method identifier
                  'eps_meters': float,        # Epsilon in meters used
                  'min_samples': int,         # Min samples parameter used
                  'n_core_points': int,       # Number of core points
                  'n_noise_points': int       # Total noise points (for info[0])
              }

    Raises:
        ValueError: If DBSCAN clustering fails to find any valid clusters.
            This ensures no silent fallback to different algorithms.

    Example:
        >>> clusters, info = simple_clustering(train_df, n_clusters=50)
        >>> print(f"Found {len(clusters)} clusters")
        Found 47 clusters
        >>> print(f"Cluster 0: center={info[0]['center']}, size={info[0]['size']}")
        Cluster 0: center=(40.7128, -74.0060), size=156

    Notes:
        - Uses Haversine distance metric for geographic accuracy
        - eps is auto-calculated based on data spread and target cluster count
        - Noise points are automatically excluded from clusters
        - No fallback algorithm - fails explicitly if DBSCAN cannot cluster
        - Computational complexity: O(n²) worst case, O(n log n) with BallTree

    Parameter Tuning Guide:
        eps (epsilon) - Maximum neighbor distance:
            - Too small: Many noise points, fragmented clusters
            - Too large: Everything merges into one cluster
            - Typical values: 0.0003-0.001 radians (20-110 meters)

        min_samples - Minimum points for core status:
            - Too small: Noise gets included in clusters
            - Too large: Valid clusters marked as noise
            - Typical values: 3-10 for transit data

    References:
        - Ester, M., Kriegel, H.P., Sander, J., Xu, X. (1996). A density-based
          algorithm for discovering clusters in large spatial databases with noise.
          KDD-96 Proceedings, pp. 226-231.
    """
    print_section("STEP 2: CLUSTERING STOPS/STATIONS (DBSCAN)")

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

    coords = coords[(coords['latitude'] != 0) & (coords['longitude'] != 0)]

    # =========================================================================
    # STEP 4: Remove Statistical Outliers
    # =========================================================================
    lat_mean, lat_std = coords['latitude'].mean(), coords['latitude'].std()
    lon_mean, lon_std = coords['longitude'].mean(), coords['longitude'].std()

    if lat_std > 0 and lon_std > 0:
        coords = coords[
            (np.abs(coords['latitude'] - lat_mean) < 3 * lat_std) &
            (np.abs(coords['longitude'] - lon_mean) < 3 * lon_std)
        ]

    if len(coords) < 10:
        print(f"✗ Too few valid coordinates ({len(coords)}) after filtering")
        return np.empty((0, 2)), {}

    print(f"✓ Using {len(coords):,} valid coordinates for clustering")

    # =========================================================================
    # STEP 5: Calculate DBSCAN Parameters (TARGET-BASED ADAPTIVE)
    # =========================================================================
    n_points = len(coords)
    coord_values = coords[['latitude', 'longitude']].values
    coords_radians = np.radians(coord_values)

    min_samples = 3

    EPS_FLOOR   = 10  / 6_371_000   # ~10 m
    EPS_CEILING = 500 / 6_371_000   # ~500 m

    # Sample for faster parameter search on large datasets
    if n_points > 10000:
        sample_size = 10000
        sample_indices = np.random.choice(n_points, sample_size, replace=False)
        sample_coords = coords_radians[sample_indices]
        print(f"\n✓ Using {sample_size:,} point sample for eps search...")
    else:
        sample_coords = coords_radians

    # Seed bounds from median nearest-neighbour distance
    from sklearn.neighbors import NearestNeighbors as _NN

    unique_sample = np.unique(sample_coords, axis=0)
    if len(unique_sample) < 2:
        median_nn_dist = EPS_FLOOR * 2
        print(f"  ⚠️  All sample points identical — using fallback bounds")
    else:
        nn_seed = _NN(n_neighbors=2, metric='haversine', algorithm='ball_tree')
        nn_seed.fit(unique_sample)
        seed_dists, _ = nn_seed.kneighbors(unique_sample)
        median_nn_dist = float(np.median(seed_dists[:, 1]))

    median_nn_meters = median_nn_dist * 6_371_000

    eps_low  = max(median_nn_dist * 0.5, EPS_FLOOR)
    eps_high = max(median_nn_dist * 4.0, EPS_FLOOR * 10)
    eps_high = min(eps_high, EPS_CEILING)

    print(f"  Median nearest-neighbour distance: {median_nn_meters:.1f}m")
    print(f"  Search bounds: [{eps_low*6_371_000:.1f}m, {eps_high*6_371_000:.1f}m]")
    print(f"\n✓ Searching for optimal eps to achieve ~{n_clusters} clusters...")

    # Binary search — capped at 10 iterations
    target_clusters = n_clusters
    best_eps = (eps_low + eps_high) / 2
    best_diff = float('inf')
    best_n_clusters = 0

    for iteration in range(10):
        eps_mid = (eps_low + eps_high) / 2

        dbscan_test = DBSCAN(
            eps=eps_mid,
            min_samples=min_samples,
            metric='haversine',
            algorithm='ball_tree',
            n_jobs=-1
        )
        test_labels = dbscan_test.fit_predict(sample_coords)

        unique_test = np.unique(test_labels)
        n_found = len(unique_test) - (1 if unique_test[0] == -1 else 0)

        diff = abs(n_found - target_clusters)
        if diff < best_diff:
            best_diff = diff
            best_eps  = eps_mid
            best_n_clusters = n_found

        if n_found > target_clusters:
            eps_low = eps_mid
        elif n_found < target_clusters:
            eps_high = eps_mid
        else:
            break

        if diff <= 5:
            break

    eps_radians = best_eps
    eps_meters  = eps_radians * 6_371_000

    lat_range = coord_values[:, 0].max() - coord_values[:, 0].min()
    lon_range = coord_values[:, 1].max() - coord_values[:, 1].min()
    area    = lat_range * lon_range
    density = n_points / (area + 1e-9)

    print(f"\n{'='*60}")
    print(f"DBSCAN PARAMETERS (TARGET-BASED ADAPTIVE)")
    print(f"{'='*60}")
    print(f"  Data points: {n_points:,}")
    print(f"  Geographic area: {lat_range:.4f}° × {lon_range:.4f}°")
    print(f"  Point density: {density:.1f} points/deg²")
    print(f"  Target clusters: {target_clusters}")
    print(f"  Estimated clusters: {best_n_clusters}")
    print(f"  eps (epsilon): {eps_radians:.6f} radians")
    print(f"  eps in meters: ~{eps_meters:.1f}m radius")
    print(f"  min_samples: {min_samples}")
    print(f"{'='*60}")

    # =========================================================================
    # STEP 6: Apply DBSCAN Clustering
    # =========================================================================
    print(f"\n✓ Running DBSCAN clustering...")

    dbscan = DBSCAN(
        eps=eps_radians,
        min_samples=min_samples,
        metric='haversine',
        algorithm='ball_tree',
        n_jobs=-1
    )

    cluster_labels = dbscan.fit_predict(coords_radians)

    # =========================================================================
    # STEP 7: Analyze Clustering Results
    # =========================================================================
    unique_labels = np.unique(cluster_labels)
    n_clusters_found = len(unique_labels) - (1 if unique_labels[0] == -1 else 0)
    n_noise = int(np.sum(cluster_labels == -1))
    n_clustered = len(cluster_labels) - n_noise

    print(f"\n  DBSCAN Results:")
    print(f"    Clusters found: {n_clusters_found}")
    print(f"    Points clustered: {n_clustered:,} ({n_clustered/len(cluster_labels)*100:.1f}%)")
    print(f"    Noise points: {n_noise:,} ({n_noise/len(cluster_labels)*100:.1f}%)")

    core_sample_indices = dbscan.core_sample_indices_
    n_core_points = len(core_sample_indices)
    print(f"    Core points: {n_core_points:,} ({n_core_points/len(cluster_labels)*100:.1f}%)")

    # =========================================================================
    # STEP 8: Extract Cluster Centers (VECTORIZED)
    # =========================================================================
    valid_cluster_ids = unique_labels[unique_labels >= 0]
    core_labels = cluster_labels[core_sample_indices]

    shifted      = cluster_labels + 1
    n_bins       = int(shifted.max()) + 1
    sizes        = np.bincount(shifted, minlength=n_bins)
    lat_sums     = np.bincount(shifted, weights=coord_values[:, 0], minlength=n_bins)
    lon_sums     = np.bincount(shifted, weights=coord_values[:, 1], minlength=n_bins)
    core_counts  = np.bincount(core_labels + 1, minlength=n_bins)

    cluster_centers = []
    cluster_info    = {}

    for cluster_id in valid_cluster_ids:
        bin_idx = cluster_id + 1
        size    = int(sizes[bin_idx])
        center  = np.array([lat_sums[bin_idx] / size,
                             lon_sums[bin_idx] / size])
        n_core  = int(core_counts[bin_idx])

        cluster_idx = len(cluster_centers)
        cluster_centers.append(center)

        cluster_info[cluster_idx] = {
            'center':         (float(center[0]), float(center[1])),
            'size':           size,
            'method':         'dbscan',
            'eps_meters':     eps_meters,
            'min_samples':    min_samples,
            'n_core_points':  n_core,
            'original_label': int(cluster_id)
        }

    if cluster_info:
        cluster_info[0]['n_noise_points_total'] = n_noise

    # =========================================================================
    # STEP 9: Validate Clustering Results
    # =========================================================================
    if len(cluster_centers) == 0:
        print("\n" + "=" * 60)
        print("✗ DBSCAN CLUSTERING FAILED")
        print("=" * 60)
        print("  No valid clusters were found by the DBSCAN algorithm.")
        print("\n  Possible causes:")
        print("    1. eps is too small - points are too far apart for given radius")
        print("    2. min_samples is too high - not enough dense regions")
        print("    3. Data is too sparse - no natural dense groupings exist")
        print("    4. Data quality issues - excessive GPS errors")
        print("\n  Suggested fixes:")
        print(f"    - Increase eps (current: {eps_meters:.1f}m)")
        print(f"    - Decrease min_samples (current: {min_samples})")
        print("    - Increase sample_fraction to include more data")
        print("    - Lower speed_threshold to include more points")
        print("=" * 60)
        raise ValueError("DBSCAN clustering failed: No valid clusters found. Cannot proceed.")

    cluster_centers = np.array(cluster_centers)

    # =========================================================================
    # STEP 10: Print Summary Statistics
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"DBSCAN CLUSTERING SUMMARY")
    print(f"{'='*60}")
    print(f"  Algorithm: DBSCAN (Density-Based Spatial Clustering)")
    print(f"  Parameters:")
    print(f"    - eps: {eps_radians:.6f} rad (~{eps_meters:.1f}m)")
    print(f"    - min_samples: {min_samples}")
    print(f"  Input points: {n_points:,}")
    print(f"  Output clusters: {len(cluster_centers)}")
    print(f"  Noise filtered: {n_noise:,} points")

    if cluster_info:
        sizes_list = [info['size'] for info in cluster_info.values()]
        print(f"\n  Cluster size statistics:")
        print(f"    Min size: {min(sizes_list)} points")
        print(f"    Max size: {max(sizes_list)} points")
        print(f"    Mean size: {np.mean(sizes_list):.1f} points")
        print(f"    Median size: {np.median(sizes_list):.1f} points")
        print(f"    Total points clustered: {sum(sizes_list):,}")

    print(f"\n  First 10 cluster centers:")
    print(f"  {'-'*50}")
    for i, (lat, lon) in enumerate(cluster_centers[:10]):
        size = cluster_info[i]['size']
        core_pts = cluster_info[i]['n_core_points']
        print(f"    Cluster {i:3d}: Lat={lat:.6f}, Lon={lon:.6f}, "
              f"Size={size}, Core={core_pts}")

    return cluster_centers, cluster_info



def event_driven_clustering_fixed(df, known_stops=None):
    """
    Adapter: replaces HDBSCAN-based clustering with DBSCAN-based clustering
    via `simple_clustering`.

    Returns the same two-value tuple expected by the rest of the pipeline:
        cluster_centers     : np.ndarray  shape (N, 2)  — [lat, lon] per cluster
        station_cluster_ids : set of int  — always empty; DBSCAN does not
                              distinguish station vs delay clusters.

    The `known_stops` argument is accepted for API compatibility but is not
    used by DBSCAN — the algorithm discovers cluster locations automatically
    from data density without requiring named anchor points.

    Key difference from HDBSCAN: eps is auto-tuned via binary search to
    target approximately 50 clusters, whereas HDBSCAN required min_cluster_size
    and min_samples. Both use Haversine distance on GPS coordinates.
    """
    print_section("EVENT-DRIVEN CLUSTERING  (DBSCAN adapter)")

    if known_stops:
        print(f"   Note: known_stops ({len(known_stops)} entries) received "
              f"but not used — DBSCAN is fully data-driven.")

    cluster_centers, cluster_info = simple_clustering(df, n_clusters=50)

    # Empty set: DBSCAN does not pre-label any cluster as a named station.
    # Downstream MAGNN code that checks station_cluster_ids still runs
    # correctly — it simply won't highlight any cluster as a known station.
    station_cluster_ids = set()

    print(f"   DBSCAN clusters produced : {len(cluster_centers)}")
    print(f"   Station cluster IDs      : {station_cluster_ids} (none labelled by DBSCAN)")

    return cluster_centers, station_cluster_ids


# =============================================================================
# SEGMENT BUILDING - CORRECTED VERSION
# =============================================================================

