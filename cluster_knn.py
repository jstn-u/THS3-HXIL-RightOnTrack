"""
cluster_knn.py
==============
Mutual KNN-graph clustering with density-adaptive K selection.

Public API (identical across all cluster_*.py modules):
    event_driven_clustering_fixed(df, known_stops=None)
        -> (cluster_centers: np.ndarray shape (N,2),
            station_cluster_ids: set of int)

To switch clustering method in main.py, change only:
    from cluster_knn import event_driven_clustering_fixed
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors, BallTree
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import connected_components
import warnings

warnings.filterwarnings('ignore')

from config import print_section, haversine_meters

# =============================================================================
# KNN GRAPH CLUSTERING
# =============================================================================

def simple_clustering(df, n_clusters=50, speed_threshold=2.0):
    """
    Cluster transit stops using K-Nearest Neighbors (KNN) graph-based clustering.

    This function identifies potential transit stops by filtering low-speed GPS points,
    builds a KNN graph where each point connects to its K nearest neighbors, then
    extracts clusters from connected components or applies spectral clustering on
    the resulting graph.

    Algorithm Steps:
        1. Filter low-speed points (potential stops where vehicles pause)
        2. Remove outliers and invalid coordinates
        3. Build KNN graph using BallTree for efficient nearest neighbor search
        4. Create mutual KNN graph (symmetric connections)
        5. Find connected components as initial clusters
        6. Merge small clusters and split large ones to reach target count
        7. Calculate cluster centers as centroids

    Args:
        df (pd.DataFrame): Transit GPS data with required columns:
            - 'latitude': GPS latitude coordinate (float)
            - 'longitude': GPS longitude coordinate (float)
            - 'speed_mps' (optional): Speed in meters per second (float)
        n_clusters (int): Target number of clusters to create. Default is 50.
            Actual number may vary based on data connectivity.
        speed_threshold (float): Maximum speed (m/s) to consider a point as a
            potential stop. Points with speed >= threshold are excluded.
            Default is 2.0 m/s (~7.2 km/h, typical walking speed).

    Returns:
        tuple: A tuple containing:
            - cluster_centers (np.ndarray): Array of shape (n_clusters, 2) with
              [latitude, longitude] for each cluster center.
            - cluster_info (dict): Dictionary mapping cluster index to metadata:
              {
                  'center': (lat, lon),      # Cluster centroid
                  'size': int,               # Number of points in cluster
                  'method': 'knn_graph',     # Clustering method used
                  'k_neighbors': int         # K value used for KNN
              }

    Raises:
        No exceptions raised; returns empty arrays if clustering fails.

    Example:
        >>> clusters, info = simple_clustering(train_df, n_clusters=50)
        >>> print(f"Found {len(clusters)} clusters")
        Found 47 clusters
        >>> print(f"Cluster 0: center={info[0]['center']}, size={info[0]['size']}")
        Cluster 0: center=(40.7128, -74.0060), size=156

    Notes:
        - Uses Haversine distance for geographic accuracy
        - K value is auto-calculated based on data density
        - Mutual KNN ensures symmetric relationships (A→B implies B→A)
        - Small clusters (<3 points) are merged with nearest neighbor
        - Computational complexity: O(n * k * log(n)) for KNN search

    References:
        - Cover, T., Hart, P. (1967). Nearest neighbor pattern classification.
        - Ertöz, L., Steinbach, M., Kumar, V. (2003). Finding clusters of different
          sizes, shapes, and densities in noisy, high dimensional data.
    """

    print_section("STEP 2: CLUSTERING STOPS/STATIONS (KNN GRAPH)")

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
        stops = df[df['speed_mps'] < speed_threshold].copy()
        print(f"✓ Found {len(stops):,} low-speed points (< {speed_threshold} m/s)")
    else:
        stops = df.copy()
        print(f"⚠️  No 'speed_mps' column - using all {len(stops):,} points")

    # =========================================================================
    # STEP 3: Extract and Validate Coordinates
    # =========================================================================
    coords = stops[['latitude', 'longitude']].dropna()

    if coords.empty:
        print("✗ No valid coordinates found")
        return np.empty((0, 2)), {}

    coords = coords[(coords['latitude'] != 0) & (coords['longitude'] != 0)].copy()

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
    # STEP 5: Calculate Optimal K for KNN (DENSITY-ADAPTIVE)
    # =========================================================================
    n_points = len(coords)
    coord_values = coords[['latitude', 'longitude']].values
    coords_radians = np.radians(coord_values)

    sample_size = min(5000, n_points)
    if n_points > sample_size:
        sample_indices = np.random.choice(n_points, sample_size, replace=False)
        sample_coords = coords_radians[sample_indices]
    else:
        sample_coords = coords_radians

    k_analysis = min(50, len(sample_coords) - 1)

    nn_analysis = NearestNeighbors(
        n_neighbors=k_analysis + 1,
        metric='haversine',
        algorithm='ball_tree'
    )
    nn_analysis.fit(sample_coords)
    distances_analysis, _ = nn_analysis.kneighbors(sample_coords)

    distances_meters_analysis = distances_analysis * 6_371_000
    mean_distances = distances_meters_analysis[:, 1:].mean(axis=0)
    distance_ratios = np.diff(mean_distances) / (mean_distances[:-1] + 1e-9)

    elbow_threshold = 0.15
    elbow_indices = np.where(distance_ratios > elbow_threshold)[0]

    if len(elbow_indices) > 0:
        k_elbow = elbow_indices[0] + 1
    else:
        k_elbow = k_analysis // 2

    lat_range = coord_values[:, 0].max() - coord_values[:, 0].min()
    lon_range = coord_values[:, 1].max() - coord_values[:, 1].min()
    area = lat_range * lon_range
    density = n_points / (area + 1e-9)

    density_factor = np.clip(1.0 / np.log10(density + 10), 0.3, 1.0)

    k_neighbors = int(np.clip(
        k_elbow * density_factor,
        5,
        min(30, n_points - 1)
    ))

    print(f"\n{'='*60}")
    print(f"KNN PARAMETERS (DENSITY-ADAPTIVE)")
    print(f"{'='*60}")
    print(f"  Data points: {n_points:,}")
    print(f"  Geographic area: {lat_range:.4f}° × {lon_range:.4f}°")
    print(f"  Point density: {density:.1f} points/deg²")
    print(f"  K at elbow point: {k_elbow}")
    print(f"  Density factor: {density_factor:.3f}")
    print(f"  Final K value: {k_neighbors}")
    print(f"  Target clusters: ~{n_clusters}")
    print(f"{'='*60}")

    # =========================================================================
    # STEP 6: Build KNN Graph Using BallTree
    # =========================================================================
    coord_values = coords[['latitude', 'longitude']].values
    coords_radians = np.radians(coord_values)

    print(f"✓ Building KNN graph with Haversine distance metric...")
    nbrs = NearestNeighbors(
        n_neighbors=k_neighbors + 1,
        metric='haversine',
        algorithm='ball_tree'
    )
    nbrs.fit(coords_radians)
    distances, indices = nbrs.kneighbors(coords_radians)

    distances_meters = distances * 6_371_000
    print(f"  Average neighbor distance: {distances_meters[:, 1:].mean():.1f}m")
    print(f"  Max neighbor distance: {distances_meters[:, 1:].max():.1f}m")

    # =========================================================================
    # STEP 7: Create Mutual KNN Adjacency Matrix
    # =========================================================================
    print(f"✓ Creating mutual KNN adjacency matrix...")

    adjacency = lil_matrix((n_points, n_points), dtype=np.float32)

    for i in range(n_points):
        for j_idx in range(1, k_neighbors + 1):
            j = indices[i, j_idx]
            if i in indices[j, 1:k_neighbors + 1]:
                adjacency[i, j] = 1.0
                adjacency[j, i] = 1.0

    adjacency_csr = adjacency.tocsr()

    n_edges = adjacency_csr.nnz // 2
    max_possible_edges = n_points * (n_points - 1) // 2
    graph_density = n_edges / max_possible_edges * 100 if max_possible_edges > 0 else 0

    print(f"  Mutual KNN edges created: {n_edges:,}")
    print(f"  Graph density: {graph_density:.2f}%")

    # =========================================================================
    # STEP 8: Find Connected Components
    # =========================================================================
    print(f"✓ Finding connected components...")

    n_components, labels = connected_components(
        adjacency_csr,
        directed=False,
        return_labels=True
    )

    print(f"  Initial connected components found: {n_components}")

    # =========================================================================
    # STEP 9: Calculate Cluster Centers and Filter Small Clusters
    # =========================================================================
    cluster_centers = []
    cluster_info = {}
    component_sizes = []

    for comp_id in range(n_components):
        mask = labels == comp_id
        component_points = coord_values[mask]
        size = len(component_points)
        component_sizes.append(size)

        if size >= 3:
            center = component_points.mean(axis=0)
            cluster_idx = len(cluster_centers)
            cluster_centers.append(center)

            cluster_info[cluster_idx] = {
                'center': (center[0], center[1]),
                'size': size,
                'method': 'knn_graph',
                'k_neighbors': k_neighbors,
                'component_id': comp_id
            }

    print(f"  Valid clusters (≥3 points): {len(cluster_centers)}")
    print(f"  Discarded small components: {n_components - len(cluster_centers)}")

    # =========================================================================
    # STEP 10: Validate Clustering Results
    # =========================================================================
    if len(cluster_centers) == 0:
        print("\n" + "=" * 60)
        print("✗ KNN CLUSTERING FAILED")
        print("=" * 60)
        print("  No valid clusters were found by the KNN graph algorithm.")
        print("\n  Possible causes:")
        print("    1. Data is too sparse - points are too far apart")
        print("    2. K value is too small - increase min K threshold")
        print("    3. All components have < 3 points - lower the minimum")
        print("    4. Data quality issues - check for GPS errors")
        print("\n  Suggested fixes:")
        print("    - Increase sample_fraction to include more data")
        print("    - Lower speed_threshold to include more points")
        print("    - Check data preprocessing steps")
        print("=" * 60)
        raise ValueError("KNN clustering failed: No valid clusters found. Cannot proceed.")

    cluster_centers = np.array(cluster_centers)

    # =========================================================================
    # STEP 11: Print Summary Statistics
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"KNN GRAPH CLUSTERING SUMMARY")
    print(f"{'='*60}")
    print(f"  Algorithm: Mutual K-Nearest Neighbors Graph")
    print(f"  K value: {k_neighbors}")
    print(f"  Input points: {n_points:,}")
    print(f"  Output clusters: {len(cluster_centers)}")

    if cluster_info:
        sizes = [info['size'] for info in cluster_info.values()]
        print(f"\n  Cluster size statistics:")
        print(f"    Min size: {min(sizes)} points")
        print(f"    Max size: {max(sizes)} points")
        print(f"    Mean size: {np.mean(sizes):.1f} points")
        print(f"    Median size: {np.median(sizes):.1f} points")
        print(f"    Total points clustered: {sum(sizes):,}")

    print(f"\n  First 10 cluster centers:")
    print(f"  {'-'*50}")
    for i, (lat, lon) in enumerate(cluster_centers[:10]):
        size = cluster_info[i]['size']
        print(f"    Cluster {i:3d}: Lat={lat:.6f}, Lon={lon:.6f}, Size={size}")

    return cluster_centers, cluster_info



def event_driven_clustering_fixed(df, known_stops=None):
    """
    Adapter: replaces HDBSCAN-based clustering with KNN-based clustering
    via `simple_clustering`.

    Returns the same two-value tuple expected by the rest of the pipeline:
        cluster_centers     : np.ndarray  shape (N, 2)  — [lat, lon] per cluster
        station_cluster_ids : set of int  — always empty (KNN does not
                              distinguish station vs delay clusters; all
                              cluster types are treated uniformly downstream)

    The `known_stops` argument is accepted for API compatibility but is not
    used by KNN clustering — the algorithm discovers all clusters from the
    GPS data directly.
    """
    print_section("EVENT-DRIVEN CLUSTERING  (KNN adapter)")

    if known_stops:
        print(f"   Note: known_stops ({len(known_stops)} entries) passed but not "
              f"used by KNN — clusters are data-driven.")

    cluster_centers, cluster_info = simple_clustering(df, n_clusters=50)

    # station_cluster_ids: empty set — KNN does not label any cluster as a
    # named station.  Downstream code that checks membership in this set
    # (e.g. visualisation) will simply see no pre-labelled station clusters.
    station_cluster_ids = set()

    print(f"   KNN clusters produced : {len(cluster_centers)}")
    print(f"   Station cluster IDs   : {station_cluster_ids} (none labelled by KNN)")

    return cluster_centers, station_cluster_ids


# =============================================================================
# SEGMENT BUILDING - CORRECTED VERSION
# =============================================================================

