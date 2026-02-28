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
from sklearn.cluster import AgglomerativeClustering
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import connected_components
import warnings

warnings.filterwarnings('ignore')

from config import print_section, haversine_meters

# =============================================================================
# KNN GRAPH CLUSTERING
# =============================================================================

def simple_clustering(df, n_clusters=50, speed_threshold=2.0, known_stops=None, station_exclusion_radius_m=300):
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

    # =========================================================================
    # STEP 4b: Exclude points near known stations
    #
    # Low-speed points concentrate heavily AT stations (trains stop there).
    # If we leave them in, KNN rediscovers the stations instead of finding
    # delay clusters between them.  We remove any point within
    # station_exclusion_radius_m of a known station so the algorithm only
    # sees inter-station delay / congestion locations.
    # =========================================================================
    if known_stops:
        station_coords = [
            (lat, lon)
            for lat, lon in known_stops.values()
            if not (np.isnan(lat) or np.isnan(lon))
        ]
        if station_coords:
            station_arr = np.array(station_coords)   # shape (S, 2)
            coord_arr   = coords[['latitude', 'longitude']].values

            # Vectorised: for each GPS point check distance to every station
            keep_mask = np.ones(len(coord_arr), dtype=bool)
            for s_lat, s_lon in station_arr:
                dists = np.array([
                    haversine_meters(r[0], r[1], s_lat, s_lon)
                    for r in coord_arr
                ])
                keep_mask &= dists > station_exclusion_radius_m

            n_before = len(coords)
            coords = coords[keep_mask].copy()
            n_excluded = n_before - len(coords)
            print(f"  Excluded {n_excluded:,} points within {station_exclusion_radius_m}m "
                  f"of known stations ({len(coords):,} remaining for delay clustering)")

            if len(coords) < 10:
                print("  ⚠️  Too few non-station points — returning empty (only stations will be used)")
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
    # STEP 9: Calculate component centres, filtering micro-components
    #
    # Dynamic min_size prevents hundreds of 3-point micro-clusters forming
    # before the merge stage.  We use n_points / (n_clusters * 3) so that
    # roughly 3x the target number of components survive, giving the
    # agglomerative step enough material to work with while discarding noise.
    # =========================================================================
    # Keep this threshold low — it only filters genuine isolated noise points
    # (1-2 GPS pings with no neighbours).  AgglomerativeClustering in Step 11
    # does all meaningful merging, so we do NOT need to be aggressive here.
    min_component_size = max(3, n_points // (n_clusters * 20))
    print(f"  Dynamic min component size: {min_component_size} points")

    component_centers = []   # [lat, lon] per surviving component
    component_sizes   = []   # point count per surviving component

    for comp_id in range(n_components):
        mask = labels == comp_id
        component_points = coord_values[mask]
        size = len(component_points)
        if size >= min_component_size:
            component_centers.append(component_points.mean(axis=0))
            component_sizes.append(size)

    n_components_kept = len(component_centers)
    print(f"  Components kept (>= {min_component_size} pts): {n_components_kept}")
    print(f"  Components discarded as noise : {n_components - n_components_kept}")

    # =========================================================================
    # STEP 10: Validate
    # =========================================================================
    if n_components_kept == 0:
        print("\n" + "=" * 60)
        print("\u2717 KNN CLUSTERING FAILED")
        print("=" * 60)
        print("  No valid clusters were found by the KNN graph algorithm.")
        print("\n  Possible causes:")
        print("    1. Data is too sparse - points are too far apart")
        print("    2. K value is too small - increase min K threshold")
        print("    3. All components have < min_component_size points")
        print("    4. Data quality issues - check for GPS errors")
        print("\n  Suggested fixes:")
        print("    - Increase sample_fraction to include more data")
        print("    - Lower speed_threshold to include more points")
        print("    - Check data preprocessing steps")
        print("=" * 60)
        raise ValueError("KNN clustering failed: No valid clusters found. Cannot proceed.")

    component_centers = np.array(component_centers)   # shape (m, 2)
    component_sizes   = np.array(component_sizes)

    # =========================================================================
    # STEP 11: Agglomerative clustering on component centroids -> n_clusters
    #
    # WHY THIS IS BETTER THAN THE PREVIOUS GREEDY MERGE
    # --------------------------------------------------
    # The old approach was O(n^3): for each of (m - target) merge steps it
    # scanned all pairwise centroid distances from scratch.  With m=500 and
    # target=50 that is ~28 million haversine calls.
    #
    # AgglomerativeClustering uses Ward linkage on the centroid array and
    # runs in O(m^2 log m).  For m=500 that is ~1 million operations -- 28x
    # faster -- and scipy's implementation is in optimised C.
    #
    # We work on centroids (not raw GPS points) so n here is the number of
    # surviving components (typically 50-300), not the raw point count
    # (which can be tens of thousands).  This keeps memory small too.
    #
    # The final cluster centre for each agglomerative group is the
    # size-weighted mean of its component centroids, which is identical to
    # what the greedy merge produced but computed correctly in one pass.
    # =========================================================================
    actual_n = min(n_clusters, n_components_kept)

    if n_components_kept > actual_n:
        print(f"  Agglomerative merge: {n_components_kept} components -> {actual_n} clusters ...")
        agg = AgglomerativeClustering(n_clusters=actual_n, linkage='ward')
        agg_labels = agg.fit_predict(component_centers)
    else:
        # Already at or below target — no merge needed
        agg_labels = np.arange(n_components_kept)
        print(f"  Components ({n_components_kept}) already <= target ({actual_n}), no merge needed.")

    cluster_centers = []
    cluster_info    = {}

    for cid in range(actual_n):
        mask   = agg_labels == cid
        if not mask.any():
            continue
        sizes_in_group  = component_sizes[mask]
        centers_in_group = component_centers[mask]
        total_size = sizes_in_group.sum()
        # Size-weighted centroid
        weighted_center = (centers_in_group * sizes_in_group[:, None]).sum(axis=0) / total_size
        idx = len(cluster_centers)
        cluster_centers.append(weighted_center)
        cluster_info[idx] = {
            'center': (weighted_center[0], weighted_center[1]),
            'size': int(total_size),
            'method': 'knn_graph+agglomerative',
            'k_neighbors': k_neighbors,
            'n_components_merged': int(mask.sum()),
        }

    cluster_centers = np.array(cluster_centers)
    print(f"  After agglomerative merge: {len(cluster_centers)} clusters")

    # =========================================================================
    # STEP 12: Print Summary Statistics
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
        print(f"    Min size:    {min(sizes)} points")
        print(f"    Max size:    {max(sizes)} points")
        print(f"    Mean size:   {np.mean(sizes):.1f} points")
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
    Adapter: replaces HDBSCAN-based clustering with KNN-based clustering.

    Known stops injection
    ---------------------
    Every entry in `known_stops` (dict: station_name -> (lat, lon)) is
    ALWAYS included as a cluster centre, prepended at indices 0...n_stations-1,
    matching the HDBSCAN module's guarantee.  Any KNN cluster whose centroid
    falls within 300 m of a known station is dropped (station takes over).

    Returns
    -------
    cluster_centers     : np.ndarray  shape (N, 2)  -- [lat, lon] per cluster
    station_cluster_ids : set of int  -- indices that correspond to real stations
    """
    print_section("EVENT-DRIVEN CLUSTERING  (KNN adapter)")

    MERGE_RADIUS_M = 300  # same radius as HDBSCAN module

    # ------------------------------------------------------------------
    # 1. Build station array from known_stops
    # ------------------------------------------------------------------
    station_centers = []
    if known_stops:
        for name, (lat, lon) in known_stops.items():
            if not (np.isnan(lat) or np.isnan(lon)):
                station_centers.append((lat, lon, name))
        print(f"   Station centres to inject: {len(station_centers)}")
    else:
        print("   No known_stops provided -- clusters are fully data-driven.")

    # ------------------------------------------------------------------
    # 2. KNN clustering on the full dataframe
    # ------------------------------------------------------------------
    knn_centers, _ = simple_clustering(df, n_clusters=50,
                                             known_stops=known_stops,
                                             station_exclusion_radius_m=MERGE_RADIUS_M)

    # ------------------------------------------------------------------
    # 3. Drop any KNN cluster within MERGE_RADIUS_M of a known station
    # ------------------------------------------------------------------
    filtered_knn = []
    for kc in knn_centers:
        too_close = any(
            haversine_meters(kc[0], kc[1], s_lat, s_lon) <= MERGE_RADIUS_M
            for s_lat, s_lon, _ in station_centers
        )
        if not too_close:
            filtered_knn.append(kc)

    dropped = len(knn_centers) - len(filtered_knn)
    if dropped:
        print(f"   Dropped {dropped} KNN cluster(s) absorbed by station zones")

    # ------------------------------------------------------------------
    # 4. Assemble: stations first (indices 0...n-1), then KNN delay clusters
    # ------------------------------------------------------------------
    final_centers = []
    station_cluster_ids = set()

    for i, (s_lat, s_lon, _) in enumerate(station_centers):
        final_centers.append([s_lat, s_lon])
        station_cluster_ids.add(i)

    for kc in filtered_knn:
        final_centers.append(list(kc))

    cluster_centers = np.array(final_centers) if final_centers else np.array([])

    print(f"   Station clusters  : {len(station_cluster_ids)}")
    print(f"   KNN delay clusters: {len(filtered_knn)}")
    print(f"   Total clusters    : {len(cluster_centers)}")

    return cluster_centers, station_cluster_ids


# =============================================================================
# SEGMENT BUILDING - CORRECTED VERSION
# =============================================================================