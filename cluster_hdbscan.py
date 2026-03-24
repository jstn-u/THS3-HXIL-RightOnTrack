"""
cluster_hdbscan.py
==================
HDBSCAN-based event-driven clustering.

Public API (identical across all cluster_*.py modules):
    event_driven_clustering_fixed(df, known_stops=None)
        -> (cluster_centers: np.ndarray shape (N,2),
            station_cluster_ids: set of int)

To switch clustering method in main.py, change only:
    from cluster_hdbscan import event_driven_clustering_fixed
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from sklearn.cluster import AgglomerativeClustering
import warnings

warnings.filterwarnings('ignore')

try:
    import hdbscan
    HAS_HDBSCAN = True
    print("✓ HDBSCAN available")
except ImportError:
    HAS_HDBSCAN = False
    print("⚠️  WARNING: hdbscan not available")

from config import print_section, haversine_meters


def event_driven_clustering_fixed(df, known_stops=None, merge_radius_m=450):
    """
    Event-driven clustering on dwell/slowdown points via HDBSCAN.

    Guaranteed station injection
    ─────────────────────────────
    Every entry in `known_stops` (dict: station_name → (lat, lon)) is
    ALWAYS included as a cluster centre, regardless of whether HDBSCAN
    finds enough points nearby.  This ensures all 13 real stations are
    present in the cluster list even when the sample fraction is small.

    After HDBSCAN, the pipeline runs in this order:
      1. HDBSCAN finds delay hotspot candidates from event points.
      2. Drop any candidate within merge_radius_m of a known station.
         Every ABSORBED/KEPT decision is printed with the distance to the
         nearest station so the caller can tune the radius.
      3. Merge remaining candidates using complete-linkage agglomerative
         clustering (all pairs in a group must be within 50m — prevents
         chain merging that produces centroids far from any real hotspot).
      4. Enforce minimum spacing (300m) between all surviving delay clusters
         and between delay clusters and stations.  Every SPACING DROP/KEEP
         decision is printed with the distance to the nearest kept coord.
      5. Assemble final array: stations first (indices 0…n_stations-1),
         delay clusters follow at higher indices.

    Parameters
    ──────────
    df            : GPS event DataFrame
    known_stops   : dict  station_name → (lat, lon)
    merge_radius_m: int   absorption radius in metres (default 300).
                    300 = aggressive (matches cluster-assign radius).
                    200 = moderate.
                    150 = tight (only immediate platform area).

    Returns
    ───────
    cluster_centers     : np.ndarray  shape (N, 2)  — [lat, lon] per cluster
    station_cluster_ids : set of int — indices that are real stations
    """
    print_section("EVENT-DRIVEN CLUSTERING")

    if not HAS_HDBSCAN:
        print("❌ HDBSCAN not available")
        return np.array([]), set()

    station_centers = []
    if known_stops:
        for name, (lat, lon) in known_stops.items():
            if not (np.isnan(lat) or np.isnan(lon)):
                station_centers.append((lat, lon, name))

    print(f"   Station centres to inject: {len(station_centers)}")

    event_mask = (
        (df['is_long_dwell'] == 1) |
        (df['is_slow_speed'] == 1) |
        (df['is_slowdown'] == 1)
    ) & (df['is_gps_valid'] == 1)

    event_points = df[event_mask].copy()
    print(f"   Event points: {len(event_points):,} "
          f"({len(event_points) / max(len(df), 1) * 100:.1f}%)")

    delay_centers = []

    if len(event_points) >= 50:
        use_slowdown = (event_points['slowdown_lat'].notna() &
                        event_points['slowdown_lon'].notna()).sum()

        if use_slowdown > len(event_points) * 0.5:
            coords = event_points[['slowdown_lat', 'slowdown_lon']].fillna(
                event_points[['latitude', 'longitude']]
            ).values
        else:
            coords = event_points[['latitude', 'longitude']].values

        coords_rad = np.radians(coords)

        clusterer = hdbscan.HDBSCAN(
            metric='haversine',
            min_cluster_size=30,
            min_samples=5
        )
        labels = clusterer.fit_predict(coords_rad)

        n_raw = len(set(labels) - {-1})
        print(f"   HDBSCAN clusters found: {n_raw}")

        for label in range(n_raw):
            mask = labels == label
            if mask.sum() > 0:
                delay_centers.append([coords[mask, 0].mean(),
                                       coords[mask, 1].mean()])
    else:
        print("   ⚠️  Too few event points for HDBSCAN — only station centres used")

    MERGE_RADIUS_M = merge_radius_m

    print(f"   Station absorption radius : {MERGE_RADIUS_M}m")
    pre_filtered = []
    for dc in delay_centers:
        too_close = False
        closest_station = None
        closest_dist = float('inf')
        for (s_lat, s_lon, s_name) in station_centers:
            dist = haversine_meters(dc[0], dc[1], s_lat, s_lon)
            if dist < closest_dist:
                closest_dist = dist
                closest_station = s_name
            if dist <= MERGE_RADIUS_M:
                too_close = True

        if too_close:
            print(f"     ABSORBED: delay cluster ({dc[0]:.6f}, {dc[1]:.6f}) "
                  f"→ nearest station '{closest_station}' ({closest_dist:.0f}m)")
        else:
            print(f"     KEPT:     delay cluster ({dc[0]:.6f}, {dc[1]:.6f}) "
                  f"→ nearest station '{closest_station}' ({closest_dist:.0f}m)")
            pre_filtered.append(dc)

    dropped_by_station = len(delay_centers) - len(pre_filtered)
    if dropped_by_station:
        print(f"   Dropped {dropped_by_station} delay cluster(s) absorbed by "
              f"station zones (radius={MERGE_RADIUS_M}m)")

    filtered_delay = pre_filtered

    if len(pre_filtered) > 1:
        delay_arr = np.array(pre_filtered)
        n = len(delay_arr)
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dist_matrix[i, j] = haversine_meters(
                    delay_arr[i, 0], delay_arr[i, 1],
                    delay_arr[j, 0], delay_arr[j, 1]
                )

        agg = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=50,
            metric='precomputed',
            linkage='complete'
        )
        group_labels = agg.fit_predict(dist_matrix)

        merged = []
        for label in np.unique(group_labels):
            members = delay_arr[group_labels == label]
            merged.append([members[:, 0].mean(), members[:, 1].mean()])

        filtered_delay = merged
        print(f"   Complete-linkage merging: {len(pre_filtered)} → "
              f"{len(filtered_delay)} delay clusters (threshold=50m)")
    elif len(pre_filtered) == 1:
        print(f"   Single delay cluster — no merging needed")
    else:
        print(f"   No delay clusters survived station absorption")

    MIN_CLUSTER_SPACING_M = 150

    final_delay = []
    all_kept_coords = [(s_lat, s_lon) for (s_lat, s_lon, _) in station_centers]

    for dc in filtered_delay:
        too_close = False
        closest_dist = float('inf')
        closest_coord = None
        for kept_lat, kept_lon in all_kept_coords:
            dist = haversine_meters(dc[0], dc[1], kept_lat, kept_lon)
            if dist < closest_dist:
                closest_dist = dist
                closest_coord = (kept_lat, kept_lon)
            if dist < MIN_CLUSTER_SPACING_M:
                too_close = True

        if too_close:
            print(f"     SPACING DROP: delay cluster ({dc[0]:.6f}, {dc[1]:.6f}) "
                  f"too close to kept coord ({closest_dist:.0f}m < {MIN_CLUSTER_SPACING_M}m)")
        else:
            print(f"     SPACING KEEP: delay cluster ({dc[0]:.6f}, {dc[1]:.6f}) "
                  f"→ nearest kept ({closest_dist:.0f}m) ✓")
            final_delay.append(dc)
            all_kept_coords.append((dc[0], dc[1]))

    dropped_by_spacing = len(filtered_delay) - len(final_delay)
    if dropped_by_spacing:
        print(f"   Dropped {dropped_by_spacing} delay cluster(s) violating "
              f"minimum spacing ({MIN_CLUSTER_SPACING_M}m)")

    filtered_delay = final_delay
    final_centers = []
    station_cluster_ids = set()

    for i, (s_lat, s_lon, _) in enumerate(station_centers):
        final_centers.append([s_lat, s_lon])
        station_cluster_ids.add(i)

    for dc in filtered_delay:
        final_centers.append(dc)

    cluster_centers = np.array(final_centers) if final_centers else np.array([])

    print(f"   Station clusters  : {len(station_cluster_ids)}")
    print(f"   Delay clusters    : {len(filtered_delay)}")
    print(f"   Total clusters    : {len(cluster_centers)}")

    return cluster_centers, station_cluster_ids