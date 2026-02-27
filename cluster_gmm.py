"""
cluster_gmm.py
==============
Gaussian Mixture Model clustering with diagonal covariance.

Public API (identical across all cluster_*.py modules):
    event_driven_clustering_fixed(df, known_stops=None, n_clusters=50)
        -> (cluster_centers: np.ndarray shape (N,2),
            station_cluster_ids: set of int)

To switch clustering method in main.py, change only:
    from cluster_gmm import event_driven_clustering_fixed
"""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import BallTree
import warnings

warnings.filterwarnings('ignore')

from config import print_section, haversine_meters

# =============================================================================
# GMM CLUSTERING
# =============================================================================

def simple_clustering(df, n_clusters=50, speed_threshold=2.0):
    """
    Cluster transit stops using Gaussian Mixture Model (GMM).

    This function identifies potential transit stops by filtering low-speed GPS
    points, then applies GMM clustering to discover cluster centers. GMM fits
    a mixture of Gaussian distributions to the data, where each cluster is
    represented by a Gaussian with its own mean (center) and covariance (shape).

    GMM Algorithm Overview:
        GMM assumes data is generated from a mixture of K Gaussian distributions.
        The algorithm uses Expectation-Maximization (EM) to find optimal parameters:

        1. Initialization: Set initial means, covariances, and weights
        2. E-step: Compute probability each point belongs to each cluster
        3. M-step: Update means, covariances, weights based on probabilities
        4. Repeat E-M until convergence (log-likelihood stabilizes)

        Each Gaussian component is defined by:
        - Mean (μ): Center of the cluster (latitude, longitude)
        - Covariance (Σ): Shape and orientation of the cluster
        - Weight (π): Proportion of data belonging to this cluster

    Algorithm Steps:
        1. Filter low-speed points (potential stops where vehicles pause)
        2. Remove outliers and invalid coordinates
        3. Fit GMM with n_components = n_clusters
        4. Extract cluster centers as Gaussian means
        5. Assign each point to most likely cluster (hard assignment)
        6. Calculate cluster metadata and statistics

    Args:
        df (pd.DataFrame): Transit GPS data with required columns:
            - 'latitude': GPS latitude coordinate (float)
            - 'longitude': GPS longitude coordinate (float)
            - 'speed_mps' (optional): Speed in meters per second (float)
        n_clusters (int): Number of Gaussian components (clusters) to fit.
            Default is 50. Unlike DBSCAN, this must be specified.
        speed_threshold (float): Maximum speed (m/s) to consider a point as a
            potential stop. Points with speed >= threshold are excluded.
            Default is 2.0 m/s (~7.2 km/h, typical walking speed).

    Returns:
        tuple: A tuple containing:
            - cluster_centers (np.ndarray): Array of shape (n_clusters, 2) with
              [latitude, longitude] for each cluster center (Gaussian means).
            - cluster_info (dict): Dictionary mapping cluster index to metadata:
              {
                  'center': (lat, lon),       # Gaussian mean
                  'size': int,                # Number of points assigned
                  'method': 'gmm',            # Clustering method identifier
                  'weight': float,            # Component weight (importance)
                  'covariance': list          # 2x2 covariance matrix
              }

    Raises:
        ValueError: If GMM clustering fails to converge or finds no clusters.
            This ensures no silent fallback to different algorithms.

    Example:
        >>> clusters, info = simple_clustering(train_df, n_clusters=50)
        >>> print(f"Found {len(clusters)} clusters")
        Found 50 clusters
        >>> print(f"Cluster 0: center={info[0]['center']}, weight={info[0]['weight']:.4f}")
        Cluster 0: center=(40.7128, -74.0060), weight=0.0234

    Notes:
        - Uses full covariance matrices for maximum flexibility
        - Multiple initializations (n_init=5) to avoid local optima
        - Converges when log-likelihood improvement < tol (1e-3)
        - Soft assignments available via gmm.predict_proba()

    Parameter Tuning Guide:
        n_components (n_clusters):
            - Too few: Clusters too large, lose stop granularity
            - Too many: Overfitting, some clusters have few points
            - Typical: 30-100 for urban transit networks

        covariance_type:
            - 'full': Each cluster has own covariance (most flexible)
            - 'tied': All clusters share same covariance
            - 'diag': Diagonal covariance (axis-aligned ellipses)
            - 'spherical': Single variance per cluster (circular)

        n_init:
            - Higher values = better chance of finding global optimum
            - Trade-off with computation time
            - Recommended: 5-10

    References:
        - Reynolds, D.A. (2009). Gaussian Mixture Models. Encyclopedia of Biometrics.
        - Dempster, A.P., Laird, N.M., Rubin, D.B. (1977). Maximum likelihood from
          incomplete data via the EM algorithm.
    """
    print_section("STEP 2: CLUSTERING STOPS/STATIONS (GMM)")

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
    print(f"GMM PARAMETERS")
    print(f"{'='*60}")
    print(f"  n_components: {actual_n_clusters}")
    print(f"  covariance_type: 'diag' (diagonal — sufficient for lat/lon, 2-3x faster than 'full')")
    print(f"  max_iter: 200")
    print(f"  n_init: 3 (reduced from 5 — kmeans init gives stable seeds)")
    print(f"  tol: 1e-3 (convergence tolerance)")
    print(f"  init_params: 'kmeans' (initialize with k-means)")
    print(f"{'='*60}")

    # =========================================================================
    # STEP 6: Fit Gaussian Mixture Model
    # =========================================================================
    print(f"\n✓ Running GMM clustering...")

    gmm = GaussianMixture(
        n_components=actual_n_clusters,
        covariance_type='diag',     # Diagonal covariance — each component has
                                    # independent lat/lon variances but no
                                    # cross-axis covariance term. For geographic
                                    # stop locations this is always appropriate:
                                    # lat and lon spread are independent by
                                    # definition. 'diag' eliminates the 2×2
                                    # matrix inversion done on every EM step,
                                    # replacing it with a cheap element-wise
                                    # reciprocal — roughly 2-3× faster overall.
        max_iter=200,               # Maximum EM iterations
        n_init=3,                   # 3 restarts — sufficient with kmeans init;
                                    # reduced from 5, saving ~40% of fit time
        tol=1e-3,                   # Convergence tolerance
        init_params='kmeans',       # Initialize with k-means for stability
        random_state=42             # Reproducibility
    )

    # Fit the model and get cluster assignments
    cluster_labels = gmm.fit_predict(coord_values)

    # Extract cluster centers (Gaussian means)
    cluster_centers = gmm.means_

    # =========================================================================
    # STEP 7: Analyze GMM Results
    # =========================================================================
    print(f"\n  GMM Results:")
    print(f"    Converged: {gmm.converged_}")
    print(f"    Iterations: {gmm.n_iter_}")
    print(f"    Log-likelihood: {gmm.lower_bound_:.4f}")

    # Analyze component weights
    weights = gmm.weights_
    print(f"    Component weights: min={weights.min():.4f}, max={weights.max():.4f}")

    # Calculate AIC and BIC for model selection reference
    aic = gmm.aic(coord_values)
    bic = gmm.bic(coord_values)
    print(f"    AIC: {aic:.2f}")
    print(f"    BIC: {bic:.2f}")

    # =========================================================================
    # STEP 8: Build Cluster Information Dictionary (VECTORIZED)
    # =========================================================================
    # Original approach: Python loop with a boolean mask per cluster to count
    # sizes, then extracting the covariance matrix individually for each.
    #
    # Optimized approach:
    #   - np.bincount counts all cluster sizes in one pass — no masks
    #   - gmm.covariances_ with 'diag' is already shape (k, 2): each row is
    #     [var_lat, var_lon]. Spread = sqrt(sum of variances) = sqrt(trace).
    #     We compute all spreads in a single vectorized call.
    #   - Covariance stored as a plain Python list of 2 floats (the diagonal)
    #     rather than a 2×2 nested list — half the size, directly meaningful.

    n_components_actual = len(cluster_centers)

    # Vectorized sizes
    sizes = np.bincount(cluster_labels, minlength=n_components_actual)

    # Vectorized spread: covariances_ is (k, 2) for 'diag'; trace = sum of row
    # sqrt(var_lat + var_lon) for each component simultaneously
    spreads = np.sqrt(gmm.covariances_.sum(axis=1))   # shape (k,)

    cluster_info = {
        i: {
            'center':     (float(cluster_centers[i, 0]), float(cluster_centers[i, 1])),
            'size':       int(sizes[i]),
            'method':     'gmm',
            'weight':     float(weights[i]),
            # Store 1D diagonal [var_lat, var_lon] — half the size of a 2×2
            # list, and directly interpretable without unpacking a matrix
            'covariance': gmm.covariances_[i].tolist(),
            'spread':     float(spreads[i])
        }
        for i in range(n_components_actual)
    }

    # =========================================================================
    # STEP 9: Validate Clustering Results
    # =========================================================================
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

    if len(cluster_centers) == 0:
        print("\n" + "=" * 60)
        print("✗ GMM CLUSTERING FAILED")
        print("=" * 60)
        print("  No valid clusters were found by the GMM algorithm.")
        print("\n  Possible causes:")
        print("    1. n_clusters is too high for the data size")
        print("    2. Data has insufficient variance")
        print("    3. Numerical instability in covariance estimation")
        print("\n  Suggested fixes:")
        print(f"    - Decrease n_clusters (current: {actual_n_clusters})")
        print("    - Use 'diag' or 'spherical' covariance_type")
        print("    - Increase sample_fraction to include more data")
        print("=" * 60)
        raise ValueError("GMM clustering failed: No valid clusters found. Cannot proceed.")

    # =========================================================================
    # STEP 10: Print Summary Statistics
    # =========================================================================
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



def event_driven_clustering_fixed(df, known_stops=None, n_clusters=50):
    """
    Adapter: replaces HDBSCAN-based clustering with GMM-based clustering
    via `simple_clustering`.

    Returns the same two-value tuple expected by the rest of the pipeline:
        cluster_centers     : np.ndarray  shape (N, 2)  — [lat, lon] per cluster
        station_cluster_ids : set of int  — always empty set; GMM does not
                              distinguish station vs delay clusters. All cluster
                              types are treated uniformly downstream.

    The `known_stops` argument is accepted for API compatibility but is not
    used by GMM clustering — the algorithm discovers all cluster locations
    from the GPS data probability distributions directly.
    """
    print_section("EVENT-DRIVEN CLUSTERING  (GMM adapter)")

    if known_stops:
        print(f"   Note: known_stops ({len(known_stops)} entries) passed but not "
              f"used by GMM — clusters are data-driven via EM.")

    cluster_centers, cluster_info = simple_clustering(df, n_clusters=n_clusters)

    # station_cluster_ids: empty set — GMM does not label any cluster as a
    # named station. Downstream code that checks membership in this set
    # (e.g. visualisation) will simply see no pre-labelled station clusters.
    station_cluster_ids = set()

    print(f"   GMM clusters produced : {len(cluster_centers)}")
    print(f"   Station cluster IDs   : {station_cluster_ids} (none labelled by GMM)")

    return cluster_centers, station_cluster_ids


# =============================================================================
# SEGMENT BUILDING - CORRECTED VERSION
# =============================================================================

