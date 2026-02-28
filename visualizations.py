"""
visualizations.py
=================
Cluster and segment visualisation functions — shared across all methods.

Public API:
    plot_clusters(cluster_centers, cluster_info, algorithm_name, save_path)
    plot_segments(segments_df, cluster_centers, max_segments, algorithm_name, save_path)
    plot_segment_statistics(segments_df, algorithm_name, save_path)
"""

import numpy as np
import pandas as pd
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import contextily as ctx

from config import print_section

warnings.filterwarnings('ignore')

# =============================================================================
# VISUALISATION FUNCTIONS
# =============================================================================

def plot_clusters(cluster_centers, cluster_info, algorithm_name='HDBSCAN', save_path='clusters_plot.png'):
    """
    Visualise cluster locations on a geographic map with CartoDB basemap.
    Falls back to a plain lat/lon plot if basemap cannot be loaded.

    Parameters
    ----------
    algorithm_name : str
        Name of the clustering algorithm (e.g. 'HDBSCAN', 'KNN', 'DBSCAN').
        Used in all plot titles, labels, and print messages.
    """
    print_section(f"VISUALISING {algorithm_name.upper()} CLUSTERS")

    if len(cluster_centers) == 0:
        print("✗ No clusters to visualise")
        return

    # Hardcoded Canberra light rail stations for reference
    stations_data = {
        'GGN': ('Gungahlin Place',        -35.185639, 149.135481),
        'MCK': ('Manning Clark Crescent', -35.186986, 149.143372),
        'MPN': ('Mapleton Avenue',         -35.193381, 149.150972),
        'NLR': ('Nullarbor Avenue',        -35.20055,  149.149294),
        'WSN': ('Well Station Drive',      -35.20905,  149.14735),
        'SFD': ('Sandford Street',         -35.221631, 149.144661),
        'EPC': ('EPIC and Racecourse',     -35.2285,   149.14422),
        'PLP': ('Phillip Avenue',          -35.235794, 149.143928),
        'SWN': ('Swinden Street',          -35.24447,  149.13462),
        'DKN': ('Dickson Interchange',     -35.250558, 149.133739),
        'MCR': ('Macarthur Avenue',        -35.260158, 149.132228),
        'IPA': ('Ipima Street',            -35.265897, 149.131283),
        'ELA': ('Elouera Street',          -35.272617, 149.130172),
        'ALG': ('Alinga Street',           -35.277933, 149.129331),
    }

    fig, ax = plt.subplots(figsize=(16, 14))

    lats = cluster_centers[:, 0]
    lons = cluster_centers[:, 1]

    cluster_sizes = np.array([cluster_info.get(i, {}).get('size', 1)
                               for i in range(len(cluster_centers))])
    size_min, size_max = cluster_sizes.min(), cluster_sizes.max()
    if size_max > size_min:
        normalized_density = (cluster_sizes - size_min) / (size_max - size_min)
        gray_colors = 0.3 - (normalized_density * 0.3)
    else:
        gray_colors = np.full(len(cluster_centers), 0.15)

    sizes = np.log1p(cluster_sizes) * 80

    try:
        from pyproj import Transformer
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

        lons_merc, lats_merc = transformer.transform(lons, lats)

        station_lats  = [d[1] for d in stations_data.values()]
        station_lons  = [d[2] for d in stations_data.values()]
        station_names = [d[0] for d in stations_data.values()]
        station_lons_merc, station_lats_merc = transformer.transform(
            station_lons, station_lats)

        all_lats_merc = np.concatenate([lats_merc, station_lats_merc])
        all_lons_merc = np.concatenate([lons_merc, station_lons_merc])
        lat_range = all_lats_merc.max() - all_lats_merc.min()
        lon_range = all_lons_merc.max() - all_lons_merc.min()

        ax.set_xlim(all_lons_merc.min() - lon_range * 0.90,
                    all_lons_merc.max() + lon_range * 0.90)
        ax.set_ylim(all_lats_merc.min() - lat_range * 0.10,
                    all_lats_merc.max() + lat_range * 0.10)
        ax.set_aspect('equal', adjustable='box')

        scatter = ax.scatter(lons_merc, lats_merc,
                             c=gray_colors, cmap='gray', vmin=0, vmax=1,
                             s=sizes, alpha=0.85,
                             edgecolors='black', linewidth=1.5, zorder=5)

        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron,
                        attribution=False, alpha=0.9, zoom='auto')

        ax.scatter(station_lons_merc, station_lats_merc,
                   c='red', s=10, alpha=0.7, linewidth=3, zorder=10,
                   marker='o', label='Light Rail Stations')

        for name, lon_m, lat_m in zip(station_names,
                                      station_lons_merc, station_lats_merc):
            ax.annotate(name, (lon_m, lat_m),
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=10, ha='left', va='bottom',
                        color='darkred', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                                  edgecolor='red', linewidth=2, alpha=0.95),
                        zorder=11)

        print("✓ Geographic basemap added successfully")

    except Exception as e:
        print(f"⚠️  Could not load basemap (using simple plot): {e}")
        ax.clear()

        station_lats  = [d[1] for d in stations_data.values()]
        station_lons  = [d[2] for d in stations_data.values()]
        station_names = [d[0] for d in stations_data.values()]

        all_lats = np.concatenate([lats, station_lats])
        all_lons = np.concatenate([lons, station_lons])
        lat_range = all_lats.max() - all_lats.min()
        lon_range = all_lons.max() - all_lons.min()

        ax.set_xlim(all_lons.min() - lon_range * 0.90,
                    all_lons.max() + lon_range * 0.90)
        ax.set_ylim(all_lats.min() - lat_range * 0.10,
                    all_lats.max() + lat_range * 0.10)
        ax.set_aspect('equal', adjustable='box')

        scatter = ax.scatter(lons, lats,
                             c=gray_colors, cmap='gray', vmin=0, vmax=1,
                             s=sizes, alpha=0.85,
                             edgecolors='black', linewidth=1.5, zorder=5)

        ax.scatter(station_lons, station_lats,
                   c='red', s=10, alpha=0.7, linewidth=3, zorder=10,
                   marker='o', label='Light Rail Stations')

        for name, lon, lat in zip(station_names, station_lons, station_lats):
            ax.annotate(name, (lon, lat),
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=10, ha='left', va='bottom',
                        color='darkred', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                                  edgecolor='red', linewidth=2, alpha=0.95),
                        zorder=11)

        ax.set_facecolor('#e8e8e8')

    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, zorder=1)

    from matplotlib.lines import Line2D
    station_legend = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor='red', markeredgecolor='white',
               markersize=12, label='Light Rail Stations', markeredgewidth=2)
    ]
    cluster_legend = [
        Line2D([0], [0], color='none',
               label=f'{algorithm_name} Clusters: {len(cluster_centers)}')
    ]
    leg1 = ax.legend(handles=station_legend, loc='upper right', fontsize=11)
    ax.add_artist(leg1)
    ax.legend(handles=cluster_legend, loc='upper left', fontsize=11, handlelength=0, handletextpad=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved {algorithm_name} cluster plot to: {save_path}")
    plt.close()



def plot_segments(segments_df, cluster_centers, max_segments=100,
                  algorithm_name='HDBSCAN', save_path='segments_plot.png'):
    """
    Two-panel figure: full view and 60%-zoom magnified view of segment connections.
    Line width and opacity scale with segment frequency.

    Parameters
    ----------
    algorithm_name : str
        Name of the clustering algorithm. Used in titles and print messages.
    """
    print_section(f"VISUALISING {algorithm_name.upper()} SEGMENTS")

    if segments_df is None or len(segments_df) == 0 or len(cluster_centers) == 0:
        print("✗ No segments to visualise")
        return

    fig = plt.figure(figsize=(18, 10))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)

    lats = cluster_centers[:, 0]
    lons = cluster_centers[:, 1]

    seg_counts = (segments_df
                  .groupby(['origin_cluster', 'dest_cluster'])
                  .size()
                  .reset_index(name='count')
                  .nlargest(max_segments, 'count'))

    max_count = seg_counts['count'].max()
    min_count = seg_counts['count'].min()

    for ax, view_name, zoom in [(ax1, 'Full View', False),
                                (ax2, 'Magnified View', True)]:
        ax.scatter(lons, lats, c='red', s=10, alpha=0.9,
                   edgecolors='darkred', linewidth=2, zorder=5,
                   label='Station Clusters')

        for i in range(len(cluster_centers)):
            ax.annotate(str(i), (lons[i], lats[i]),
                        fontsize=9 if not zoom else 11,
                        ha='center', va='center',
                        color='white', fontweight='bold', zorder=6)

        for _, row in seg_counts.iterrows():
            from_idx = int(row['origin_cluster'])
            to_idx   = int(row['dest_cluster'])
            if from_idx < len(cluster_centers) and to_idx < len(cluster_centers):
                from_lat, from_lon = cluster_centers[from_idx]
                to_lat,   to_lon   = cluster_centers[to_idx]
                norm      = (row['count'] - min_count) / (max_count - min_count + 1)
                linewidth = 1.5 + norm * 3.5
                alpha     = 0.5 + norm * 0.4
                color     = (0.1, 0.3 + norm * 0.5, 0.8, alpha)
                ax.plot([from_lon, to_lon], [from_lat, to_lat],
                        color=color, linewidth=linewidth, zorder=2)

        ax.set_xlabel('Longitude', fontsize=12, fontweight='bold')
        ax.set_ylabel('Latitude',  fontsize=12, fontweight='bold')
        ax.set_title(f'{algorithm_name} Transit Segments — {view_name}',
                     fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.4, linestyle='--')
        ax.legend(loc='upper right', fontsize=10)

        if zoom and len(lats) > 0:
            lat_c = np.median(lats)
            lon_c = np.median(lons)
            lat_r = lats.max() - lats.min()
            lon_r = lons.max() - lons.min()
            zf    = 0.6
            ax.set_xlim(lon_c - lon_r * zf / 2, lon_c + lon_r * zf / 2)
            ax.set_ylim(lat_c - lat_r * zf / 2, lat_c + lat_r * zf / 2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"✓ Saved {algorithm_name} segments plot to: {save_path}")
    plt.close()



def plot_segment_statistics(segments_df, algorithm_name='HDBSCAN', save_path='segment_stats.png'):
    """
    2×2 grid: travel-time histogram, distance histogram, speed histogram,
    and a horizontal bar chart of the top-20 most-frequent segment connections.

    Parameters
    ----------
    algorithm_name : str
        Name of the clustering algorithm. Used in titles and print messages.
    """
    print_section(f"VISUALISING {algorithm_name.upper()} SEGMENT STATISTICS")

    if segments_df is None or len(segments_df) == 0:
        print("✗ No segments to visualise")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].hist(segments_df['duration_sec'], bins=50,
                    color='skyblue', edgecolor='black')
    axes[0, 0].set_xlabel('Travel Time (seconds)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title(f'{algorithm_name} — Travel Time Distribution')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].hist(segments_df['distance_m'], bins=50,
                    color='lightgreen', edgecolor='black')
    axes[0, 1].set_xlabel('Distance (meters)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title(f'{algorithm_name} — Distance Distribution')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].hist(segments_df['speed_mps'], bins=50,
                    color='salmon', edgecolor='black')
    axes[1, 0].set_xlabel('Speed (m/s)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title(f'{algorithm_name} — Speed Distribution')
    axes[1, 0].grid(True, alpha=0.3)

    seg_counts = (segments_df
                  .groupby(['origin_cluster', 'dest_cluster'])
                  .size()
                  .reset_index(name='count')
                  .nlargest(20, 'count'))

    y_labels = [f"{int(r['origin_cluster'])}→{int(r['dest_cluster'])}"
                for _, r in seg_counts.iterrows()]

    axes[1, 1].barh(range(len(seg_counts)), seg_counts['count'],
                    color='orchid', edgecolor='black')
    axes[1, 1].set_yticks(range(len(seg_counts)))
    axes[1, 1].set_yticklabels(y_labels, fontsize=8)
    axes[1, 1].set_xlabel('Frequency')
    axes[1, 1].set_title(f'{algorithm_name} — Top 20 Segment Connections')
    axes[1, 1].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved {algorithm_name} segment statistics plot to: {save_path}")
    plt.close()