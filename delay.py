"""
visualize_flags.py
==================
Visualize which segments/stations get binary flags and their impact on delays

Usage:
    python visualize_flags.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
import os

from data_loader import load_train_test_val_data_fixed, get_known_stops
from cluster_hdbscan import event_driven_clustering_fixed
from segments import build_segments_fixed
from config import print_section


# =============================================================================
# 1. SEGMENT-LEVEL ANALYSIS
# =============================================================================

def analyze_flags_impact():
    """Analyze how binary flags correlate with delays"""

    print_section("LOADING DATA FOR FLAG ANALYSIS")

    # Load data
    train_df, test_df, val_df = load_train_test_val_data_fixed(
        data_folder='./data',
        sample_fraction=0.1  # 10% sample for faster visualization
    )

    print(f"   Loaded {len(train_df):,} train records")

    # Clustering
    known_stops = get_known_stops(train_df)
    clusters, station_cluster_ids = event_driven_clustering_fixed(
        train_df, known_stops=known_stops
    )

    print(f"   Created {len(clusters)} clusters")

    # Build segments
    segments = build_segments_fixed(train_df, clusters)

    print(f"   Created {len(segments):,} segments")
    print(f"   Segments with is_weekend flag: {'is_weekend' in segments.columns}")
    print(f"   Segments with is_peak_hour flag: {'is_peak_hour' in segments.columns}")

    return segments, clusters, known_stops


def create_flag_statistics(segments):
    """Generate statistical summary of flags and delays"""

    print_section("FLAG STATISTICS")

    # Overall statistics
    total_segments = len(segments)

    print(f"\n📊 Overall Statistics:")
    print(f"   Total segments: {total_segments:,}")

    # Weekend statistics
    if 'is_weekend' in segments.columns:
        weekend_count = segments['is_weekend'].sum()
        weekday_count = total_segments - weekend_count

        print(f"\n🗓️  Weekend vs Weekday:")
        print(f"   Weekend segments: {weekend_count:,} ({weekend_count / total_segments * 100:.1f}%)")
        print(f"   Weekday segments: {weekday_count:,} ({weekday_count / total_segments * 100:.1f}%)")

        # Delay analysis
        weekend_segs = segments[segments['is_weekend'] == 1]
        weekday_segs = segments[segments['is_weekend'] == 0]

        if len(weekend_segs) > 0:
            print(f"\n   Weekend delays:")
            print(f"     Avg arrival delay: {weekend_segs['arrivalDelay'].mean():.2f}s")
            print(f"     Avg departure delay: {weekend_segs['departureDelay'].mean():.2f}s")
            print(f"     Avg duration: {weekend_segs['duration_sec'].mean():.2f}s")

        if len(weekday_segs) > 0:
            print(f"\n   Weekday delays:")
            print(f"     Avg arrival delay: {weekday_segs['arrivalDelay'].mean():.2f}s")
            print(f"     Avg departure delay: {weekday_segs['departureDelay'].mean():.2f}s")
            print(f"     Avg duration: {weekday_segs['duration_sec'].mean():.2f}s")

    # Peak hour statistics
    if 'is_peak_hour' in segments.columns:
        peak_count = segments['is_peak_hour'].sum()
        offpeak_count = total_segments - peak_count

        print(f"\n⏰ Peak vs Off-Peak:")
        print(f"   Peak hour segments: {peak_count:,} ({peak_count / total_segments * 100:.1f}%)")
        print(f"   Off-peak segments: {offpeak_count:,} ({offpeak_count / total_segments * 100:.1f}%)")

        # Delay analysis
        peak_segs = segments[segments['is_peak_hour'] == 1]
        offpeak_segs = segments[segments['is_peak_hour'] == 0]

        if len(peak_segs) > 0:
            print(f"\n   Peak hour delays:")
            print(f"     Avg arrival delay: {peak_segs['arrivalDelay'].mean():.2f}s")
            print(f"     Avg departure delay: {peak_segs['departureDelay'].mean():.2f}s")
            print(f"     Avg duration: {peak_segs['duration_sec'].mean():.2f}s")

        if len(offpeak_segs) > 0:
            print(f"\n   Off-peak delays:")
            print(f"     Avg arrival delay: {offpeak_segs['arrivalDelay'].mean():.2f}s")
            print(f"     Avg departure delay: {offpeak_segs['departureDelay'].mean():.2f}s")
            print(f"     Avg duration: {offpeak_segs['duration_sec'].mean():.2f}s")

    # Combined analysis
    if 'is_weekend' in segments.columns and 'is_peak_hour' in segments.columns:
        print(f"\n🔍 Combined Analysis:")

        categories = [
            ("Weekday Off-Peak", (segments['is_weekend'] == 0) & (segments['is_peak_hour'] == 0)),
            ("Weekday Peak", (segments['is_weekend'] == 0) & (segments['is_peak_hour'] == 1)),
            ("Weekend Off-Peak", (segments['is_weekend'] == 1) & (segments['is_peak_hour'] == 0)),
            ("Weekend Peak", (segments['is_weekend'] == 1) & (segments['is_peak_hour'] == 1)),
        ]

        for name, mask in categories:
            subset = segments[mask]
            if len(subset) > 0:
                print(f"\n   {name}: {len(subset):,} segments")
                print(f"     Avg arrival delay: {subset['arrivalDelay'].mean():.2f}s")
                print(f"     Avg departure delay: {subset['departureDelay'].mean():.2f}s")
                print(f"     Avg duration: {subset['duration_sec'].mean():.2f}s")


# =============================================================================
# 2. VISUALIZATIONS
# =============================================================================

def plot_flag_distribution(segments, output_folder='outputs/flag_analysis'):
    """Plot distribution of flags across segments"""

    os.makedirs(output_folder, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Binary Flag Distribution and Impact Analysis', fontsize=16, fontweight='bold')

    # 1. Weekend distribution
    if 'is_weekend' in segments.columns:
        weekend_counts = segments['is_weekend'].value_counts()
        axes[0, 0].bar(['Weekday', 'Weekend'],
                       [weekend_counts.get(0, 0), weekend_counts.get(1, 0)],
                       color=['skyblue', 'coral'])
        axes[0, 0].set_title('Weekend Flag Distribution')
        axes[0, 0].set_ylabel('Number of Segments')
        axes[0, 0].grid(True, alpha=0.3)

    # 2. Peak hour distribution
    if 'is_peak_hour' in segments.columns:
        peak_counts = segments['is_peak_hour'].value_counts()
        axes[0, 1].bar(['Off-Peak', 'Peak'],
                       [peak_counts.get(0, 0), peak_counts.get(1, 0)],
                       color=['lightgreen', 'orange'])
        axes[0, 1].set_title('Peak Hour Flag Distribution')
        axes[0, 1].set_ylabel('Number of Segments')
        axes[0, 1].grid(True, alpha=0.3)

    # 3. Combined distribution
    if 'is_weekend' in segments.columns and 'is_peak_hour' in segments.columns:
        categories = ['Weekday\nOff-Peak', 'Weekday\nPeak', 'Weekend\nOff-Peak', 'Weekend\nPeak']
        masks = [
            (segments['is_weekend'] == 0) & (segments['is_peak_hour'] == 0),
            (segments['is_weekend'] == 0) & (segments['is_peak_hour'] == 1),
            (segments['is_weekend'] == 1) & (segments['is_peak_hour'] == 0),
            (segments['is_weekend'] == 1) & (segments['is_peak_hour'] == 1),
        ]
        counts = [segments[mask].shape[0] for mask in masks]
        colors = ['skyblue', 'orange', 'lightcoral', 'red']
        axes[0, 2].bar(categories, counts, color=colors)
        axes[0, 2].set_title('Combined Flag Distribution')
        axes[0, 2].set_ylabel('Number of Segments')
        axes[0, 2].tick_params(axis='x', rotation=45)
        axes[0, 2].grid(True, alpha=0.3)

    # 4. Arrival delay by weekend
    if 'is_weekend' in segments.columns:
        weekend_delays = [
            segments[segments['is_weekend'] == 0]['arrivalDelay'].mean(),
            segments[segments['is_weekend'] == 1]['arrivalDelay'].mean()
        ]
        axes[1, 0].bar(['Weekday', 'Weekend'], weekend_delays, color=['skyblue', 'coral'])
        axes[1, 0].set_title('Avg Arrival Delay by Weekend Flag')
        axes[1, 0].set_ylabel('Delay (seconds)')
        axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[1, 0].grid(True, alpha=0.3)

    # 5. Departure delay by peak hour
    if 'is_peak_hour' in segments.columns:
        peak_delays = [
            segments[segments['is_peak_hour'] == 0]['departureDelay'].mean(),
            segments[segments['is_peak_hour'] == 1]['departureDelay'].mean()
        ]
        axes[1, 1].bar(['Off-Peak', 'Peak'], peak_delays, color=['lightgreen', 'orange'])
        axes[1, 1].set_title('Avg Departure Delay by Peak Hour Flag')
        axes[1, 1].set_ylabel('Delay (seconds)')
        axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[1, 1].grid(True, alpha=0.3)

    # 6. Combined delay analysis
    if 'is_weekend' in segments.columns and 'is_peak_hour' in segments.columns:
        combined_delays = []
        for mask in masks:
            subset = segments[mask]
            if len(subset) > 0:
                combined_delays.append(subset['departureDelay'].mean())
            else:
                combined_delays.append(0)

        axes[1, 2].bar(categories, combined_delays, color=colors)
        axes[1, 2].set_title('Avg Departure Delay (Combined)')
        axes[1, 2].set_ylabel('Delay (seconds)')
        axes[1, 2].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[1, 2].tick_params(axis='x', rotation=45)
        axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'flag_distribution.png'), dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {output_folder}/flag_distribution.png")
    plt.close()


def plot_segment_map_with_flags(segments, clusters, known_stops, output_folder='outputs/flag_analysis'):
    """Plot map of segments colored by flags"""

    os.makedirs(output_folder, exist_ok=True)

    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Segment Network Colored by Binary Flags', fontsize=16, fontweight='bold')

    # Parse segment origins and destinations
    segment_coords = []
    for _, seg in segments.iterrows():
        origin_id, dest_id = map(int, seg['segment_id'].split('_'))
        if origin_id < len(clusters) and dest_id < len(clusters):
            segment_coords.append({
                'origin': clusters[origin_id],
                'dest': clusters[dest_id],
                'is_weekend': seg.get('is_weekend', 0),
                'is_peak_hour': seg.get('is_peak_hour', 0),
                'arrivalDelay': seg.get('arrivalDelay', 0),
                'departureDelay': seg.get('departureDelay', 0)
            })

    print(f"   Found {len(segment_coords)} valid segments to plot")

    # 1. All segments (baseline)
    ax = axes[0, 0]
    for seg in segment_coords:
        ax.plot([seg['origin'][1], seg['dest'][1]],
                [seg['origin'][0], seg['dest'][0]],
                'gray', alpha=0.3, linewidth=0.5)

    for i, (lat, lon) in enumerate(clusters):
        ax.plot(lon, lat, 'o', color='darkblue', markersize=8, zorder=10)

    ax.set_title('All Segments (Baseline)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(True, alpha=0.3)

    # 2. Weekend segments highlighted
    ax = axes[0, 1]
    weekend_count = 0

    for seg in segment_coords:
        if seg['is_weekend'] == 1:
            ax.plot([seg['origin'][1], seg['dest'][1]],
                    [seg['origin'][0], seg['dest'][0]],
                    'red', alpha=0.8, linewidth=2)
            weekend_count += 1
        else:
            ax.plot([seg['origin'][1], seg['dest'][1]],
                    [seg['origin'][0], seg['dest'][0]],
                    'lightgray', alpha=0.2, linewidth=0.5)

    for i, (lat, lon) in enumerate(clusters):
        ax.plot(lon, lat, 'o', color='darkblue', markersize=8, zorder=10)

    ax.set_title(f'Weekend Segments (Red = {weekend_count} segments)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(True, alpha=0.3)

    # 3. Peak hour segments highlighted
    ax = axes[1, 0]
    peak_count = 0

    for seg in segment_coords:
        if seg['is_peak_hour'] == 1:
            ax.plot([seg['origin'][1], seg['dest'][1]],
                    [seg['origin'][0], seg['dest'][0]],
                    'orange', alpha=0.8, linewidth=2)
            peak_count += 1
        else:
            ax.plot([seg['origin'][1], seg['dest'][1]],
                    [seg['origin'][0], seg['dest'][0]],
                    'lightgray', alpha=0.2, linewidth=0.5)

    for i, (lat, lon) in enumerate(clusters):
        ax.plot(lon, lat, 'o', color='darkblue', markersize=8, zorder=10)

    ax.set_title(f'Peak Hour Segments (Orange = {peak_count} segments)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(True, alpha=0.3)

    # 4. Delay magnitude
    ax = axes[1, 1]

    delays = [seg['departureDelay'] for seg in segment_coords]
    if len(delays) > 0:
        vmin, vmax = np.percentile(delays, [5, 95])

        for seg in segment_coords:
            delay = seg['departureDelay']
            if delay < 0:
                color = plt.cm.Blues(abs(delay) / abs(vmin) if vmin != 0 else 0)
            else:
                color = plt.cm.Reds(delay / vmax if vmax != 0 else 0)

            ax.plot([seg['origin'][1], seg['dest'][1]],
                    [seg['origin'][0], seg['dest'][0]],
                    color=color, alpha=0.6, linewidth=1.5)

        for i, (lat, lon) in enumerate(clusters):
            ax.plot(lon, lat, 'o', color='black', markersize=8, zorder=10)

        ax.set_title('Segments by Departure Delay (Blue=Early, Red=Late)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.grid(True, alpha=0.3)

        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdBu_r,
                                   norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Departure Delay (s)', rotation=270, labelpad=20)

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'segment_map_flags.png'), dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_folder}/segment_map_flags.png")
    plt.close()


def plot_temporal_heatmap(segments, output_folder='outputs/flag_analysis'):
    """Plot heatmap of delays by hour and day of week"""

    os.makedirs(output_folder, exist_ok=True)

    if 'hour' not in segments.columns or 'day_of_week' not in segments.columns:
        print("⚠️  Missing temporal columns, skipping heatmap")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Delay Heatmaps by Time', fontsize=16, fontweight='bold')

    # Create pivot tables
    arrival_pivot = segments.pivot_table(
        values='arrivalDelay',
        index='hour',
        columns='day_of_week',
        aggfunc='mean'
    )

    departure_pivot = segments.pivot_table(
        values='departureDelay',
        index='hour',
        columns='day_of_week',
        aggfunc='mean'
    )

    # ✅ FIX: Ensure all 7 days are present (reindex to 0-6)
    all_days = list(range(7))
    arrival_pivot = arrival_pivot.reindex(columns=all_days)
    departure_pivot = departure_pivot.reindex(columns=all_days)

    # Day names
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    # Arrival delay heatmap
    sns.heatmap(arrival_pivot, annot=True, fmt='.1f', cmap='RdYlGn_r',
                center=0, ax=axes[0], cbar_kws={'label': 'Delay (s)'},
                mask=arrival_pivot.isna())  # ✅ Mask NaN cells
    axes[0].set_title('Average Arrival Delay')
    axes[0].set_xlabel('Day of Week')
    axes[0].set_ylabel('Hour of Day')
    axes[0].set_xticklabels(day_names, rotation=0)

    # Departure delay heatmap
    sns.heatmap(departure_pivot, annot=True, fmt='.1f', cmap='RdYlGn_r',
                center=0, ax=axes[1], cbar_kws={'label': 'Delay (s)'},
                mask=departure_pivot.isna())  # ✅ Mask NaN cells
    axes[1].set_title('Average Departure Delay')
    axes[1].set_xlabel('Day of Week')
    axes[1].set_ylabel('Hour of Day')
    axes[1].set_xticklabels(day_names, rotation=0)

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'temporal_heatmap.png'), dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_folder}/temporal_heatmap.png")
    plt.close()


def plot_station_level_analysis(segments, clusters, known_stops, output_folder='outputs/flag_analysis'):
    """Analyze which stations are most affected by flags"""

    os.makedirs(output_folder, exist_ok=True)

    # Aggregate by station (cluster)
    station_stats = []

    for cluster_id in range(len(clusters)):
        # Find segments originating from this cluster
        origin_segs = segments[segments['segment_id'].str.startswith(f'{cluster_id}_')]

        if len(origin_segs) == 0:
            continue

        station_stats.append({
            'cluster_id': cluster_id,
            'lat': clusters[cluster_id][0],
            'lon': clusters[cluster_id][1],
            'n_segments': len(origin_segs),
            'avg_arrival_delay': origin_segs['arrivalDelay'].mean(),
            'avg_departure_delay': origin_segs['departureDelay'].mean(),
            'pct_weekend': origin_segs['is_weekend'].mean() * 100 if 'is_weekend' in origin_segs else 0,
            'pct_peak': origin_segs['is_peak_hour'].mean() * 100 if 'is_peak_hour' in origin_segs else 0,
        })

    station_df = pd.DataFrame(station_stats)

    # Plot top 10 most delayed stations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Station-Level Flag Analysis', fontsize=16, fontweight='bold')

    # 1. Top stations by departure delay
    top_delayed = station_df.nlargest(10, 'avg_departure_delay')
    axes[0, 0].barh(range(len(top_delayed)), top_delayed['avg_departure_delay'].values)
    axes[0, 0].set_yticks(range(len(top_delayed)))
    axes[0, 0].set_yticklabels([f'Cluster {int(cid)}' for cid in top_delayed['cluster_id'].values])
    axes[0, 0].set_xlabel('Avg Departure Delay (s)')
    axes[0, 0].set_title('Top 10 Most Delayed Stations')
    axes[0, 0].grid(True, alpha=0.3, axis='x')

    # 2. Stations with most weekend traffic
    top_weekend = station_df.nlargest(10, 'pct_weekend')
    axes[0, 1].barh(range(len(top_weekend)), top_weekend['pct_weekend'].values, color='coral')
    axes[0, 1].set_yticks(range(len(top_weekend)))
    axes[0, 1].set_yticklabels([f'Cluster {int(cid)}' for cid in top_weekend['cluster_id'].values])
    axes[0, 1].set_xlabel('Weekend Traffic (%)')
    axes[0, 1].set_title('Top 10 Stations by Weekend Traffic')
    axes[0, 1].grid(True, alpha=0.3, axis='x')

    # 3. Stations with most peak hour traffic
    top_peak = station_df.nlargest(10, 'pct_peak')
    axes[1, 0].barh(range(len(top_peak)), top_peak['pct_peak'].values, color='orange')
    axes[1, 0].set_yticks(range(len(top_peak)))
    axes[1, 0].set_yticklabels([f'Cluster {int(cid)}' for cid in top_peak['cluster_id'].values])
    axes[1, 0].set_xlabel('Peak Hour Traffic (%)')
    axes[1, 0].set_title('Top 10 Stations by Peak Hour Traffic')
    axes[1, 0].grid(True, alpha=0.3, axis='x')

    # 4. Scatter: Peak traffic vs Delay
    axes[1, 1].scatter(station_df['pct_peak'], station_df['avg_departure_delay'],
                       s=station_df['n_segments'] * 10, alpha=0.6, c=station_df['pct_weekend'],
                       cmap='coolwarm')
    axes[1, 1].set_xlabel('Peak Hour Traffic (%)')
    axes[1, 1].set_ylabel('Avg Departure Delay (s)')
    axes[1, 1].set_title('Peak Traffic vs Delay (size=traffic volume, color=weekend%)')
    axes[1, 1].grid(True, alpha=0.3)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='coolwarm',
                               norm=plt.Normalize(vmin=station_df['pct_weekend'].min(),
                                                  vmax=station_df['pct_weekend'].max()))
    sm.set_array([])
    plt.colorbar(sm, ax=axes[1, 1], label='Weekend %')

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'station_analysis.png'), dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_folder}/station_analysis.png")
    plt.close()

    # Save CSV
    station_df.to_csv(os.path.join(output_folder, 'station_statistics.csv'), index=False)
    print(f"✓ Saved: {output_folder}/station_statistics.csv")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run complete flag analysis and visualization"""

    print("=" * 80)
    print("BINARY FLAG IMPACT ANALYSIS")
    print("=" * 80)

    # Load and analyze
    segments, clusters, known_stops = analyze_flags_impact()

    # Statistics
    create_flag_statistics(segments)

    # Visualizations
    print_section("GENERATING VISUALIZATIONS")

    output_folder = 'outputs/flag_analysis'
    os.makedirs(output_folder, exist_ok=True)

    print("\n1. Creating flag distribution plots...")
    plot_flag_distribution(segments, output_folder)

    print("\n2. Creating segment network maps...")
    plot_segment_map_with_flags(segments, clusters, known_stops, output_folder)

    print("\n3. Creating temporal heatmaps...")
    plot_temporal_heatmap(segments, output_folder)

    print("\n4. Creating station-level analysis...")
    plot_station_level_analysis(segments, clusters, known_stops, output_folder)

    print_section("✅ ANALYSIS COMPLETE")
    print(f"\n📁 All outputs saved to: {output_folder}/")
    print(f"\n📊 Generated files:")
    print(f"   • flag_distribution.png    - Bar charts of flag distributions and delays")
    print(f"   • segment_map_flags.png    - Geographic maps showing flagged segments")
    print(f"   • temporal_heatmap.png     - Hour/day heatmaps of delays")
    print(f"   • station_analysis.png     - Station-level statistics")
    print(f"   • station_statistics.csv   - Detailed station data (CSV)")
    print("=" * 80)


if __name__ == '__main__':
    main()