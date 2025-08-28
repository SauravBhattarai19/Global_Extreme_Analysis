#!/usr/bin/env python3
"""
Visualization of Heatwave-Precipitation Analysis (Output from 07_Heatwave_Precipitation_Analysis.py)

Creates comprehensive scientific visualizations of heatwave-precipitation interactions:
- Compound event patterns (drought-heat, wet heatwaves)
- Pre/during/post heatwave precipitation analysis
- Recovery patterns and persistent drought
- Regional compound event hotspots
- Temporal trends in precipitation-heatwave interactions

Input files:
- heatwave_events_precipitation_{year}.parquet (enhanced events with precipitation)
- heatwave_precipitation_analysis_{year}.parquet (detailed precipitation time series)
- heatwave_precipitation_aggregations_{year}.nc (annual spatial summaries)
"""

import sys
import argparse
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
import pandas as pd
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Compound event colors
COMPOUND_COLORS = {
    'compound_drought_heat': '#8B0000',      # Dark Red
    'dry_heatwave': '#FF4500',               # Orange Red
    'wet_heatwave': '#4682B4',               # Steel Blue
    'mixed_heatwave': '#9370DB',             # Medium Purple
    'extreme_drought_heat': '#8B0000',       # Dark Red
    'humid_heat_no_precip': '#4169E1',       # Royal Blue
    'humid_heat_with_precip': '#00CED1'      # Dark Turquoise
}

def load_precipitation_analysis_data(precip_dir, years=None):
    """Load heatwave-precipitation analysis data."""
    precip_dir = Path(precip_dir)
    
    if years is None:
        # Find available years
        event_files = list(precip_dir.glob('heatwave_events_precipitation_*.parquet'))
        years = sorted(set([int(f.name.split('_')[3].split('.')[0]) for f in event_files]))
        print(f"Found data for years: {years[0]}-{years[-1]}")
    
    print(f"Loading precipitation analysis data for years {years}...")
    
    events_data = []
    analysis_data = []
    aggregation_data = {}
    
    for year in years:
        # Load enhanced events
        events_file = precip_dir / f'heatwave_events_precipitation_{year}.parquet'
        if events_file.exists():
            events_df = pd.read_parquet(events_file)
            events_df['year'] = year
            events_data.append(events_df)
            print(f"  Loaded events: {events_file.name}")
        
        # Load detailed analysis
        analysis_file = precip_dir / f'heatwave_precipitation_analysis_{year}.parquet'
        if analysis_file.exists():
            analysis_df = pd.read_parquet(analysis_file)
            analysis_df['year'] = year
            analysis_data.append(analysis_df)
            print(f"  Loaded analysis: {analysis_file.name}")
        
        # Load aggregations
        agg_file = precip_dir / f'heatwave_precipitation_aggregations_{year}.nc'
        if agg_file.exists():
            agg_ds = xr.open_dataset(agg_file)
            agg_ds = agg_ds.expand_dims('year').assign_coords(year=[year])
            if year == years[0]:
                aggregation_data = agg_ds
            else:
                aggregation_data = xr.concat([aggregation_data, agg_ds], dim='year')
            print(f"  Loaded aggregations: {agg_file.name}")
    
    # Combine data
    combined_events = pd.concat(events_data, ignore_index=True) if events_data else pd.DataFrame()
    combined_analysis = pd.concat(analysis_data, ignore_index=True) if analysis_data else pd.DataFrame()
    
    print(f"Combined events: {len(combined_events):,} records")
    print(f"Combined analysis: {len(combined_analysis):,} records")
    
    return combined_events, combined_analysis, aggregation_data

def plot_compound_event_patterns(aggregation_data, output_dir):
    """Create global maps of compound event patterns."""
    output_dir = Path(output_dir)
    
    if not aggregation_data or len(aggregation_data.data_vars) == 0:
        print("No aggregation data available for compound event analysis")
        return
    
    # Calculate multi-year means
    compound_vars = [
        ('compound_drought_heat_events', 'Compound Drought-Heat Events'),
        ('dry_heatwave_events', 'Dry Heatwave Events'),
        ('wet_heatwave_events', 'Wet Heatwave Events'),
        ('extreme_drought_heat_events', 'Extreme Drought-Heat Events')
    ]
    
    fig = plt.figure(figsize=(20, 16))
    
    for i, (var_name, title) in enumerate(compound_vars):
        if var_name not in aggregation_data.data_vars:
            continue
        
        ax = plt.subplot(2, 2, i+1, projection=ccrs.Robinson())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax.set_global()
        
        data = aggregation_data[var_name].mean(dim='year')
        
        # Color scheme based on event type
        if 'drought' in var_name:
            cmap = plt.cm.Reds
            levels = np.arange(0, 3.1, 0.2)
        elif 'wet' in var_name:
            cmap = plt.cm.Blues
            levels = np.arange(0, 1.1, 0.1)
        else:
            cmap = plt.cm.YlOrRd
            levels = np.arange(0, 2.1, 0.2)
        
        im = ax.contourf(data.longitude, data.latitude, data,
                        levels=levels, cmap=cmap, transform=ccrs.PlateCarree(),
                        extend='max')
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Events per year', fontsize=10)
    
    plt.suptitle(f'Compound Event Patterns\n'
                f'Period: {aggregation_data.year.min().values}-{aggregation_data.year.max().values}',
                fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    
    # Save
    filename = 'compound_event_patterns.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {filename}")

def plot_precipitation_recovery_analysis(aggregation_data, output_dir):
    """Analyze precipitation recovery patterns after heatwaves."""
    output_dir = Path(output_dir)
    
    if not aggregation_data or len(aggregation_data.data_vars) == 0:
        print("No aggregation data available for recovery analysis")
        return
    
    fig = plt.figure(figsize=(20, 12))
    
    # Recovery pattern variables
    recovery_vars = [
        ('immediate_recovery_events', 'Immediate Recovery (<3 days)'),
        ('delayed_recovery_events', 'Delayed Recovery (7-30 days)'),
        ('persistent_drought_events', 'Persistent Drought (>30 days)')
    ]
    
    for i, (var_name, title) in enumerate(recovery_vars):
        if var_name not in aggregation_data.data_vars:
            continue
        
        ax = plt.subplot(2, 3, i+1, projection=ccrs.Robinson())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax.set_global()
        
        data = aggregation_data[var_name].mean(dim='year')
        
        # Color scheme based on recovery type
        if 'immediate' in var_name:
            cmap = plt.cm.Greens
        elif 'delayed' in var_name:
            cmap = plt.cm.Oranges
        else:  # persistent
            cmap = plt.cm.Reds
        
        levels = np.arange(0, 2.1, 0.2)
        
        im = ax.contourf(data.longitude, data.latitude, data,
                        levels=levels, cmap=cmap, transform=ccrs.PlateCarree(),
                        extend='max')
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Events per year', fontsize=10)
    
    # Precipitation statistics after heatwaves
    precip_vars = [
        ('mean_precip_3d_after_heatwave', '3-Day Post-Heatwave Precipitation'),
        ('mean_precip_7d_after_heatwave', '7-Day Post-Heatwave Precipitation'),
        ('mean_days_to_first_precip', 'Days to First Precipitation')
    ]
    
    for i, (var_name, title) in enumerate(precip_vars):
        if var_name not in aggregation_data.data_vars:
            continue
        
        ax = plt.subplot(2, 3, i+4, projection=ccrs.Robinson())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax.set_global()
        
        data = aggregation_data[var_name].mean(dim='year')
        
        if 'days_to_first' in var_name:
            cmap = plt.cm.plasma
            levels = np.arange(0, 31, 2)
            cbar_label = 'Days'
        else:
            cmap = plt.cm.Blues
            levels = np.arange(0, 21, 2)
            cbar_label = 'Precipitation (mm)'
        
        im = ax.contourf(data.longitude, data.latitude, data,
                        levels=levels, cmap=cmap, transform=ccrs.PlateCarree(),
                        extend='max')
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label(cbar_label, fontsize=10)
    
    plt.suptitle(f'Precipitation Recovery Analysis\n'
                f'Period: {aggregation_data.year.min().values}-{aggregation_data.year.max().values}',
                fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    
    # Save
    filename = 'precipitation_recovery_analysis.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {filename}")

def plot_event_precipitation_analysis(events_data, output_dir):
    """Analyze precipitation characteristics of individual events."""
    output_dir = Path(output_dir)
    
    if events_data.empty:
        print("No event data available for precipitation analysis")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Compound event type distribution
    ax1 = axes[0, 0]
    
    if 'compound_event_type' in events_data.columns:
        event_counts = events_data['compound_event_type'].value_counts()
        colors = [COMPOUND_COLORS.get(event_type, 'gray') for event_type in event_counts.index]
        
        wedges, texts, autotexts = ax1.pie(event_counts.values, labels=event_counts.index,
                                          colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Compound Event Type Distribution', fontweight='bold')
    
    # 2. Precipitation pattern distribution
    ax2 = axes[0, 1]
    
    if 'precip_pattern' in events_data.columns:
        pattern_counts = events_data['precip_pattern'].value_counts()
        colors = [COMPOUND_COLORS.get(pattern, 'gray') for pattern in pattern_counts.index]
        
        ax2.bar(pattern_counts.index, pattern_counts.values, color=colors, alpha=0.7)
        ax2.set_xlabel('Precipitation Pattern')
        ax2.set_ylabel('Number of Events')
        ax2.set_title('Precipitation Pattern Distribution', fontweight='bold')
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
    
    # 3. Pre-heatwave vs during-heatwave precipitation
    ax3 = axes[1, 0]
    
    if all(col in events_data.columns for col in ['pre_hw_total_precip', 'during_hw_total_precip']):
        pre_precip = events_data['pre_hw_total_precip'].dropna()
        during_precip = events_data['during_hw_total_precip'].dropna()
        
        ax3.scatter(pre_precip, during_precip, alpha=0.6, s=20)
        ax3.set_xlabel('Pre-Heatwave Precipitation (mm)')
        ax3.set_ylabel('During-Heatwave Precipitation (mm)')
        ax3.set_title('Pre vs During Heatwave Precipitation', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Add diagonal line
        max_val = max(pre_precip.max(), during_precip.max())
        ax3.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='1:1 line')
        ax3.legend()
    
    # 4. Recovery time distribution
    ax4 = axes[1, 1]
    
    recovery_cols = [col for col in events_data.columns if 'first_precip_day' in col]
    if recovery_cols:
        # Use the shortest recovery time available
        recovery_col = recovery_cols[0]  # Usually 'post_3d_first_precip_day'
        recovery_times = events_data[recovery_col].dropna()
        
        if len(recovery_times) > 0:
            ax4.hist(recovery_times, bins=range(1, 32), alpha=0.7, color='skyblue', edgecolor='black')
            ax4.set_xlabel('Days to First Precipitation')
            ax4.set_ylabel('Number of Events')
            ax4.set_title('Post-Heatwave Recovery Time', fontweight='bold')
            ax4.grid(True, alpha=0.3)
            
            # Add statistics
            mean_recovery = recovery_times.mean()
            ax4.axvline(mean_recovery, color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {mean_recovery:.1f} days')
            ax4.legend()
    
    plt.suptitle(f'Event-Level Precipitation Analysis\n'
                f'Total Events: {len(events_data):,}',
                fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    
    # Save
    filename = 'event_precipitation_analysis.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {filename}")

def plot_temporal_precipitation_trends(aggregation_data, events_data, output_dir):
    """Plot temporal trends in precipitation-heatwave interactions."""
    output_dir = Path(output_dir)
    
    if not aggregation_data or 'year' not in aggregation_data.dims or len(aggregation_data.year) < 3:
        print("Insufficient years for trend analysis")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    years = aggregation_data.year.values
    
    # 1. Compound event trends
    ax1 = axes[0, 0]
    
    compound_vars = ['compound_drought_heat_events', 'extreme_drought_heat_events']
    colors = ['red', 'darkred']
    
    for var, color in zip(compound_vars, colors):
        if var in aggregation_data.data_vars:
            global_mean = aggregation_data[var].mean(dim=['latitude', 'longitude'])
            ax1.plot(years, global_mean, 'o-', color=color, 
                    label=var.replace('_', ' ').title(), linewidth=2, markersize=6)
    
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Global Mean Events/Year')
    ax1.set_title('Compound Event Trends', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Recovery pattern trends
    ax2 = axes[0, 1]
    
    recovery_vars = ['immediate_recovery_events', 'delayed_recovery_events', 'persistent_drought_events']
    colors = ['green', 'orange', 'red']
    
    for var, color in zip(recovery_vars, colors):
        if var in aggregation_data.data_vars:
            global_mean = aggregation_data[var].mean(dim=['latitude', 'longitude'])
            ax2.plot(years, global_mean, 'o-', color=color,
                    label=var.replace('_', ' ').title(), linewidth=2, markersize=6)
    
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Global Mean Events/Year')
    ax2.set_title('Recovery Pattern Trends', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Post-heatwave precipitation trends
    ax3 = axes[1, 0]
    
    precip_vars = ['mean_precip_3d_after_heatwave', 'mean_precip_7d_after_heatwave']
    colors = ['blue', 'navy']
    labels = ['3-Day Post', '7-Day Post']
    
    for var, color, label in zip(precip_vars, colors, labels):
        if var in aggregation_data.data_vars:
            global_mean = aggregation_data[var].mean(dim=['latitude', 'longitude'])
            ax3.plot(years, global_mean, 'o-', color=color,
                    label=label, linewidth=2, markersize=6)
    
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Mean Precipitation (mm)')
    ax3.set_title('Post-Heatwave Precipitation Trends', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Annual event counts by pattern
    ax4 = axes[1, 1]
    
    if not events_data.empty and 'precip_pattern' in events_data.columns:
        annual_patterns = events_data.groupby(['year', 'precip_pattern']).size().unstack(fill_value=0)
        
        if not annual_patterns.empty and len(annual_patterns) > 1:
            for pattern in annual_patterns.columns:
                color = COMPOUND_COLORS.get(pattern, 'gray')
                ax4.plot(annual_patterns.index, annual_patterns[pattern], 'o-', 
                        color=color, label=pattern, linewidth=2, markersize=6)
            
            ax4.set_xlabel('Year')
            ax4.set_ylabel('Number of Events')
            ax4.set_title('Precipitation Pattern Trends', fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Insufficient data\nfor pattern trends', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=14)
            ax4.set_title('Precipitation Pattern Analysis', fontweight='bold')
    
    plt.suptitle(f'Temporal Trends in Precipitation-Heatwave Interactions\n'
                f'Period: {years[0]}-{years[-1]}',
                fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    
    # Save
    filename = 'precipitation_temporal_trends.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {filename}")

def plot_regional_compound_analysis(events_data, output_dir):
    """Analyze compound events by region."""
    output_dir = Path(output_dir)
    
    if events_data.empty:
        print("No event data available for regional analysis")
        return
    
    # Define regions (simplified based on lat/lon if available)
    regions = {
        'Global': {'lat_range': (-90, 90), 'lon_range': (-180, 180), 'color': 'black'}
    }
    
    # If we have lat/lon coordinates, add more specific regions
    if 'grid_y' in events_data.columns and 'grid_x' in events_data.columns:
        # For now, use a simplified global analysis
        # In practice, you'd convert grid indices to lat/lon
        pass
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Compound event frequency by type
    ax1 = axes[0, 0]
    
    if 'compound_event_type' in events_data.columns:
        compound_counts = events_data['compound_event_type'].value_counts()
        colors = [COMPOUND_COLORS.get(event_type, 'gray') for event_type in compound_counts.index]
        
        ax1.barh(compound_counts.index, compound_counts.values, color=colors, alpha=0.7)
        ax1.set_xlabel('Number of Events')
        ax1.set_title('Compound Event Frequency by Type', fontweight='bold')
        ax1.grid(True, alpha=0.3)
    
    # 2. Precipitation pattern seasonality
    ax2 = axes[0, 1]
    
    if 'year_start' in events_data.columns and 'precip_pattern' in events_data.columns:
        events_data['start_month'] = pd.to_datetime(events_data['year_start']).dt.month
        
        monthly_patterns = events_data.groupby(['start_month', 'precip_pattern']).size().unstack(fill_value=0)
        
        if not monthly_patterns.empty:
            months = range(1, 13)
            month_labels = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
            
            bottom = np.zeros(len(months))
            for pattern in monthly_patterns.columns:
                values = [monthly_patterns.loc[m, pattern] if m in monthly_patterns.index else 0 
                         for m in months]
                color = COMPOUND_COLORS.get(pattern, 'gray')
                ax2.bar(months, values, bottom=bottom, label=pattern, color=color, alpha=0.8)
                bottom += values
            
            ax2.set_xlabel('Month')
            ax2.set_ylabel('Number of Events')
            ax2.set_title('Seasonal Distribution of Precipitation Patterns', fontweight='bold')
            ax2.set_xticks(months)
            ax2.set_xticklabels(month_labels)
            ax2.legend()
    
    # 3. Duration vs precipitation relationship
    ax3 = axes[1, 0]
    
    if all(col in events_data.columns for col in ['duration_days', 'compound_event_type', 'during_hw_total_precip']):
        for event_type in events_data['compound_event_type'].unique():
            if pd.isna(event_type):
                continue
            
            subset = events_data[events_data['compound_event_type'] == event_type]
            if len(subset) > 0:
                color = COMPOUND_COLORS.get(event_type, 'gray')
                ax3.scatter(subset['duration_days'], subset['during_hw_total_precip'], 
                           alpha=0.6, label=event_type, color=color, s=20)
        
        ax3.set_xlabel('Duration (days)')
        ax3.set_ylabel('During-Heatwave Precipitation (mm)')
        ax3.set_title('Duration vs Precipitation by Event Type', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. Summary statistics table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    if 'compound_event_type' in events_data.columns:
        stats_data = [['Event Type', 'Count', 'Mean Duration', 'Mean Precip (mm)']]
        
        for event_type in events_data['compound_event_type'].unique():
            if pd.isna(event_type):
                continue
            
            subset = events_data[events_data['compound_event_type'] == event_type]
            if len(subset) > 0:
                count = len(subset)
                mean_duration = subset['duration_days'].mean() if 'duration_days' in subset.columns else np.nan
                mean_precip = subset['during_hw_total_precip'].mean() if 'during_hw_total_precip' in subset.columns else np.nan
                
                stats_data.append([
                    event_type.replace('_', ' ').title(),
                    f'{count:,}',
                    f'{mean_duration:.1f}' if not np.isnan(mean_duration) else 'N/A',
                    f'{mean_precip:.1f}' if not np.isnan(mean_precip) else 'N/A'
                ])
        
        if len(stats_data) > 1:
            table = ax4.table(cellText=stats_data[1:], colLabels=stats_data[0],
                             cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2)
            
            # Style the table
            for i in range(len(stats_data)):
                for j in range(len(stats_data[0])):
                    cell = table[(i, j)]
                    if i == 0:  # Header
                        cell.set_facecolor('#4472C4')
                        cell.set_text_props(weight='bold', color='white')
                    else:
                        event_type = stats_data[i][0].lower().replace(' ', '_')
                        color = COMPOUND_COLORS.get(event_type, 'white')
                        cell.set_facecolor(color)
                        cell.set_alpha(0.3)
    
    ax4.set_title('Compound Event Statistics', fontweight='bold', pad=20)
    
    plt.suptitle(f'Regional Compound Event Analysis\n'
                f'Total Events: {len(events_data):,}',
                fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    
    # Save
    filename = 'regional_compound_analysis.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {filename}")

def main():
    """Main visualization function for precipitation analysis."""
    parser = argparse.ArgumentParser(description='Visualize heatwave-precipitation interactions')
    
    parser.add_argument('--precip-dir', default='data/processed/precipitation_analysis',
                       help='Directory containing precipitation analysis files')
    parser.add_argument('--output-dir', default='visualizations/output/precipitation_analysis',
                       help='Output directory for plots')
    parser.add_argument('--years', nargs='+', type=int,
                       help='Specific years to analyze (default: all available)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("HEATWAVE-PRECIPITATION ANALYSIS VISUALIZATION")
    print("="*80)
    print(f"Input directory: {args.precip_dir}")
    print(f"Output directory: {args.output_dir}")
    print("="*80)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load data
        events_data, analysis_data, aggregation_data = load_precipitation_analysis_data(
            args.precip_dir, args.years
        )
        
        if events_data.empty and analysis_data.empty and not aggregation_data:
            print("No precipitation analysis data found!")
            return 1
        
        print("\nCreating visualizations...")
        
        # 1. Compound event patterns
        if aggregation_data and len(aggregation_data.data_vars) > 0:
            print("1. Compound event patterns...")
            plot_compound_event_patterns(aggregation_data, output_dir)
        
        # 2. Recovery analysis
        if aggregation_data and len(aggregation_data.data_vars) > 0:
            print("2. Precipitation recovery analysis...")
            plot_precipitation_recovery_analysis(aggregation_data, output_dir)
        
        # 3. Event-level analysis
        if not events_data.empty:
            print("3. Event-level precipitation analysis...")
            plot_event_precipitation_analysis(events_data, output_dir)
        
        # 4. Temporal trends
        if aggregation_data and 'year' in aggregation_data.dims:
            print("4. Temporal trends...")
            plot_temporal_precipitation_trends(aggregation_data, events_data, output_dir)
        
        # 5. Regional analysis
        if not events_data.empty:
            print("5. Regional compound analysis...")
            plot_regional_compound_analysis(events_data, output_dir)
        
        print("\n" + "="*80)
        print("PRECIPITATION ANALYSIS VISUALIZATION COMPLETED!")
        print("="*80)
        print(f"Output files saved in: {output_dir}")
        print("\nGenerated plots:")
        for plot_file in output_dir.glob('*.png'):
            print(f"  - {plot_file.name}")
        
        # Close datasets
        if aggregation_data:
            aggregation_data.close()
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
