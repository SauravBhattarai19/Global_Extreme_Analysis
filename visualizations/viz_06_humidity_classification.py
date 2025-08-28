#!/usr/bin/env python3
"""
Visualization of Humidity Classification (Output from 06_Humidity_Classification.py)

Creates comprehensive scientific visualizations of humidity-classified heatwaves:
- Global maps of humid vs dry heatwave patterns
- Seasonal and regional analysis of humidity categories
- Event-level humidity statistics and distributions
- Comparison between absolute and percentile methods
- Trends in heatwave humidity characteristics

Input files:
- heatwave_events_humidity_{year}.parquet (events with humidity stats)
- heatwave_days_humidity_{year}.parquet (days with humidity classes)
- heatwave_humidity_aggregations_{year}.nc (annual spatial summaries)
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

# Humidity category colors
HUMIDITY_COLORS = {
    'dry-hot': '#D2691E',      # Saddle Brown
    'humid-hot': '#4682B4',    # Steel Blue
    'mixed-hot': '#9370DB',    # Medium Purple
    'insufficient-RH': '#808080'  # Gray
}

def load_humidity_data(humidity_dir, years=None, variables=['tmax', 'tmin']):
    """Load humidity classification data."""
    humidity_dir = Path(humidity_dir)
    
    if years is None:
        # Find available years
        event_files = list(humidity_dir.glob('heatwave_events_humidity_*.parquet'))
        years = sorted(set([int(f.name.split('_')[3].split('.')[0]) for f in event_files]))
        print(f"Found data for years: {years[0]}-{years[-1]}")
    
    print(f"Loading humidity data for years {years}...")
    
    events_data = []
    days_data = []
    aggregation_data = {}
    
    for year in years:
        # Load event data
        events_file = humidity_dir / f'heatwave_events_humidity_{year}.parquet'
        if events_file.exists():
            events_df = pd.read_parquet(events_file)
            events_df['year'] = year
            events_data.append(events_df)
            print(f"  Loaded events: {events_file.name}")
        
        # Load day data
        days_file = humidity_dir / f'heatwave_days_humidity_{year}.parquet'
        if days_file.exists():
            days_df = pd.read_parquet(days_file)
            days_df['year'] = year
            days_data.append(days_df)
            print(f"  Loaded days: {days_file.name}")
        
        # Load aggregation data
        agg_file = humidity_dir / f'heatwave_humidity_aggregations_{year}.nc'
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
    combined_days = pd.concat(days_data, ignore_index=True) if days_data else pd.DataFrame()
    
    print(f"Combined events: {len(combined_events):,} records")
    print(f"Combined days: {len(combined_days):,} records")
    print(f"Aggregation years: {aggregation_data.year.values if 'year' in aggregation_data.dims else 'None'}")
    
    return combined_events, combined_days, aggregation_data

def plot_global_humidity_patterns(aggregation_data, output_dir, variables=['tmax', 'tmin']):
    """Create global maps of humidity-classified heatwave patterns."""
    output_dir = Path(output_dir)
    
    for var in variables:
        # Check which humidity variables are available
        var_suffix = 'day' if var == 'tmax' else 'night'
        
        humid_var = f'HWN_{var_suffix}_humid'
        dry_var = f'HWN_{var_suffix}_dry'
        mixed_var = f'HWN_{var_suffix}_mixed'
        
        if humid_var not in aggregation_data.data_vars:
            print(f"Variables for {var} not found in aggregation data")
            continue
        
        # Calculate multi-year means
        humid_mean = aggregation_data[humid_var].mean(dim='year')
        dry_mean = aggregation_data[dry_var].mean(dim='year')
        mixed_mean = aggregation_data[mixed_var].mean(dim='year')
        total_events = humid_mean + dry_mean + mixed_mean
        
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Humid heatwaves
        ax1 = plt.subplot(2, 2, 1, projection=ccrs.Robinson())
        ax1.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax1.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax1.set_global()
        
        levels = np.arange(0, 6, 0.5)
        cmap = plt.cm.Blues
        
        im1 = ax1.contourf(humid_mean.longitude, humid_mean.latitude, humid_mean,
                          levels=levels, cmap=cmap, transform=ccrs.PlateCarree(),
                          extend='max')
        
        ax1.set_title(f'Humid Heatwave Frequency ({var.upper()})\n'
                     f'Period: {aggregation_data.year.min().values}-{aggregation_data.year.max().values}',
                     fontsize=12, fontweight='bold')
        
        cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
        cbar1.set_label('Events per year', fontsize=10)
        
        # 2. Dry heatwaves
        ax2 = plt.subplot(2, 2, 2, projection=ccrs.Robinson())
        ax2.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax2.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax2.set_global()
        
        cmap2 = plt.cm.Oranges
        
        im2 = ax2.contourf(dry_mean.longitude, dry_mean.latitude, dry_mean,
                          levels=levels, cmap=cmap2, transform=ccrs.PlateCarree(),
                          extend='max')
        
        ax2.set_title(f'Dry Heatwave Frequency ({var.upper()})', fontsize=12, fontweight='bold')
        
        cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
        cbar2.set_label('Events per year', fontsize=10)
        
        # 3. Dominant humidity type
        ax3 = plt.subplot(2, 2, 3, projection=ccrs.Robinson())
        ax3.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax3.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax3.set_global()
        
        # Calculate dominant type (where total events > 0.5/year)
        significant_mask = total_events > 0.5
        
        # Create dominant type array
        dominant_type = np.full_like(humid_mean.values, 0, dtype=int)
        
        # Where significant heatwave activity exists
        mask = significant_mask.values
        humid_dom = (humid_mean >= dry_mean) & (humid_mean >= mixed_mean) & significant_mask
        dry_dom = (dry_mean >= humid_mean) & (dry_mean >= mixed_mean) & significant_mask
        mixed_dom = (mixed_mean >= humid_mean) & (mixed_mean >= dry_mean) & significant_mask
        
        dominant_type[humid_dom.values] = 1  # Humid
        dominant_type[dry_dom.values] = 2    # Dry
        dominant_type[mixed_dom.values] = 3  # Mixed
        
        # Create custom colormap
        colors = ['white', HUMIDITY_COLORS['humid-hot'], HUMIDITY_COLORS['dry-hot'], 
                 HUMIDITY_COLORS['mixed-hot']]
        cmap3 = mcolors.ListedColormap(colors[1:])  # Exclude white
        
        im3 = ax3.imshow(dominant_type, cmap=cmap3, vmin=1, vmax=3,
                        extent=[aggregation_data.longitude.min(), aggregation_data.longitude.max(),
                               aggregation_data.latitude.min(), aggregation_data.latitude.max()],
                        transform=ccrs.PlateCarree())
        
        ax3.set_title(f'Dominant Heatwave Type ({var.upper()})', fontsize=12, fontweight='bold')
        
        cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.8, ticks=[1, 2, 3])
        cbar3.set_ticklabels(['Humid', 'Dry', 'Mixed'])
        
        # 4. Humidity ratio (humid / (humid + dry))
        ax4 = plt.subplot(2, 2, 4, projection=ccrs.Robinson())
        ax4.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax4.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax4.set_global()
        
        # Calculate humidity ratio
        humidity_ratio = humid_mean / (humid_mean + dry_mean + 0.1)  # Add small value to avoid division by zero
        
        levels_ratio = np.linspace(0, 1, 11)
        cmap4 = plt.cm.RdYlBu
        
        im4 = ax4.contourf(humidity_ratio.longitude, humidity_ratio.latitude, humidity_ratio,
                          levels=levels_ratio, cmap=cmap4, transform=ccrs.PlateCarree(),
                          extend='both')
        
        ax4.set_title(f'Humidity Ratio ({var.upper()})\nHumid / (Humid + Dry)', 
                     fontsize=12, fontweight='bold')
        
        cbar4 = plt.colorbar(im4, ax=ax4, shrink=0.8)
        cbar4.set_label('Ratio (0=Dry, 1=Humid)', fontsize=10)
        
        plt.suptitle(f'Heatwave Humidity Patterns - {var.upper()}\n'
                    f'Period: {aggregation_data.year.min().values}-{aggregation_data.year.max().values}',
                    fontsize=16, fontweight='bold', y=0.95)
        
        plt.tight_layout()
        
        # Save
        filename = f'humidity_patterns_{var}.png'
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {filename}")

def plot_humidity_event_analysis(events_data, output_dir):
    """Analyze humidity characteristics of individual events."""
    output_dir = Path(output_dir)
    
    if events_data.empty:
        print("No event data available for humidity analysis")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Humidity category distribution
    ax1 = axes[0, 0]
    
    # Combine day and night labels
    all_labels = []
    if 'label_day' in events_data.columns:
        day_labels = events_data['label_day'].dropna()
        all_labels.extend(day_labels.tolist())
    if 'label_night' in events_data.columns:
        night_labels = events_data['label_night'].dropna()
        all_labels.extend(night_labels.tolist())
    
    if all_labels:
        label_counts = pd.Series(all_labels).value_counts()
        colors = [HUMIDITY_COLORS.get(label, 'gray') for label in label_counts.index]
        
        wedges, texts, autotexts = ax1.pie(label_counts.values, labels=label_counts.index, 
                                          colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Heatwave Humidity Category Distribution', fontweight='bold')
    
    # 2. Humidity vs Duration
    ax2 = axes[0, 1]
    
    if 'label_day' in events_data.columns and 'duration_days' in events_data.columns:
        for label in ['dry-hot', 'humid-hot', 'mixed-hot']:
            if label in events_data['label_day'].values:
                subset = events_data[events_data['label_day'] == label]
                durations = subset['duration_days']
                ax2.hist(durations, bins=range(1, 21), alpha=0.7, label=label, 
                        color=HUMIDITY_COLORS[label], density=True)
        
        ax2.set_xlabel('Duration (days)')
        ax2.set_ylabel('Density')
        ax2.set_title('Duration by Humidity Category', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. Seasonal humidity patterns
    ax3 = axes[1, 0]
    
    if 'year_start' in events_data.columns and 'label_day' in events_data.columns:
        events_data['start_month'] = pd.to_datetime(events_data['year_start']).dt.month
        
        # Monthly distribution by humidity type
        months = range(1, 13)
        month_labels = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
        
        for label in ['dry-hot', 'humid-hot', 'mixed-hot']:
            if label in events_data['label_day'].values:
                subset = events_data[events_data['label_day'] == label]
                monthly_counts = subset['start_month'].value_counts().sort_index()
                counts = [monthly_counts.get(m, 0) for m in months]
                
                ax3.plot(months, counts, 'o-', color=HUMIDITY_COLORS[label], 
                        label=label, linewidth=2, markersize=6)
        
        ax3.set_xlabel('Month')
        ax3.set_ylabel('Number of Events')
        ax3.set_title('Seasonal Humidity Patterns', fontweight='bold')
        ax3.set_xticks(months)
        ax3.set_xticklabels(month_labels)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. Humidity statistics table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate statistics by humidity category
    stats_data = [['Category', 'Count', 'Mean Duration', 'Mean RH (%)', 'Max Duration']]
    
    if 'label_day' in events_data.columns:
        for label in ['dry-hot', 'humid-hot', 'mixed-hot', 'insufficient-RH']:
            subset = events_data[events_data['label_day'] == label]
            if len(subset) > 0:
                count = len(subset)
                mean_duration = subset['duration_days'].mean() if 'duration_days' in subset.columns else np.nan
                max_duration = subset['duration_days'].max() if 'duration_days' in subset.columns else np.nan
                
                # Try to get RH statistics
                rh_col = 'mean_RH_day' if 'mean_RH_day' in subset.columns else None
                mean_rh = subset[rh_col].mean() if rh_col and not subset[rh_col].isna().all() else np.nan
                
                stats_data.append([
                    label.replace('-', ' ').title(),
                    f'{count:,}',
                    f'{mean_duration:.1f}' if not np.isnan(mean_duration) else 'N/A',
                    f'{mean_rh:.1f}' if not np.isnan(mean_rh) else 'N/A',
                    f'{max_duration:.0f}' if not np.isnan(max_duration) else 'N/A'
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
                    # Color by category
                    category = stats_data[i][0].lower().replace(' ', '-')
                    color = HUMIDITY_COLORS.get(category, 'white')
                    cell.set_facecolor(color)
                    cell.set_alpha(0.3)
    
    ax4.set_title('Humidity Category Statistics', fontweight='bold', pad=20)
    
    plt.suptitle(f'Heatwave Humidity Event Analysis\n'
                f'Total Events: {len(events_data):,}',
                fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    
    # Save
    filename = 'humidity_event_analysis.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {filename}")

def plot_humidity_trends(aggregation_data, output_dir, variables=['tmax', 'tmin']):
    """Plot temporal trends in humidity-classified heatwaves."""
    output_dir = Path(output_dir)
    
    if 'year' not in aggregation_data.dims or len(aggregation_data.year) < 3:
        print("Insufficient years for trend analysis")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    years = aggregation_data.year.values
    
    for i, var in enumerate(variables):
        var_suffix = 'day' if var == 'tmax' else 'night'
        
        humid_var = f'HWN_{var_suffix}_humid'
        dry_var = f'HWN_{var_suffix}_dry'
        mixed_var = f'HWN_{var_suffix}_mixed'
        
        if humid_var not in aggregation_data.data_vars:
            continue
        
        # Calculate global means
        humid_global = aggregation_data[humid_var].mean(dim=['latitude', 'longitude'])
        dry_global = aggregation_data[dry_var].mean(dim=['latitude', 'longitude'])
        mixed_global = aggregation_data[mixed_var].mean(dim=['latitude', 'longitude'])
        total_global = humid_global + dry_global + mixed_global
        
        # Plot absolute trends
        ax1 = axes[i, 0]
        
        ax1.plot(years, humid_global, 'o-', color=HUMIDITY_COLORS['humid-hot'], 
                label='Humid', linewidth=2, markersize=6)
        ax1.plot(years, dry_global, 'o-', color=HUMIDITY_COLORS['dry-hot'], 
                label='Dry', linewidth=2, markersize=6)
        ax1.plot(years, mixed_global, 'o-', color=HUMIDITY_COLORS['mixed-hot'], 
                label='Mixed', linewidth=2, markersize=6)
        
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Global Mean Events/Year')
        ax1.set_title(f'Humidity Category Trends ({var.upper()})', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot relative trends (percentages)
        ax2 = axes[i, 1]
        
        humid_pct = humid_global / (total_global + 0.001) * 100  # Avoid division by zero
        dry_pct = dry_global / (total_global + 0.001) * 100
        mixed_pct = mixed_global / (total_global + 0.001) * 100
        
        ax2.plot(years, humid_pct, 'o-', color=HUMIDITY_COLORS['humid-hot'], 
                label='Humid', linewidth=2, markersize=6)
        ax2.plot(years, dry_pct, 'o-', color=HUMIDITY_COLORS['dry-hot'], 
                label='Dry', linewidth=2, markersize=6)
        ax2.plot(years, mixed_pct, 'o-', color=HUMIDITY_COLORS['mixed-hot'], 
                label='Mixed', linewidth=2, markersize=6)
        
        ax2.set_xlabel('Year')
        ax2.set_ylabel('Percentage of Total Events (%)')
        ax2.set_title(f'Humidity Category Percentages ({var.upper()})', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)
    
    plt.suptitle(f'Humidity-Classified Heatwave Trends\n'
                f'Period: {years[0]}-{years[-1]}',
                fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    
    # Save
    filename = 'humidity_trends.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {filename}")

def plot_regional_humidity_comparison(events_data, output_dir):
    """Compare humidity patterns across different regions."""
    output_dir = Path(output_dir)
    
    if events_data.empty or 'grid_y' not in events_data.columns:
        print("Insufficient data for regional analysis")
        return
    
    # Define regions based on latitude (simplified)
    regions = {
        'Tropical': {'lat_range': (-23.5, 23.5), 'color': 'green'},
        'Northern Subtropics': {'lat_range': (23.5, 40), 'color': 'orange'},
        'Northern Mid-Latitudes': {'lat_range': (40, 60), 'color': 'red'},
        'Southern Subtropics': {'lat_range': (-40, -23.5), 'color': 'blue'},
        'Southern Mid-Latitudes': {'lat_range': (-60, -40), 'color': 'purple'}
    }
    
    # Assuming grid_y corresponds to latitude index (need to convert)
    # This is a simplification - in practice, you'd need the actual lat/lon coordinates
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # For this example, we'll use a simplified regional analysis
    # based on grid_y values (assuming they correlate with latitude)
    
    # 1. Humidity distribution by variable
    ax1 = axes[0, 0]
    
    if 'var' in events_data.columns and 'label_day' in events_data.columns:
        # Count events by variable and humidity type
        var_humidity = events_data.groupby(['var', 'label_day']).size().unstack(fill_value=0)
        
        if not var_humidity.empty:
            var_humidity.plot(kind='bar', ax=ax1, 
                             color=[HUMIDITY_COLORS.get(col, 'gray') for col in var_humidity.columns])
            ax1.set_xlabel('Variable')
            ax1.set_ylabel('Number of Events')
            ax1.set_title('Humidity Distribution by Variable', fontweight='bold')
            ax1.legend(title='Humidity Type')
            plt.setp(ax1.get_xticklabels(), rotation=0)
    
    # 2. Seasonal humidity patterns
    ax2 = axes[0, 1]
    
    if 'year_start' in events_data.columns and 'label_day' in events_data.columns:
        events_data['start_month'] = pd.to_datetime(events_data['year_start']).dt.month
        
        # Create seasonal humidity distribution
        seasonal_humidity = events_data.groupby(['start_month', 'label_day']).size().unstack(fill_value=0)
        
        if not seasonal_humidity.empty:
            # Normalize to percentages
            seasonal_humidity_pct = seasonal_humidity.div(seasonal_humidity.sum(axis=1), axis=0) * 100
            
            months = range(1, 13)
            month_labels = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
            
            bottom = np.zeros(len(months))
            for humidity_type in seasonal_humidity_pct.columns:
                values = [seasonal_humidity_pct.loc[m, humidity_type] if m in seasonal_humidity_pct.index else 0 
                         for m in months]
                ax2.bar(months, values, bottom=bottom, label=humidity_type,
                       color=HUMIDITY_COLORS.get(humidity_type, 'gray'), alpha=0.8)
                bottom += values
            
            ax2.set_xlabel('Month')
            ax2.set_ylabel('Percentage (%)')
            ax2.set_title('Seasonal Humidity Distribution', fontweight='bold')
            ax2.set_xticks(months)
            ax2.set_xticklabels(month_labels)
            ax2.legend()
    
    # 3. Duration vs humidity intensity
    ax3 = axes[1, 0]
    
    if all(col in events_data.columns for col in ['duration_days', 'label_day', 'mean_RH_day']):
        for label in ['dry-hot', 'humid-hot', 'mixed-hot']:
            subset = events_data[events_data['label_day'] == label]
            if len(subset) > 0 and not subset['mean_RH_day'].isna().all():
                ax3.scatter(subset['duration_days'], subset['mean_RH_day'], 
                           alpha=0.6, label=label, color=HUMIDITY_COLORS[label], s=20)
        
        ax3.set_xlabel('Duration (days)')
        ax3.set_ylabel('Mean RH (%)')
        ax3.set_title('Duration vs Humidity Intensity', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. Annual humidity statistics
    ax4 = axes[1, 1]
    
    if 'year' in events_data.columns and 'label_day' in events_data.columns:
        annual_humidity = events_data.groupby(['year', 'label_day']).size().unstack(fill_value=0)
        
        if not annual_humidity.empty and len(annual_humidity) > 1:
            # Plot trends for each humidity type
            for humidity_type in annual_humidity.columns:
                ax4.plot(annual_humidity.index, annual_humidity[humidity_type], 
                        'o-', color=HUMIDITY_COLORS.get(humidity_type, 'gray'),
                        label=humidity_type, linewidth=2, markersize=6)
            
            ax4.set_xlabel('Year')
            ax4.set_ylabel('Number of Events')
            ax4.set_title('Annual Humidity Trends', fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Insufficient years\nfor trend analysis', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=14)
            ax4.set_title('Annual Humidity Analysis', fontweight='bold')
    
    plt.suptitle(f'Regional Humidity Analysis\n'
                f'Total Events: {len(events_data):,}',
                fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    
    # Save
    filename = 'humidity_regional_analysis.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {filename}")

def plot_humidity_day_analysis(days_data, output_dir):
    """Analyze day-level humidity classifications."""
    output_dir = Path(output_dir)
    
    if days_data.empty:
        print("No day-level data available for humidity analysis")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Daily humidity distribution
    ax1 = axes[0, 0]
    
    humidity_cols = ['humidity_class_day', 'humidity_class_night']
    all_classes = []
    
    for col in humidity_cols:
        if col in days_data.columns:
            classes = days_data[col].dropna()
            all_classes.extend(classes.tolist())
    
    if all_classes:
        class_counts = pd.Series(all_classes).value_counts()
        colors = ['#228B22' if 'humid' in cls else '#D2691E' if 'dry' in cls else '#9370DB' 
                 for cls in class_counts.index]
        
        ax1.bar(class_counts.index, class_counts.values, color=colors, alpha=0.7)
        ax1.set_xlabel('Humidity Class')
        ax1.set_ylabel('Number of Days')
        ax1.set_title('Daily Humidity Classification Distribution', fontweight='bold')
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
    
    # 2. RH value distributions
    ax2 = axes[0, 1]
    
    rh_cols = ['RH_day', 'RH_night']
    for i, col in enumerate(rh_cols):
        if col in days_data.columns:
            rh_values = days_data[col].dropna()
            if len(rh_values) > 0:
                ax2.hist(rh_values, bins=50, alpha=0.7, 
                        label=col.replace('_', ' ').title(), density=True)
    
    ax2.set_xlabel('Relative Humidity (%)')
    ax2.set_ylabel('Density')
    ax2.set_title('RH Value Distributions', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Temperature vs RH relationship
    ax3 = axes[1, 0]
    
    if 'temp' in days_data.columns and 'RH_day' in days_data.columns:
        # Convert temperature to Celsius if needed
        temps = days_data['temp'].values
        if np.nanmean(temps) > 100:  # Likely in Kelvin
            temps = temps - 273.15
        
        rh_vals = days_data['RH_day'].values
        
        # Remove NaN values
        mask = ~(np.isnan(temps) | np.isnan(rh_vals))
        if np.sum(mask) > 0:
            ax3.scatter(temps[mask], rh_vals[mask], alpha=0.5, s=1, c='blue')
            ax3.set_xlabel('Temperature (°C)')
            ax3.set_ylabel('Relative Humidity (%)')
            ax3.set_title('Temperature vs RH Relationship', fontweight='bold')
            ax3.grid(True, alpha=0.3)
    
    # 4. Temporal patterns
    ax4 = axes[1, 1]
    
    if 'date' in days_data.columns and 'humidity_class_day' in days_data.columns:
        # Convert date and extract month
        days_data['date'] = pd.to_datetime(days_data['date'])
        days_data['month'] = days_data['date'].dt.month
        
        monthly_humidity = days_data.groupby(['month', 'humidity_class_day']).size().unstack(fill_value=0)
        
        if not monthly_humidity.empty:
            # Normalize to percentages
            monthly_humidity_pct = monthly_humidity.div(monthly_humidity.sum(axis=1), axis=0) * 100
            
            months = range(1, 13)
            month_labels = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
            
            bottom = np.zeros(len(months))
            colors_map = {'humid': '#228B22', 'dry': '#D2691E', 'moderate': '#9370DB', 'missing': '#808080'}
            
            for humidity_class in monthly_humidity_pct.columns:
                values = [monthly_humidity_pct.loc[m, humidity_class] if m in monthly_humidity_pct.index else 0 
                         for m in months]
                color = colors_map.get(humidity_class, 'gray')
                ax4.bar(months, values, bottom=bottom, label=humidity_class,
                       color=color, alpha=0.8)
                bottom += values
            
            ax4.set_xlabel('Month')
            ax4.set_ylabel('Percentage (%)')
            ax4.set_title('Monthly Humidity Classification', fontweight='bold')
            ax4.set_xticks(months)
            ax4.set_xticklabels(month_labels)
            ax4.legend()
    
    plt.suptitle(f'Day-Level Humidity Analysis\n'
                f'Total Days: {len(days_data):,}',
                fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    
    # Save
    filename = 'humidity_day_analysis.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {filename}")

def main():
    """Main visualization function for humidity classification."""
    parser = argparse.ArgumentParser(description='Visualize humidity-classified heatwaves')
    
    parser.add_argument('--humidity-dir', default='data/processed/humidity_classification',
                       help='Directory containing humidity classification files')
    parser.add_argument('--output-dir', default='visualizations/output/humidity_classification',
                       help='Output directory for plots')
    parser.add_argument('--years', nargs='+', type=int,
                       help='Specific years to analyze (default: all available)')
    parser.add_argument('--variables', nargs='+', default=['tmax', 'tmin'],
                       help='Variables to analyze (default: tmax tmin)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("HUMIDITY CLASSIFICATION VISUALIZATION")
    print("="*80)
    print(f"Input directory: {args.humidity_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Variables: {args.variables}")
    print("="*80)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load data
        events_data, days_data, aggregation_data = load_humidity_data(
            args.humidity_dir, args.years, args.variables
        )
        
        if events_data.empty and days_data.empty and not aggregation_data:
            print("No humidity classification data found!")
            return 1
        
        print("\nCreating visualizations...")
        
        # 1. Global humidity patterns
        if aggregation_data and len(aggregation_data.data_vars) > 0:
            print("1. Global humidity patterns...")
            plot_global_humidity_patterns(aggregation_data, output_dir, args.variables)
        
        # 2. Event-level analysis
        if not events_data.empty:
            print("2. Event-level humidity analysis...")
            plot_humidity_event_analysis(events_data, output_dir)
        
        # 3. Temporal trends
        if aggregation_data and 'year' in aggregation_data.dims:
            print("3. Humidity trends...")
            plot_humidity_trends(aggregation_data, output_dir, args.variables)
        
        # 4. Regional comparison
        if not events_data.empty:
            print("4. Regional humidity comparison...")
            plot_regional_humidity_comparison(events_data, output_dir)
        
        # 5. Day-level analysis
        if not days_data.empty:
            print("5. Day-level humidity analysis...")
            plot_humidity_day_analysis(days_data, output_dir)
        
        print("\n" + "="*80)
        print("HUMIDITY CLASSIFICATION VISUALIZATION COMPLETED!")
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
