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
try:
    import regionmask
    HAS_REGIONMASK = True
except ImportError:
    HAS_REGIONMASK = False
    print("Note: regionmask not available - using simplified land/ocean mask")
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def create_land_ocean_mask(ds, mask_type='both'):
    """
    Create accurate land/ocean mask using regionmask (if available) or cartopy.
    
    Args:
        ds: xarray Dataset with latitude/longitude coordinates
        mask_type: 'land', 'ocean', or 'both'
    
    Returns:
        mask: Boolean array where True = include pixel, False = exclude
    """
    if mask_type == 'both':
        return np.ones((len(ds.latitude), len(ds.longitude)), dtype=bool)
    
    lat = ds.latitude.values
    lon = ds.longitude.values
    
    print(f"Creating {mask_type} mask...")
    print(f"  Grid: {len(lat)} x {len(lon)} pixels")
    
    if HAS_REGIONMASK:
        try:
            # Use regionmask with Natural Earth land boundaries
            print("  Using regionmask with Natural Earth boundaries...")
            
            # Create a dummy dataset with coordinates named as regionmask expects
            dummy_ds = xr.Dataset(coords={'lat': lat, 'lon': lon})
            
            # Get Natural Earth land boundaries
            land_mask_rm = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(dummy_ds)
            
            # Convert to boolean (regionmask returns integers)
            land_mask = ~np.isnan(land_mask_rm.values)
            
            print(f"  Land pixels: {np.sum(land_mask):,} ({np.sum(land_mask)/land_mask.size*100:.1f}%)")
            print(f"  Ocean pixels: {np.sum(~land_mask):,} ({np.sum(~land_mask)/land_mask.size*100:.1f}%)")
            
            if mask_type == 'land':
                return land_mask
            elif mask_type == 'ocean':
                return ~land_mask
            else:
                return np.ones_like(land_mask, dtype=bool)
                
        except Exception as e:
            print(f"  regionmask failed: {e}")
            print("  Falling back to cartopy...")
    
    # Fallback: Use cartopy's built-in land feature
    print("  Using cartopy Natural Earth features...")
    
    try:
        # Use cartopy's natural earth land feature
        import cartopy.io.shapereader as shpreader
        
        # Get land geometries
        land_shp = shpreader.natural_earth(resolution='50m', category='physical', name='land')
        land_geoms = list(shpreader.Reader(land_shp).geometries())
        
        # Create mask efficiently
        land_mask = np.zeros((len(lat), len(lon)), dtype=bool)
        
        # Create coordinate meshgrid
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        
        # Vectorized point-in-polygon test (more efficient)
        from shapely.geometry import Point
        from shapely.prepared import prep
        
        # Prepare geometries for faster intersection
        prepared_geoms = [prep(geom) for geom in land_geoms]
        
        # Check each point
        for i in range(len(lat)):
            for j in range(len(lon)):
                point = Point(lon[j], lat[i])
                for prep_geom in prepared_geoms:
                    if prep_geom.contains(point):
                        land_mask[i, j] = True
                        break
        
        print(f"  Land pixels: {np.sum(land_mask):,} ({np.sum(land_mask)/land_mask.size*100:.1f}%)")
        print(f"  Ocean pixels: {np.sum(~land_mask):,} ({np.sum(~land_mask)/land_mask.size*100:.1f}%)")
        
        if mask_type == 'land':
            return land_mask
        elif mask_type == 'ocean':
            return ~land_mask
        else:
            return np.ones_like(land_mask, dtype=bool)
            
    except Exception as e:
        print(f"  Error creating land mask: {e}")
        print("  Using simplified approximation...")
        
        # Last resort: very simplified land/ocean approximation
        lat_grid, lon_grid = np.meshgrid(lat, lon, indexing='ij')
        
        # Very basic land approximation (major continents only)
        land_mask = np.zeros_like(lat_grid, dtype=bool)
        
        # Continents (very rough)
        land_mask |= (lat_grid >= -60) & (lat_grid <= 75) & (lon_grid >= -170) & (lon_grid <= -30)  # Americas
        land_mask |= (lat_grid >= -35) & (lat_grid <= 75) & (lon_grid >= -20) & (lon_grid <= 180)   # Eurasia+Africa
        land_mask |= (lat_grid >= -50) & (lat_grid <= -10) & (lon_grid >= 110) & (lon_grid <= 180)  # Australia
        
        if mask_type == 'land':
            return land_mask
        elif mask_type == 'ocean':
            return ~land_mask
        else:
            return np.ones_like(land_mask, dtype=bool)

def apply_mask_and_adjust_colorbar(data, mask, percentile_range=(0.5, 99.5)):
    """
    Apply mask to data and calculate appropriate colorbar range.
    
    Args:
        data: numpy array of data values
        mask: boolean mask (True = include)
        percentile_range: tuple of (low, high) percentiles for colorbar
    
    Returns:
        masked_data: data with mask applied (NaN where mask is False)
        vmin, vmax: colorbar range values
        valid_count: number of valid pixels
    """
    # Apply mask
    masked_data = data.copy()
    masked_data[~mask] = np.nan
    
    # Calculate colorbar range from masked data
    valid_data = masked_data[~np.isnan(masked_data)]
    
    if len(valid_data) == 0:
        return masked_data, 0, 1, 0
    
    vmin = np.percentile(valid_data, percentile_range[0])
    vmax = np.percentile(valid_data, percentile_range[1])
    
    return masked_data, vmin, vmax, len(valid_data)

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

def plot_global_humidity_patterns(aggregation_data, output_dir, mask_type='both', variables=['tmax', 'tmin']):
    """Create global maps of humidity-classified heatwave patterns."""
    output_dir = Path(output_dir)
    
    # Create land/ocean mask
    mask = create_land_ocean_mask(aggregation_data, mask_type)
    
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
        
        # Apply mask and calculate levels
        humid_masked, humid_vmin, humid_vmax, humid_count = apply_mask_and_adjust_colorbar(
            humid_mean.values, mask, percentile_range=(0.5, 99.5)
        )
        
        levels = np.linspace(humid_vmin, humid_vmax, 20)
        cmap = plt.cm.Blues
        
        im1 = ax1.contourf(humid_mean.longitude, humid_mean.latitude, humid_masked,
                          levels=levels, cmap=cmap, transform=ccrs.PlateCarree(),
                          extend='max')
        
        ax1.set_title(f'Humid Heatwave Frequency ({var.upper()}, {mask_type.capitalize()})\n'
                     f'Period: {aggregation_data.year.min().values}-{aggregation_data.year.max().values}\n'
                     f'({humid_count:,} valid pixels)',
                     fontsize=12, fontweight='bold')
        
        cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
        cbar1.set_label('Events per year', fontsize=10)
        
        # 2. Dry heatwaves
        ax2 = plt.subplot(2, 2, 2, projection=ccrs.Robinson())
        ax2.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax2.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax2.set_global()
        
        # Apply mask to dry data
        dry_masked, dry_vmin, dry_vmax, dry_count = apply_mask_and_adjust_colorbar(
            dry_mean.values, mask, percentile_range=(0.5, 99.5)
        )
        
        levels_dry = np.linspace(dry_vmin, dry_vmax, 20)
        cmap2 = plt.cm.Oranges
        
        im2 = ax2.contourf(dry_mean.longitude, dry_mean.latitude, dry_masked,
                          levels=levels_dry, cmap=cmap2, transform=ccrs.PlateCarree(),
                          extend='max')
        
        ax2.set_title(f'Dry Heatwave Frequency ({var.upper()}, {mask_type.capitalize()})\n'
                     f'({dry_count:,} valid pixels)', 
                     fontsize=12, fontweight='bold')
        
        cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
        cbar2.set_label('Events per year', fontsize=10)
        
        # 3. Dominant humidity type
        ax3 = plt.subplot(2, 2, 3, projection=ccrs.Robinson())
        ax3.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax3.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax3.set_global()
        
        # Calculate dominant type (where total events > 0.5/year)
        total_masked, _, _, _ = apply_mask_and_adjust_colorbar(
            total_events.values, mask, percentile_range=(0.5, 99.5)
        )
        significant_mask = (total_masked > 0.5) & (~np.isnan(total_masked))
        
        # Create dominant type array
        dominant_type = np.zeros_like(humid_mean.values, dtype=int)
        
        # Apply masking first
        humid_masked_dom = humid_mean.values.copy()
        dry_masked_dom = dry_mean.values.copy()
        mixed_masked_dom = mixed_mean.values.copy()
        humid_masked_dom[~mask] = np.nan
        dry_masked_dom[~mask] = np.nan
        mixed_masked_dom[~mask] = np.nan
        
        # Where significant heatwave activity exists
        humid_dom = (humid_masked_dom >= dry_masked_dom) & (humid_masked_dom >= mixed_masked_dom) & significant_mask
        dry_dom = (dry_masked_dom >= humid_masked_dom) & (dry_masked_dom >= mixed_masked_dom) & significant_mask
        mixed_dom = (mixed_masked_dom >= humid_masked_dom) & (mixed_masked_dom >= dry_masked_dom) & significant_mask
        
        dominant_type[humid_dom] = 1  # Humid
        dominant_type[dry_dom] = 2    # Dry
        dominant_type[mixed_dom] = 3  # Mixed
        
        # Create custom colormap
        colors = ['white', HUMIDITY_COLORS['humid-hot'], HUMIDITY_COLORS['dry-hot'], 
                 HUMIDITY_COLORS['mixed-hot']]
        cmap3 = mcolors.ListedColormap(colors[1:])  # Exclude white
        
        im3 = ax3.imshow(dominant_type, cmap=cmap3, vmin=1, vmax=3,
                        extent=[aggregation_data.longitude.min(), aggregation_data.longitude.max(),
                               aggregation_data.latitude.min(), aggregation_data.latitude.max()],
                        transform=ccrs.PlateCarree())
        
        ax3.set_title(f'Dominant Heatwave Type ({var.upper()}, {mask_type.capitalize()})', 
                     fontsize=12, fontweight='bold')
        
        cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.8, ticks=[1, 2, 3])
        cbar3.set_ticklabels(['Humid', 'Dry', 'Mixed'])
        
        # 4. Humidity ratio (humid / (humid + dry))
        ax4 = plt.subplot(2, 2, 4, projection=ccrs.Robinson())
        ax4.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax4.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax4.set_global()
        
        # Calculate humidity ratio with masking
        ratio_denominator = humid_mean + dry_mean + 0.1  # Add small value to avoid division by zero
        humidity_ratio = humid_mean / ratio_denominator
        
        # Apply mask to ratio data
        ratio_masked, ratio_vmin, ratio_vmax, ratio_count = apply_mask_and_adjust_colorbar(
            humidity_ratio.values, mask, percentile_range=(0.5, 99.5)
        )
        
        levels_ratio = np.linspace(0, 1, 11)
        cmap4 = plt.cm.RdYlBu
        
        im4 = ax4.contourf(humidity_ratio.longitude, humidity_ratio.latitude, ratio_masked,
                          levels=levels_ratio, cmap=cmap4, transform=ccrs.PlateCarree(),
                          extend='both')
        
        ax4.set_title(f'Humidity Ratio ({var.upper()}, {mask_type.capitalize()})\nHumid / (Humid + Dry)\n'
                     f'({ratio_count:,} valid pixels)', 
                     fontsize=12, fontweight='bold')
        
        cbar4 = plt.colorbar(im4, ax=ax4, shrink=0.8)
        cbar4.set_label('Ratio (0=Dry, 1=Humid)', fontsize=10)
        
        plt.suptitle(f'Heatwave Humidity Patterns - {var.upper()} ({mask_type.capitalize()})\n'
                    f'Period: {aggregation_data.year.min().values}-{aggregation_data.year.max().values}',
                    fontsize=16, fontweight='bold', y=0.95)
        
        plt.tight_layout()
        
        # Save
        filename = f'humidity_patterns_{var}_{mask_type}.png'
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {filename}")

def plot_humidity_event_analysis(events_data, output_dir, mask_type='both'):
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
        ax1.set_title(f'Heatwave Humidity Category Distribution ({mask_type.capitalize()})', fontweight='bold')
    
    # 2. Humidity vs Duration
    ax2 = axes[0, 1]
    
    if 'label_day' in events_data.columns and 'duration_days' in events_data.columns:
        for label in ['dry-hot', 'humid-hot', 'mixed-hot']:
            if label in events_data['label_day'].values:
                subset = events_data[events_data['label_day'] == label]
                durations = subset['duration_days']
                if len(durations) > 0:
                    ax2.hist(durations, bins=range(1, 21), alpha=0.7, label=label, 
                            color=HUMIDITY_COLORS[label], density=True)
        
        ax2.set_xlabel('Duration (days)')
        ax2.set_ylabel('Density')
        ax2.set_title(f'Duration by Humidity Category ({mask_type.capitalize()})', fontweight='bold')
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
                if len(subset) > 0:
                    monthly_counts = subset['start_month'].value_counts().sort_index()
                    counts = [monthly_counts.get(m, 0) for m in months]
                    
                    ax3.plot(months, counts, 'o-', color=HUMIDITY_COLORS[label], 
                            label=label, linewidth=2, markersize=6)
        
        ax3.set_xlabel('Month')
        ax3.set_ylabel('Number of Events')
        ax3.set_title(f'Seasonal Humidity Patterns ({mask_type.capitalize()})', fontweight='bold')
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
    
    plt.suptitle(f'Heatwave Humidity Event Analysis ({mask_type.capitalize()})\n'
                f'Total Events: {len(events_data):,}',
                fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    
    # Save
    filename = f'humidity_event_analysis_{mask_type}.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {filename}")

def plot_humidity_trends(aggregation_data, output_dir, mask_type='both', variables=['tmax', 'tmin']):
    """Plot temporal trends in humidity-classified heatwaves."""
    output_dir = Path(output_dir)
    
    if 'year' not in aggregation_data.dims or len(aggregation_data.year) < 3:
        print("Insufficient years for trend analysis")
        return
    
    # Create land/ocean mask
    mask = create_land_ocean_mask(aggregation_data, mask_type)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    years = aggregation_data.year.values
    
    for i, var in enumerate(variables):
        var_suffix = 'day' if var == 'tmax' else 'night'
        
        humid_var = f'HWN_{var_suffix}_humid'
        dry_var = f'HWN_{var_suffix}_dry'
        mixed_var = f'HWN_{var_suffix}_mixed'
        
        if humid_var not in aggregation_data.data_vars:
            continue
        
        # Calculate global means with masking
        humid_global = []
        dry_global = []
        mixed_global = []
        
        for year in years:
            # Apply mask to each year's data
            humid_year = aggregation_data[humid_var].sel(year=year).values
            dry_year = aggregation_data[dry_var].sel(year=year).values
            mixed_year = aggregation_data[mixed_var].sel(year=year).values
            
            humid_masked = humid_year.copy()
            dry_masked = dry_year.copy()
            mixed_masked = mixed_year.copy()
            
            humid_masked[~mask] = np.nan
            dry_masked[~mask] = np.nan
            mixed_masked[~mask] = np.nan
            
            humid_global.append(np.nanmean(humid_masked))
            dry_global.append(np.nanmean(dry_masked))
            mixed_global.append(np.nanmean(mixed_masked))
        
        humid_global = np.array(humid_global)
        dry_global = np.array(dry_global)
        mixed_global = np.array(mixed_global)
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
        ax1.set_ylabel('Mean Events/Year')
        ax1.set_title(f'Humidity Category Trends ({var.upper()}, {mask_type.capitalize()})', fontweight='bold')
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
        ax2.set_title(f'Humidity Category Percentages ({var.upper()}, {mask_type.capitalize()})', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)
    
    plt.suptitle(f'Humidity-Classified Heatwave Trends ({mask_type.capitalize()})\n'
                f'Period: {years[0]}-{years[-1]}',
                fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    
    # Save
    filename = f'humidity_trends_{mask_type}.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {filename}")

def plot_regional_humidity_comparison(events_data, output_dir, mask_type='both'):
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
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
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
            ax1.set_title(f'Humidity Distribution by Variable ({mask_type.capitalize()})', fontweight='bold')
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
            ax2.set_title(f'Seasonal Humidity Distribution ({mask_type.capitalize()})', fontweight='bold')
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
        ax3.set_title(f'Duration vs Humidity Intensity ({mask_type.capitalize()})', fontweight='bold')
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
            ax4.set_title(f'Annual Humidity Trends ({mask_type.capitalize()})', fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Insufficient years\nfor trend analysis', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=14)
            ax4.set_title(f'Annual Humidity Analysis ({mask_type.capitalize()})', fontweight='bold')
    
    plt.suptitle(f'Regional Humidity Analysis ({mask_type.capitalize()})\n'
                f'Total Events: {len(events_data):,}',
                fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    
    # Save
    filename = f'humidity_regional_analysis_{mask_type}.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {filename}")

def plot_humidity_day_analysis(days_data, output_dir, mask_type='both'):
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
        ax1.set_title(f'Daily Humidity Classification Distribution ({mask_type.capitalize()})', fontweight='bold')
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
    ax2.set_title(f'RH Value Distributions ({mask_type.capitalize()})', fontweight='bold')
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
        mask_temp_rh = ~(np.isnan(temps) | np.isnan(rh_vals))
        if np.sum(mask_temp_rh) > 0:
            ax3.scatter(temps[mask_temp_rh], rh_vals[mask_temp_rh], alpha=0.5, s=1, c='blue')
            ax3.set_xlabel('Temperature (°C)')
            ax3.set_ylabel('Relative Humidity (%)')
            ax3.set_title(f'Temperature vs RH Relationship ({mask_type.capitalize()})', fontweight='bold')
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
            ax4.set_title(f'Monthly Humidity Classification ({mask_type.capitalize()})', fontweight='bold')
            ax4.set_xticks(months)
            ax4.set_xticklabels(month_labels)
            ax4.legend()
    
    plt.suptitle(f'Day-Level Humidity Analysis ({mask_type.capitalize()})\n'
                f'Total Days: {len(days_data):,}',
                fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    
    # Save
    filename = f'humidity_day_analysis_{mask_type}.png'
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
    parser.add_argument('--mask-type', choices=['land', 'ocean', 'both'], default='both',
                       help='Analysis domain: land-only, ocean-only, or both (default: both)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("HUMIDITY CLASSIFICATION VISUALIZATION")
    print("="*80)
    print(f"Input directory: {args.humidity_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Variables: {args.variables}")
    print(f"Analysis domain: {args.mask_type}")
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
            plot_global_humidity_patterns(aggregation_data, output_dir, args.mask_type, args.variables)
        
        # 2. Event-level analysis
        if not events_data.empty:
            print("2. Event-level humidity analysis...")
            plot_humidity_event_analysis(events_data, output_dir, args.mask_type)
        
        # 3. Temporal trends
        if aggregation_data and 'year' in aggregation_data.dims:
            print("3. Humidity trends...")
            plot_humidity_trends(aggregation_data, output_dir, args.mask_type, args.variables)
        
        # 4. Regional comparison
        if not events_data.empty:
            print("4. Regional humidity comparison...")
            plot_regional_humidity_comparison(events_data, output_dir, args.mask_type)
        
        # 5. Day-level analysis
        if not days_data.empty:
            print("5. Day-level humidity analysis...")
            plot_humidity_day_analysis(days_data, output_dir, args.mask_type)
        
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