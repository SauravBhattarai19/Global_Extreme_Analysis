#!/usr/bin/env python3
"""
Visualization of Heatwave Metrics (Output from 05_Heatwave_Metrics.py)

Creates comprehensive scientific visualizations of heatwave characteristics:
- Global maps of heatwave frequency, intensity, and duration
- Multi-year trends and variability
- Seasonal patterns and hotspots
- Event-level analysis and statistics
- Comparison between tmax and tmin heatwaves

Input files:
- heatwave_metrics_{var}_{year}.nc (gridded annual metrics)
- heatwave_events_{var}_{year}.parquet (event-level data)
- heatwave_days_{var}_{year}.parquet (day-level data)
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
from scipy import stats
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
    """Create accurate land/ocean mask using regionmask Natural Earth boundaries."""
    if mask_type == 'both':
        return np.ones((len(ds.latitude), len(ds.longitude)), dtype=bool)
    
    lat = ds.latitude.values
    lon = ds.longitude.values
    
    print(f"Creating {mask_type} mask...")
    print(f"  Grid: {len(lat)} x {len(lon)} pixels")
    
    if HAS_REGIONMASK:
        try:
            print("  Using regionmask with Natural Earth boundaries...")
            dummy_ds = xr.Dataset(coords={'lat': lat, 'lon': lon})
            land_mask_rm = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(dummy_ds)
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
    
    # Fallback: simplified approximation
    print("  Using simplified land/ocean boundaries...")
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing='ij')
    land_mask = np.zeros_like(lat_grid, dtype=bool)
    
    # Major continents (improved boundaries)
    land_mask |= (lat_grid >= -60) & (lat_grid <= 75) & (lon_grid >= -170) & (lon_grid <= -30)  # Americas
    land_mask |= (lat_grid >= -35) & (lat_grid <= 75) & (lon_grid >= -20) & (lon_grid <= 180)   # Eurasia+Africa
    land_mask |= (lat_grid >= -50) & (lat_grid <= -10) & (lon_grid >= 110) & (lon_grid <= 180)  # Australia
    
    if mask_type == 'land':
        return land_mask
    elif mask_type == 'ocean':
        return ~land_mask
    else:
        return np.ones_like(land_mask, dtype=bool)

def calculate_pixel_trends(data_array, years, significance_level=0.05):
    """Calculate trends and significance for each pixel."""
    if len(years) < 5:
        print("Warning: Less than 5 years of data - trend analysis may be unreliable")
    
    trends = np.full(data_array.shape[1:], np.nan)
    p_values = np.full(data_array.shape[1:], np.nan)
    
    for i in range(data_array.shape[1]):
        for j in range(data_array.shape[2]):
            pixel_data = data_array[:, i, j].values
            valid_mask = ~np.isnan(pixel_data)
            if np.sum(valid_mask) < 5:
                continue
            
            valid_years = years[valid_mask]
            valid_data = pixel_data[valid_mask]
            slope, intercept, r_value, p_value, std_err = stats.linregress(valid_years, valid_data)
            trends[i, j] = slope
            p_values[i, j] = p_value
    
    significant = p_values < significance_level
    return trends, p_values, significant

def apply_mask_and_adjust_colorbar(data, mask, percentile_range=(0.5, 99.5)):
    """Apply mask to data and calculate appropriate colorbar range."""
    masked_data = data.copy()
    masked_data[~mask] = np.nan
    valid_data = masked_data[~np.isnan(masked_data)]
    
    if len(valid_data) == 0:
        return masked_data, 0, 1, 0
    
    vmin = np.percentile(valid_data, percentile_range[0])
    vmax = np.percentile(valid_data, percentile_range[1])
    return masked_data, vmin, vmax, len(valid_data)

def add_significance_stippling(ax, ds, significant, mask):
    """Add simple stippling to show significant areas."""
    significant_masked = significant & mask
    if not np.any(significant_masked):
        return
    
    # Use stippling (dots) for significant areas
    lon_grid, lat_grid = np.meshgrid(ds.longitude, ds.latitude)
    
    # Subsample for performance (every nth point)
    skip = max(1, len(ds.latitude) // 100)  # Adaptive subsampling
    
    sig_lons = lon_grid[significant_masked][::skip]
    sig_lats = lat_grid[significant_masked][::skip]
    
    if len(sig_lons) > 0:
        ax.scatter(sig_lons, sig_lats, s=0.5, c='black', marker='.', 
                  alpha=0.7, transform=ccrs.PlateCarree())

def create_trend_categories(trends_decade, significant, mask):
    """Create discrete trend categories for cleaner visualization."""
    significant_masked = significant & mask
    
    if not np.any(significant_masked):
        return np.zeros_like(trends_decade, dtype=int)
    
    increasing_sig = significant_masked & (trends_decade > 0)
    decreasing_sig = significant_masked & (trends_decade < 0)
    
    abs_trends = np.abs(trends_decade)
    strong_threshold = np.nanpercentile(abs_trends[significant_masked], 75)
    
    # Create categories
    categories = np.zeros_like(trends_decade, dtype=int)
    categories[significant_masked & (trends_decade > 0) & (abs_trends >= strong_threshold)] = 4  # Strong increase
    categories[significant_masked & (trends_decade > 0) & (abs_trends < strong_threshold)] = 3   # Weak increase
    categories[significant_masked & (trends_decade < 0) & (abs_trends < strong_threshold)] = 2   # Weak decrease
    categories[significant_masked & (trends_decade < 0) & (abs_trends >= strong_threshold)] = 1  # Strong decrease
    
    return categories

def load_heatwave_metrics(metrics_dir, years=None, variables=['tmax', 'tmin']):
    """Load heatwave metrics data."""
    metrics_dir = Path(metrics_dir)
    
    if years is None:
        # Find available years
        metric_files = list(metrics_dir.glob('heatwave_metrics_*.nc'))
        years = sorted(set([int(f.name.split('_')[3].split('.')[0]) for f in metric_files]))
        print(f"Found data for years: {years[0]}-{years[-1]}")
    
    print(f"Loading heatwave metrics for years {years} and variables {variables}...")
    
    metrics_data = {}
    events_data = {}
    
    for var in variables:
        var_metrics = []
        var_events = []
        
        for year in years:
            # Load gridded metrics
            metrics_file = metrics_dir / f'heatwave_metrics_{var}_{year}.nc'
            if metrics_file.exists():
                ds = xr.open_dataset(metrics_file)
                ds = ds.expand_dims('year').assign_coords(year=[year])
                var_metrics.append(ds)
                print(f"  Loaded metrics: {metrics_file.name}")
            
            # Load event data
            events_file = metrics_dir / f'heatwave_events_{var}_{year}.parquet'
            if events_file.exists():
                events_df = pd.read_parquet(events_file)
                events_df['year'] = year
                var_events.append(events_df)
                print(f"  Loaded events: {events_file.name}")
        
        if var_metrics:
            metrics_data[var] = xr.concat(var_metrics, dim='year')
        if var_events:
            events_data[var] = pd.concat(var_events, ignore_index=True)
    
    print(f"Loaded {len(metrics_data)} variables with metrics")
    print(f"Loaded {len(events_data)} variables with events")
    
    return metrics_data, events_data

def plot_global_heatwave_climatology(metrics_data, output_dir, variables=['tmax', 'tmin']):
    """Create global maps of heatwave climatology."""
    output_dir = Path(output_dir)
    
    for var in variables:
        if var not in metrics_data:
            continue
        
        ds = metrics_data[var]
        
        # Calculate multi-year means
        hwn_mean = ds[f'hwn_{var}'].mean(dim='year')
        hwmt_mean = ds[f'hwmt_{var}'].mean(dim='year') - 273.15  # Convert to Celsius
        
        # Convert timedelta to days for duration metrics
        hwtd_mean = ds[f'hwtd_{var}'].mean(dim='year') / np.timedelta64(1, 'D')
        hwld_mean = ds[f'hwld_{var}'].mean(dim='year') / np.timedelta64(1, 'D')
        
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Heatwave Number (HWN)
        ax1 = plt.subplot(2, 2, 1, projection=ccrs.Robinson())
        ax1.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax1.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax1.set_global()
        
        levels_hwn = np.arange(0, 11, 0.5)
        cmap_hwn = plt.cm.YlOrRd
        
        im1 = ax1.contourf(hwn_mean.longitude, hwn_mean.latitude, hwn_mean,
                          levels=levels_hwn, cmap=cmap_hwn, transform=ccrs.PlateCarree(),
                          extend='max')
        
        ax1.set_title(f'Mean Annual Heatwave Number ({var.upper()})\n'
                     f'Period: {ds.year.min().values}-{ds.year.max().values}',
                     fontsize=12, fontweight='bold')
        
        cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
        cbar1.set_label('Events per year', fontsize=10)
        
        # 2. Heatwave Mean Temperature (HWMT)
        ax2 = plt.subplot(2, 2, 2, projection=ccrs.Robinson())
        ax2.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax2.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax2.set_global()
        
        levels_hwmt = np.arange(-10, 51, 2)
        cmap_hwmt = plt.cm.RdYlBu_r
        
        im2 = ax2.contourf(hwmt_mean.longitude, hwmt_mean.latitude, hwmt_mean,
                          levels=levels_hwmt, cmap=cmap_hwmt, transform=ccrs.PlateCarree(),
                          extend='both')
        
        ax2.set_title(f'Mean Heatwave Temperature ({var.upper()})', fontsize=12, fontweight='bold')
        
        cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
        cbar2.set_label('Temperature (°C)', fontsize=10)
        
        # 3. Heatwave Total Duration (HWTD)
        ax3 = plt.subplot(2, 2, 3, projection=ccrs.Robinson())
        ax3.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax3.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax3.set_global()
        
        levels_hwtd = np.arange(0, 51, 2)
        cmap_hwtd = plt.cm.plasma
        
        im3 = ax3.contourf(hwtd_mean.longitude, hwtd_mean.latitude, hwtd_mean,
                          levels=levels_hwtd, cmap=cmap_hwtd, transform=ccrs.PlateCarree(),
                          extend='max')
        
        ax3.set_title(f'Mean Annual Heatwave Duration ({var.upper()})', fontsize=12, fontweight='bold')
        
        cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.8)
        cbar3.set_label('Days per year', fontsize=10)
        
        # 4. Heatwave Longest Duration (HWLD)
        ax4 = plt.subplot(2, 2, 4, projection=ccrs.Robinson())
        ax4.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax4.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax4.set_global()
        
        levels_hwld = np.arange(0, 21, 1)
        cmap_hwld = plt.cm.Reds
        
        im4 = ax4.contourf(hwld_mean.longitude, hwld_mean.latitude, hwld_mean,
                          levels=levels_hwld, cmap=cmap_hwld, transform=ccrs.PlateCarree(),
                          extend='max')
        
        ax4.set_title(f'Mean Longest Heatwave Duration ({var.upper()})', fontsize=12, fontweight='bold')
        
        cbar4 = plt.colorbar(im4, ax=ax4, shrink=0.8)
        cbar4.set_label('Days', fontsize=10)
        
        plt.suptitle(f'Heatwave Climatology - {var.upper()}\n'
                    f'Period: {ds.year.min().values}-{ds.year.max().values}',
                    fontsize=16, fontweight='bold', y=0.95)
        
        plt.tight_layout()
        
        # Save
        filename = f'heatwave_climatology_{var}.png'
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {filename}")

def plot_heatwave_trends(metrics_data, output_dir, variables=['tmax', 'tmin']):
    """Plot temporal trends in heatwave metrics."""
    output_dir = Path(output_dir)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = {'tmax': 'red', 'tmin': 'blue'}
    
    for var in variables:
        if var not in metrics_data:
            continue
        
        ds = metrics_data[var]
        years = ds.year.values
        
        if len(years) < 3:
            print(f"Not enough years for trend analysis ({len(years)} years)")
            continue
        
        # Calculate global means for each year
        hwn_global = ds[f'hwn_{var}'].mean(dim=['latitude', 'longitude'])
        hwmt_global = ds[f'hwmt_{var}'].mean(dim=['latitude', 'longitude']) - 273.15
        hwtd_global = ds[f'hwtd_{var}'].mean(dim=['latitude', 'longitude']) / np.timedelta64(1, 'D')
        hwld_global = ds[f'hwld_{var}'].mean(dim=['latitude', 'longitude']) / np.timedelta64(1, 'D')
        
        # 1. Heatwave Number trend
        ax1 = axes[0, 0]
        ax1.plot(years, hwn_global, 'o-', color=colors[var], label=var.upper(), 
                linewidth=2, markersize=6, alpha=0.8)
        
        # Add trend line
        if len(years) > 2:
            z = np.polyfit(years, hwn_global.values, 1)
            p = np.poly1d(z)
            ax1.plot(years, p(years), '--', color=colors[var], alpha=0.7)
        
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Global Mean HWN (events/year)')
        ax1.set_title('Heatwave Number Trend', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. Heatwave Mean Temperature trend
        ax2 = axes[0, 1]
        ax2.plot(years, hwmt_global, 'o-', color=colors[var], label=var.upper(), 
                linewidth=2, markersize=6, alpha=0.8)
        
        # Add trend line
        if len(years) > 2:
            z = np.polyfit(years, hwmt_global.values, 1)
            p = np.poly1d(z)
            ax2.plot(years, p(years), '--', color=colors[var], alpha=0.7)
        
        ax2.set_xlabel('Year')
        ax2.set_ylabel('Global Mean HWMT (°C)')
        ax2.set_title('Heatwave Temperature Trend', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 3. Heatwave Total Duration trend
        ax3 = axes[1, 0]
        ax3.plot(years, hwtd_global, 'o-', color=colors[var], label=var.upper(), 
                linewidth=2, markersize=6, alpha=0.8)
        
        # Add trend line
        if len(years) > 2:
            z = np.polyfit(years, hwtd_global.values, 1)
            p = np.poly1d(z)
            ax3.plot(years, p(years), '--', color=colors[var], alpha=0.7)
        
        ax3.set_xlabel('Year')
        ax3.set_ylabel('Global Mean HWTD (days/year)')
        ax3.set_title('Heatwave Duration Trend', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 4. Heatwave Longest Duration trend
        ax4 = axes[1, 1]
        ax4.plot(years, hwld_global, 'o-', color=colors[var], label=var.upper(), 
                linewidth=2, markersize=6, alpha=0.8)
        
        # Add trend line
        if len(years) > 2:
            z = np.polyfit(years, hwld_global.values, 1)
            p = np.poly1d(z)
            ax4.plot(years, p(years), '--', color=colors[var], alpha=0.7)
        
        ax4.set_xlabel('Year')
        ax4.set_ylabel('Global Mean HWLD (days)')
        ax4.set_title('Longest Heatwave Trend', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
    
    plt.suptitle(f'Global Heatwave Trends\n'
                f'Period: {years[0]}-{years[-1]}',
                fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    
    # Save
    filename = 'heatwave_trends.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {filename}")

def plot_heatwave_hotspots(metrics_data, output_dir, variables=['tmax', 'tmin']):
    """Identify and visualize heatwave hotspots."""
    output_dir = Path(output_dir)
    
    for var in variables:
        if var not in metrics_data:
            continue
        
        ds = metrics_data[var]
        
        # Calculate multi-year statistics
        hwn_mean = ds[f'hwn_{var}'].mean(dim='year')
        hwn_std = ds[f'hwn_{var}'].std(dim='year')
        hwtd_mean = ds[f'hwtd_{var}'].mean(dim='year') / np.timedelta64(1, 'D')
        hwld_max = ds[f'hwld_{var}'].max(dim='year') / np.timedelta64(1, 'D')
        
        # Define hotspots (high frequency AND high intensity)
        hwn_threshold = np.nanpercentile(hwn_mean.values, 90)  # Top 10% for frequency
        hwtd_threshold = np.nanpercentile(hwtd_mean.values, 90)  # Top 10% for duration
        
        hotspot_mask = (hwn_mean >= hwn_threshold) & (hwtd_mean >= hwtd_threshold)
        
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Hotspot identification map
        ax1 = plt.subplot(2, 1, 1, projection=ccrs.Robinson())
        ax1.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax1.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax1.set_global()
        
        # Create composite hotspot index
        hotspot_index = (hwn_mean / hwn_mean.max()) + (hwtd_mean / hwtd_mean.max())
        
        levels = np.linspace(0, 2, 21)
        cmap = plt.cm.hot_r
        
        im1 = ax1.contourf(hotspot_index.longitude, hotspot_index.latitude, hotspot_index,
                          levels=levels, cmap=cmap, transform=ccrs.PlateCarree(),
                          extend='max')
        
        # Highlight extreme hotspots
        ax1.contour(hotspot_mask.longitude, hotspot_mask.latitude, hotspot_mask.astype(int),
                   levels=[0.5], colors=['black'], linewidths=2, 
                   transform=ccrs.PlateCarree())
        
        ax1.set_title(f'Heatwave Hotspots ({var.upper()})\n'
                     f'Black contours: Top 10% frequency AND duration',
                     fontsize=14, fontweight='bold', pad=20)
        
        cbar1 = plt.colorbar(im1, ax=ax1, orientation='horizontal', pad=0.08, shrink=0.8)
        cbar1.set_label('Hotspot Index (normalized frequency + duration)', fontsize=12)
        
        # 2. Variability analysis
        ax2 = plt.subplot(2, 1, 2, projection=ccrs.Robinson())
        ax2.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax2.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax2.set_global()
        
        # Coefficient of variation (std/mean)
        cv = hwn_std / (hwn_mean + 0.1)  # Add small value to avoid division by zero
        
        levels_cv = np.linspace(0, 2, 21)
        cmap_cv = plt.cm.viridis
        
        im2 = ax2.contourf(cv.longitude, cv.latitude, cv,
                          levels=levels_cv, cmap=cmap_cv, transform=ccrs.PlateCarree(),
                          extend='max')
        
        ax2.set_title(f'Heatwave Variability ({var.upper()})\n'
                     f'Coefficient of Variation (std/mean)',
                     fontsize=14, fontweight='bold', pad=20)
        
        cbar2 = plt.colorbar(im2, ax=ax2, orientation='horizontal', pad=0.08, shrink=0.8)
        cbar2.set_label('Coefficient of Variation', fontsize=12)
        
        plt.tight_layout()
        
        # Save
        filename = f'heatwave_hotspots_{var}.png'
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {filename}")

def plot_event_analysis(events_data, output_dir, variables=['tmax', 'tmin']):
    """Analyze individual heatwave events."""
    output_dir = Path(output_dir)
    
    if not events_data:
        print("No event data available for analysis")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = {'tmax': 'red', 'tmin': 'blue'}
    
    # Combine all events
    all_events = []
    for var in variables:
        if var in events_data:
            df = events_data[var].copy()
            df['variable'] = var
            all_events.append(df)
    
    if not all_events:
        print("No event data to analyze")
        return
    
    combined_events = pd.concat(all_events, ignore_index=True)
    
    # 1. Duration distribution
    ax1 = axes[0, 0]
    
    for var in variables:
        if var in events_data:
            durations = events_data[var]['duration_days']
            ax1.hist(durations, bins=range(1, 31), alpha=0.7, label=var.upper(), 
                    color=colors[var], density=True)
    
    ax1.set_xlabel('Duration (days)')
    ax1.set_ylabel('Density')
    ax1.set_title('Heatwave Duration Distribution', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Seasonal timing
    ax2 = axes[0, 1]
    
    # Extract month from start date
    combined_events['start_month'] = pd.to_datetime(combined_events['year_start']).dt.month
    
    for var in variables:
        if var in events_data:
            var_events = combined_events[combined_events['variable'] == var]
            month_counts = var_events['start_month'].value_counts().sort_index()
            
            months = range(1, 13)
            counts = [month_counts.get(m, 0) for m in months]
            
            ax2.plot(months, counts, 'o-', color=colors[var], label=var.upper(), 
                    linewidth=2, markersize=6)
    
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Number of Events')
    ax2.set_title('Seasonal Distribution of Heatwave Onset', fontweight='bold')
    ax2.set_xticks(range(1, 13))
    ax2.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Intensity vs Duration scatter
    ax3 = axes[1, 0]
    
    for var in variables:
        if var in events_data:
            df = events_data[var]
            if 'mean_temp_clipped' in df.columns:
                # Convert to Celsius if needed
                temps = df['mean_temp_clipped'].values
                if np.nanmean(temps) > 100:  # Likely in Kelvin
                    temps = temps - 273.15
                
                ax3.scatter(df['duration_days'], temps, alpha=0.6, 
                           color=colors[var], label=var.upper(), s=20)
    
    ax3.set_xlabel('Duration (days)')
    ax3.set_ylabel('Mean Temperature (°C)')
    ax3.set_title('Heatwave Intensity vs Duration', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Annual event counts
    ax4 = axes[1, 1]
    
    annual_counts = combined_events.groupby(['year', 'variable']).size().unstack(fill_value=0)
    
    if len(annual_counts) > 1:
        for var in variables:
            if var in annual_counts.columns:
                ax4.plot(annual_counts.index, annual_counts[var], 'o-', 
                        color=colors[var], label=var.upper(), linewidth=2, markersize=6)
        
        ax4.set_xlabel('Year')
        ax4.set_ylabel('Number of Events')
        ax4.set_title('Annual Heatwave Event Counts', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        # Single year - show spatial distribution
        ax4.text(0.5, 0.5, 'Insufficient years\nfor trend analysis', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=14)
        ax4.set_title('Annual Event Analysis', fontweight='bold')
    
    plt.suptitle(f'Heatwave Event Analysis\n'
                f'Total Events: {len(combined_events):,}',
                fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    
    # Save
    filename = 'heatwave_event_analysis.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {filename}")

def plot_tmax_tmin_comparison(metrics_data, events_data, output_dir):
    """Compare tmax and tmin heatwave characteristics."""
    output_dir = Path(output_dir)
    
    if 'tmax' not in metrics_data or 'tmin' not in metrics_data:
        print("Both tmax and tmin data required for comparison")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Spatial correlation of metrics
    ax1 = axes[0, 0]
    
    # Calculate correlations
    tmax_hwn = metrics_data['tmax']['hwn_tmax'].mean(dim='year')
    tmin_hwn = metrics_data['tmin']['hwn_tmin'].mean(dim='year')
    
    # Flatten and remove NaN for correlation
    tmax_flat = tmax_hwn.values.flatten()
    tmin_flat = tmin_hwn.values.flatten()
    
    mask = ~(np.isnan(tmax_flat) | np.isnan(tmin_flat))
    tmax_clean = tmax_flat[mask]
    tmin_clean = tmin_flat[mask]
    
    if len(tmax_clean) > 0:
        correlation = np.corrcoef(tmax_clean, tmin_clean)[0, 1]
        
        ax1.scatter(tmax_clean, tmin_clean, alpha=0.6, s=1, c='purple')
        ax1.set_xlabel('Tmax Heatwave Number')
        ax1.set_ylabel('Tmin Heatwave Number')
        ax1.set_title(f'Heatwave Number Correlation\nr = {correlation:.3f}', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add 1:1 line
        max_val = max(np.max(tmax_clean), np.max(tmin_clean))
        ax1.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='1:1 line')
        ax1.legend()
    
    # 2. Duration comparison
    ax2 = axes[0, 1]
    
    if 'tmax' in events_data and 'tmin' in events_data:
        tmax_durations = events_data['tmax']['duration_days']
        tmin_durations = events_data['tmin']['duration_days']
        
        ax2.hist(tmax_durations, bins=range(1, 21), alpha=0.7, label='Tmax', 
                color='red', density=True)
        ax2.hist(tmin_durations, bins=range(1, 21), alpha=0.7, label='Tmin', 
                color='blue', density=True)
        
        ax2.set_xlabel('Duration (days)')
        ax2.set_ylabel('Density')
        ax2.set_title('Duration Distribution Comparison', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. Seasonal patterns
    ax3 = axes[1, 0]
    
    if 'tmax' in events_data and 'tmin' in events_data:
        # Monthly event counts
        tmax_events = events_data['tmax'].copy()
        tmin_events = events_data['tmin'].copy()
        
        tmax_events['start_month'] = pd.to_datetime(tmax_events['year_start']).dt.month
        tmin_events['start_month'] = pd.to_datetime(tmin_events['year_start']).dt.month
        
        tmax_monthly = tmax_events['start_month'].value_counts().sort_index()
        tmin_monthly = tmin_events['start_month'].value_counts().sort_index()
        
        months = range(1, 13)
        tmax_counts = [tmax_monthly.get(m, 0) for m in months]
        tmin_counts = [tmin_monthly.get(m, 0) for m in months]
        
        x = np.arange(len(months))
        width = 0.35
        
        ax3.bar(x - width/2, tmax_counts, width, label='Tmax', color='red', alpha=0.7)
        ax3.bar(x + width/2, tmin_counts, width, label='Tmin', color='blue', alpha=0.7)
        
        ax3.set_xlabel('Month')
        ax3.set_ylabel('Number of Events')
        ax3.set_title('Seasonal Event Distribution', fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. Summary statistics table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    stats_data = [['Metric', 'Tmax', 'Tmin', 'Ratio (Tmax/Tmin)']]
    
    # Calculate summary statistics
    if 'tmax' in events_data and 'tmin' in events_data:
        tmax_events = events_data['tmax']
        tmin_events = events_data['tmin']
        
        metrics = [
            ('Total Events', len(tmax_events), len(tmin_events)),
            ('Mean Duration', tmax_events['duration_days'].mean(), tmin_events['duration_days'].mean()),
            ('Max Duration', tmax_events['duration_days'].max(), tmin_events['duration_days'].max()),
            ('Events/Year', len(tmax_events)/len(tmax_events['year'].unique()), 
             len(tmin_events)/len(tmin_events['year'].unique()))
        ]
        
        for metric_name, tmax_val, tmin_val in metrics:
            if metric_name == 'Total Events':
                ratio_str = f'{tmax_val/max(tmin_val, 1):.2f}'
                stats_data.append([metric_name, f'{tmax_val:,}', f'{tmin_val:,}', ratio_str])
            else:
                ratio_str = f'{tmax_val/max(tmin_val, 0.001):.2f}'
                stats_data.append([metric_name, f'{tmax_val:.1f}', f'{tmin_val:.1f}', ratio_str])
    
    table = ax4.table(cellText=stats_data[1:], colLabels=stats_data[0],
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(stats_data)):
        for j in range(len(stats_data[0])):
            cell = table[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor('#4472C4')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#F2F2F2' if i % 2 == 0 else 'white')
    
    ax4.set_title('Tmax vs Tmin Comparison', fontweight='bold', pad=20)
    
    plt.suptitle('Tmax vs Tmin Heatwave Comparison',
                fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    
    # Save
    filename = 'heatwave_tmax_tmin_comparison.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {filename}")

def plot_heatwave_spatial_trends(metrics_data, output_dir, mask_type='both', variables=['tmax', 'tmin']):
    """Plot improved spatial trends with separate magnitude and significance maps."""
    output_dir = Path(output_dir)
    
    for var in variables:
        if var not in metrics_data:
            continue
        
        ds = metrics_data[var]
        
        if len(ds.year) < 5:
            print(f"Insufficient years for {var} trend analysis")
            continue
        
        # Create land/ocean mask
        mask = create_land_ocean_mask(ds, mask_type)
        years = ds.year.values
        
        # Variables to analyze
        heatwave_vars = [f'hwn_{var}', f'hwmt_{var}', f'hwtd_{var}', f'hwld_{var}']
        var_titles = ['Heatwave Number', 'Mean Temperature', 'Total Duration', 'Longest Duration']
        var_units = ['events/decade', '°C/decade', 'days/decade', 'days/decade']
        
        # Create two separate figures
        
        # FIGURE 1: Trend Magnitude with Significance Stippling
        fig1 = plt.figure(figsize=(20, 16))
        
        for idx, (hvar, title, units) in enumerate(zip(heatwave_vars, var_titles, var_units)):
            if hvar not in ds.data_vars:
                continue
            
            ax = plt.subplot(2, 2, idx+1, projection=ccrs.Robinson())
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.3)
            ax.set_global()
            
            print(f"  Calculating trends for {hvar}...")
            
            # Handle different data types
            data_var = ds[hvar]
            if 'timedelta' in str(data_var.dtype):
                data_var = data_var / np.timedelta64(1, 'D')
            elif 'hwmt' in hvar:  # Convert temperature to Celsius
                data_var = data_var - 273.15
            
            trends, p_values, significant = calculate_pixel_trends(data_var, years)
            
            # Apply mask and adjust colorbar
            trends_masked, vmin, vmax, valid_count = apply_mask_and_adjust_colorbar(
                trends, mask, percentile_range=(2.5, 97.5)
            )
            
            # Convert to per-decade
            trends_decade = trends_masked * 10
            vmin_decade = vmin * 10
            vmax_decade = vmax * 10
            
            # Create symmetric colorbar
            vmax_abs = max(abs(vmin_decade), abs(vmax_decade)) if vmax_decade != vmin_decade else 1
            levels = np.linspace(-vmax_abs, vmax_abs, 21)
            
            # Plot trends with clean colormap
            cmap = plt.cm.RdBu_r
            im = ax.contourf(ds.longitude, ds.latitude, trends_decade,
                            levels=levels, cmap=cmap, transform=ccrs.PlateCarree(),
                            extend='both')
            
            # Add simple stippling for significant areas
            add_significance_stippling(ax, ds, significant, mask)
            
            # Calculate significance statistics
            sig_percentage = np.sum(significant & mask) / np.sum(mask) * 100
            
            ax.set_title(f'{title} ({var.upper()}) Trend\\n'
                        f'({mask_type.capitalize()}, {valid_count:,} pixels, {sig_percentage:.1f}% significant)',
                        fontsize=12, fontweight='bold')
            
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label(f'Trend ({units})', fontsize=10)
        
        # Overall title for Figure 1
        period_str = f"{years[0]}-{years[-1]}"
        fig1.suptitle(f'{var.upper()} Heatwave Trend Magnitude ({period_str}) | {mask_type.capitalize()}\\n'
                     f'Black dots show statistically significant areas (p < 0.05)',
                     fontsize=14, fontweight='bold', y=0.96)
        
        plt.tight_layout()
        
        filename1 = f'heatwave_trends_magnitude_{var}_{mask_type}.png'
        plt.savefig(output_dir / filename1, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {filename1}")
        
        # FIGURE 2: Trend Categories
        fig2 = plt.figure(figsize=(20, 16))
        
        for idx, (hvar, title, units) in enumerate(zip(heatwave_vars, var_titles, var_units)):
            if hvar not in ds.data_vars:
                continue
            
            ax = plt.subplot(2, 2, idx+1, projection=ccrs.Robinson())
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.3)
            ax.set_global()
            
            # Handle different data types (same as before)
            data_var = ds[hvar]
            if 'timedelta' in str(data_var.dtype):
                data_var = data_var / np.timedelta64(1, 'D')
            elif 'hwmt' in hvar:
                data_var = data_var - 273.15
            
            trends, p_values, significant = calculate_pixel_trends(data_var, years)
            trends_masked, vmin, vmax, valid_count = apply_mask_and_adjust_colorbar(
                trends, mask, percentile_range=(2.5, 97.5)
            )
            trends_decade = trends_masked * 10
            
            # Create trend categories
            categories = create_trend_categories(trends_decade, significant, mask)
            
            # Define colors for categories
            colors = ['white', 'darkblue', 'lightblue', 'lightcoral', 'darkred']
            cmap_discrete = mcolors.ListedColormap(colors)
            
            # Plot categories
            im = ax.contourf(ds.longitude, ds.latitude, categories,
                            levels=np.arange(-0.5, 5.5, 1), cmap=cmap_discrete, 
                            transform=ccrs.PlateCarree())
            
            # Calculate category statistics
            cat_counts = [(categories == i).sum() for i in range(5)]
            total_valid = np.sum(mask)
            cat_percentages = [count/total_valid*100 for count in cat_counts]
            
            ax.set_title(f'{title} ({var.upper()}) Trend Categories\\n'
                        f'Strong↓({cat_percentages[1]:.1f}%) | Weak↓({cat_percentages[2]:.1f}%) | '
                        f'Weak↑({cat_percentages[3]:.1f}%) | Strong↑({cat_percentages[4]:.1f}%)',
                        fontsize=11, fontweight='bold')
        
        # Add custom legend for Figure 2
        legend_elements = [
            plt.Rectangle((0,0),1,1, facecolor='darkblue', label='Strong Decrease'),
            plt.Rectangle((0,0),1,1, facecolor='lightblue', label='Weak Decrease'),
            plt.Rectangle((0,0),1,1, facecolor='white', edgecolor='black', label='Non-significant'),
            plt.Rectangle((0,0),1,1, facecolor='lightcoral', label='Weak Increase'),
            plt.Rectangle((0,0),1,1, facecolor='darkred', label='Strong Increase')
        ]
        fig2.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.02), ncol=5)
        
        fig2.suptitle(f'{var.upper()} Heatwave Trend Categories ({period_str}) | {mask_type.capitalize()}\\n'
                     f'Strong = Top 25% of trend magnitudes among significant pixels',
                     fontsize=14, fontweight='bold', y=0.96)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)  # Make room for legend
        
        filename2 = f'heatwave_trends_categories_{var}_{mask_type}.png'
        plt.savefig(output_dir / filename2, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {filename2}")

def main():
    """Main visualization function for heatwave metrics."""
    parser = argparse.ArgumentParser(description='Visualize heatwave metrics')
    
    parser.add_argument('--metrics-dir', default='data/processed/heatwave_metrics',
                       help='Directory containing heatwave metrics files')
    parser.add_argument('--output-dir', default='visualizations/output/heatwave_metrics',
                       help='Output directory for plots')
    parser.add_argument('--years', nargs='+', type=int,
                       help='Specific years to analyze (default: all available)')
    parser.add_argument('--variables', nargs='+', default=['tmax', 'tmin'],
                       help='Variables to analyze (default: tmax tmin)')
    parser.add_argument('--mask-type', choices=['land', 'ocean', 'both'], default='both',
                       help='Analysis domain: land-only, ocean-only, or both (default: both)')
    parser.add_argument('--include-trends', action='store_true',
                       help='Include spatial trend analysis with significance testing')
    parser.add_argument('--percentile-cap', type=float, default=99.5,
                       help='Percentile for capping extreme values in colorbars (default: 99.5)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("HEATWAVE METRICS VISUALIZATION")
    print("="*80)
    print(f"Input directory: {args.metrics_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Variables: {args.variables}")
    print(f"Analysis domain: {args.mask_type}")
    print(f"Include trends: {args.include_trends}")
    print(f"Percentile cap: {args.percentile_cap}%")
    print("="*80)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load data
        metrics_data, events_data = load_heatwave_metrics(args.metrics_dir, args.years, args.variables)
        
        if not metrics_data and not events_data:
            print("No heatwave data found!")
            return 1
        
        print("\nCreating visualizations...")
        
        # 1. Global climatology maps
        if metrics_data:
            print("1. Global climatology maps...")
            plot_global_heatwave_climatology(metrics_data, output_dir, args.variables)
        
        # 2. Temporal trends
        if metrics_data:
            print("2. Temporal trends...")
            plot_heatwave_trends(metrics_data, output_dir, args.variables)
        
        # 3. Hotspot analysis
        if metrics_data:
            print("3. Hotspot analysis...")
            plot_heatwave_hotspots(metrics_data, output_dir, args.variables)
        
        # 4. Event analysis
        if events_data:
            print("4. Event analysis...")
            plot_event_analysis(events_data, output_dir, args.variables)
        
        # 5. Tmax vs Tmin comparison
        if len(args.variables) >= 2 and 'tmax' in args.variables and 'tmin' in args.variables:
            print("5. Tmax vs Tmin comparison...")
            plot_tmax_tmin_comparison(metrics_data, events_data, output_dir)
        
        # 6. Spatial trends (if requested)
        if args.include_trends:
            print("6. Spatial trend analysis...")
            plot_heatwave_spatial_trends(metrics_data, output_dir, args.mask_type, args.variables)
        
        print("\n" + "="*80)
        print("HEATWAVE METRICS VISUALIZATION COMPLETED!")
        print("="*80)
        print(f"Output files saved in: {output_dir}")
        print("\nGenerated plots:")
        for plot_file in output_dir.glob('*.png'):
            print(f"  - {plot_file.name}")
        
        # Close datasets
        for ds in metrics_data.values():
            ds.close()
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
