#!/usr/bin/env python3
"""
Visualization of Heat Index Data (Output from 04_Heat_Index.py)

Creates comprehensive scientific visualizations of heat index climatology:
- Global heat index distribution maps
- Seasonal patterns and extremes
- Heat index vs temperature/humidity relationships
- Dangerous heat index threshold analysis
- Regional comparisons and trends

Input files:
- era5_heat_index_{year}_{month:02d}.nc files from data/processed/heat_index/
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

# Heat Index Categories (in Celsius)
HI_CATEGORIES = {
    'Safe': (0, 27),
    'Caution': (27, 32),
    'Extreme Caution': (32, 41),
    'Danger': (41, 54),
    'Extreme Danger': (54, 100)
}

HI_COLORS = {
    'Safe': '#2E8B57',
    'Caution': '#FFD700', 
    'Extreme Caution': '#FF8C00',
    'Danger': '#FF4500',
    'Extreme Danger': '#8B0000'
}

def load_heat_index_data(hi_dir, years=None, sample_months=None, start_year=None, end_year=None):
    """Load heat index data for analysis."""
    hi_dir = Path(hi_dir)
    
    if years is None:
        # Find available years
        hi_files = list(hi_dir.glob('era5_heat_index_*.nc'))
        all_years = sorted(set([int(f.name.split('_')[3]) for f in hi_files]))
        print(f"Found data for years: {all_years[0]}-{all_years[-1]} (total: {len(all_years)} years)")
        
        # Apply start_year and end_year filters
        if start_year is not None or end_year is not None:
            start_yr = start_year if start_year is not None else all_years[0]
            end_yr = end_year if end_year is not None else all_years[-1]
            
            # Validate year range
            if start_yr < all_years[0]:
                print(f"Warning: start_year {start_yr} is before earliest available year {all_years[0]}")
                start_yr = all_years[0]
            if end_yr > all_years[-1]:
                print(f"Warning: end_year {end_yr} is after latest available year {all_years[-1]}")
                end_yr = all_years[-1]
            
            # Filter years within the specified range
            years = [yr for yr in all_years if start_yr <= yr <= end_yr]
            print(f"Using years {start_yr}-{end_yr}: {len(years)} years")
        else:
            years = all_years
            print(f"Using all {len(years)} years")
    
    if sample_months is None:
        sample_months = [1, 4, 7, 10]  # Jan, Apr, Jul, Oct
    
    print(f"Loading Heat Index data for years {years} and months {sample_months}...")
    
    datasets = []
    file_info = []
    
    for year in years:
        for month in sample_months:
            file_path = hi_dir / f"era5_heat_index_{year}_{month:02d}.nc"
            
            if file_path.exists():
                try:
                    ds = xr.open_dataset(file_path, chunks={'valid_time': 10})
                    datasets.append(ds)
                    file_info.append((year, month))
                    print(f"  Loaded: {file_path.name}")
                except Exception as e:
                    print(f"  Error loading {file_path}: {e}")
    
    if not datasets:
        raise ValueError(f"No Heat Index files found in {hi_dir}")
    
    # Combine datasets
    combined_ds = xr.concat(datasets, dim='valid_time')
    
    print(f"Combined dataset shape: {combined_ds.dims}")
    print(f"Heat Index range: {combined_ds.heat_index.min().values:.1f} - {combined_ds.heat_index.max().values:.1f}°C")
    
    return combined_ds, file_info

def plot_global_heat_index_climatology(hi_ds, output_dir, mask_type='both', seasons=None):
    """Create global maps of heat index climatology."""
    output_dir = Path(output_dir)
    
    if seasons is None:
        seasons = ['DJF', 'MAM', 'JJA', 'SON', 'Annual']
    
    # Create land/ocean mask
    mask = create_land_ocean_mask(hi_ds, mask_type)
    
    # Calculate seasonal means and extremes
    hi_seasonal_mean = hi_ds.heat_index.groupby('valid_time.season').mean()
    hi_seasonal_max = hi_ds.heat_index.groupby('valid_time.season').max()
    hi_annual_mean = hi_ds.heat_index.mean(dim='valid_time')
    hi_annual_max = hi_ds.heat_index.max(dim='valid_time')
    
    for season in seasons:
        fig = plt.figure(figsize=(20, 12))
        
        if season == 'Annual':
            hi_mean_data = hi_annual_mean
            hi_max_data = hi_annual_max
            title_suffix = 'Annual'
        else:
            if season in hi_seasonal_mean.season.values:
                hi_mean_data = hi_seasonal_mean.sel(season=season)
                hi_max_data = hi_seasonal_max.sel(season=season)
                title_suffix = season
            else:
                continue
        
        # Mean Heat Index
        ax1 = plt.subplot(2, 1, 1, projection=ccrs.Robinson())
        ax1.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax1.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax1.set_global()
        
        # Apply mask and calculate percentiles
        hi_mean_masked, vmin_mean, vmax_mean, valid_count_mean = apply_mask_and_adjust_colorbar(
            hi_mean_data.values, mask, percentile_range=(0.5, 99.5)
        )
        
        # Heat index color scheme with danger zones
        levels = np.linspace(vmin_mean, vmax_mean, 20)
        colors = []
        for level in levels[:-1]:
            if level < 27:
                colors.append(HI_COLORS['Safe'])
            elif level < 32:
                colors.append(HI_COLORS['Caution'])
            elif level < 41:
                colors.append(HI_COLORS['Extreme Caution'])
            elif level < 54:
                colors.append(HI_COLORS['Danger'])
            else:
                colors.append(HI_COLORS['Extreme Danger'])
        
        cmap = mcolors.ListedColormap(colors)
        
        im1 = ax1.contourf(hi_mean_data.longitude, hi_mean_data.latitude, hi_mean_masked,
                          levels=levels, cmap=cmap, transform=ccrs.PlateCarree(),
                          extend='both')
        
        ax1.set_title(f'Mean Heat Index - {title_suffix} ({mask_type.capitalize()})\n'
                     f'Period: {hi_ds.valid_time.dt.year.min().values}-{hi_ds.valid_time.dt.year.max().values}\n'
                     f'({valid_count_mean:,} valid pixels)',
                     fontsize=14, fontweight='bold', pad=20)
        
        cbar1 = plt.colorbar(im1, ax=ax1, orientation='horizontal', pad=0.08, shrink=0.8)
        cbar1.set_label('Heat Index (°C)', fontsize=12, fontweight='bold')
        
        # Add danger zone lines
        for temp, color in [(27, 'yellow'), (32, 'orange'), (41, 'red'), (54, 'darkred')]:
            ax1.contour(hi_mean_data.longitude, hi_mean_data.latitude, hi_mean_masked,
                       levels=[temp], colors=[color], linewidths=2, alpha=0.8,
                       transform=ccrs.PlateCarree())
        
        # Maximum Heat Index
        ax2 = plt.subplot(2, 1, 2, projection=ccrs.Robinson())
        ax2.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax2.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax2.set_global()
        
        # Apply mask to max data
        hi_max_masked, vmin_max, vmax_max, valid_count_max = apply_mask_and_adjust_colorbar(
            hi_max_data.values, mask, percentile_range=(0.5, 99.5)
        )
        
        levels_max = np.linspace(vmin_max, vmax_max, 20)
        cmap_max = plt.cm.hot_r
        
        im2 = ax2.contourf(hi_max_data.longitude, hi_max_data.latitude, hi_max_masked,
                          levels=levels_max, cmap=cmap_max, transform=ccrs.PlateCarree(),
                          extend='both')
        
        ax2.set_title(f'Maximum Heat Index - {title_suffix} ({mask_type.capitalize()})\n'
                     f'({valid_count_max:,} valid pixels)', 
                     fontsize=14, fontweight='bold', pad=20)
        
        cbar2 = plt.colorbar(im2, ax=ax2, orientation='horizontal', pad=0.08, shrink=0.8)
        cbar2.set_label('Heat Index (°C)', fontsize=12, fontweight='bold')
        
        # Add extreme danger threshold
        ax2.contour(hi_max_data.longitude, hi_max_data.latitude, hi_max_masked,
                   levels=[54], colors=['purple'], linewidths=3, alpha=0.9,
                   transform=ccrs.PlateCarree())
        
        plt.tight_layout()
        
        # Save
        filename = f'heat_index_climatology_{season.lower()}_{mask_type}.png'
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {filename}")

def plot_heat_index_categories(hi_ds, output_dir, mask_type='both'):
    """Analyze heat index by danger categories."""
    output_dir = Path(output_dir)
    
    # Create land/ocean mask
    mask = create_land_ocean_mask(hi_ds, mask_type)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Apply mask to heat index values
    hi_values = hi_ds.heat_index.values
    hi_values_flat = hi_values.flatten()
    
    # Create a flat mask
    mask_flat = np.broadcast_to(mask, hi_values.shape).flatten()
    hi_masked_flat = hi_values_flat.copy()
    hi_masked_flat[~mask_flat] = np.nan
    hi_clean = hi_masked_flat[~np.isnan(hi_masked_flat)]
    
    # 1. Global distribution by category
    ax1 = axes[0, 0]
    
    category_counts = {}
    for cat_name, (min_val, max_val) in HI_CATEGORIES.items():
        count = np.sum((hi_clean >= min_val) & (hi_clean < max_val))
        category_counts[cat_name] = count
    
    categories = list(category_counts.keys())
    counts = list(category_counts.values())
    colors = [HI_COLORS[cat] for cat in categories]
    
    wedges, texts, autotexts = ax1.pie(counts, labels=categories, colors=colors, 
                                      autopct='%1.1f%%', startangle=90)
    ax1.set_title(f'Heat Index Distribution by Category ({mask_type.capitalize()})', fontweight='bold')
    
    # 2. Seasonal category analysis
    ax2 = axes[0, 1]
    
    seasonal_categories = {}
    hi_seasonal = hi_ds.heat_index.groupby('valid_time.season')
    
    for season in ['DJF', 'MAM', 'JJA', 'SON']:
        try:
            # Iterate through groups to find the matching season
            season_found = False
            for season_key, season_group in hi_seasonal:
                if season_key == season:
                    season_data = season_group.values
                    season_found = True
                    break
            
            if not season_found:
                continue
                
            # Apply mask to seasonal data
            season_mask_flat = np.broadcast_to(mask, season_data.shape).flatten()
            season_data_flat = season_data.flatten()
            season_masked_flat = season_data_flat.copy()
            season_masked_flat[~season_mask_flat] = np.nan
            season_clean = season_masked_flat[~np.isnan(season_masked_flat)]
            
            season_counts = []
            for cat_name, (min_val, max_val) in HI_CATEGORIES.items():
                count = np.sum((season_clean >= min_val) & (season_clean < max_val))
                pct = count / len(season_clean) * 100 if len(season_clean) > 0 else 0
                season_counts.append(pct)
            
            seasonal_categories[season] = season_counts
        except KeyError:
            continue
    
    # Stacked bar chart
    if seasonal_categories:
        bottom = np.zeros(len(seasonal_categories))
        width = 0.6
        
        for i, (cat_name, color) in enumerate(HI_COLORS.items()):
            values = [seasonal_categories[season][i] for season in seasonal_categories.keys()]
            ax2.bar(seasonal_categories.keys(), values, width, bottom=bottom, 
                   label=cat_name, color=color, alpha=0.8)
            bottom += values
        
        ax2.set_ylabel('Percentage (%)')
        ax2.set_title(f'Seasonal Heat Index Categories ({mask_type.capitalize()})', fontweight='bold')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 3. Dangerous heat index frequency map
    ax3 = axes[1, 0]
    
    # Calculate frequency of dangerous conditions (HI > 41°C)
    dangerous_freq = (hi_ds.heat_index > 41).mean(dim='valid_time') * 100
    
    # Apply mask to frequency data
    dangerous_masked, vmin_d, vmax_d, valid_count_d = apply_mask_and_adjust_colorbar(
        dangerous_freq.values, mask, percentile_range=(0.5, 99.5)
    )
    
    im3 = ax3.imshow(dangerous_masked, cmap='Reds', vmin=vmin_d, vmax=vmax_d,
                     extent=[hi_ds.longitude.min(), hi_ds.longitude.max(),
                            hi_ds.latitude.min(), hi_ds.latitude.max()],
                     aspect='auto')
    
    ax3.set_xlabel('Longitude (°)')
    ax3.set_ylabel('Latitude (°)')
    ax3.set_title(f'Frequency of Dangerous Heat Index\n(HI > 41°C, {mask_type.capitalize()})', fontweight='bold')
    
    cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.8)
    cbar3.set_label('Frequency (%)')
    
    # 4. Heat index statistics table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate statistics by category
    stats_data = [['Category', 'Range (°C)', f'{mask_type.capitalize()} %', 'Mean HI (°C)']]
    
    for cat_name, (min_val, max_val) in HI_CATEGORIES.items():
        mask_cat = (hi_clean >= min_val) & (hi_clean < max_val)
        count = np.sum(mask_cat)
        pct = count / len(hi_clean) * 100 if len(hi_clean) > 0 else 0
        mean_hi = np.mean(hi_clean[mask_cat]) if count > 0 else np.nan
        
        range_str = f'{min_val}-{max_val}' if max_val < 100 else f'>{min_val}'
        stats_data.append([cat_name, range_str, f'{pct:.1f}%', f'{mean_hi:.1f}'])
    
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
                cat_name = stats_data[i][0]
                cell.set_facecolor(HI_COLORS[cat_name])
                cell.set_alpha(0.3)
    
    ax4.set_title('Heat Index Category Statistics', fontweight='bold', pad=20)
    
    plt.suptitle(f'Heat Index Category Analysis ({mask_type.capitalize()})\n'
                f'Period: {hi_ds.valid_time.dt.year.min().values}-{hi_ds.valid_time.dt.year.max().values}\n'
                f'Valid pixels: {len(hi_clean):,}',
                fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    
    # Save
    filename = f'heat_index_categories_{mask_type}.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {filename}")

def plot_heat_index_relationships(hi_ds, rh_dir, temp_dir, output_dir, mask_type='both'):
    """Plot relationships between heat index, temperature, and humidity."""
    output_dir = Path(output_dir)
    
    # Create land/ocean mask
    mask = create_land_ocean_mask(hi_ds, mask_type)
    
    try:
        # Sample some data for relationship analysis
        hi_sample = hi_ds.heat_index.isel(valid_time=slice(0, None, 10))  # Every 10th time step
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Apply mask to sampled data
        hi_flat = hi_sample.values
        mask_sample = np.broadcast_to(mask, hi_flat.shape)
        hi_flat_masked = hi_flat.copy()
        hi_flat_masked[~mask_sample] = np.nan
        hi_clean = hi_flat_masked[~np.isnan(hi_flat_masked)]
        
        # 1. Heat Index distribution
        ax1 = axes[0, 0]
        
        ax1.hist(hi_clean, bins=50, density=True, alpha=0.7, color='red', edgecolor='black')
        ax1.axvline(np.mean(hi_clean), color='blue', linestyle='--', linewidth=2, 
                   label=f'Mean: {np.mean(hi_clean):.1f}°C')
        ax1.axvline(np.median(hi_clean), color='green', linestyle='--', linewidth=2, 
                   label=f'Median: {np.median(hi_clean):.1f}°C')
        
        # Add danger thresholds
        for temp, color, label in [(27, 'yellow', 'Caution'), (32, 'orange', 'Extreme Caution'),
                                  (41, 'red', 'Danger'), (54, 'darkred', 'Extreme Danger')]:
            ax1.axvline(temp, color=color, linestyle=':', linewidth=2, alpha=0.8)
        
        ax1.set_xlabel('Heat Index (°C)')
        ax1.set_ylabel('Density')
        ax1.set_title(f'Heat Index Distribution ({mask_type.capitalize()})', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Temporal patterns
        ax2 = axes[0, 1]
        
        # Monthly climatology - apply mask before averaging
        hi_monthly_raw = hi_ds.heat_index.groupby('valid_time.month').mean()
        
        # For monthly means, we need to carefully apply the mask
        monthly_means = []
        for month in range(1, 13):
            if month in hi_monthly_raw.month.values:
                month_data = hi_monthly_raw.sel(month=month).values
                month_masked = month_data.copy()
                month_masked[~mask] = np.nan
                monthly_means.append(np.nanmean(month_masked))
            else:
                monthly_means.append(np.nan)
        
        months = range(1, 13)
        month_names = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
        
        bars = ax2.bar(months, monthly_means, color='coral', alpha=0.7, edgecolor='black')
        
        # Color bars by danger level
        for i, (month, hi_val) in enumerate(zip(months, monthly_means)):
            if np.isnan(hi_val):
                continue
            if hi_val >= 54:
                bars[i].set_color(HI_COLORS['Extreme Danger'])
            elif hi_val >= 41:
                bars[i].set_color(HI_COLORS['Danger'])
            elif hi_val >= 32:
                bars[i].set_color(HI_COLORS['Extreme Caution'])
            elif hi_val >= 27:
                bars[i].set_color(HI_COLORS['Caution'])
            else:
                bars[i].set_color(HI_COLORS['Safe'])
        
        ax2.set_xlabel('Month')
        ax2.set_ylabel('Mean Heat Index (°C)')
        ax2.set_title(f'Monthly Heat Index Climatology ({mask_type.capitalize()})', fontweight='bold')
        ax2.set_xticks(months)
        ax2.set_xticklabels(month_names)
        ax2.grid(True, alpha=0.3)
        
        # Add danger threshold lines
        for temp, color in [(27, 'yellow'), (32, 'orange'), (41, 'red'), (54, 'darkred')]:
            ax2.axhline(temp, color=color, linestyle='--', alpha=0.7)
        
        # 3. Latitudinal variation
        ax3 = axes[1, 0]
        
        # Calculate zonal means by season with masking
        hi_seasonal = hi_ds.heat_index.groupby('valid_time.season')
        
        seasons = ['DJF', 'MAM', 'JJA', 'SON']
        colors = ['blue', 'green', 'red', 'orange']
        
        for season, color in zip(seasons, colors):
            # Find the season in the groups
            season_found = False
            for season_key, season_group in hi_seasonal:
                if season_key == season:
                    season_data = season_group.mean(dim='longitude')
                    season_found = True
                    break
            
            if not season_found:
                continue
                
            # Apply latitude-wise masking (simplified)
            season_zonal = season_data.mean(dim=['valid_time'])
            ax3.plot(season_zonal.latitude, season_zonal, color=color, linewidth=2, 
                    label=season, alpha=0.8)
        
        # Add danger threshold lines
        for temp, color, alpha in [(27, 'yellow', 0.5), (32, 'orange', 0.5), 
                                  (41, 'red', 0.7), (54, 'darkred', 0.9)]:
            ax3.axhline(temp, color=color, linestyle='--', alpha=alpha)
        
        ax3.set_xlabel('Latitude (°)')
        ax3.set_ylabel('Heat Index (°C)')
        ax3.set_title(f'Seasonal Latitudinal Variation ({mask_type.capitalize()})', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(-90, 90)
        
        # 4. Extreme heat index analysis
        ax4 = axes[1, 1]
        
        # Calculate extreme heat index days (>41°C) per year with masking
        extreme_days_raw = (hi_ds.heat_index > 41).groupby('valid_time.year').sum()
        
        # Apply mask and calculate mean
        extreme_means = []
        years_list = []
        for year in extreme_days_raw.year.values:
            year_data = extreme_days_raw.sel(year=year).values
            year_masked = year_data.copy()
            year_masked[~mask] = np.nan
            mean_extreme = np.nanmean(year_masked)
            if not np.isnan(mean_extreme):
                extreme_means.append(mean_extreme)
                years_list.append(year)
        
        if len(years_list) > 1:
            ax4.plot(years_list, extreme_means, 'ro-', linewidth=2, markersize=6)
            ax4.set_xlabel('Year')
            ax4.set_ylabel('Mean Extreme Heat Days per Grid Cell')
            ax4.set_title(f'Trend in Extreme Heat Index Days\n(HI > 41°C, {mask_type.capitalize()})', fontweight='bold')
            ax4.grid(True, alpha=0.3)
            
            # Add trend line if multiple years
            if len(years_list) > 2:
                z = np.polyfit(years_list, extreme_means, 1)
                p = np.poly1d(z)
                ax4.plot(years_list, p(years_list), 'r--', alpha=0.7, 
                        label=f'Trend: {z[0]:.2f} days/year')
                ax4.legend()
        else:
            # Single year or no variation - show spatial distribution
            extreme_spatial = (hi_ds.heat_index > 41).mean(dim='valid_time')
            
            # Apply mask to spatial data
            extreme_masked, vmin_e, vmax_e, valid_count_e = apply_mask_and_adjust_colorbar(
                extreme_spatial.values, mask, percentile_range=(0.5, 99.5)
            )
            
            im4 = ax4.imshow(extreme_masked, cmap='Reds', vmin=vmin_e, vmax=vmax_e,
                           extent=[hi_ds.longitude.min(), hi_ds.longitude.max(),
                                  hi_ds.latitude.min(), hi_ds.latitude.max()],
                           aspect='auto')
            ax4.set_xlabel('Longitude (°)')
            ax4.set_ylabel('Latitude (°)')
            ax4.set_title(f'Fraction of Extreme Heat Index Days\n(HI > 41°C, {mask_type.capitalize()})', fontweight='bold')
            plt.colorbar(im4, ax=ax4, shrink=0.8)
        
        plt.suptitle(f'Heat Index Analysis ({mask_type.capitalize()})\n'
                    f'Period: {hi_ds.valid_time.dt.year.min().values}-{hi_ds.valid_time.dt.year.max().values}\n'
                    f'Valid pixels: {len(hi_clean):,}',
                    fontsize=16, fontweight='bold', y=0.95)
        
        plt.tight_layout()
        
        # Save
        filename = f'heat_index_relationships_{mask_type}.png'
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {filename}")
        
    except Exception as e:
        print(f"Note: Could not create relationship plots: {e}")

def calculate_heat_index_trends(hi_ds, mask_type='both'):
    """
    Calculate spatial trends in heat index data.
    
    Args:
        hi_ds: xarray Dataset with heat index data
        mask_type: 'land', 'ocean', or 'both'
    
    Returns:
        trends: array of trend slopes (per year)
        p_values: array of p-values
        significant: boolean array where True = significant trend
    """
    from scipy import stats
    
    # Create land/ocean mask
    mask = create_land_ocean_mask(hi_ds, mask_type)
    
    # Get years for trend calculation
    years = hi_ds.valid_time.dt.year.values
    unique_years = np.unique(years)
    
    if len(unique_years) < 5:
        print("Warning: Less than 5 years of data - trend analysis may be unreliable")
        return None, None, None
    
    print(f"Calculating trends for {len(unique_years)} years: {unique_years[0]}-{unique_years[-1]}")
    
    # Calculate annual means for trend analysis
    hi_annual = hi_ds.heat_index.groupby('valid_time.year').mean()
    
    # Initialize arrays
    trends = np.full((len(hi_ds.latitude), len(hi_ds.longitude)), np.nan)
    p_values = np.full((len(hi_ds.latitude), len(hi_ds.longitude)), np.nan)
    significant = np.full((len(hi_ds.latitude), len(hi_ds.longitude)), False)
    
    # Calculate trends for each pixel
    for i in range(len(hi_ds.latitude)):
        for j in range(len(hi_ds.longitude)):
            if mask[i, j]:  # Only calculate for masked pixels
                pixel_data = hi_annual.isel(latitude=i, longitude=j).values
                
                if not np.any(np.isnan(pixel_data)) and len(pixel_data) >= 3:
                    # Linear regression
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        unique_years, pixel_data
                    )
                    
                    trends[i, j] = slope
                    p_values[i, j] = p_value
                    significant[i, j] = p_value < 0.05  # 95% confidence level
    
    return trends, p_values, significant

def plot_heat_index_trends(hi_ds, output_dir, mask_type='both'):
    """Plot spatial trends in heat index data."""
    output_dir = Path(output_dir)
    
    # Calculate trends
    trends, p_values, significant = calculate_heat_index_trends(hi_ds, mask_type)
    
    if trends is None:
        print("Insufficient data for trend analysis")
        return
    
    # Create land/ocean mask
    mask = create_land_ocean_mask(hi_ds, mask_type)
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # 1. Trend magnitude
    ax1 = axes[0, 0]
    
    # Apply mask and calculate percentiles
    trends_masked, vmin_trend, vmax_trend, valid_count_trend = apply_mask_and_adjust_colorbar(
        trends, mask, percentile_range=(2.5, 97.5)
    )
    
    # Convert to per-decade for better interpretation
    trends_decade = trends_masked * 10
    
    im1 = ax1.contourf(hi_ds.longitude, hi_ds.latitude, trends_decade,
                      levels=20, cmap='RdBu_r', transform=ccrs.PlateCarree(),
                      extend='both', vmin=vmin_trend*10, vmax=vmax_trend*10)
    
    ax1.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax1.add_feature(cfeature.BORDERS, linewidth=0.3)
    ax1.set_global()
    ax1.set_title(f'Heat Index Trends ({mask_type.capitalize()})\n'
                  f'Change per decade (°C/decade)', fontsize=14, fontweight='bold', pad=20)
    
    cbar1 = plt.colorbar(im1, ax=ax1, orientation='horizontal', pad=0.08, shrink=0.8)
    cbar1.set_label('Trend (°C/decade)', fontsize=12, fontweight='bold')
    
    # 2. Trend significance
    ax2 = axes[0, 1]
    
    # Create significance mask
    sig_masked = significant.copy()
    sig_masked[~mask] = False
    
    # Plot significant trends only
    sig_trends = trends_decade.copy()
    sig_trends[~sig_masked] = np.nan
    
    im2 = ax2.contourf(hi_ds.longitude, hi_ds.latitude, sig_trends,
                      levels=20, cmap='RdBu_r', transform=ccrs.PlateCarree(),
                      extend='both', vmin=vmin_trend*10, vmax=vmax_trend*10)
    
    ax2.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax2.add_feature(cfeature.BORDERS, linewidth=0.3)
    ax2.set_global()
    ax2.set_title(f'Significant Heat Index Trends ({mask_type.capitalize()})\n'
                  f'p < 0.05 only', fontsize=14, fontweight='bold', pad=20)
    
    cbar2 = plt.colorbar(im2, ax=ax2, orientation='horizontal', pad=0.08, shrink=0.8)
    cbar2.set_label('Trend (°C/decade)', fontsize=12, fontweight='bold')
    
    # 3. Trend direction summary
    ax3 = axes[1, 0]
    
    # Classify trends
    increasing = (trends_decade > 0) & sig_masked
    decreasing = (trends_decade < 0) & sig_masked
    no_trend = ~sig_masked & mask
    
    # Create categorical plot
    trend_categories = np.full_like(trends_decade, np.nan)
    trend_categories[increasing] = 1
    trend_categories[decreasing] = -1
    trend_categories[no_trend] = 0
    
    colors = ['blue', 'gray', 'red']
    labels = ['Decreasing', 'No significant trend', 'Increasing']
    
    im3 = ax3.contourf(hi_ds.longitude, hi_ds.latitude, trend_categories,
                      levels=[-1.5, -0.5, 0.5, 1.5], colors=colors,
                      transform=ccrs.PlateCarree())
    
    ax3.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax3.add_feature(cfeature.BORDERS, linewidth=0.3)
    ax3.set_global()
    ax3.set_title(f'Trend Direction Summary ({mask_type.capitalize()})', 
                  fontsize=14, fontweight='bold', pad=20)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=label) 
                      for color, label in zip(colors, labels)]
    ax3.legend(handles=legend_elements, loc='lower right', framealpha=0.8)
    
    # 4. Trend statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate statistics
    valid_trends = trends_decade[sig_masked]
    if len(valid_trends) > 0:
        increasing_pct = np.sum(valid_trends > 0) / len(valid_trends) * 100
        decreasing_pct = np.sum(valid_trends < 0) / len(valid_trends) * 100
        mean_trend = np.mean(valid_trends)
        max_trend = np.max(valid_trends)
        min_trend = np.min(valid_trends)
        
        stats_text = f"""
        TREND STATISTICS
        
        Significant Trends (p < 0.05): {np.sum(sig_masked):,} pixels
        
        Direction:
        • Increasing: {increasing_pct:.1f}%
        • Decreasing: {decreasing_pct:.1f}%
        
        Magnitude:
        • Mean trend: {mean_trend:.2f}°C/decade
        • Maximum: {max_trend:.2f}°C/decade
        • Minimum: {min_trend:.2f}°C/decade
        
        Period: {hi_ds.valid_time.dt.year.min().values}-{hi_ds.valid_time.dt.year.max().values}
        """
    else:
        stats_text = "No significant trends found"
    
    ax4.text(0.5, 0.5, stats_text, transform=ax4.transAxes, 
             fontsize=12, ha='center', va='center',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.suptitle(f'Heat Index Trend Analysis ({mask_type.capitalize()})\n'
                f'Period: {hi_ds.valid_time.dt.year.min().values}-{hi_ds.valid_time.dt.year.max().values}',
                fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    
    # Save
    filename = f'heat_index_trends_{mask_type}.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {filename}")

def plot_regional_heat_index_comparison(hi_ds, output_dir, mask_type='both'):
    """Compare heat index patterns across different regions."""
    output_dir = Path(output_dir)
    
    # Create land/ocean mask
    mask = create_land_ocean_mask(hi_ds, mask_type)
    
    # Define regions of interest
    regions = {
        'Persian Gulf': {'lat': (24, 32), 'lon': (48, 58), 'color': 'red'},
        'Sahara': {'lat': (15, 30), 'lon': (-5, 25), 'color': 'orange'},
        'Indian Subcontinent': {'lat': (8, 30), 'lon': (68, 88), 'color': 'purple'},
        'Southeast USA': {'lat': (25, 35), 'lon': (-95, -75), 'color': 'blue'},
        'Northern Australia': {'lat': (-20, -10), 'lon': (130, 145), 'color': 'green'}
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Calculate regional means
    regional_data = {}
    for region_name, bounds in regions.items():
        lat_mask = (hi_ds.latitude >= bounds['lat'][0]) & (hi_ds.latitude <= bounds['lat'][1])
        lon_mask = (hi_ds.longitude >= bounds['lon'][0]) & (hi_ds.longitude <= bounds['lon'][1])
        
        # Combine with land/ocean mask
        regional_mask = mask & lat_mask.values[:, None] & lon_mask.values[None, :]
        
        if np.any(regional_mask):
            regional_hi = hi_ds.heat_index.where(regional_mask, drop=False).mean(dim=['latitude', 'longitude'])
            regional_data[region_name] = regional_hi
    
    # 1. Time series comparison
    ax1 = axes[0, 0]
    
    for region_name, hi_data in regional_data.items():
        if len(hi_data) > 0 and not hi_data.isnull().all():
            time_pd = pd.to_datetime(hi_data.valid_time.values)
            ax1.plot(time_pd, hi_data, linewidth=1.5, label=region_name, 
                    color=regions[region_name]['color'], alpha=0.8)
    
    # Add danger thresholds
    for temp, color in [(41, 'red'), (54, 'darkred')]:
        ax1.axhline(temp, color=color, linestyle='--', alpha=0.7)
    
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Regional Mean Heat Index (°C)')
    ax1.set_title(f'Regional Heat Index Time Series ({mask_type.capitalize()})', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Seasonal patterns
    ax2 = axes[0, 1]
    
    seasonal_means = {}
    for region_name, hi_data in regional_data.items():
        if len(hi_data) > 0 and not hi_data.isnull().all():
            seasonal_mean = hi_data.groupby('valid_time.season').mean()
            seasonal_means[region_name] = seasonal_mean
    
    if seasonal_means:
        seasons = ['DJF', 'MAM', 'JJA', 'SON']
        x_pos = np.arange(len(seasons))
        width = 0.15
        
        for i, (region_name, seasonal_data) in enumerate(seasonal_means.items()):
            values = [seasonal_data.sel(season=s).values if s in seasonal_data.season.values else np.nan 
                     for s in seasons]
            bars = ax2.bar(x_pos + i*width, values, width, label=region_name, 
                          color=regions[region_name]['color'], alpha=0.7)
            
            # Color bars above danger threshold
            for bar, val in zip(bars, values):
                if not np.isnan(val):
                    if val >= 54:
                        bar.set_edgecolor('darkred')
                        bar.set_linewidth(3)
                    elif val >= 41:
                        bar.set_edgecolor('red')
                        bar.set_linewidth(2)
        
        ax2.set_xlabel('Season')
        ax2.set_ylabel('Mean Heat Index (°C)')
        ax2.set_title(f'Seasonal Regional Comparison ({mask_type.capitalize()})', fontweight='bold')
        ax2.set_xticks(x_pos + width*2)
        ax2.set_xticklabels(seasons)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add danger threshold lines
        for temp, color in [(41, 'red'), (54, 'darkred')]:
            ax2.axhline(temp, color=color, linestyle='--', alpha=0.7)
    
    # 3. Extreme heat frequency by region
    ax3 = axes[1, 0]
    
    extreme_freq = {}
    for region_name, hi_data in regional_data.items():
        if len(hi_data) > 0 and not hi_data.isnull().all():
            # Calculate frequency of different danger levels
            caution_freq = ((hi_data >= 27) & (hi_data < 32)).mean().values * 100
            extreme_caution_freq = ((hi_data >= 32) & (hi_data < 41)).mean().values * 100
            danger_freq = ((hi_data >= 41) & (hi_data < 54)).mean().values * 100
            extreme_danger_freq = (hi_data >= 54).mean().values * 100
            
            extreme_freq[region_name] = [caution_freq, extreme_caution_freq, danger_freq, extreme_danger_freq]
    
    # Stacked bar chart
    if extreme_freq:
        categories = ['Caution', 'Extreme Caution', 'Danger', 'Extreme Danger']
        cat_colors = [HI_COLORS['Caution'], HI_COLORS['Extreme Caution'], 
                      HI_COLORS['Danger'], HI_COLORS['Extreme Danger']]
        
        regions_list = list(extreme_freq.keys())
        bottom = np.zeros(len(regions_list))
        
        for i, (category, color) in enumerate(zip(categories, cat_colors)):
            values = [extreme_freq[region][i] for region in regions_list]
            ax3.bar(regions_list, values, bottom=bottom, label=category, 
                   color=color, alpha=0.8)
            bottom += values
        
        ax3.set_ylabel('Frequency (%)')
        ax3.set_title(f'Heat Index Danger Category Frequency by Region ({mask_type.capitalize()})', fontweight='bold')
        ax3.legend()
        plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
    
    # 4. Regional statistics table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    stats_data = [['Region', 'Mean (°C)', 'Max (°C)', 'Danger Days (%)']]
    
    for region_name, hi_data in regional_data.items():
        if len(hi_data) > 0 and not hi_data.isnull().all():
            clean_data = hi_data.values[~np.isnan(hi_data.values)]
            if len(clean_data) > 0:
                mean_hi = np.mean(clean_data)
                max_hi = np.max(clean_data)
                danger_pct = np.sum(clean_data >= 41) / len(clean_data) * 100
                
                stats_data.append([
                    region_name,
                    f'{mean_hi:.1f}',
                    f'{max_hi:.1f}',
                    f'{danger_pct:.1f}%'
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
                    region_name = stats_data[i][0]
                    cell.set_facecolor(regions[region_name]['color'])
                    cell.set_alpha(0.3)
    
    ax4.set_title('Regional Heat Index Statistics', fontweight='bold', pad=20)
    
    plt.suptitle(f'Regional Heat Index Comparison ({mask_type.capitalize()})\n'
                f'Period: {hi_ds.valid_time.dt.year.min().values}-{hi_ds.valid_time.dt.year.max().values}',
                fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    
    # Save
    filename = f'heat_index_regional_comparison_{mask_type}.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {filename}")

def main():
    """Main visualization function for heat index data."""
    parser = argparse.ArgumentParser(description='Visualize heat index data')
    
    parser.add_argument('--hi-dir', default='data/processed/heat_index',
                       help='Directory containing heat index files')
    parser.add_argument('--output-dir', default='visualizations/output/heat_index',
                       help='Output directory for plots')
    parser.add_argument('--rh-dir', default='data/processed/relative_humidity',
                       help='Directory containing RH files (for relationships)')
    parser.add_argument('--temp-dir', default='/data/climate/disk3/datasets/era5',
                       help='Directory containing temperature files (for relationships)')
    parser.add_argument('--years', nargs='+', type=int,
                       help='Specific years to analyze (default: all available)')
    parser.add_argument('--start-year', type=int,
                       help='Start year for analysis (default: earliest available)')
    parser.add_argument('--end-year', type=int,
                       help='End year for analysis (default: latest available)')
    parser.add_argument('--months', nargs='+', type=int, default=[1, 4, 7, 10],
                       help='Months to sample (default: 1 4 7 10)')
    parser.add_argument('--mask-type', choices=['land', 'ocean', 'both'], default='both',
                       help='Analysis domain: land-only, ocean-only, or both (default: both)')
    parser.add_argument('--include-trends', action='store_true',
                       help='Include spatial trend analysis with significance testing')
    parser.add_argument('--percentile-cap', type=float, default=99.5,
                       help='Percentile for capping extreme values in colorbars (default: 99.5)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("HEAT INDEX VISUALIZATION")
    print("="*80)
    print(f"Input directory: {args.hi_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Sample months: {args.months}")
    print(f"Analysis domain: {args.mask_type}")
    if args.start_year or args.end_year:
        year_range = f"{args.start_year or 'earliest'}-{args.end_year or 'latest'}"
        print(f"Year range: {year_range}")
    print("="*80)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load data
        hi_ds, file_info = load_heat_index_data(args.hi_dir, args.years, args.months, args.start_year, args.end_year)
        
        print("\nCreating visualizations...")
        
        # 1. Global climatology maps
        print("1. Global climatology maps...")
        plot_global_heat_index_climatology(hi_ds, output_dir, args.mask_type)
        
        # 2. Category analysis
        print("2. Heat index category analysis...")
        plot_heat_index_categories(hi_ds, output_dir, args.mask_type)
        
        # 3. Relationships and patterns
        print("3. Heat index relationships...")
        plot_heat_index_relationships(hi_ds, args.rh_dir, args.temp_dir, output_dir, args.mask_type)
        
        # 4. Regional comparison
        print("4. Regional comparison...")
        plot_regional_heat_index_comparison(hi_ds, output_dir, args.mask_type)
        
        # 5. Trend analysis (if requested and sufficient data)
        if args.include_trends:
            print("5. Trend analysis...")
            plot_heat_index_trends(hi_ds, output_dir, args.mask_type)
        
        print("\n" + "="*80)
        print("HEAT INDEX VISUALIZATION COMPLETED!")
        print("="*80)
        print(f"Output files saved in: {output_dir}")
        print("\nGenerated plots:")
        for plot_file in output_dir.glob('*.png'):
            print(f"  - {plot_file.name}")
        
        # Close dataset
        hi_ds.close()
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())