#!/usr/bin/env python3
"""
Visualization of ETCCDI Precipitation Indices (Output from 08_ETCCDI_Precipitation_Indices.py)

Creates comprehensive scientific visualizations of ETCCDI precipitation indices:
- Global maps of PRCPTOT, R95p, R95pTOT, WD50
- Temporal trends and climate change signals
- Regional analysis and hotspot identification
- Extreme precipitation concentration analysis
- Comparison with climate change projections

Input files:
- etccdi_precipitation_indices_{year}.nc files from data/processed/etccdi_indices/
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
            # Use regionmask with Natural Earth land boundaries (much faster!)
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
        
        # Remove obvious ocean areas
        land_mask &= ~((lat_grid >= -10) & (lat_grid <= 10) & (lon_grid >= -40) & (lon_grid <= 20))  # Atlantic
        land_mask &= ~((lat_grid >= -10) & (lat_grid <= 10) & (lon_grid >= 80) & (lon_grid <= 120))   # Indian Ocean
        
        if mask_type == 'land':
            return land_mask
        elif mask_type == 'ocean':
            return ~land_mask
        else:
            return np.ones_like(land_mask, dtype=bool)

def calculate_pixel_trends(data_array, years, significance_level=0.05):
    """
    Calculate trends and significance for each pixel.
    
    Args:
        data_array: xarray DataArray with time dimension
        years: array of years
        significance_level: p-value threshold for significance (default 0.05)
    
    Returns:
        trends: array of trend slopes (per year)
        p_values: array of p-values
        significant: boolean array where True = significant trend
    """
    if len(years) < 5:
        print("Warning: Less than 5 years of data - trend analysis may be unreliable")
    
    # Initialize output arrays
    trends = np.full(data_array.shape[1:], np.nan)
    p_values = np.full(data_array.shape[1:], np.nan)
    
    # Calculate trends for each pixel
    for i in range(data_array.shape[1]):
        for j in range(data_array.shape[2]):
            pixel_data = data_array[:, i, j].values
            
            # Skip if too many NaN values
            valid_mask = ~np.isnan(pixel_data)
            if np.sum(valid_mask) < 5:  # Need at least 5 years
                continue
            
            valid_years = years[valid_mask]
            valid_data = pixel_data[valid_mask]
            
            # Linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(valid_years, valid_data)
            
            trends[i, j] = slope
            p_values[i, j] = p_value
    
    # Determine significance
    significant = p_values < significance_level
    
    return trends, p_values, significant

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

def load_etccdi_data(etccdi_dir, years=None):
    """Load ETCCDI precipitation indices data."""
    etccdi_dir = Path(etccdi_dir)
    
    if years is None:
        # Find available years
        etccdi_files = list(etccdi_dir.glob('etccdi_precipitation_indices_*.nc'))
        years = sorted(set([int(f.name.split('_')[3].split('.')[0]) for f in etccdi_files]))
        print(f"Found data for years: {years[0]}-{years[-1]}")
    
    print(f"Loading ETCCDI data for years {years}...")
    
    datasets = []
    
    for year in years:
        file_path = etccdi_dir / f'etccdi_precipitation_indices_{year}.nc'
        
        if file_path.exists():
            try:
                ds = xr.open_dataset(file_path)
                
                # Fix timedelta variables by converting to days (float)
                timedelta_vars = ['WD50', 'wet_days', 'very_wet_days']
                for var in timedelta_vars:
                    if var in ds.data_vars and 'timedelta' in str(ds[var].dtype):
                        print(f"    Converting {var} from timedelta to days")
                        ds[var] = ds[var] / np.timedelta64(1, 'D')
                        ds[var] = ds[var].astype(np.float64)
                
                ds = ds.expand_dims('year').assign_coords(year=[year])
                datasets.append(ds)
                print(f"  Loaded: {file_path.name}")
            except Exception as e:
                print(f"  Error loading {file_path}: {e}")
    
    if not datasets:
        raise ValueError(f"No ETCCDI files found in {etccdi_dir}")
    
    # Combine datasets
    combined_ds = xr.concat(datasets, dim='year')
    
    print(f"Combined dataset: {len(years)} years")
    print(f"Variables: {list(combined_ds.data_vars)}")
    print(f"Grid dimensions: {combined_ds.dims}")
    
    return combined_ds

def plot_global_etccdi_climatology(etccdi_ds, output_dir, mask_type='both'):
    """Create global maps of ETCCDI indices climatology."""
    output_dir = Path(output_dir)
    
    # Create land/ocean mask
    mask = create_land_ocean_mask(etccdi_ds, mask_type)
    
    # Calculate multi-year means
    prcptot_mean = etccdi_ds.PRCPTOT.mean(dim='year')
    r95p_mean = etccdi_ds.R95p.mean(dim='year')
    r95ptot_mean = etccdi_ds.R95pTOT.mean(dim='year')
    wd50_mean = etccdi_ds.WD50.mean(dim='year')
    
    fig = plt.figure(figsize=(20, 16))
    
    # 1. PRCPTOT (Annual precipitation from wet days)
    ax1 = plt.subplot(2, 2, 1, projection=ccrs.Robinson())
    ax1.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax1.add_feature(cfeature.BORDERS, linewidth=0.3)
    ax1.set_global()
    
    # Apply mask and calculate percentiles from masked data
    prcptot_masked, prcptot_p01, prcptot_p99, prcptot_count = apply_mask_and_adjust_colorbar(
        prcptot_mean.values, mask, percentile_range=(0.5, 99.5)
    )
    
    print(f"  PRCPTOT ({mask_type}): {prcptot_count:,} pixels, "
          f"range {prcptot_p01:.0f}-{prcptot_p99:.0f} mm")
    
    levels_prcptot = np.linspace(max(prcptot_p01, 0), prcptot_p99, 20)
    cmap_prcptot = plt.cm.Blues
    
    # Use masked data for plotting
    prcptot_plot = np.clip(prcptot_masked, prcptot_p01, prcptot_p99)
    
    im1 = ax1.contourf(prcptot_mean.longitude, prcptot_mean.latitude, prcptot_plot,
                      levels=levels_prcptot, cmap=cmap_prcptot, transform=ccrs.PlateCarree(),
                      extend='both')
    
    ax1.set_title('PRCPTOT: Annual Precipitation from Wet Days\n'
                 f'Period: {etccdi_ds.year.min().values}-{etccdi_ds.year.max().values}\n'
                 f'({mask_type.capitalize()}, capped at {prcptot_p99:.0f} mm)',
                 fontsize=12, fontweight='bold')
    
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('Precipitation (mm/year)', fontsize=10)
    
    # 2. R95p (Annual precipitation from very wet days)
    ax2 = plt.subplot(2, 2, 2, projection=ccrs.Robinson())
    ax2.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax2.add_feature(cfeature.BORDERS, linewidth=0.3)
    ax2.set_global()
    
    # Use percentile-based levels for R95p as well
    r95p_valid = r95p_mean.values[~np.isnan(r95p_mean.values)]
    r95p_p99 = np.percentile(r95p_valid, 99.5)  # Cap at 99.5th percentile
    r95p_p01 = np.percentile(r95p_valid, 0.5)   # Floor at 0.5th percentile
    
    print(f"  R95p: Range {r95p_valid.min():.0f}-{r95p_valid.max():.0f} mm, "
          f"capped at {r95p_p01:.0f}-{r95p_p99:.0f} mm")
    
    levels_r95p = np.linspace(max(r95p_p01, 0), r95p_p99, 20)
    cmap_r95p = plt.cm.Reds
    
    # Cap extreme values for better visualization
    r95p_capped = np.clip(r95p_mean, r95p_p01, r95p_p99)
    
    im2 = ax2.contourf(r95p_capped.longitude, r95p_capped.latitude, r95p_capped,
                      levels=levels_r95p, cmap=cmap_r95p, transform=ccrs.PlateCarree(),
                      extend='both')
    
    ax2.set_title('R95p: Annual Precipitation from Very Wet Days\n(>P95 threshold)\n'
                 f'(Capped at 99.5th percentile: {r95p_p99:.0f} mm)',
                 fontsize=12, fontweight='bold')
    
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label('Precipitation (mm/year)', fontsize=10)
    
    # 3. R95pTOT (Contribution from very wet days)
    ax3 = plt.subplot(2, 2, 3, projection=ccrs.Robinson())
    ax3.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax3.add_feature(cfeature.BORDERS, linewidth=0.3)
    ax3.set_global()
    
    levels_r95ptot = np.linspace(0, 1, 21)
    cmap_r95ptot = plt.cm.plasma
    
    im3 = ax3.contourf(r95ptot_mean.longitude, r95ptot_mean.latitude, r95ptot_mean,
                      levels=levels_r95ptot, cmap=cmap_r95ptot, transform=ccrs.PlateCarree(),
                      extend='both')
    
    ax3.set_title('R95pTOT: Contribution from Very Wet Days\n(R95p/PRCPTOT)',
                 fontsize=12, fontweight='bold')
    
    cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.8)
    cbar3.set_label('Fraction', fontsize=10)
    
    # 4. WD50 (Precipitation concentration)
    ax4 = plt.subplot(2, 2, 4, projection=ccrs.Robinson())
    ax4.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax4.add_feature(cfeature.BORDERS, linewidth=0.3)
    ax4.set_global()
    
    levels_wd50 = np.arange(1, 51, 2)
    cmap_wd50 = plt.cm.viridis_r
    
    im4 = ax4.contourf(wd50_mean.longitude, wd50_mean.latitude, wd50_mean,
                      levels=levels_wd50, cmap=cmap_wd50, transform=ccrs.PlateCarree(),
                      extend='both')
    
    ax4.set_title('WD50: Precipitation Concentration\n(Days for 50% of annual total)',
                 fontsize=12, fontweight='bold')
    
    cbar4 = plt.colorbar(im4, ax=ax4, shrink=0.8)
    cbar4.set_label('Days', fontsize=10)
    
    plt.suptitle(f'ETCCDI Precipitation Indices Climatology\n'
                f'Period: {etccdi_ds.year.min().values}-{etccdi_ds.year.max().values}',
                fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    
    # Save
    filename = 'etccdi_indices_climatology.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {filename}")

def plot_etccdi_trends(etccdi_ds, output_dir):
    """Plot temporal trends in ETCCDI indices."""
    output_dir = Path(output_dir)
    
    if len(etccdi_ds.year) < 3:
        print("Insufficient years for trend analysis")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    years = etccdi_ds.year.values
    
    # Calculate global means for each year
    indices = ['PRCPTOT', 'R95p', 'R95pTOT', 'WD50']
    colors = ['blue', 'red', 'purple', 'green']
    
    for i, (index, color) in enumerate(zip(indices, colors)):
        ax = axes[i//2, i%2]
        
        if index in etccdi_ds.data_vars:
            global_mean = etccdi_ds[index].mean(dim=['latitude', 'longitude'])
            
            ax.plot(years, global_mean, 'o-', color=color, linewidth=2, markersize=6)
            
            # Add trend line if enough years
            if len(years) > 5:
                # Remove NaN values for trend calculation
                valid_mask = ~np.isnan(global_mean.values)
                if np.sum(valid_mask) > 3:
                    valid_years = years[valid_mask]
                    valid_values = global_mean.values[valid_mask]
                    
                    z = np.polyfit(valid_years, valid_values, 1)
                    p = np.poly1d(z)
                    ax.plot(years, p(years), '--', color=color, alpha=0.7,
                           label=f'Trend: {z[0]:.3f}/year')
                    ax.legend()
            
            ax.set_xlabel('Year')
            
            # Set appropriate y-label and title
            if index == 'PRCPTOT':
                ax.set_ylabel('Annual Precipitation (mm)')
                ax.set_title('PRCPTOT Trend (Wet Day Precipitation)', fontweight='bold')
            elif index == 'R95p':
                ax.set_ylabel('Annual Precipitation (mm)')
                ax.set_title('R95p Trend (Very Wet Day Precipitation)', fontweight='bold')
            elif index == 'R95pTOT':
                ax.set_ylabel('Fraction')
                ax.set_title('R95pTOT Trend (Extreme Precip Contribution)', fontweight='bold')
                ax.set_ylim(0, 1)
            elif index == 'WD50':
                ax.set_ylabel('Days')
                ax.set_title('WD50 Trend (Precipitation Concentration)', fontweight='bold')
            
            ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'ETCCDI Precipitation Index Trends\n'
                f'Period: {years[0]}-{years[-1]}',
                fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    
    # Save
    filename = 'etccdi_trends.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {filename}")

def plot_extreme_precipitation_analysis(etccdi_ds, output_dir):
    """Analyze extreme precipitation patterns using ETCCDI indices."""
    output_dir = Path(output_dir)
    
    # Calculate multi-year statistics
    r95p_mean = etccdi_ds.R95p.mean(dim='year')
    r95ptot_mean = etccdi_ds.R95pTOT.mean(dim='year')
    wd50_mean = etccdi_ds.WD50.mean(dim='year')
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Extreme precipitation hotspots
    ax1 = axes[0, 0]
    
    # Define hotspots as areas with high R95p AND high R95pTOT
    r95p_threshold = np.nanpercentile(r95p_mean.values, 90)
    r95ptot_threshold = np.nanpercentile(r95ptot_mean.values, 75)
    
    hotspot_mask = (r95p_mean >= r95p_threshold) & (r95ptot_mean >= r95ptot_threshold)
    
    # Create hotspot index
    hotspot_index = (r95p_mean / np.nanmax(r95p_mean.values)) + (r95ptot_mean / np.nanmax(r95ptot_mean.values))
    
    im1 = ax1.imshow(hotspot_index, cmap='hot_r', vmin=0, vmax=2,
                     extent=[etccdi_ds.longitude.min(), etccdi_ds.longitude.max(),
                            etccdi_ds.latitude.min(), etccdi_ds.latitude.max()],
                     aspect='auto')
    
    # Highlight extreme hotspots
    ax1.contour(hotspot_mask.longitude, hotspot_mask.latitude, hotspot_mask.astype(int),
               levels=[0.5], colors=['black'], linewidths=2)
    
    ax1.set_xlabel('Longitude (°)')
    ax1.set_ylabel('Latitude (°)')
    ax1.set_title('Extreme Precipitation Hotspots\n(High R95p + High R95pTOT)', fontweight='bold')
    
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('Hotspot Index', fontsize=10)
    
    # 2. Precipitation concentration analysis
    ax2 = axes[0, 1]
    
    # WD50 analysis - lower values indicate more concentrated precipitation
    im2 = ax2.imshow(wd50_mean, cmap='RdYlBu', vmin=1, vmax=50,
                     extent=[etccdi_ds.longitude.min(), etccdi_ds.longitude.max(),
                            etccdi_ds.latitude.min(), etccdi_ds.latitude.max()],
                     aspect='auto')
    
    ax2.set_xlabel('Longitude (°)')
    ax2.set_ylabel('Latitude (°)')
    ax2.set_title('Precipitation Concentration (WD50)\nLower = More Concentrated', fontweight='bold')
    
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label('Days for 50% of Annual Precipitation', fontsize=10)
    
    # 3. R95pTOT distribution
    ax3 = axes[1, 0]
    
    r95ptot_flat = r95ptot_mean.values.flatten()
    r95ptot_clean = r95ptot_flat[~np.isnan(r95ptot_flat)]
    
    ax3.hist(r95ptot_clean, bins=50, density=True, alpha=0.7, color='purple', edgecolor='black')
    ax3.axvline(np.mean(r95ptot_clean), color='red', linestyle='--', linewidth=2,
               label=f'Global Mean: {np.mean(r95ptot_clean):.2f}')
    ax3.axvline(np.median(r95ptot_clean), color='orange', linestyle='--', linewidth=2,
               label=f'Global Median: {np.median(r95ptot_clean):.2f}')
    
    ax3.set_xlabel('R95pTOT (Fraction)')
    ax3.set_ylabel('Density')
    ax3.set_title('Global Distribution of Extreme Precipitation Contribution', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 1)
    
    # 4. Regional statistics table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate regional statistics
    regions = {
        'Global': {'lat': (-90, 90), 'lon': (-180, 180)},
        'Tropics': {'lat': (-23.5, 23.5), 'lon': (-180, 180)},
        'NH Mid-Lat': {'lat': (30, 60), 'lon': (-180, 180)},
        'SH Mid-Lat': {'lat': (-60, -30), 'lon': (-180, 180)}
    }
    
    stats_data = [['Region', 'PRCPTOT (mm)', 'R95p (mm)', 'R95pTOT', 'WD50 (days)']]
    
    for region_name, bounds in regions.items():
        lat_mask = (etccdi_ds.latitude >= bounds['lat'][0]) & (etccdi_ds.latitude <= bounds['lat'][1])
        lon_mask = (etccdi_ds.longitude >= bounds['lon'][0]) & (etccdi_ds.longitude <= bounds['lon'][1])
        
        regional_data = etccdi_ds.where(lat_mask & lon_mask, drop=True).mean(dim=['latitude', 'longitude', 'year'])
        
        stats_data.append([
            region_name,
            f"{regional_data.PRCPTOT.values:.0f}" if not np.isnan(regional_data.PRCPTOT.values) else "N/A",
            f"{regional_data.R95p.values:.0f}" if not np.isnan(regional_data.R95p.values) else "N/A",
            f"{regional_data.R95pTOT.values:.2f}" if not np.isnan(regional_data.R95pTOT.values) else "N/A",
            f"{regional_data.WD50.values:.0f}" if not np.isnan(regional_data.WD50.values) else "N/A"
        ])
    
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
                cell.set_facecolor('#F2F2F2' if i % 2 == 0 else 'white')
    
    ax4.set_title('Regional ETCCDI Statistics', fontweight='bold', pad=20)
    
    plt.suptitle(f'Extreme Precipitation Analysis (ETCCDI)\n'
                f'Period: {etccdi_ds.year.min().values}-{etccdi_ds.year.max().values}',
                fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    
    # Save
    filename = 'etccdi_extreme_precipitation_analysis.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {filename}")

def plot_precipitation_concentration_analysis(etccdi_ds, output_dir):
    """Analyze precipitation concentration patterns using WD50 and R95pTOT."""
    output_dir = Path(output_dir)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Calculate multi-year means
    wd50_mean = etccdi_ds.WD50.mean(dim='year')
    r95ptot_mean = etccdi_ds.R95pTOT.mean(dim='year')
    prcptot_mean = etccdi_ds.PRCPTOT.mean(dim='year')
    
    # 1. WD50 vs R95pTOT relationship
    ax1 = axes[0, 0]
    
    # Sample data for scatter plot
    wd50_flat = wd50_mean.values.flatten()[::100]  # Every 100th point
    r95ptot_flat = r95ptot_mean.values.flatten()[::100]
    
    # Remove NaN values
    mask = ~(np.isnan(wd50_flat) | np.isnan(r95ptot_flat))
    wd50_clean = wd50_flat[mask]
    r95ptot_clean = r95ptot_flat[mask]
    
    if len(wd50_clean) > 0:
        ax1.scatter(wd50_clean, r95ptot_clean, alpha=0.6, s=10, c='blue')
        
        # Add correlation
        correlation = np.corrcoef(wd50_clean, r95ptot_clean)[0, 1]
        ax1.text(0.05, 0.95, f'r = {correlation:.3f}', transform=ax1.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax1.set_xlabel('WD50 (Days)')
    ax1.set_ylabel('R95pTOT (Fraction)')
    ax1.set_title('Precipitation Concentration Relationship', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. Concentration categories
    ax2 = axes[0, 1]
    
    # Define concentration categories based on WD50
    # Highly concentrated: WD50 < 10 days
    # Moderately concentrated: 10 <= WD50 < 20 days  
    # Distributed: WD50 >= 20 days
    
    concentration_categories = np.full_like(wd50_mean.values, 0, dtype=int)
    
    high_conc_mask = (wd50_mean < 10).values
    mod_conc_mask = ((wd50_mean >= 10) & (wd50_mean < 20)).values
    distributed_mask = (wd50_mean >= 20).values
    
    concentration_categories[high_conc_mask] = 1
    concentration_categories[mod_conc_mask] = 2
    concentration_categories[distributed_mask] = 3
    
    # Custom colormap
    colors = ['red', 'orange', 'green']
    cmap = mcolors.ListedColormap(colors)
    
    im2 = ax2.imshow(concentration_categories, cmap=cmap, vmin=1, vmax=3,
                     extent=[etccdi_ds.longitude.min(), etccdi_ds.longitude.max(),
                            etccdi_ds.latitude.min(), etccdi_ds.latitude.max()],
                     aspect='auto')
    
    ax2.set_xlabel('Longitude (°)')
    ax2.set_ylabel('Latitude (°)')
    ax2.set_title('Precipitation Concentration Categories', fontweight='bold')
    
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8, ticks=[1, 2, 3])
    cbar2.set_ticklabels(['Highly Concentrated', 'Moderate', 'Distributed'])
    
    # 3. Temporal variability
    ax3 = axes[1, 0]
    
    # Calculate coefficient of variation for each index
    indices_cv = {}
    for index in ['PRCPTOT', 'R95p', 'R95pTOT', 'WD50']:
        if index in etccdi_ds.data_vars:
            temporal_std = etccdi_ds[index].std(dim='year').mean(dim=['latitude', 'longitude'])
            temporal_mean = etccdi_ds[index].mean(dim='year').mean(dim=['latitude', 'longitude'])
            cv = temporal_std / (temporal_mean + 0.001)  # Coefficient of variation
            indices_cv[index] = cv.values
    
    if indices_cv:
        indices_names = list(indices_cv.keys())
        cv_values = list(indices_cv.values())
        colors = ['blue', 'red', 'purple', 'green'][:len(indices_names)]
        
        bars = ax3.bar(indices_names, cv_values, color=colors, alpha=0.7)
        ax3.set_ylabel('Coefficient of Variation')
        ax3.set_title('Temporal Variability of ETCCDI Indices', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Add values on bars
        for bar, value in zip(bars, cv_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
    
    # 4. Climate change indicators
    ax4 = axes[1, 1]
    
    if len(etccdi_ds.year) > 10:
        # Calculate trends for climate change detection
        years = etccdi_ds.year.values
        
        trend_data = []
        for index in ['PRCPTOT', 'R95p', 'R95pTOT']:
            if index in etccdi_ds.data_vars:
                global_series = etccdi_ds[index].mean(dim=['latitude', 'longitude'])
                valid_mask = ~np.isnan(global_series.values)
                
                if np.sum(valid_mask) > 5:
                    valid_years = years[valid_mask]
                    valid_values = global_series.values[valid_mask]
                    
                    # Calculate trend (per decade)
                    z = np.polyfit(valid_years, valid_values, 1)
                    trend_per_decade = z[0] * 10
                    
                    # Calculate relative trend (% per decade)
                    mean_value = np.mean(valid_values)
                    relative_trend = (trend_per_decade / mean_value) * 100 if mean_value != 0 else 0
                    
                    trend_data.append((index, trend_per_decade, relative_trend))
        
        if trend_data:
            indices_list = [item[0] for item in trend_data]
            abs_trends = [item[1] for item in trend_data]
            rel_trends = [item[2] for item in trend_data]
            
            x = np.arange(len(indices_list))
            width = 0.35
            
            bars1 = ax4.bar(x - width/2, abs_trends, width, label='Absolute Trend', alpha=0.7)
            
            # Secondary y-axis for relative trends
            ax4_twin = ax4.twinx()
            bars2 = ax4_twin.bar(x + width/2, rel_trends, width, label='Relative Trend (%)', 
                               alpha=0.7, color='red')
            
            ax4.set_xlabel('ETCCDI Index')
            ax4.set_ylabel('Absolute Trend (per decade)')
            ax4_twin.set_ylabel('Relative Trend (% per decade)', color='red')
            ax4.set_title('Climate Change Trends (per decade)', fontweight='bold')
            ax4.set_xticks(x)
            ax4.set_xticklabels(indices_list)
            ax4.grid(True, alpha=0.3)
            
            # Add zero line
            ax4.axhline(0, color='black', linestyle='-', alpha=0.5)
            ax4_twin.axhline(0, color='red', linestyle='-', alpha=0.5)
        else:
            ax4.text(0.5, 0.5, 'Insufficient data\nfor trend analysis', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=14)
            ax4.set_title('Climate Change Trends', fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'Insufficient years\nfor trend analysis', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=14)
        ax4.set_title('Climate Change Analysis', fontweight='bold')
    
    plt.suptitle(f'Extreme Precipitation Concentration Analysis\n'
                f'Period: {etccdi_ds.year.min().values}-{etccdi_ds.year.max().values}',
                fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    
    # Save
    filename = 'etccdi_concentration_analysis.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {filename}")

def plot_seasonal_etccdi_patterns(etccdi_ds, output_dir):
    """Analyze seasonal patterns in ETCCDI indices.""" 
    output_dir = Path(output_dir)
    
    # For annual indices, we'll analyze latitudinal and regional patterns
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Latitudinal profiles
    ax1 = axes[0, 0]
    
    indices = ['PRCPTOT', 'R95p']
    colors = ['blue', 'red']
    
    for index, color in zip(indices, colors):
        if index in etccdi_ds.data_vars:
            zonal_mean = etccdi_ds[index].mean(dim=['year', 'longitude'])
            ax1.plot(zonal_mean.latitude, zonal_mean, color=color, linewidth=2, 
                    label=index, alpha=0.8)
    
    ax1.set_xlabel('Latitude (°)')
    ax1.set_ylabel('Annual Precipitation (mm)')
    ax1.set_title('Latitudinal Precipitation Profiles', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-90, 90)
    
    # 2. R95pTOT latitudinal profile
    ax2 = axes[0, 1]
    
    if 'R95pTOT' in etccdi_ds.data_vars:
        r95ptot_zonal = etccdi_ds.R95pTOT.mean(dim=['year', 'longitude'])
        ax2.plot(r95ptot_zonal.latitude, r95ptot_zonal, 'purple', linewidth=2)
        
        # Add global mean line
        global_mean = etccdi_ds.R95pTOT.mean(dim=['year', 'latitude', 'longitude'])
        ax2.axhline(global_mean, color='orange', linestyle='--', 
                   label=f'Global Mean: {global_mean.values:.2f}')
    
    ax2.set_xlabel('Latitude (°)')
    ax2.set_ylabel('R95pTOT (Fraction)')
    ax2.set_title('Extreme Precipitation Contribution by Latitude', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-90, 90)
    ax2.set_ylim(0, 1)
    
    # 3. WD50 latitudinal profile
    ax3 = axes[1, 0]
    
    if 'WD50' in etccdi_ds.data_vars:
        wd50_zonal = etccdi_ds.WD50.mean(dim=['year', 'longitude'])
        ax3.plot(wd50_zonal.latitude, wd50_zonal, 'green', linewidth=2)
        
        # Add global mean line
        global_mean_wd50 = etccdi_ds.WD50.mean(dim=['year', 'latitude', 'longitude'])
        ax3.axhline(global_mean_wd50, color='orange', linestyle='--',
                   label=f'Global Mean: {global_mean_wd50.values:.1f} days')
    
    ax3.set_xlabel('Latitude (°)')
    ax3.set_ylabel('WD50 (Days)')
    ax3.set_title('Precipitation Concentration by Latitude', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-90, 90)
    
    # 4. Index relationships
    ax4 = axes[1, 1]
    
    # Scatter plot: PRCPTOT vs R95p
    if 'PRCPTOT' in etccdi_ds.data_vars and 'R95p' in etccdi_ds.data_vars:
        prcptot_sample = etccdi_ds.PRCPTOT.mean(dim='year').values.flatten()[::100]
        r95p_sample = etccdi_ds.R95p.mean(dim='year').values.flatten()[::100]
        
        mask = ~(np.isnan(prcptot_sample) | np.isnan(r95p_sample))
        prcptot_clean = prcptot_sample[mask]
        r95p_clean = r95p_sample[mask]
        
        if len(prcptot_clean) > 0:
            ax4.scatter(prcptot_clean, r95p_clean, alpha=0.6, s=5, c='purple')
            
            # Add 1:1 line and fit line
            max_val = max(np.max(prcptot_clean), np.max(r95p_clean))
            ax4.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='1:1 line')
            
            # Linear fit
            z = np.polyfit(prcptot_clean, r95p_clean, 1)
            p = np.poly1d(z)
            ax4.plot(prcptot_clean, p(prcptot_clean), 'r-', alpha=0.7,
                    label=f'Fit: y = {z[0]:.2f}x + {z[1]:.0f}')
            
            correlation = np.corrcoef(prcptot_clean, r95p_clean)[0, 1]
            ax4.text(0.05, 0.95, f'r = {correlation:.3f}', transform=ax4.transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax4.set_xlabel('PRCPTOT (mm/year)')
    ax4.set_ylabel('R95p (mm/year)')
    ax4.set_title('Total vs Extreme Precipitation Relationship', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'ETCCDI Precipitation Concentration Analysis\n'
                f'Period: {etccdi_ds.year.min().values}-{etccdi_ds.year.max().values}',
                fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    
    # Save
    filename = 'etccdi_concentration_patterns.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {filename}")

def plot_spatial_trends(etccdi_ds, output_dir, mask_type='both'):
    """Plot improved spatial trends with separate magnitude and significance maps."""
    output_dir = Path(output_dir)
    
    if len(etccdi_ds.year) < 5:
        print("Insufficient years for spatial trend analysis")
        return
    
    # Create land/ocean mask
    mask = create_land_ocean_mask(etccdi_ds, mask_type)
    
    years = etccdi_ds.year.values
    indices = ['PRCPTOT', 'R95p', 'R95pTOT', 'WD50']
    
    # FIGURE 1: Trend Magnitude with Significance Stippling
    fig1 = plt.figure(figsize=(20, 16))
    
    for idx, index in enumerate(indices):
        if index not in etccdi_ds.data_vars:
            continue
        
        ax = plt.subplot(2, 2, idx+1, projection=ccrs.Robinson())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax.set_global()
        
        # Calculate pixel-wise trends
        print(f"  Calculating trends for {index}...")
        trends, p_values, significant = calculate_pixel_trends(etccdi_ds[index], years)
        
        # Apply mask and adjust colorbar
        trends_masked, vmin, vmax, valid_count = apply_mask_and_adjust_colorbar(
            trends, mask, percentile_range=(2.5, 97.5)
        )
        
        # Convert trends to per-decade
        trends_decade = trends_masked * 10
        vmin_decade = vmin * 10
        vmax_decade = vmax * 10
        
        # Create symmetric colorbar around zero
        vmax_abs = max(abs(vmin_decade), abs(vmax_decade)) if vmax_decade != vmin_decade else 1
        levels = np.linspace(-vmax_abs, vmax_abs, 21)
        
        # Plot trends with clean colormap
        cmap = plt.cm.RdBu_r
        im = ax.contourf(etccdi_ds.longitude, etccdi_ds.latitude, trends_decade,
                        levels=levels, cmap=cmap, transform=ccrs.PlateCarree(),
                        extend='both')
        
        # Add simple stippling for significant areas
        add_significance_stippling(ax, etccdi_ds, significant, mask)
        
        # Calculate significance statistics
        sig_percentage = np.sum(significant & mask) / np.sum(mask) * 100
        
        # Set title with units
        if index == 'PRCPTOT':
            units = 'mm/decade'
            title = f'{index}: Precipitation Trend'
        elif index == 'R95p':
            units = 'mm/decade'
            title = f'{index}: Extreme Precipitation Trend'
        elif index == 'R95pTOT':
            units = 'fraction/decade'
            title = f'{index}: Extreme Contribution Trend'
        elif index == 'WD50':
            units = 'days/decade'
            title = f'{index}: Concentration Trend'
        else:
            units = 'units/decade'
            title = f'{index}: Trend'
        
        ax.set_title(f'{title}\\n({mask_type.capitalize()}, {valid_count:,} pixels, {sig_percentage:.1f}% significant)',
                    fontsize=12, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label(f'Trend ({units})', fontsize=10)
    
    # Add overall title for Figure 1
    period_str = f"{years[0]}-{years[-1]}"
    fig1.suptitle(f'ETCCDI Trend Magnitude ({period_str}) | Analysis: {mask_type.capitalize()}\\n'
                 f'Black dots show statistically significant areas (p < 0.05)',
                 fontsize=14, fontweight='bold', y=0.96)
    
    plt.tight_layout()
    
    # Save Figure 1
    filename1 = f'etccdi_trends_magnitude_{mask_type}.png'
    plt.savefig(output_dir / filename1, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {filename1}")
    
    # FIGURE 2: Trend Categories
    fig2 = plt.figure(figsize=(20, 16))
    
    for idx, index in enumerate(indices):
        if index not in etccdi_ds.data_vars:
            continue
        
        ax = plt.subplot(2, 2, idx+1, projection=ccrs.Robinson())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax.set_global()
        
        # Calculate pixel-wise trends (same as Figure 1)
        trends, p_values, significant = calculate_pixel_trends(etccdi_ds[index], years)
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
        im = ax.contourf(etccdi_ds.longitude, etccdi_ds.latitude, categories,
                        levels=np.arange(-0.5, 5.5, 1), cmap=cmap_discrete, 
                        transform=ccrs.PlateCarree())
        
        # Calculate category statistics
        cat_counts = [(categories == i).sum() for i in range(5)]
        total_valid = np.sum(mask)
        cat_percentages = [count/total_valid*100 for count in cat_counts]
        
        # Set title with units
        if index == 'PRCPTOT':
            title = f'{index}: Precipitation Categories'
        elif index == 'R95p':
            title = f'{index}: Extreme Precipitation Categories'
        elif index == 'R95pTOT':
            title = f'{index}: Extreme Contribution Categories'
        elif index == 'WD50':
            title = f'{index}: Concentration Categories'
        else:
            title = f'{index}: Categories'
        
        ax.set_title(f'{title}\\n'
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
    
    fig2.suptitle(f'ETCCDI Trend Categories ({period_str}) | Analysis: {mask_type.capitalize()}\\n'
                 f'Strong = Top 25% of trend magnitudes among significant pixels',
                 fontsize=14, fontweight='bold', y=0.96)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)  # Make room for legend
    
    # Save Figure 2
    filename2 = f'etccdi_trends_categories_{mask_type}.png'
    plt.savefig(output_dir / filename2, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {filename2}")

def main():
    """Main visualization function for ETCCDI indices."""
    parser = argparse.ArgumentParser(description='Visualize ETCCDI precipitation indices')
    
    parser.add_argument('--etccdi-dir', default='data/processed/etccdi_indices',
                       help='Directory containing ETCCDI index files')
    parser.add_argument('--output-dir', default='visualizations/output/etccdi_indices',
                       help='Output directory for plots')
    parser.add_argument('--years', nargs='+', type=int,
                       help='Specific years to analyze (default: all available)')
    parser.add_argument('--mask-type', choices=['land', 'ocean', 'both'], default='both',
                       help='Analysis domain: land-only, ocean-only, or both (default: both)')
    parser.add_argument('--include-trends', action='store_true',
                       help='Include spatial trend analysis with significance testing')
    
    args = parser.parse_args()
    
    print("="*80)
    print("ETCCDI PRECIPITATION INDICES VISUALIZATION")
    print("="*80)
    print(f"Input directory: {args.etccdi_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Analysis domain: {args.mask_type}")
    print(f"Include trends: {args.include_trends}")
    print("="*80)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load data
        etccdi_ds = load_etccdi_data(args.etccdi_dir, args.years)
        
        print("\nCreating visualizations...")
        
        # 1. Global climatology
        print("1. Global ETCCDI climatology...")
        plot_global_etccdi_climatology(etccdi_ds, output_dir, args.mask_type)
        
        # 2. Temporal trends
        print("2. ETCCDI temporal trends...")
        plot_etccdi_trends(etccdi_ds, output_dir)
        
        # 3. Extreme precipitation analysis
        print("3. Extreme precipitation analysis...")
        plot_extreme_precipitation_analysis(etccdi_ds, output_dir)
        
        # 4. Concentration analysis
        print("4. Precipitation concentration analysis...")
        plot_precipitation_concentration_analysis(etccdi_ds, output_dir)
        
        # 5. Spatial trends (if requested)
        if args.include_trends:
            print("5. Spatial trend analysis...")
            plot_spatial_trends(etccdi_ds, output_dir, args.mask_type)
        
        print("\n" + "="*80)
        print("ETCCDI VISUALIZATION COMPLETED!")
        print("="*80)
        print(f"Output files saved in: {output_dir}")
        print("\nGenerated plots:")
        for plot_file in output_dir.glob('*.png'):
            print(f"  - {plot_file.name}")
        
        # Close dataset
        etccdi_ds.close()
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
