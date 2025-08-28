#!/usr/bin/env python3
"""
Enhanced Visualization Features Template
=======================================

This template provides the core functions that should be added to all visualization scripts:
1. Accurate Land/Ocean masking using Natural Earth boundaries
2. Spatial trend analysis with direction, magnitude, and significance
3. Improved colorbar scaling with percentile capping
4. Enhanced argument parsing

USAGE: Copy these functions to other visualization scripts and adapt as needed.

REQUIRED IMPORTS:
```python
import numpy as np
from scipy import stats
try:
    import regionmask
    HAS_REGIONMASK = True
except ImportError:
    HAS_REGIONMASK = False
```
"""

import numpy as np
from scipy import stats
try:
    import regionmask
    HAS_REGIONMASK = True
except ImportError:
    HAS_REGIONMASK = False

def create_land_ocean_mask(ds, mask_type='both'):
    """
    Create accurate land/ocean mask using regionmask Natural Earth boundaries.
    
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
            print("  Using simplified approximation...")
    
    # Fallback: simplified land/ocean approximation
    print("  Using simplified land/ocean boundaries...")
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing='ij')
    
    # Simplified continental outlines
    land_mask = np.zeros_like(lat_grid, dtype=bool)
    
    # Major continents (improved boundaries)
    land_mask |= (lat_grid >= -60) & (lat_grid <= 75) & (lon_grid >= -170) & (lon_grid <= -30)  # Americas
    land_mask |= (lat_grid >= -35) & (lat_grid <= 75) & (lon_grid >= -20) & (lon_grid <= 180)   # Eurasia+Africa
    land_mask |= (lat_grid >= -50) & (lat_grid <= -10) & (lon_grid >= 110) & (lon_grid <= 180)  # Australia
    
    # Remove major ocean areas
    land_mask &= ~((lat_grid >= -30) & (lat_grid <= 30) & (lon_grid >= -60) & (lon_grid <= 20))  # Atlantic
    land_mask &= ~((lat_grid >= -30) & (lat_grid <= 30) & (lon_grid >= 60) & (lon_grid <= 120))   # Indian Ocean
    land_mask &= ~((lat_grid >= -30) & (lat_grid <= 30) & (lon_grid >= 120) & (lon_grid <= -120))  # Pacific
    
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

# Template argument parser additions:
"""
Add these arguments to your ArgumentParser:

parser.add_argument('--mask-type', choices=['land', 'ocean', 'both'], default='both',
                   help='Analysis domain: land-only, ocean-only, or both (default: both)')
parser.add_argument('--include-trends', action='store_true',
                   help='Include spatial trend analysis with significance testing')
"""

def add_enhanced_trend_indicators(ax, ds, trends_decade, significant, mask):
    """
    Add enhanced trend indicators showing direction, magnitude, and significance.
    
    Args:
        ax: matplotlib axis
        ds: xarray dataset (for coordinates)
        trends_decade: trend values per decade
        significant: boolean array of significant pixels
        mask: land/ocean mask
    """
    significant_masked = significant & mask
    if not np.any(significant_masked):
        return
    
    # Separate increasing and decreasing significant trends
    increasing_sig = significant_masked & (trends_decade > 0)
    decreasing_sig = significant_masked & (trends_decade < 0)
    
    # Define magnitude thresholds (strong = top 25% of absolute trends)
    abs_trends = np.abs(trends_decade)
    strong_threshold = np.nanpercentile(abs_trends[significant_masked], 75) if np.any(significant_masked) else 0
    
    # Classify trends by direction and magnitude
    strong_increasing = increasing_sig & (abs_trends >= strong_threshold)
    weak_increasing = increasing_sig & (abs_trends < strong_threshold)
    strong_decreasing = decreasing_sig & (abs_trends >= strong_threshold)
    weak_decreasing = decreasing_sig & (abs_trends < strong_threshold)
    
    # Plot different patterns for different trend types
    if np.any(strong_increasing):
        ax.contourf(ds.longitude, ds.latitude, strong_increasing.astype(int),
                   levels=[0.5, 1.5], colors='none', hatches=['^^^'], alpha=0.8,
                   transform=ccrs.PlateCarree())
    
    if np.any(weak_increasing):
        ax.contourf(ds.longitude, ds.latitude, weak_increasing.astype(int),
                   levels=[0.5, 1.5], colors='none', hatches=['...'], alpha=0.6,
                   transform=ccrs.PlateCarree())
    
    if np.any(strong_decreasing):
        ax.contourf(ds.longitude, ds.latitude, strong_decreasing.astype(int),
                   levels=[0.5, 1.5], colors='none', hatches=['vvv'], alpha=0.8,
                   transform=ccrs.PlateCarree())
    
    if np.any(weak_decreasing):
        ax.contourf(ds.longitude, ds.latitude, weak_decreasing.astype(int),
                   levels=[0.5, 1.5], colors='none', hatches=['///'], alpha=0.6,
                   transform=ccrs.PlateCarree())

# Template for enhanced spatial trends function:
def plot_spatial_trends_enhanced(data_ds, output_dir, mask_type='both', variables=None):
    """
    Plot spatial trends with enhanced direction/magnitude indicators.
    
    Args:
        data_ds: xarray Dataset with year dimension
        output_dir: Path object for output directory
        mask_type: 'land', 'ocean', or 'both'
        variables: list of variable names to analyze
    """
    from pathlib import Path
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    
    output_dir = Path(output_dir)
    
    if len(data_ds.year) < 5:
        print("Insufficient years for spatial trend analysis")
        return
    
    # Create land/ocean mask
    mask = create_land_ocean_mask(data_ds, mask_type)
    
    years = data_ds.year.values
    
    if variables is None:
        variables = list(data_ds.data_vars)[:4]  # Limit to first 4 variables
    
    fig = plt.figure(figsize=(20, 16))
    
    for idx, variable in enumerate(variables[:4]):  # Max 4 subplots
        if variable not in data_ds.data_vars:
            continue
        
        ax = plt.subplot(2, 2, idx+1, projection=ccrs.Robinson())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax.set_global()
        
        # Calculate pixel-wise trends
        print(f"  Calculating trends for {variable}...")
        
        # Handle timedelta variables
        data_var = data_ds[variable]
        if 'timedelta' in str(data_var.dtype):
            data_var = data_var / np.timedelta64(1, 'D')
        
        trends, p_values, significant = calculate_pixel_trends(data_var, years)
        
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
        
        # Plot trends
        cmap = plt.cm.RdBu_r
        im = ax.contourf(data_ds.longitude, data_ds.latitude, trends_decade,
                        levels=levels, cmap=cmap, transform=ccrs.PlateCarree(),
                        extend='both')
        
        # Add enhanced significance indicators
        add_enhanced_trend_indicators(ax, data_ds, trends_decade, significant, mask)
        
        # Customize title and units based on variable
        units = 'units/decade'  # Default - customize for each script
        title = f'{variable}: Trend'
        
        ax.set_title(f'{title}\\n({mask_type.capitalize()}, {valid_count:,} pixels)',
                    fontsize=12, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label(f'Trend ({units})', fontsize=10)
    
    # Add comprehensive title with legend
    period_str = f"{years[0]}-{years[-1]}"
    fig.suptitle(f'Spatial Trends ({period_str}) | Analysis: {mask_type.capitalize()}\\n'
                f'Patterns: ^^^ Strong Increase | ... Weak Increase | /// Weak Decrease | vvv Strong Decrease',
                fontsize=14, fontweight='bold', y=0.96)
    
    plt.tight_layout()
    
    # Save
    filename = f'spatial_trends_{mask_type}.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {filename}")

# Enhanced argument parser template:
ENHANCED_ARGUMENTS = '''
# Add these arguments to your ArgumentParser:

parser.add_argument('--mask-type', choices=['land', 'ocean', 'both'], default='both',
                   help='Analysis domain: land-only, ocean-only, or both (default: both)')
parser.add_argument('--include-trends', action='store_true',
                   help='Include spatial trend analysis with significance testing')
parser.add_argument('--percentile-cap', type=float, default=99.5,
                   help='Percentile for capping extreme values in colorbars (default: 99.5)')
'''
