#!/usr/bin/env python3
"""
Visualization of Precipitation Percentiles (Output from 00_calculate_precipitation_percentiles.py)

Creates comprehensive scientific visualizations of precipitation percentile climatology:
- Global maps of precipitation percentiles (P10, P25, P50, P75, P90)
- Seasonal patterns and drought/flood thresholds
- Dry vs wet zone identification
- Comparison with absolute thresholds
- Land/ocean masking capabilities

Input files:
- precipitation_percentiles.nc containing precip_p10, precip_p25, precip_p50, precip_p75, precip_p90
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

def load_precipitation_percentiles(percentile_file):
    """Load precipitation percentile data."""
    percentile_file = Path(percentile_file)
    
    if not percentile_file.exists():
        raise ValueError(f"Percentile file not found: {percentile_file}")
    
    ds = xr.open_dataset(percentile_file)
    
    print(f"Loaded precipitation percentiles from: {percentile_file}")
    print(f"Variables: {list(ds.data_vars)}")
    print(f"Period: {ds.attrs.get('climatology_period', 'Unknown')}")
    print(f"Grid: {ds.dims}")
    print(f"Day of year range: {ds.dayofyear.min().values} to {ds.dayofyear.max().values}")
    
    return ds

def plot_global_precipitation_percentiles(precip_pct_ds, output_dir, mask_type='both', season_months=None):
    """Create global maps of precipitation percentiles with masking."""
    output_dir = Path(output_dir)
    
    # Create land/ocean mask
    mask = create_land_ocean_mask(precip_pct_ds, mask_type)
    
    # Select season or annual mean
    if season_months:
        season_name = f"{'_'.join(season_months)}"
        # Map season to day-of-year ranges (approximate)
        season_doy_map = {
            'DJF': list(range(1, 32)) + list(range(335, 366)) + list(range(60, 91)),
            'MAM': list(range(60, 152)),
            'JJA': list(range(152, 244)), 
            'SON': list(range(244, 335))
        }
        
        if season_months[0] in season_doy_map:
            doy_indices = season_doy_map[season_months[0]]
            # Get available percentiles
            percentiles_data = {}
            for var in precip_pct_ds.data_vars:
                if 'precip_p' in var:
                    percentiles_data[var] = precip_pct_ds[var].isel(
                        dayofyear=[d-1 for d in doy_indices if d-1 < len(precip_pct_ds.dayofyear)]
                    ).mean(dim='dayofyear')
        else:
            percentiles_data = {var: precip_pct_ds[var].mean(dim='dayofyear') for var in precip_pct_ds.data_vars if 'precip_p' in var}
    else:
        season_name = "Annual"
        percentiles_data = {var: precip_pct_ds[var].mean(dim='dayofyear') for var in precip_pct_ds.data_vars if 'precip_p' in var}
    
    # Create figure with multiple percentiles
    fig = plt.figure(figsize=(20, 24))
    
    # Define percentiles to plot
    percentile_vars = ['precip_p10', 'precip_p25', 'precip_p50', 'precip_p75', 'precip_p90']
    percentile_names = ['P10 (Drought)', 'P25', 'P50 (Median)', 'P75', 'P90 (Intense)']
    
    for i, (var, name) in enumerate(zip(percentile_vars, percentile_names)):
        if var not in percentiles_data:
            continue
        
        ax = plt.subplot(3, 2, i+1, projection=ccrs.Robinson())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax.set_global()
        
        data = percentiles_data[var]
        
        # Apply mask and calculate percentiles from masked data
        data_masked, vmin, vmax, valid_count = apply_mask_and_adjust_colorbar(
            data.values, mask, percentile_range=(0.5, 99.5)
        )
        
        print(f"  {var} ({mask_type}): {valid_count:,} pixels, range {vmin:.2f}-{vmax:.2f} mm/day")
        
        # Different color schemes for different percentiles
        if 'p10' in var:
            levels = np.logspace(np.log10(max(vmin, 0.01)), np.log10(max(vmax, 1)), 20)
            cmap = plt.cm.YlOrBr
        elif 'p90' in var:
            levels = np.logspace(np.log10(max(vmin, 0.1)), np.log10(max(vmax, 10)), 20)
            cmap = plt.cm.Blues
        else:
            levels = np.logspace(np.log10(max(vmin, 0.01)), np.log10(max(vmax, 5)), 20)
            cmap = plt.cm.viridis
        
        # Use masked data for plotting
        data_plot = np.clip(data_masked, vmin, vmax)
        
        im = ax.contourf(data.longitude, data.latitude, data_plot,
                        levels=levels, cmap=cmap, transform=ccrs.PlateCarree(),
                        extend='both')
        
        ax.set_title(f'Precipitation {name} - {season_name}\n'
                    f'({mask_type.capitalize()}, {valid_count:,} pixels)', 
                    fontsize=12, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Precipitation (mm/day)', fontsize=10)
    
    plt.suptitle(f'Precipitation Percentile Climatology - {season_name}\n'
                f'Period: {precip_pct_ds.attrs.get("climatology_period", "Unknown")} | Domain: {mask_type.capitalize()}',
                fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    
    # Save
    filename = f'precipitation_percentiles_global_{season_name.lower()}_{mask_type}.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {filename}")

def plot_drought_flood_analysis(precip_pct_ds, output_dir, mask_type='both'):
    """Analyze drought and flood thresholds with masking."""
    output_dir = Path(output_dir)
    
    # Create land/ocean mask
    mask = create_land_ocean_mask(precip_pct_ds, mask_type)
    
    # Calculate annual means
    p10_annual = precip_pct_ds.precip_p10.mean(dim='dayofyear')
    p90_annual = precip_pct_ds.precip_p90.mean(dim='dayofyear')
    p50_annual = precip_pct_ds.precip_p50.mean(dim='dayofyear')
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Drought threshold map (P10)
    ax1 = axes[0, 0]
    
    p10_masked, p10_vmin, p10_vmax, p10_count = apply_mask_and_adjust_colorbar(
        p10_annual.values, mask, percentile_range=(1, 99)
    )
    
    im1 = ax1.imshow(p10_masked, cmap='YlOrBr', vmin=p10_vmin, vmax=min(p10_vmax, 10),
                     extent=[precip_pct_ds.longitude.min(), precip_pct_ds.longitude.max(),
                            precip_pct_ds.latitude.min(), precip_pct_ds.latitude.max()],
                     aspect='auto')
    
    ax1.set_xlabel('Longitude (°)')
    ax1.set_ylabel('Latitude (°)')
    ax1.set_title(f'Drought Threshold (P10)\n({mask_type.capitalize()}, {p10_count:,} pixels)', fontweight='bold')
    
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('Precipitation (mm/day)')
    
    # 2. Intense precipitation threshold (P90)
    ax2 = axes[0, 1]
    
    p90_masked, p90_vmin, p90_vmax, p90_count = apply_mask_and_adjust_colorbar(
        p90_annual.values, mask, percentile_range=(1, 99)
    )
    
    im2 = ax2.imshow(p90_masked, cmap='Blues', vmin=p90_vmin, vmax=min(p90_vmax, 50),
                     extent=[precip_pct_ds.longitude.min(), precip_pct_ds.longitude.max(),
                            precip_pct_ds.latitude.min(), precip_pct_ds.latitude.max()],
                     aspect='auto')
    
    ax2.set_xlabel('Longitude (°)')
    ax2.set_ylabel('Latitude (°)')
    ax2.set_title(f'Intense Precipitation Threshold (P90)\n({mask_type.capitalize()}, {p90_count:,} pixels)', fontweight='bold')
    
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label('Precipitation (mm/day)')
    
    # 3. Precipitation variability (P90/P10 ratio)
    ax3 = axes[1, 0]
    
    variability = p90_annual / (p10_annual + 0.1)  # Add small value to avoid division by zero
    var_masked, var_vmin, var_vmax, var_count = apply_mask_and_adjust_colorbar(
        variability.values, mask, percentile_range=(5, 95)
    )
    
    im3 = ax3.imshow(var_masked, cmap='plasma', vmin=var_vmin, vmax=min(var_vmax, 100),
                     extent=[precip_pct_ds.longitude.min(), precip_pct_ds.longitude.max(),
                            precip_pct_ds.latitude.min(), precip_pct_ds.latitude.max()],
                     aspect='auto')
    
    ax3.set_xlabel('Longitude (°)')
    ax3.set_ylabel('Latitude (°)')
    ax3.set_title(f'Precipitation Variability (P90/P10)\n({mask_type.capitalize()}, {var_count:,} pixels)', fontweight='bold')
    
    cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.8)
    cbar3.set_label('Ratio')
    
    # 4. Climate classification based on precipitation (only for masked pixels)
    ax4 = axes[1, 1]
    
    # Define climate zones based on median precipitation
    # Arid: P50 < 1 mm/day
    # Semi-arid: 1 <= P50 < 3 mm/day  
    # Humid: P50 >= 3 mm/day
    
    climate_zones = np.full_like(p50_annual.values, 0, dtype=int)
    
    # Apply mask first
    p50_masked_values = p50_annual.values.copy()
    p50_masked_values[~mask] = np.nan
    
    arid_mask = (p50_masked_values < 1) & mask
    semi_arid_mask = ((p50_masked_values >= 1) & (p50_masked_values < 3)) & mask
    humid_mask = (p50_masked_values >= 3) & mask
    
    climate_zones[arid_mask] = 1
    climate_zones[semi_arid_mask] = 2
    climate_zones[humid_mask] = 3
    climate_zones[~mask] = 0  # Set masked areas to 0
    
    # Custom colormap
    colors = ['white', 'brown', 'orange', 'green']
    cmap = mcolors.ListedColormap(colors)
    
    im4 = ax4.imshow(climate_zones, cmap=cmap, vmin=0, vmax=3,
                     extent=[precip_pct_ds.longitude.min(), precip_pct_ds.longitude.max(),
                            precip_pct_ds.latitude.min(), precip_pct_ds.latitude.max()],
                     aspect='auto')
    
    ax4.set_xlabel('Longitude (°)')
    ax4.set_ylabel('Latitude (°)')
    ax4.set_title(f'Precipitation-Based Climate Zones\n({mask_type.capitalize()})', fontweight='bold')
    
    # Calculate zone statistics
    arid_pct = np.sum(arid_mask) / np.sum(mask) * 100 if np.sum(mask) > 0 else 0
    semi_arid_pct = np.sum(semi_arid_mask) / np.sum(mask) * 100 if np.sum(mask) > 0 else 0
    humid_pct = np.sum(humid_mask) / np.sum(mask) * 100 if np.sum(mask) > 0 else 0
    
    cbar4 = plt.colorbar(im4, ax=ax4, shrink=0.8, ticks=[0, 1, 2, 3])
    cbar4.set_ticklabels([f'Masked', f'Arid ({arid_pct:.1f}%)', f'Semi-Arid ({semi_arid_pct:.1f}%)', f'Humid ({humid_pct:.1f}%)'])
    
    plt.suptitle(f'Drought and Flood Analysis\n'
                f'Period: {precip_pct_ds.attrs.get("climatology_period", "Unknown")} | Domain: {mask_type.capitalize()}',
                fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    
    # Save
    filename = f'precipitation_drought_flood_analysis_{mask_type}.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {filename}")

def plot_seasonal_precipitation_cycle(precip_pct_ds, output_dir, mask_type='both'):
    """Plot seasonal cycle of precipitation percentiles with masking."""
    output_dir = Path(output_dir)
    
    # Create land/ocean mask
    mask = create_land_ocean_mask(precip_pct_ds, mask_type)
    
    # Define latitude bands
    lat_bands = [
        ('Tropical', -23.5, 23.5),
        ('Northern Subtropics', 23.5, 40),
        ('Northern Mid-Latitudes', 40, 60),
        ('Southern Subtropics', -40, -23.5),
        ('Southern Mid-Latitudes', -60, -40)
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, (band_name, lat_min, lat_max) in enumerate(lat_bands):
        ax = axes[i]
        
        # Create 2D latitude mask that matches the shape of the land/ocean mask
        lat_1d = precip_pct_ds.latitude.values
        lon_1d = precip_pct_ds.longitude.values
        lat_2d, lon_2d = np.meshgrid(lat_1d, lon_1d, indexing='ij')
        
        # Select latitude band (2D mask)
        lat_mask_2d = (lat_2d >= lat_min) & (lat_2d <= lat_max)
        
        # Calculate zonal means for each percentile with mask applied
        percentiles_to_plot = ['precip_p10', 'precip_p50', 'precip_p90']
        colors = ['brown', 'green', 'blue']
        labels = ['P10 (Drought)', 'P50 (Median)', 'P90 (Intense)']
        
        for var, color, label in zip(percentiles_to_plot, colors, labels):
            if var in precip_pct_ds.data_vars:
                # Apply both latitude and land/ocean mask
                combined_mask = lat_mask_2d & mask
                
                # Calculate area-weighted mean only for valid (non-masked) pixels
                data_masked = precip_pct_ds[var].where(combined_mask)
                zonal_mean = data_masked.mean(dim=['latitude', 'longitude'], skipna=True)
                
                days = precip_pct_ds.dayofyear.values
                ax.plot(days, zonal_mean, color=color, linewidth=2, label=label, alpha=0.8)
        
        # Calculate valid pixel count for this band
        band_mask_count = np.sum(lat_mask_2d & mask)
        total_band_pixels = np.sum(lat_mask_2d)
        
        ax.set_title(f'{band_name} ({mask_type.capitalize()})\n'
                    f'({lat_min}° to {lat_max}°) | {band_mask_count:,}/{total_band_pixels:,} pixels', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Day of Year')
        ax.set_ylabel('Precipitation (mm/day)')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
        # Add month labels
        month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
        month_labels = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
        ax.set_xticks(month_starts)
        ax.set_xticklabels(month_labels)
        ax.set_xlim(1, 365)
    
    # Remove empty subplot
    axes[-1].remove()
    
    plt.suptitle(f'Seasonal Cycle of Precipitation Percentiles by Latitude Band\n'
                f'Period: {precip_pct_ds.attrs.get("climatology_period", "Unknown")} | Domain: {mask_type.capitalize()}',
                fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    
    # Save
    filename = f'precipitation_percentiles_seasonal_cycle_{mask_type}.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {filename}")

def plot_precipitation_statistics(precip_pct_ds, output_dir, mask_type='both'):
    """Plot statistical analysis of precipitation percentiles with masking."""
    output_dir = Path(output_dir)
    
    # Create land/ocean mask
    mask = create_land_ocean_mask(precip_pct_ds, mask_type)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Global distribution of percentiles
    ax1 = axes[0, 0]
    
    percentiles_to_plot = ['precip_p10', 'precip_p50', 'precip_p90']
    colors = ['brown', 'green', 'blue']
    labels = ['P10', 'P50', 'P90']
    
    for var, color, label in zip(percentiles_to_plot, colors, labels):
        if var in precip_pct_ds.data_vars:
            annual_mean = precip_pct_ds[var].mean(dim='dayofyear')
            
            # Apply mask
            data_masked = annual_mean.values.copy()
            data_masked[~mask] = np.nan
            data_clean = data_masked[~np.isnan(data_masked)]
            
            if len(data_clean) > 0:
                ax1.hist(data_clean, bins=50, alpha=0.7, color=color, label=f'{label} (n={len(data_clean):,})', 
                        density=True, range=(0, min(50, np.percentile(data_clean, 99))))
    
    ax1.set_xlabel('Precipitation (mm/day)')
    ax1.set_ylabel('Density')
    ax1.set_title(f'Global Distribution of Precipitation Percentiles\n({mask_type.capitalize()})', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Latitudinal profiles
    ax2 = axes[0, 1]
    
    for var, color, label in zip(percentiles_to_plot, colors, labels):
        if var in precip_pct_ds.data_vars:
            annual_mean = precip_pct_ds[var].mean(dim='dayofyear')
            
            # Apply mask and calculate zonal means
            data_masked = annual_mean.where(mask)
            zonal_mean = data_masked.mean(dim='longitude', skipna=True)
            
            ax2.plot(zonal_mean.latitude, zonal_mean, color=color, linewidth=2, 
                    label=label, alpha=0.8)
    
    ax2.set_xlabel('Latitude (°)')
    ax2.set_ylabel('Precipitation (mm/day)')
    ax2.set_title(f'Latitudinal Precipitation Profiles\n({mask_type.capitalize()})', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-90, 90)
    
    # 3. Seasonal variability
    ax3 = axes[1, 0]
    
    # Calculate coefficient of variation for each percentile (masked)
    cv_values = []
    cv_labels = []
    
    for var, color, label in zip(percentiles_to_plot, colors, labels):
        if var in precip_pct_ds.data_vars:
            # Apply mask to all calculations
            data_masked = precip_pct_ds[var].where(mask)
            
            seasonal_std = data_masked.std(dim='dayofyear').mean(dim=['latitude', 'longitude'], skipna=True)
            seasonal_mean = data_masked.mean(dim='dayofyear').mean(dim=['latitude', 'longitude'], skipna=True)
            cv = seasonal_std / (seasonal_mean + 0.1)  # Coefficient of variation
            
            if not np.isnan(cv):
                cv_values.append(cv)
                cv_labels.append(label)
                ax3.bar(label, cv, color=color, alpha=0.7)
    
    ax3.set_ylabel('Coefficient of Variation')
    ax3.set_title(f'Seasonal Variability by Percentile\n({mask_type.capitalize()})', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Summary statistics table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    stats_data = [['Percentile', 'Global Mean', 'Global Std', 'Min', 'Max', 'Valid Pixels']]
    
    for var, label in zip(percentiles_to_plot, labels):
        if var in precip_pct_ds.data_vars:
            annual_mean = precip_pct_ds[var].mean(dim='dayofyear')
            
            # Apply mask
            data_masked = annual_mean.values.copy()
            data_masked[~mask] = np.nan
            data_clean = data_masked[~np.isnan(data_masked)]
            
            if len(data_clean) > 0:
                stats_data.append([
                    label,
                    f'{np.mean(data_clean):.2f}',
                    f'{np.std(data_clean):.2f}',
                    f'{np.min(data_clean):.2f}',
                    f'{np.max(data_clean):.2f}',
                    f'{len(data_clean):,}'
                ])
    
    table = ax4.table(cellText=stats_data[1:], colLabels=stats_data[0],
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    # Style the table
    for i in range(len(stats_data)):
        for j in range(len(stats_data[0])):
            cell = table[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor('#4472C4')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#F2F2F2' if i % 2 == 0 else 'white')
    
    ax4.set_title(f'Precipitation Percentile Statistics\n({mask_type.capitalize()})', fontweight='bold', pad=20)
    
    plt.suptitle(f'Precipitation Percentile Analysis\n'
                f'Period: {precip_pct_ds.attrs.get("climatology_period", "Unknown")} | Domain: {mask_type.capitalize()}',
                fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    
    # Save
    filename = f'precipitation_percentiles_statistics_{mask_type}.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {filename}")

def main():
    """Main visualization function for precipitation percentiles."""
    parser = argparse.ArgumentParser(description='Visualize precipitation percentile climatology')
    
    parser.add_argument('--percentile-file', default='data/processed/precipitation_percentiles.nc',
                       help='Precipitation percentile file')
    parser.add_argument('--output-dir', default='visualizations/output/precipitation_percentiles',
                       help='Output directory for plots')
    parser.add_argument('--mask-type', choices=['land', 'ocean', 'both'], default='both',
                       help='Analysis domain: land-only, ocean-only, or both (default: both)')
    parser.add_argument('--create-seasonal', action='store_true',
                       help='Create seasonal plots (DJF, MAM, JJA, SON)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("PRECIPITATION PERCENTILE VISUALIZATION")
    print("="*80)
    print(f"Input file: {args.percentile_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"Analysis domain: {args.mask_type}")
    print("="*80)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load data
        precip_pct_ds = load_precipitation_percentiles(args.percentile_file)
        
        print("\nCreating visualizations...")
        
        # 1. Global maps (annual)
        print("1. Global annual maps...")
        plot_global_precipitation_percentiles(precip_pct_ds, output_dir, args.mask_type)
        
        # 2. Seasonal maps
        if args.create_seasonal:
            print("2. Seasonal maps...")
            seasons = ['DJF', 'MAM', 'JJA', 'SON']
            for season in seasons:
                plot_global_precipitation_percentiles(precip_pct_ds, output_dir, args.mask_type, [season])
        
        # 3. Drought and flood analysis
        print("3. Drought and flood analysis...")
        plot_drought_flood_analysis(precip_pct_ds, output_dir, args.mask_type)
        
        # 4. Seasonal cycle by latitude
        print("4. Seasonal cycle analysis...")
        plot_seasonal_precipitation_cycle(precip_pct_ds, output_dir, args.mask_type)
        
        # 5. Statistical analysis
        print("5. Statistical analysis...")
        plot_precipitation_statistics(precip_pct_ds, output_dir, args.mask_type)
        
        print("\n" + "="*80)
        print("PRECIPITATION PERCENTILE VISUALIZATION COMPLETED!")
        print("="*80)
        print(f"Output files saved in: {output_dir}")
        print(f"Analysis domain: {args.mask_type}")
        print("\nGenerated plots:")
        for plot_file in output_dir.glob('*.png'):
            print(f"  - {plot_file.name}")
        
        # Close dataset
        precip_pct_ds.close()
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())