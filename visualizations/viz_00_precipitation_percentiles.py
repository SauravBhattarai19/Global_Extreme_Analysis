#!/usr/bin/env python3
"""
Visualization of Precipitation Percentiles (Output from 00_calculate_precipitation_percentiles.py)

Creates comprehensive scientific visualizations of precipitation percentile climatology:
- Global maps of precipitation percentiles (P10, P25, P50, P75, P90)
- Seasonal patterns and drought/flood thresholds
- Dry vs wet zone identification
- Comparison with absolute thresholds

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
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

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

def plot_global_precipitation_percentiles(precip_pct_ds, output_dir, season_months=None):
    """Create global maps of precipitation percentiles."""
    output_dir = Path(output_dir)
    
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
        
        # Different color schemes for different percentiles
        if 'p10' in var:
            levels = np.logspace(-1, 2, 20)  # 0.1 to 100 mm
            cmap = plt.cm.YlOrBr
        elif 'p90' in var:
            levels = np.logspace(0, 2.5, 20)  # 1 to 300 mm
            cmap = plt.cm.Blues
        else:
            levels = np.logspace(-0.5, 2, 20)  # 0.3 to 100 mm
            cmap = plt.cm.viridis
        
        im = ax.contourf(data.longitude, data.latitude, data,
                        levels=levels, cmap=cmap, transform=ccrs.PlateCarree(),
                        extend='both')
        
        ax.set_title(f'Precipitation {name} - {season_name}', 
                    fontsize=12, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Precipitation (mm/day)', fontsize=10)
    
    plt.suptitle(f'Precipitation Percentile Climatology - {season_name}\n'
                f'Period: {ds.attrs.get("climatology_period", "Unknown")}',
                fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    
    # Save
    filename = f'precipitation_percentiles_global_{season_name.lower()}.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {filename}")

def plot_drought_flood_analysis(precip_pct_ds, output_dir):
    """Analyze drought and flood thresholds."""
    output_dir = Path(output_dir)
    
    # Calculate annual means
    p10_annual = precip_pct_ds.precip_p10.mean(dim='dayofyear')
    p90_annual = precip_pct_ds.precip_p90.mean(dim='dayofyear')
    p50_annual = precip_pct_ds.precip_p50.mean(dim='dayofyear')
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Drought threshold map (P10)
    ax1 = axes[0, 0]
    
    im1 = ax1.imshow(p10_annual, cmap='YlOrBr', vmin=0, vmax=10,
                     extent=[precip_pct_ds.longitude.min(), precip_pct_ds.longitude.max(),
                            precip_pct_ds.latitude.min(), precip_pct_ds.latitude.max()],
                     aspect='auto')
    
    ax1.set_xlabel('Longitude (°)')
    ax1.set_ylabel('Latitude (°)')
    ax1.set_title('Drought Threshold (P10)', fontweight='bold')
    
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('Precipitation (mm/day)')
    
    # 2. Intense precipitation threshold (P90)
    ax2 = axes[0, 1]
    
    im2 = ax2.imshow(p90_annual, cmap='Blues', vmin=0, vmax=50,
                     extent=[precip_pct_ds.longitude.min(), precip_pct_ds.longitude.max(),
                            precip_pct_ds.latitude.min(), precip_pct_ds.latitude.max()],
                     aspect='auto')
    
    ax2.set_xlabel('Longitude (°)')
    ax2.set_ylabel('Latitude (°)')
    ax2.set_title('Intense Precipitation Threshold (P90)', fontweight='bold')
    
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label('Precipitation (mm/day)')
    
    # 3. Precipitation variability (P90/P10 ratio)
    ax3 = axes[1, 0]
    
    variability = p90_annual / (p10_annual + 0.1)  # Add small value to avoid division by zero
    
    im3 = ax3.imshow(variability, cmap='plasma', vmin=1, vmax=100,
                     extent=[precip_pct_ds.longitude.min(), precip_pct_ds.longitude.max(),
                            precip_pct_ds.latitude.min(), precip_pct_ds.latitude.max()],
                     aspect='auto')
    
    ax3.set_xlabel('Longitude (°)')
    ax3.set_ylabel('Latitude (°)')
    ax3.set_title('Precipitation Variability (P90/P10)', fontweight='bold')
    
    cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.8)
    cbar3.set_label('Ratio')
    
    # 4. Climate classification based on precipitation
    ax4 = axes[1, 1]
    
    # Define climate zones based on median precipitation
    # Arid: P50 < 1 mm/day
    # Semi-arid: 1 <= P50 < 3 mm/day  
    # Humid: P50 >= 3 mm/day
    
    climate_zones = np.full_like(p50_annual.values, 0, dtype=int)
    
    arid_mask = (p50_annual < 1).values
    semi_arid_mask = ((p50_annual >= 1) & (p50_annual < 3)).values
    humid_mask = (p50_annual >= 3).values
    
    climate_zones[arid_mask] = 1
    climate_zones[semi_arid_mask] = 2
    climate_zones[humid_mask] = 3
    
    # Custom colormap
    colors = ['brown', 'orange', 'green']
    cmap = mcolors.ListedColormap(colors)
    
    im4 = ax4.imshow(climate_zones, cmap=cmap, vmin=1, vmax=3,
                     extent=[precip_pct_ds.longitude.min(), precip_pct_ds.longitude.max(),
                            precip_pct_ds.latitude.min(), precip_pct_ds.latitude.max()],
                     aspect='auto')
    
    ax4.set_xlabel('Longitude (°)')
    ax4.set_ylabel('Latitude (°)')
    ax4.set_title('Precipitation-Based Climate Zones', fontweight='bold')
    
    cbar4 = plt.colorbar(im4, ax=ax4, shrink=0.8, ticks=[1, 2, 3])
    cbar4.set_ticklabels(['Arid', 'Semi-Arid', 'Humid'])
    
    plt.suptitle(f'Drought and Flood Analysis\n'
                f'Period: {precip_pct_ds.attrs.get("climatology_period", "Unknown")}',
                fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    
    # Save
    filename = 'precipitation_drought_flood_analysis.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {filename}")

def plot_seasonal_precipitation_cycle(precip_pct_ds, output_dir):
    """Plot seasonal cycle of precipitation percentiles."""
    output_dir = Path(output_dir)
    
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
        
        # Select latitude band
        lat_mask = (precip_pct_ds.latitude >= lat_min) & (precip_pct_ds.latitude <= lat_max)
        
        # Calculate zonal means for each percentile
        percentiles_to_plot = ['precip_p10', 'precip_p50', 'precip_p90']
        colors = ['brown', 'green', 'blue']
        labels = ['P10 (Drought)', 'P50 (Median)', 'P90 (Intense)']
        
        for var, color, label in zip(percentiles_to_plot, colors, labels):
            if var in precip_pct_ds.data_vars:
                zonal_mean = precip_pct_ds[var].where(lat_mask, drop=True).mean(dim=['latitude', 'longitude'])
                days = precip_pct_ds.dayofyear.values
                ax.plot(days, zonal_mean, color=color, linewidth=2, label=label, alpha=0.8)
        
        ax.set_title(f'{band_name}\n({lat_min}° to {lat_max}°)', fontsize=12, fontweight='bold')
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
    
    plt.suptitle('Seasonal Cycle of Precipitation Percentiles by Latitude Band\n'
                f'Period: {precip_pct_ds.attrs.get("climatology_period", "Unknown")}',
                fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    
    # Save
    filename = 'precipitation_percentiles_seasonal_cycle.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {filename}")

def plot_precipitation_statistics(precip_pct_ds, output_dir):
    """Plot statistical analysis of precipitation percentiles."""
    output_dir = Path(output_dir)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Global distribution of percentiles
    ax1 = axes[0, 0]
    
    percentiles_to_plot = ['precip_p10', 'precip_p50', 'precip_p90']
    colors = ['brown', 'green', 'blue']
    labels = ['P10', 'P50', 'P90']
    
    for var, color, label in zip(percentiles_to_plot, colors, labels):
        if var in ds.data_vars:
            annual_mean = ds[var].mean(dim='dayofyear')
            data_flat = annual_mean.values.flatten()
            data_clean = data_flat[~np.isnan(data_flat)]
            
            ax1.hist(data_clean, bins=50, alpha=0.7, color=color, label=label, 
                    density=True, range=(0, 50))
    
    ax1.set_xlabel('Precipitation (mm/day)')
    ax1.set_ylabel('Density')
    ax1.set_title('Global Distribution of Precipitation Percentiles', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 50)
    
    # 2. Latitudinal profiles
    ax2 = axes[0, 1]
    
    for var, color, label in zip(percentiles_to_plot, colors, labels):
        if var in precip_pct_ds.data_vars:
            annual_mean = precip_pct_ds[var].mean(dim='dayofyear')
            zonal_mean = annual_mean.mean(dim='longitude')
            ax2.plot(zonal_mean.latitude, zonal_mean, color=color, linewidth=2, 
                    label=label, alpha=0.8)
    
    ax2.set_xlabel('Latitude (°)')
    ax2.set_ylabel('Precipitation (mm/day)')
    ax2.set_title('Latitudinal Precipitation Profiles', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-90, 90)
    
    # 3. Seasonal variability
    ax3 = axes[1, 0]
    
    # Calculate coefficient of variation for each percentile
    for var, color, label in zip(percentiles_to_plot, colors, labels):
        if var in precip_pct_ds.data_vars:
            seasonal_std = precip_pct_ds[var].std(dim='dayofyear').mean(dim=['latitude', 'longitude'])
            seasonal_mean = precip_pct_ds[var].mean(dim='dayofyear').mean(dim=['latitude', 'longitude'])
            cv = seasonal_std / (seasonal_mean + 0.1)  # Coefficient of variation
            
            ax3.bar(label, cv, color=color, alpha=0.7)
    
    ax3.set_ylabel('Coefficient of Variation')
    ax3.set_title('Seasonal Variability by Percentile', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Summary statistics table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    stats_data = [['Percentile', 'Global Mean', 'Global Std', 'Min', 'Max']]
    
    for var, label in zip(percentiles_to_plot, labels):
        if var in precip_pct_ds.data_vars:
            annual_mean = precip_pct_ds[var].mean(dim='dayofyear')
            data_flat = annual_mean.values.flatten()
            data_clean = data_flat[~np.isnan(data_flat)]
            
            if len(data_clean) > 0:
                stats_data.append([
                    label,
                    f'{np.mean(data_clean):.2f}',
                    f'{np.std(data_clean):.2f}',
                    f'{np.min(data_clean):.2f}',
                    f'{np.max(data_clean):.2f}'
                ])
    
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
    
    ax4.set_title('Precipitation Percentile Statistics', fontweight='bold', pad=20)
    
    plt.suptitle(f'Precipitation Percentile Analysis\n'
                f'Period: {precip_pct_ds.attrs.get("climatology_period", "Unknown")}',
                fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    
    # Save
    filename = 'precipitation_percentiles_statistics.png'
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
    parser.add_argument('--create-seasonal', action='store_true',
                       help='Create seasonal plots (DJF, MAM, JJA, SON)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("PRECIPITATION PERCENTILE VISUALIZATION")
    print("="*80)
    print(f"Input file: {args.percentile_file}")
    print(f"Output directory: {args.output_dir}")
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
        plot_global_precipitation_percentiles(precip_pct_ds, output_dir)
        
        # 2. Seasonal maps
        if args.create_seasonal:
            print("2. Seasonal maps...")
            seasons = ['DJF', 'MAM', 'JJA', 'SON']
            for season in seasons:
                plot_global_precipitation_percentiles(precip_pct_ds, output_dir, [season])
        
        # 3. Drought and flood analysis
        print("3. Drought and flood analysis...")
        plot_drought_flood_analysis(precip_pct_ds, output_dir)
        
        # 4. Seasonal cycle by latitude
        print("4. Seasonal cycle analysis...")
        plot_seasonal_precipitation_cycle(precip_pct_ds, output_dir)
        
        # 5. Statistical analysis
        print("5. Statistical analysis...")
        plot_precipitation_statistics(precip_pct_ds, output_dir)
        
        print("\n" + "="*80)
        print("PRECIPITATION PERCENTILE VISUALIZATION COMPLETED!")
        print("="*80)
        print(f"Output files saved in: {output_dir}")
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
