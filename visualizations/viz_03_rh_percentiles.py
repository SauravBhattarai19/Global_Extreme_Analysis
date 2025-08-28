#!/usr/bin/env python3
"""
Visualization of RH Percentiles (Output from 03_calculate_RH_percentiles.py)

Creates comprehensive scientific visualizations of relative humidity percentile climatology:
- Global maps of RH percentiles (P33, P66)
- Seasonal variations in humidity thresholds
- Latitudinal gradients
- Dry/humid zone identification
- Comparison with absolute thresholds

Input files:
- rh_percentiles.nc containing rh_p33 and rh_p66 by day-of-year
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

def load_rh_percentiles(percentile_file):
    """Load RH percentile data."""
    percentile_file = Path(percentile_file)
    
    if not percentile_file.exists():
        raise ValueError(f"Percentile file not found: {percentile_file}")
    
    ds = xr.open_dataset(percentile_file)
    
    print(f"Loaded RH percentiles from: {percentile_file}")
    print(f"Variables: {list(ds.data_vars)}")
    print(f"Period: {ds.attrs.get('climatology_period', 'Unknown')}")
    print(f"Grid: {ds.dims}")
    print(f"Day of year range: {ds.dayofyear.min().values} to {ds.dayofyear.max().values}")
    
    return ds

def plot_global_percentile_maps(rh_pct_ds, output_dir, season_months=None):
    """Create global maps of RH percentiles."""
    output_dir = Path(output_dir)
    
    # Select season or annual mean
    if season_months:
        season_name = f"{'_'.join(season_months)}"
        # Map season to day-of-year ranges (approximate)
        season_doy_map = {
            'DJF': list(range(1, 32)) + list(range(335, 366)) + list(range(60, 91)),  # Dec-Feb
            'MAM': list(range(60, 152)),   # Mar-May
            'JJA': list(range(152, 244)),  # Jun-Aug
            'SON': list(range(244, 335))   # Sep-Nov
        }
        
        if season_months[0] in season_doy_map:
            doy_indices = season_doy_map[season_months[0]]
            p33_data = rh_pct_ds.rh_p33.isel(dayofyear=[d-1 for d in doy_indices if d-1 < len(rh_pct_ds.dayofyear)]).mean(dim='dayofyear')
            p66_data = rh_pct_ds.rh_p66.isel(dayofyear=[d-1 for d in doy_indices if d-1 < len(rh_pct_ds.dayofyear)]).mean(dim='dayofyear')
        else:
            p33_data = rh_pct_ds.rh_p33.mean(dim='dayofyear')
            p66_data = rh_pct_ds.rh_p66.mean(dim='dayofyear')
    else:
        season_name = "Annual"
        p33_data = rh_pct_ds.rh_p33.mean(dim='dayofyear')
        p66_data = rh_pct_ds.rh_p66.mean(dim='dayofyear')
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # P33 (Dry threshold) map
    ax1 = plt.subplot(3, 1, 1, projection=ccrs.Robinson())
    ax1.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax1.add_feature(cfeature.BORDERS, linewidth=0.3)
    ax1.set_global()
    
    levels_p33 = np.arange(0, 81, 5)
    cmap_dry = plt.cm.YlOrBr
    
    im1 = ax1.contourf(p33_data.longitude, p33_data.latitude, p33_data,
                       levels=levels_p33, cmap=cmap_dry, transform=ccrs.PlateCarree(),
                       extend='both')
    
    ax1.set_title(f'RH 33rd Percentile (Dry Threshold) - {season_name}\n'
                 f'Period: {rh_pct_ds.attrs.get("climatology_period", "Unknown")}',
                 fontsize=14, fontweight='bold', pad=20)
    
    cbar1 = plt.colorbar(im1, ax=ax1, orientation='horizontal', pad=0.08, shrink=0.8)
    cbar1.set_label('Relative Humidity (%)', fontsize=12, fontweight='bold')
    
    # P66 (Humid threshold) map
    ax2 = plt.subplot(3, 1, 2, projection=ccrs.Robinson())
    ax2.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax2.add_feature(cfeature.BORDERS, linewidth=0.3)
    ax2.set_global()
    
    levels_p66 = np.arange(20, 101, 5)
    cmap_humid = plt.cm.BuGn
    
    im2 = ax2.contourf(p66_data.longitude, p66_data.latitude, p66_data,
                       levels=levels_p66, cmap=cmap_humid, transform=ccrs.PlateCarree(),
                       extend='both')
    
    ax2.set_title(f'RH 66th Percentile (Humid Threshold) - {season_name}',
                 fontsize=14, fontweight='bold', pad=20)
    
    cbar2 = plt.colorbar(im2, ax=ax2, orientation='horizontal', pad=0.08, shrink=0.8)
    cbar2.set_label('Relative Humidity (%)', fontsize=12, fontweight='bold')
    
    # Humidity range (P66 - P33)
    ax3 = plt.subplot(3, 1, 3, projection=ccrs.Robinson())
    ax3.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax3.add_feature(cfeature.BORDERS, linewidth=0.3)
    ax3.set_global()
    
    humidity_range = p66_data - p33_data
    levels_range = np.arange(0, 61, 3)
    cmap_range = plt.cm.plasma
    
    im3 = ax3.contourf(humidity_range.longitude, humidity_range.latitude, humidity_range,
                       levels=levels_range, cmap=cmap_range, transform=ccrs.PlateCarree(),
                       extend='both')
    
    ax3.set_title(f'Humidity Range (P66 - P33) - {season_name}',
                 fontsize=14, fontweight='bold', pad=20)
    
    cbar3 = plt.colorbar(im3, ax=ax3, orientation='horizontal', pad=0.08, shrink=0.8)
    cbar3.set_label('Humidity Range (%)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    filename = f'rh_percentiles_global_{season_name.lower()}.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {filename}")

def plot_seasonal_percentile_cycle(rh_pct_ds, output_dir, lat_bands=None):
    """Plot seasonal cycle of RH percentiles for different latitude bands."""
    output_dir = Path(output_dir)
    
    if lat_bands is None:
        lat_bands = [
            ('Tropical', -23.5, 23.5),
            ('Northern Subtropics', 23.5, 40),
            ('Northern Mid-Latitudes', 40, 60),
            ('Southern Subtropics', -40, -23.5),
            ('Southern Mid-Latitudes', -60, -40)
        ]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    colors_p33 = plt.cm.Oranges(np.linspace(0.4, 0.9, len(lat_bands)))
    colors_p66 = plt.cm.Blues(np.linspace(0.4, 0.9, len(lat_bands)))
    
    for i, (band_name, lat_min, lat_max) in enumerate(lat_bands):
        ax = axes[i]
        
        # Select latitude band
        lat_mask = (rh_pct_ds.latitude >= lat_min) & (rh_pct_ds.latitude <= lat_max)
        
        # Calculate zonal means
        p33_zonal = rh_pct_ds.rh_p33.where(lat_mask, drop=True).mean(dim=['latitude', 'longitude'])
        p66_zonal = rh_pct_ds.rh_p66.where(lat_mask, drop=True).mean(dim=['latitude', 'longitude'])
        
        # Create day-of-year axis
        days = rh_pct_ds.dayofyear.values
        
        # Plot
        ax.plot(days, p33_zonal, color='orange', linewidth=2, label='P33 (Dry)', alpha=0.8)
        ax.plot(days, p66_zonal, color='blue', linewidth=2, label='P66 (Humid)', alpha=0.8)
        ax.fill_between(days, p33_zonal, p66_zonal, alpha=0.2, color='green', label='Moderate Zone')
        
        # Add absolute thresholds for comparison
        ax.axhline(33, color='red', linestyle='--', alpha=0.7, label='Absolute 33%')
        ax.axhline(66, color='darkblue', linestyle='--', alpha=0.7, label='Absolute 66%')
        
        ax.set_title(f'{band_name}\n({lat_min}° to {lat_max}°)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Day of Year')
        ax.set_ylabel('RH Percentile (%)')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
        # Add month labels
        month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
        month_labels = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
        ax.set_xticks(month_starts)
        ax.set_xticklabels(month_labels)
        ax.set_xlim(1, 365)
        ax.set_ylim(0, 100)
    
    # Remove empty subplot
    axes[-1].remove()
    
    plt.suptitle('Seasonal Cycle of RH Percentiles by Latitude Band\n'
                f'Period: {rh_pct_ds.attrs.get("climatology_period", "Unknown")}',
                fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    
    # Save
    filename = 'rh_percentiles_seasonal_cycle.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {filename}")

def plot_threshold_comparison(rh_pct_ds, output_dir, absolute_thresholds=[33, 66]):
    """Compare percentile-based vs absolute thresholds."""
    output_dir = Path(output_dir)
    
    # Calculate annual means
    p33_annual = rh_pct_ds.rh_p33.mean(dim='dayofyear')
    p66_annual = rh_pct_ds.rh_p66.mean(dim='dayofyear')
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Difference maps: Percentile - Absolute
    ax1 = axes[0, 0]
    
    diff_p33 = p33_annual - absolute_thresholds[0]
    
    im1 = ax1.imshow(diff_p33, cmap='RdBu_r', vmin=-20, vmax=20,
                     extent=[rh_pct_ds.longitude.min(), rh_pct_ds.longitude.max(),
                            rh_pct_ds.latitude.min(), rh_pct_ds.latitude.max()],
                     aspect='auto')
    
    ax1.set_xlabel('Longitude (°)')
    ax1.set_ylabel('Latitude (°)')
    ax1.set_title(f'P33 - Absolute 33% Threshold\n(Percentile minus Absolute)', fontweight='bold')
    
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('Difference (%)')
    
    ax2 = axes[0, 1]
    
    diff_p66 = p66_annual - absolute_thresholds[1]
    
    im2 = ax2.imshow(diff_p66, cmap='RdBu_r', vmin=-20, vmax=20,
                     extent=[rh_pct_ds.longitude.min(), rh_pct_ds.longitude.max(),
                            rh_pct_ds.latitude.min(), rh_pct_ds.latitude.max()],
                     aspect='auto')
    
    ax2.set_xlabel('Longitude (°)')
    ax2.set_ylabel('Latitude (°)')
    ax2.set_title(f'P66 - Absolute 66% Threshold\n(Percentile minus Absolute)', fontweight='bold')
    
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label('Difference (%)')
    
    # 2. Scatter plots: Percentile vs Absolute
    ax3 = axes[1, 0]
    
    # Sample data to avoid memory issues
    p33_sample = p33_annual.values.flatten()[::100]
    abs_33 = np.full_like(p33_sample, absolute_thresholds[0])
    
    # Remove NaN values
    mask = ~np.isnan(p33_sample)
    p33_clean = p33_sample[mask]
    abs_33_clean = abs_33[mask]
    
    ax3.scatter(abs_33_clean, p33_clean, alpha=0.6, s=1, c='orange')
    ax3.plot([0, 100], [0, 100], 'k--', alpha=0.5, label='1:1 line')
    ax3.set_xlabel('Absolute Threshold (33%)')
    ax3.set_ylabel('Percentile Threshold (P33)')
    ax3.set_title('P33 vs Absolute 33%', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 100)
    ax3.set_ylim(0, 100)
    
    ax4 = axes[1, 1]
    
    p66_sample = p66_annual.values.flatten()[::100]
    abs_66 = np.full_like(p66_sample, absolute_thresholds[1])
    
    # Remove NaN values
    mask = ~np.isnan(p66_sample)
    p66_clean = p66_sample[mask]
    abs_66_clean = abs_66[mask]
    
    ax4.scatter(abs_66_clean, p66_clean, alpha=0.6, s=1, c='blue')
    ax4.plot([0, 100], [0, 100], 'k--', alpha=0.5, label='1:1 line')
    ax4.set_xlabel('Absolute Threshold (66%)')
    ax4.set_ylabel('Percentile Threshold (P66)')
    ax4.set_title('P66 vs Absolute 66%', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 100)
    ax4.set_ylim(0, 100)
    
    plt.suptitle(f'Percentile vs Absolute Threshold Comparison\n'
                f'Period: {rh_pct_ds.attrs.get("climatology_period", "Unknown")}',
                fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    
    # Save
    filename = 'rh_percentiles_threshold_comparison.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {filename}")

def plot_climate_zone_analysis(rh_pct_ds, output_dir):
    """Analyze RH percentiles by climate zones."""
    output_dir = Path(output_dir)
    
    # Calculate annual means
    p33_annual = rh_pct_ds.rh_p33.mean(dim='dayofyear')
    p66_annual = rh_pct_ds.rh_p66.mean(dim='dayofyear')
    humidity_range = p66_annual - p33_annual
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Climate zone classification based on humidity characteristics
    ax1 = axes[0, 0]
    
    # Define climate zones based on P33 and P66 values
    # Arid: P66 < 50
    # Semi-arid: P33 < 40 and P66 >= 50
    # Humid: P33 >= 40 and P66 >= 70
    # Moderate: Others
    
    climate_zones = np.full_like(p33_annual.values, 0, dtype=int)
    
    # Arid zones (very dry)
    arid_mask = (p66_annual < 50).values
    climate_zones[arid_mask] = 1
    
    # Semi-arid zones
    semi_arid_mask = ((p33_annual < 40) & (p66_annual >= 50)).values
    climate_zones[semi_arid_mask] = 2
    
    # Humid zones
    humid_mask = ((p33_annual >= 40) & (p66_annual >= 70)).values
    climate_zones[humid_mask] = 3
    
    # Moderate zones (everything else)
    moderate_mask = ~(arid_mask | semi_arid_mask | humid_mask)
    climate_zones[moderate_mask] = 4
    
    # Create custom colormap
    colors = ['white', 'brown', 'orange', 'green', 'blue']
    cmap = mcolors.ListedColormap(colors[1:])  # Exclude white (for NaN)
    
    im1 = ax1.imshow(climate_zones, cmap=cmap, vmin=1, vmax=4,
                     extent=[rh_pct_ds.longitude.min(), rh_pct_ds.longitude.max(),
                            rh_pct_ds.latitude.min(), rh_pct_ds.latitude.max()],
                     aspect='auto')
    
    ax1.set_xlabel('Longitude (°)')
    ax1.set_ylabel('Latitude (°)')
    ax1.set_title('Climate Zones by Humidity Characteristics', fontweight='bold')
    
    # Custom colorbar
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8, ticks=[1, 2, 3, 4])
    cbar1.set_ticklabels(['Arid', 'Semi-Arid', 'Humid', 'Moderate'])
    
    # 2. Humidity variability (seasonal range)
    ax2 = axes[0, 1]
    
    # Calculate seasonal variability
    p33_seasonal_std = rh_pct_ds.rh_p33.std(dim='dayofyear')
    p66_seasonal_std = rh_pct_ds.rh_p66.std(dim='dayofyear')
    total_variability = p33_seasonal_std + p66_seasonal_std
    
    im2 = ax2.imshow(total_variability, cmap='viridis', vmin=0, vmax=20,
                     extent=[rh_pct_ds.longitude.min(), rh_pct_ds.longitude.max(),
                            rh_pct_ds.latitude.min(), rh_pct_ds.latitude.max()],
                     aspect='auto')
    
    ax2.set_xlabel('Longitude (°)')
    ax2.set_ylabel('Latitude (°)')
    ax2.set_title('Seasonal Humidity Variability\n(P33 + P66 std dev)', fontweight='bold')
    
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label('Variability (%)')
    
    # 3. Latitudinal profiles
    ax3 = axes[1, 0]
    
    # Calculate zonal means
    p33_zonal = p33_annual.mean(dim='longitude')
    p66_zonal = p66_annual.mean(dim='longitude')
    range_zonal = humidity_range.mean(dim='longitude')
    
    ax3.plot(p33_zonal.latitude, p33_zonal, 'o-', color='orange', label='P33', linewidth=2)
    ax3.plot(p66_zonal.latitude, p66_zonal, 'o-', color='blue', label='P66', linewidth=2)
    ax3.plot(range_zonal.latitude, range_zonal, 'o-', color='green', label='Range (P66-P33)', linewidth=2)
    
    # Add absolute thresholds
    ax3.axhline(33, color='red', linestyle='--', alpha=0.7, label='Absolute 33%')
    ax3.axhline(66, color='darkblue', linestyle='--', alpha=0.7, label='Absolute 66%')
    
    ax3.set_xlabel('Latitude (°)')
    ax3.set_ylabel('Relative Humidity (%)')
    ax3.set_title('Latitudinal RH Percentile Profiles', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-90, 90)
    
    # 4. Statistics by climate zone
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate statistics for each zone
    zone_names = ['Arid', 'Semi-Arid', 'Humid', 'Moderate']
    zone_stats = []
    
    for i, zone_name in enumerate(zone_names, 1):
        zone_mask = climate_zones == i
        if np.any(zone_mask):
            area_pct = np.sum(zone_mask) / np.sum(~np.isnan(climate_zones)) * 100
            p33_mean = np.nanmean(p33_annual.values[zone_mask])
            p66_mean = np.nanmean(p66_annual.values[zone_mask])
            range_mean = np.nanmean(humidity_range.values[zone_mask])
            
            zone_stats.append([
                zone_name,
                f'{area_pct:.1f}%',
                f'{p33_mean:.1f}%',
                f'{p66_mean:.1f}%',
                f'{range_mean:.1f}%'
            ])
    
    if zone_stats:
        headers = ['Climate Zone', 'Area', 'Mean P33', 'Mean P66', 'Mean Range']
        table_data = [headers] + zone_stats
        
        table = ax4.table(cellText=zone_stats, colLabels=headers,
                         cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style the table
        for i in range(len(table_data)):
            for j in range(len(headers)):
                cell = table[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor('#4472C4')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    # Color by climate zone
                    zone_colors = {'Arid': '#D2691E', 'Semi-Arid': '#FF8C00', 
                                  'Humid': '#228B22', 'Moderate': '#4682B4'}
                    zone_name = zone_stats[i-1][0]
                    cell.set_facecolor(zone_colors.get(zone_name, 'white'))
                    cell.set_alpha(0.3)
    
    ax4.set_title('Climate Zone Statistics', fontsize=12, fontweight='bold', pad=20)
    
    plt.suptitle(f'Climate Zone Analysis Based on RH Percentiles\n'
                f'Period: {rh_pct_ds.attrs.get("climatology_period", "Unknown")}',
                fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    
    # Save
    filename = 'rh_percentiles_climate_zones.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {filename}")

def main():
    """Main visualization function for RH percentiles."""
    parser = argparse.ArgumentParser(description='Visualize RH percentile climatology')
    
    parser.add_argument('--percentile-file', default='data/processed/rh_percentiles.nc',
                       help='RH percentile file')
    parser.add_argument('--output-dir', default='visualizations/output/rh_percentiles',
                       help='Output directory for plots')
    parser.add_argument('--create-seasonal', action='store_true',
                       help='Create seasonal plots (DJF, MAM, JJA, SON)')
    parser.add_argument('--absolute-thresholds', nargs=2, type=float, default=[33, 66],
                       help='Absolute thresholds for comparison (default: 33 66)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("RH PERCENTILE VISUALIZATION")
    print("="*80)
    print(f"Input file: {args.percentile_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"Absolute thresholds: {args.absolute_thresholds}")
    print("="*80)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load data
        rh_pct_ds = load_rh_percentiles(args.percentile_file)
        
        print("\nCreating visualizations...")
        
        # 1. Global maps (annual)
        print("1. Global annual maps...")
        plot_global_percentile_maps(rh_pct_ds, output_dir)
        
        # 2. Seasonal maps
        if args.create_seasonal:
            print("2. Seasonal maps...")
            seasons = ['DJF', 'MAM', 'JJA', 'SON']
            for season in seasons:
                plot_global_percentile_maps(rh_pct_ds, output_dir, [season])
        
        # 3. Seasonal cycle by latitude
        print("3. Seasonal cycle analysis...")
        plot_seasonal_percentile_cycle(rh_pct_ds, output_dir)
        
        # 4. Threshold comparison
        print("4. Threshold comparison...")
        plot_threshold_comparison(rh_pct_ds, output_dir, args.absolute_thresholds)
        
        # 5. Climate zone analysis
        print("5. Climate zone analysis...")
        plot_climate_zone_analysis(rh_pct_ds, output_dir)
        
        print("\n" + "="*80)
        print("RH PERCENTILE VISUALIZATION COMPLETED!")
        print("="*80)
        print(f"Output files saved in: {output_dir}")
        print("\nGenerated plots:")
        for plot_file in output_dir.glob('*.png'):
            print(f"  - {plot_file.name}")
        
        # Close dataset
        rh_pct_ds.close()
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
