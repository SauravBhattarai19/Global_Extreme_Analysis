#!/usr/bin/env python3
"""
Visualization of Temperature Percentiles (Output from 01_calculate_percentiles.py)

Creates comprehensive scientific visualizations of temperature percentile climatology:
- Global maps of temperature percentiles
- Seasonal variations
- Latitudinal gradients
- Comparison between tmax and tmin percentiles
- Statistical summaries

Input files:
- tmax_p95_1980-2000.nc
- tmin_p95_1980-2000.nc
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

def load_percentile_data(percentile_dir):
    """Load temperature percentile data."""
    percentile_dir = Path(percentile_dir)
    
    # Find percentile files
    tmax_files = list(percentile_dir.glob('tmax_p95_*.nc'))
    tmin_files = list(percentile_dir.glob('tmin_p95_*.nc'))
    
    if not tmax_files or not tmin_files:
        raise ValueError(f"No percentile files found in {percentile_dir}")
    
    tmax_ds = xr.open_dataset(tmax_files[0])
    tmin_ds = xr.open_dataset(tmin_files[0])
    
    print(f"Loaded temperature percentiles:")
    print(f"  Tmax: {tmax_files[0]}")
    print(f"  Tmin: {tmin_files[0]}")
    print(f"  Period: {tmax_ds.attrs.get('baseline_period', 'Unknown')}")
    print(f"  Grid: {tmax_ds.dims}")
    
    return tmax_ds, tmin_ds

def plot_global_percentile_maps(tmax_ds, tmin_ds, output_dir, season_months=None):
    """Create global maps of temperature percentiles."""
    output_dir = Path(output_dir)
    
    # Select season or annual mean
    if season_months:
        season_name = f"{'_'.join(season_months)}"
        tmax_data = tmax_ds.tmax_p95.isel(dayofyear=slice(0, 365)).groupby('dayofyear.season').mean()
        tmin_data = tmin_ds.tmin_p95.isel(dayofyear=slice(0, 365)).groupby('dayofyear.season').mean()
        
        # Map season names to indices
        season_map = {'DJF': 0, 'MAM': 1, 'JJA': 2, 'SON': 3}
        if season_months[0] in season_map:
            tmax_data = tmax_data.isel(season=season_map[season_months[0]])
            tmin_data = tmin_data.isel(season=season_map[season_months[0]])
    else:
        season_name = "Annual"
        tmax_data = tmax_ds.tmax_p95.mean(dim='dayofyear')
        tmin_data = tmin_ds.tmin_p95.mean(dim='dayofyear')
    
    # Convert from Kelvin to Celsius
    tmax_data_c = tmax_data - 273.15
    tmin_data_c = tmin_data - 273.15
    
    # Create figure
    fig = plt.figure(figsize=(20, 12))
    
    # Tmax plot
    ax1 = plt.subplot(2, 1, 1, projection=ccrs.Robinson())
    ax1.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax1.add_feature(cfeature.BORDERS, linewidth=0.3)
    ax1.set_global()
    
    # Color scheme for temperature
    levels_tmax = np.arange(-10, 51, 2)
    cmap_tmax = plt.cm.RdYlBu_r
    
    im1 = ax1.contourf(tmax_data_c.longitude, tmax_data_c.latitude, tmax_data_c,
                       levels=levels_tmax, cmap=cmap_tmax, transform=ccrs.PlateCarree(),
                       extend='both')
    
    ax1.set_title(f'Daily Maximum Temperature 95th Percentile - {season_name}\n'
                 f'Period: {tmax_ds.attrs.get("baseline_period", "Unknown")}', 
                 fontsize=14, fontweight='bold', pad=20)
    
    cbar1 = plt.colorbar(im1, ax=ax1, orientation='horizontal', pad=0.08, shrink=0.8)
    cbar1.set_label('Temperature (°C)', fontsize=12, fontweight='bold')
    
    # Tmin plot
    ax2 = plt.subplot(2, 1, 2, projection=ccrs.Robinson())
    ax2.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax2.add_feature(cfeature.BORDERS, linewidth=0.3)
    ax2.set_global()
    
    levels_tmin = np.arange(-30, 31, 2)
    cmap_tmin = plt.cm.RdYlBu_r
    
    im2 = ax2.contourf(tmin_data_c.longitude, tmin_data_c.latitude, tmin_data_c,
                       levels=levels_tmin, cmap=cmap_tmin, transform=ccrs.PlateCarree(),
                       extend='both')
    
    ax2.set_title(f'Daily Minimum Temperature 95th Percentile - {season_name}', 
                 fontsize=14, fontweight='bold', pad=20)
    
    cbar2 = plt.colorbar(im2, ax=ax2, orientation='horizontal', pad=0.08, shrink=0.8)
    cbar2.set_label('Temperature (°C)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    filename = f'temperature_percentiles_global_{season_name.lower()}.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {filename}")

def plot_seasonal_cycle(tmax_ds, tmin_ds, output_dir, lat_bands=None):
    """Plot seasonal cycle of temperature percentiles for different latitude bands."""
    output_dir = Path(output_dir)
    
    if lat_bands is None:
        lat_bands = [
            ('Arctic', 70, 90),
            ('Northern Mid-Latitudes', 30, 70),
            ('Tropics', -30, 30),
            ('Southern Mid-Latitudes', -70, -30),
            ('Antarctic', -90, -70)
        ]
    
    # Convert to Celsius
    tmax_data = tmax_ds.tmax_p95 - 273.15
    tmin_data = tmin_ds.tmin_p95 - 273.15
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(lat_bands)))
    
    for i, (band_name, lat_min, lat_max) in enumerate(lat_bands):
        ax = axes[i]
        
        # Select latitude band
        lat_mask = (tmax_ds.latitude >= lat_min) & (tmax_ds.latitude <= lat_max)
        
        # Calculate zonal means
        tmax_zonal = tmax_data.where(lat_mask, drop=True).mean(dim=['latitude', 'longitude'])
        tmin_zonal = tmin_data.where(lat_mask, drop=True).mean(dim=['latitude', 'longitude'])
        
        # Create day-of-year axis
        days = np.arange(1, len(tmax_zonal) + 1)
        
        # Plot
        ax.plot(days, tmax_zonal, 'r-', linewidth=2, label='Tmax P95', alpha=0.8)
        ax.plot(days, tmin_zonal, 'b-', linewidth=2, label='Tmin P95', alpha=0.8)
        
        ax.set_title(f'{band_name}\n({lat_min}° to {lat_max}°)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Day of Year')
        ax.set_ylabel('Temperature (°C)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add month labels
        month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
        month_labels = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
        ax.set_xticks(month_starts)
        ax.set_xticklabels(month_labels)
        
        # Set reasonable y-limits
        all_temps = np.concatenate([tmax_zonal.values, tmin_zonal.values])
        temp_range = np.nanmax(all_temps) - np.nanmin(all_temps)
        y_margin = temp_range * 0.1
        ax.set_ylim(np.nanmin(all_temps) - y_margin, np.nanmax(all_temps) + y_margin)
    
    # Remove empty subplot
    axes[-1].remove()
    
    plt.suptitle('Seasonal Cycle of Temperature Percentiles by Latitude Band\n'
                f'Period: {tmax_ds.attrs.get("baseline_period", "Unknown")}',
                fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    
    # Save
    filename = 'temperature_percentiles_seasonal_cycle.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {filename}")

def plot_latitudinal_gradients(tmax_ds, tmin_ds, output_dir):
    """Plot latitudinal gradients of temperature percentiles."""
    output_dir = Path(output_dir)
    
    # Convert to Celsius and calculate annual means
    tmax_annual = (tmax_ds.tmax_p95 - 273.15).mean(dim='dayofyear')
    tmin_annual = (tmin_ds.tmin_p95 - 273.15).mean(dim='dayofyear')
    
    # Calculate zonal means
    tmax_zonal = tmax_annual.mean(dim='longitude')
    tmin_zonal = tmin_annual.mean(dim='longitude')
    
    # Calculate statistics
    tmax_std = tmax_annual.std(dim='longitude')
    tmin_std = tmin_annual.std(dim='longitude')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Latitudinal gradient
    ax1.plot(tmax_zonal, tmax_zonal.latitude, 'r-', linewidth=3, label='Tmax P95', alpha=0.8)
    ax1.fill_betweenx(tmax_zonal.latitude, 
                      tmax_zonal - tmax_std, tmax_zonal + tmax_std, 
                      alpha=0.2, color='red', label='Tmax ±1σ')
    
    ax1.plot(tmin_zonal, tmin_zonal.latitude, 'b-', linewidth=3, label='Tmin P95', alpha=0.8)
    ax1.fill_betweenx(tmin_zonal.latitude, 
                      tmin_zonal - tmin_std, tmin_zonal + tmin_std, 
                      alpha=0.2, color='blue', label='Tmin ±1σ')
    
    ax1.set_xlabel('Temperature (°C)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Latitude (°)', fontsize=12, fontweight='bold')
    ax1.set_title('Latitudinal Temperature Gradient\n(Annual Mean ± Standard Deviation)', 
                 fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(-90, 90)
    
    # Diurnal temperature range (DTR)
    dtr = tmax_zonal - tmin_zonal
    dtr_std = np.sqrt(tmax_std**2 + tmin_std**2)  # Error propagation
    
    ax2.plot(dtr, dtr.latitude, 'g-', linewidth=3, label='DTR (Tmax - Tmin)', alpha=0.8)
    ax2.fill_betweenx(dtr.latitude, dtr - dtr_std, dtr + dtr_std, 
                      alpha=0.2, color='green', label='DTR ±1σ')
    
    ax2.set_xlabel('Diurnal Temperature Range (°C)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Latitude (°)', fontsize=12, fontweight='bold')
    ax2.set_title('Diurnal Temperature Range\n(P95 Tmax - P95 Tmin)', 
                 fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim(-90, 90)
    
    plt.suptitle(f'Temperature Percentile Analysis\n'
                f'Period: {tmax_ds.attrs.get("baseline_period", "Unknown")}',
                fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    
    # Save
    filename = 'temperature_percentiles_latitudinal_gradients.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {filename}")

def plot_percentile_statistics(tmax_ds, tmin_ds, output_dir):
    """Plot statistical summary of temperature percentiles."""
    output_dir = Path(output_dir)
    
    # Convert to Celsius
    tmax_data = tmax_ds.tmax_p95 - 273.15
    tmin_data = tmin_ds.tmin_p95 - 273.15
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Global histograms
    ax1 = axes[0, 0]
    tmax_flat = tmax_data.values.flatten()
    tmin_flat = tmin_data.values.flatten()
    
    # Remove NaN values
    tmax_clean = tmax_flat[~np.isnan(tmax_flat)]
    tmin_clean = tmin_flat[~np.isnan(tmin_flat)]
    
    ax1.hist(tmax_clean, bins=50, alpha=0.7, color='red', label='Tmax P95', density=True)
    ax1.hist(tmin_clean, bins=50, alpha=0.7, color='blue', label='Tmin P95', density=True)
    ax1.set_xlabel('Temperature (°C)', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Global Distribution of Temperature Percentiles', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Seasonal variability
    ax2 = axes[0, 1]
    
    # Calculate seasonal standard deviation
    # Create a season coordinate manually since dayofyear doesn't have season attribute
    doy_values = tmax_data.dayofyear.values
    seasons = np.full_like(doy_values, '', dtype='<U3')
    
    # Map day-of-year to seasons
    seasons[(doy_values >= 1) & (doy_values <= 59)] = 'DJF'    # Jan-Feb
    seasons[(doy_values >= 60) & (doy_values <= 151)] = 'MAM'   # Mar-May
    seasons[(doy_values >= 152) & (doy_values <= 243)] = 'JJA'  # Jun-Aug
    seasons[(doy_values >= 244) & (doy_values <= 334)] = 'SON'  # Sep-Nov
    seasons[(doy_values >= 335) & (doy_values <= 366)] = 'DJF'  # Dec
    
    # Calculate seasonal statistics manually
    seasonal_stats = {}
    for season in ['DJF', 'MAM', 'JJA', 'SON']:
        season_mask = seasons == season
        if np.any(season_mask):
            tmax_season = tmax_data.isel(dayofyear=season_mask)
            tmin_season = tmin_data.isel(dayofyear=season_mask)
            
            seasonal_stats[season] = {
                'tmax_std': tmax_season.std(dim='dayofyear').mean(dim=['latitude', 'longitude']).values,
                'tmin_std': tmin_season.std(dim='dayofyear').mean(dim=['latitude', 'longitude']).values
            }
    
    # Extract values for plotting
    tmax_seasonal_std = [seasonal_stats[s]['tmax_std'] for s in ['DJF', 'MAM', 'JJA', 'SON']]
    tmin_seasonal_std = [seasonal_stats[s]['tmin_std'] for s in ['DJF', 'MAM', 'JJA', 'SON']]
    
    seasons = ['DJF', 'MAM', 'JJA', 'SON']
    x_pos = np.arange(len(seasons))
    
    width = 0.35
    ax2.bar(x_pos - width/2, tmax_seasonal_std, width, label='Tmax P95', color='red', alpha=0.7)
    ax2.bar(x_pos + width/2, tmin_seasonal_std, width, label='Tmin P95', color='blue', alpha=0.7)
    
    ax2.set_xlabel('Season', fontsize=12)
    ax2.set_ylabel('Temperature Variability (°C)', fontsize=12)
    ax2.set_title('Seasonal Temperature Variability', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(seasons)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Land-Ocean contrast (simplified)
    ax3 = axes[1, 0]
    
    # Calculate annual means
    tmax_annual = tmax_data.mean(dim='dayofyear')
    tmin_annual = tmin_data.mean(dim='dayofyear')
    
    # Simple land-ocean mask (land typically has higher variability)
    # Use temperature range as proxy for continentality
    temp_range = tmax_annual - tmin_annual
    
    # Create scatter plot
    scatter = ax3.scatter(tmax_annual.values.flatten(), tmin_annual.values.flatten(),
                         c=temp_range.values.flatten(), cmap='viridis', alpha=0.6, s=1)
    
    ax3.set_xlabel('Tmax P95 (°C)', fontsize=12)
    ax3.set_ylabel('Tmin P95 (°C)', fontsize=12)
    ax3.set_title('Tmax vs Tmin Percentiles\n(colored by temperature range)', 
                 fontsize=14, fontweight='bold')
    
    # Add 1:1 line
    min_temp = min(np.nanmin(tmax_annual.values), np.nanmin(tmin_annual.values))
    max_temp = max(np.nanmax(tmax_annual.values), np.nanmax(tmin_annual.values))
    ax3.plot([min_temp, max_temp], [min_temp, max_temp], 'k--', alpha=0.5, label='1:1 line')
    ax3.legend()
    
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Temperature Range (°C)', fontsize=10)
    
    # Summary statistics table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    stats_data = [
        ['Statistic', 'Tmax P95 (°C)', 'Tmin P95 (°C)'],
        ['Global Mean', f'{np.nanmean(tmax_clean):.1f}', f'{np.nanmean(tmin_clean):.1f}'],
        ['Global Std', f'{np.nanstd(tmax_clean):.1f}', f'{np.nanstd(tmin_clean):.1f}'],
        ['Global Min', f'{np.nanmin(tmax_clean):.1f}', f'{np.nanmin(tmin_clean):.1f}'],
        ['Global Max', f'{np.nanmax(tmax_clean):.1f}', f'{np.nanmax(tmin_clean):.1f}'],
        ['25th Percentile', f'{np.nanpercentile(tmax_clean, 25):.1f}', f'{np.nanpercentile(tmin_clean, 25):.1f}'],
        ['75th Percentile', f'{np.nanpercentile(tmax_clean, 75):.1f}', f'{np.nanpercentile(tmin_clean, 75):.1f}']
    ]
    
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
    
    ax4.set_title('Statistical Summary', fontsize=14, fontweight='bold', pad=20)
    
    plt.suptitle(f'Temperature Percentile Analysis\n'
                f'Period: {tmax_ds.attrs.get("baseline_period", "Unknown")}',
                fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    
    # Save
    filename = 'temperature_percentiles_statistics.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {filename}")

def main():
    """Main visualization function for temperature percentiles."""
    parser = argparse.ArgumentParser(description='Visualize temperature percentile climatology')
    
    parser.add_argument('--percentile-dir', default='data/processed/percentiles',
                       help='Directory containing temperature percentile files')
    parser.add_argument('--output-dir', default='visualizations/output/temperature_percentiles',
                       help='Output directory for plots')
    parser.add_argument('--create-seasonal', action='store_true',
                       help='Create seasonal plots (DJF, MAM, JJA, SON)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("TEMPERATURE PERCENTILE VISUALIZATION")
    print("="*80)
    print(f"Input directory: {args.percentile_dir}")
    print(f"Output directory: {args.output_dir}")
    print("="*80)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load data
        tmax_ds, tmin_ds = load_percentile_data(args.percentile_dir)
        
        print("\nCreating visualizations...")
        
        # 1. Global maps (annual)
        print("1. Global annual maps...")
        plot_global_percentile_maps(tmax_ds, tmin_ds, output_dir)
        
        # 2. Seasonal maps
        if args.create_seasonal:
            print("2. Seasonal maps...")
            seasons = ['DJF', 'MAM', 'JJA', 'SON']
            for season in seasons:
                plot_global_percentile_maps(tmax_ds, tmin_ds, output_dir, [season])
        
        # 3. Seasonal cycle by latitude
        print("3. Seasonal cycle analysis...")
        plot_seasonal_cycle(tmax_ds, tmin_ds, output_dir)
        
        # 4. Latitudinal gradients
        print("4. Latitudinal gradients...")
        plot_latitudinal_gradients(tmax_ds, tmin_ds, output_dir)
        
        # 5. Statistical summary
        print("5. Statistical summary...")
        plot_percentile_statistics(tmax_ds, tmin_ds, output_dir)
        
        print("\n" + "="*80)
        print("TEMPERATURE PERCENTILE VISUALIZATION COMPLETED!")
        print("="*80)
        print(f"Output files saved in: {output_dir}")
        print("\nGenerated plots:")
        for plot_file in output_dir.glob('*.png'):
            print(f"  - {plot_file.name}")
        
        # Close datasets
        tmax_ds.close()
        tmin_ds.close()
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
