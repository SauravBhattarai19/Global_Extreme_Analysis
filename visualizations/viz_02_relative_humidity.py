#!/usr/bin/env python3
"""
Visualization of Relative Humidity Data (Output from 02_calculate_RH.py)

Creates comprehensive scientific visualizations of relative humidity climatology:
- Global RH distribution maps
- Seasonal patterns
- Diurnal variations (if available)
- Spatial-temporal statistics
- Quality assessment plots

Input files:
- era5_rh_{year}_{month:02d}.nc files from data/processed/relative_humidity/
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

def load_rh_data(rh_dir, years=None, sample_months=None):
    """Load relative humidity data for analysis."""
    rh_dir = Path(rh_dir)
    
    if years is None:
        # Find available years
        rh_files = list(rh_dir.glob('era5_rh_*.nc'))
        years = sorted(set([int(f.name.split('_')[2]) for f in rh_files]))
        print(f"Found data for years: {years[0]}-{years[-1]}")
        years = years[:5]  # Limit to first 5 years for visualization
    
    if sample_months is None:
        sample_months = [1, 4, 7, 10]  # Jan, Apr, Jul, Oct
    
    print(f"Loading RH data for years {years} and months {sample_months}...")
    
    datasets = []
    file_info = []
    
    for year in years:
        for month in sample_months:
            file_path = rh_dir / f"era5_rh_{year}_{month:02d}.nc"
            
            if file_path.exists():
                try:
                    ds = xr.open_dataset(file_path, chunks={'valid_time': 10})
                    datasets.append(ds)
                    file_info.append((year, month))
                    print(f"  Loaded: {file_path.name}")
                except Exception as e:
                    print(f"  Error loading {file_path}: {e}")
    
    if not datasets:
        raise ValueError(f"No RH files found in {rh_dir}")
    
    # Combine datasets
    combined_ds = xr.concat(datasets, dim='valid_time')
    
    print(f"Combined dataset shape: {combined_ds.dims}")
    print(f"RH range: {combined_ds.rh.min().values:.1f} - {combined_ds.rh.max().values:.1f}%")
    
    return combined_ds, file_info

def plot_global_rh_climatology(rh_ds, output_dir, seasons=None):
    """Create global maps of RH climatology."""
    output_dir = Path(output_dir)
    
    if seasons is None:
        seasons = ['DJF', 'MAM', 'JJA', 'SON', 'Annual']
    
    # Calculate seasonal means
    rh_seasonal = rh_ds.rh.groupby('valid_time.season').mean()
    rh_annual = rh_ds.rh.mean(dim='valid_time')
    
    # Create plots for each season
    for season in seasons:
        fig = plt.figure(figsize=(16, 10))
        
        if season == 'Annual':
            rh_data = rh_annual
            title_suffix = 'Annual Mean'
        else:
            if season in rh_seasonal.season.values:
                rh_data = rh_seasonal.sel(season=season)
                title_suffix = f'{season} Mean'
            else:
                continue
        
        # Main map
        ax = plt.subplot(2, 1, 1, projection=ccrs.Robinson())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax.set_global()
        
        # RH color scheme
        levels = np.arange(0, 101, 5)
        cmap = plt.cm.BrBG
        
        im = ax.contourf(rh_data.longitude, rh_data.latitude, rh_data,
                        levels=levels, cmap=cmap, transform=ccrs.PlateCarree(),
                        extend='both')
        
        ax.set_title(f'Relative Humidity Climatology - {title_suffix}\n'
                    f'Period: {rh_ds.valid_time.dt.year.min().values}-{rh_ds.valid_time.dt.year.max().values}',
                    fontsize=14, fontweight='bold', pad=20)
        
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.08, shrink=0.8)
        cbar.set_label('Relative Humidity (%)', fontsize=12, fontweight='bold')
        
        # Zonal mean
        ax2 = plt.subplot(2, 1, 2)
        rh_zonal = rh_data.mean(dim='longitude')
        ax2.plot(rh_zonal.latitude, rh_zonal, 'b-', linewidth=2)
        ax2.set_xlabel('Latitude (°)', fontsize=12)
        ax2.set_ylabel('Relative Humidity (%)', fontsize=12)
        ax2.set_title(f'Zonal Mean Relative Humidity - {title_suffix}', 
                     fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(-90, 90)
        
        plt.tight_layout()
        
        # Save
        filename = f'relative_humidity_climatology_{season.lower()}.png'
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {filename}")

def plot_rh_temporal_patterns(rh_ds, output_dir):
    """Plot temporal patterns in relative humidity."""
    output_dir = Path(output_dir)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Time series of global mean RH
    ax1 = axes[0, 0]
    rh_global_mean = rh_ds.rh.mean(dim=['latitude', 'longitude'])
    
    # Convert time to pandas for better plotting
    time_pd = pd.to_datetime(rh_global_mean.valid_time.values)
    
    ax1.plot(time_pd, rh_global_mean, 'b-', linewidth=1, alpha=0.7)
    
    # Add monthly means
    rh_monthly = rh_global_mean.groupby('valid_time.month').mean()
    available_months = rh_monthly.month.values
    month_names = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
    
    ax1_twin = ax1.twinx()
    ax1_twin.plot(available_months, rh_monthly, 'ro-', linewidth=2, markersize=6, alpha=0.8)
    ax1_twin.set_ylabel('Monthly Mean RH (%)', color='red', fontsize=10)
    ax1_twin.tick_params(axis='y', labelcolor='red')
    ax1_twin.set_xticks(available_months)
    ax1_twin.set_xticklabels([month_names[m-1] for m in available_months])
    
    ax1.set_xlabel('Time', fontsize=10)
    ax1.set_ylabel('Daily Global Mean RH (%)', fontsize=10)
    ax1.set_title('Global Mean Relative Humidity Time Series', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. Latitudinal variation by season
    ax2 = axes[0, 1]
    
    rh_seasonal_lat = rh_ds.rh.groupby('valid_time.season').mean().mean(dim='longitude')
    
    seasons = ['DJF', 'MAM', 'JJA', 'SON']
    colors = ['blue', 'green', 'red', 'orange']
    
    for season, color in zip(seasons, colors):
        if season in rh_seasonal_lat.season.values:
            data = rh_seasonal_lat.sel(season=season)
            ax2.plot(data.latitude, data, color=color, linewidth=2, label=season)
    
    ax2.set_xlabel('Latitude (°)', fontsize=10)
    ax2.set_ylabel('Relative Humidity (%)', fontsize=10)
    ax2.set_title('Seasonal Latitudinal Variation', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-90, 90)
    
    # 3. Diurnal cycle (if sub-daily data available)
    ax3 = axes[1, 0]
    
    # Check if we have sub-daily data
    time_diff = np.diff(rh_ds.valid_time.values).astype('timedelta64[h]').astype(int)
    if len(np.unique(time_diff)) > 1 or np.min(time_diff) < 24:
        # Sub-daily data available
        rh_hourly = rh_ds.rh.groupby('valid_time.hour').mean().mean(dim=['latitude', 'longitude'])
        hours = rh_hourly.hour.values
        ax3.plot(hours, rh_hourly, 'g-o', linewidth=2, markersize=4)
        ax3.set_xlabel('Hour of Day (UTC)', fontsize=10)
        ax3.set_ylabel('Global Mean RH (%)', fontsize=10)
        ax3.set_title('Diurnal Cycle of Relative Humidity', fontsize=12, fontweight='bold')
        ax3.set_xlim(0, 23)
        ax3.set_xticks(range(0, 24, 3))
    else:
        # Daily data - show monthly climatology instead
        rh_monthly_clim = rh_ds.rh.groupby('valid_time.month').mean().mean(dim=['latitude', 'longitude'])
        available_months_clim = rh_monthly_clim.month.values
        ax3.bar(available_months_clim, rh_monthly_clim, color='skyblue', alpha=0.7)
        ax3.set_xlabel('Month', fontsize=10)
        ax3.set_ylabel('Global Mean RH (%)', fontsize=10)
        ax3.set_title('Monthly Climatology', fontsize=12, fontweight='bold')
        ax3.set_xticks(available_months_clim)
        ax3.set_xticklabels([month_names[m-1] for m in available_months_clim])
    
    ax3.grid(True, alpha=0.3)
    
    # 4. RH distribution histogram
    ax4 = axes[1, 1]
    
    # Sample data to avoid memory issues
    rh_sample = rh_ds.rh.isel(valid_time=slice(0, None, 10))  # Every 10th time step
    rh_flat = rh_sample.values.flatten()
    rh_clean = rh_flat[~np.isnan(rh_flat)]
    
    ax4.hist(rh_clean, bins=50, density=True, alpha=0.7, color='lightblue', edgecolor='black')
    ax4.axvline(np.mean(rh_clean), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(rh_clean):.1f}%')
    ax4.axvline(np.median(rh_clean), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(rh_clean):.1f}%')
    
    ax4.set_xlabel('Relative Humidity (%)', fontsize=10)
    ax4.set_ylabel('Density', fontsize=10)
    ax4.set_title('Global RH Distribution', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'Relative Humidity Temporal Analysis\n'
                f'Period: {rh_ds.valid_time.dt.year.min().values}-{rh_ds.valid_time.dt.year.max().values}',
                fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    
    # Save
    filename = 'relative_humidity_temporal_patterns.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {filename}")

def plot_rh_regional_analysis(rh_ds, output_dir):
    """Analyze RH patterns for specific regions."""
    output_dir = Path(output_dir)
    
    # Define regions of interest
    regions = {
        'Amazon': {'lat': (-10, 5), 'lon': (-75, -45), 'color': 'green'},
        'Sahara': {'lat': (10, 30), 'lon': (-10, 30), 'color': 'orange'},
        'Southeast Asia': {'lat': (-10, 25), 'lon': (90, 140), 'color': 'blue'},
        'North America': {'lat': (25, 55), 'lon': (-130, -60), 'color': 'red'},
        'Europe': {'lat': (35, 70), 'lon': (-10, 40), 'color': 'purple'}
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Calculate regional means
    regional_data = {}
    for region_name, bounds in regions.items():
        lat_mask = (rh_ds.latitude >= bounds['lat'][0]) & (rh_ds.latitude <= bounds['lat'][1])
        lon_mask = (rh_ds.longitude >= bounds['lon'][0]) & (rh_ds.longitude <= bounds['lon'][1])
        
        regional_rh = rh_ds.rh.where(lat_mask & lon_mask, drop=True).mean(dim=['latitude', 'longitude'])
        regional_data[region_name] = regional_rh
    
    # 1. Time series comparison
    ax1 = axes[0, 0]
    
    for region_name, rh_data in regional_data.items():
        time_pd = pd.to_datetime(rh_data.valid_time.values)
        ax1.plot(time_pd, rh_data, linewidth=1.5, label=region_name, 
                color=regions[region_name]['color'], alpha=0.8)
    
    ax1.set_xlabel('Time', fontsize=10)
    ax1.set_ylabel('Regional Mean RH (%)', fontsize=10)
    ax1.set_title('Regional RH Time Series', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Seasonal patterns
    ax2 = axes[0, 1]
    
    seasonal_means = {}
    for region_name, rh_data in regional_data.items():
        seasonal_mean = rh_data.groupby('valid_time.season').mean()
        seasonal_means[region_name] = seasonal_mean
    
    seasons = ['DJF', 'MAM', 'JJA', 'SON']
    x_pos = np.arange(len(seasons))
    width = 0.15
    
    for i, (region_name, seasonal_data) in enumerate(seasonal_means.items()):
        values = [seasonal_data.sel(season=s).values if s in seasonal_data.season.values else 0 
                 for s in seasons]
        ax2.bar(x_pos + i*width, values, width, label=region_name, 
               color=regions[region_name]['color'], alpha=0.7)
    
    ax2.set_xlabel('Season', fontsize=10)
    ax2.set_ylabel('Mean RH (%)', fontsize=10)
    ax2.set_title('Seasonal Regional Comparison', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos + width*2)
    ax2.set_xticklabels(seasons)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Box plots of regional distributions
    ax3 = axes[1, 0]
    
    regional_distributions = []
    region_names = []
    
    for region_name, rh_data in regional_data.items():
        # Sample to avoid memory issues
        sampled_data = rh_data.values[::10]  # Every 10th value
        clean_data = sampled_data[~np.isnan(sampled_data)]
        if len(clean_data) > 0:
            regional_distributions.append(clean_data)
            region_names.append(region_name)
    
    bp = ax3.boxplot(regional_distributions, labels=region_names, patch_artist=True)
    
    # Color the boxes
    for patch, region_name in zip(bp['boxes'], region_names):
        patch.set_facecolor(regions[region_name]['color'])
        patch.set_alpha(0.7)
    
    ax3.set_ylabel('Relative Humidity (%)', fontsize=10)
    ax3.set_title('Regional RH Distributions', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
    
    # 4. Regional statistics table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    stats_data = [['Region', 'Mean (%)', 'Std (%)', 'Min (%)', 'Max (%)']]
    
    for region_name, rh_data in regional_data.items():
        clean_data = rh_data.values[~np.isnan(rh_data.values)]
        if len(clean_data) > 0:
            stats_data.append([
                region_name,
                f'{np.mean(clean_data):.1f}',
                f'{np.std(clean_data):.1f}',
                f'{np.min(clean_data):.1f}',
                f'{np.max(clean_data):.1f}'
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
    
    ax4.set_title('Regional Statistics', fontsize=12, fontweight='bold', pad=20)
    
    plt.suptitle(f'Regional Relative Humidity Analysis\n'
                f'Period: {rh_ds.valid_time.dt.year.min().values}-{rh_ds.valid_time.dt.year.max().values}',
                fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    
    # Save
    filename = 'relative_humidity_regional_analysis.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {filename}")

def plot_rh_quality_assessment(rh_ds, output_dir):
    """Create quality assessment plots for RH data."""
    output_dir = Path(output_dir)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Data coverage map
    ax1 = axes[0, 0]
    
    # Calculate percentage of valid data
    valid_data_pct = (~np.isnan(rh_ds.rh)).sum(dim='valid_time') / len(rh_ds.valid_time) * 100
    
    im1 = ax1.imshow(valid_data_pct, cmap='RdYlGn', vmin=0, vmax=100, 
                     extent=[rh_ds.longitude.min(), rh_ds.longitude.max(),
                            rh_ds.latitude.min(), rh_ds.latitude.max()],
                     aspect='auto')
    
    ax1.set_xlabel('Longitude (°)', fontsize=10)
    ax1.set_ylabel('Latitude (°)', fontsize=10)
    ax1.set_title('Data Coverage (%)', fontsize=12, fontweight='bold')
    
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('Valid Data (%)', fontsize=10)
    
    # 2. Temporal data availability
    ax2 = axes[0, 1]
    
    # Count valid data points per time step
    valid_per_time = (~np.isnan(rh_ds.rh)).sum(dim=['latitude', 'longitude'])
    total_points = len(rh_ds.latitude) * len(rh_ds.longitude)
    coverage_pct = valid_per_time / total_points * 100
    
    time_pd = pd.to_datetime(rh_ds.valid_time.values)
    ax2.plot(time_pd, coverage_pct, 'b-', linewidth=1)
    ax2.set_xlabel('Time', fontsize=10)
    ax2.set_ylabel('Spatial Coverage (%)', fontsize=10)
    ax2.set_title('Temporal Data Coverage', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    
    # 3. Value range check
    ax3 = axes[1, 0]
    
    # Check for values outside expected range [0, 100]
    rh_values = rh_ds.rh.values.flatten()
    rh_clean = rh_values[~np.isnan(rh_values)]
    
    # Histogram with range indicators
    ax3.hist(rh_clean, bins=50, density=True, alpha=0.7, color='lightblue', edgecolor='black')
    ax3.axvline(0, color='red', linestyle='--', linewidth=2, label='Expected Min (0%)')
    ax3.axvline(100, color='red', linestyle='--', linewidth=2, label='Expected Max (100%)')
    
    # Highlight problematic values
    below_zero = np.sum(rh_clean < 0)
    above_hundred = np.sum(rh_clean > 100)
    
    ax3.set_xlabel('Relative Humidity (%)', fontsize=10)
    ax3.set_ylabel('Density', fontsize=10)
    ax3.set_title(f'Value Range Check\nBelow 0%: {below_zero:,} | Above 100%: {above_hundred:,}', 
                 fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate comprehensive statistics
    total_values = len(rh_values)
    valid_values = len(rh_clean)
    missing_values = total_values - valid_values
    missing_pct = missing_values / total_values * 100
    
    stats_data = [
        ['Metric', 'Value'],
        ['Total Grid Points', f'{total_values:,}'],
        ['Valid Values', f'{valid_values:,}'],
        ['Missing Values', f'{missing_values:,} ({missing_pct:.1f}%)'],
        ['Mean RH', f'{np.mean(rh_clean):.1f}%'],
        ['Median RH', f'{np.median(rh_clean):.1f}%'],
        ['Std Dev', f'{np.std(rh_clean):.1f}%'],
        ['Min Value', f'{np.min(rh_clean):.1f}%'],
        ['Max Value', f'{np.max(rh_clean):.1f}%'],
        ['Values < 0%', f'{below_zero:,}'],
        ['Values > 100%', f'{above_hundred:,}']
    ]
    
    table = ax4.table(cellText=stats_data[1:], colLabels=stats_data[0],
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
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
                # Highlight problematic values
                if i >= 9:  # Values < 0% and > 100% rows
                    if int(stats_data[i][1].split()[0].replace(',', '')) > 0:
                        cell.set_facecolor('#FFE6E6')  # Light red for issues
    
    ax4.set_title('Data Quality Summary', fontsize=12, fontweight='bold', pad=20)
    
    plt.suptitle(f'Relative Humidity Data Quality Assessment\n'
                f'Period: {rh_ds.valid_time.dt.year.min().values}-{rh_ds.valid_time.dt.year.max().values}',
                fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    
    # Save
    filename = 'relative_humidity_quality_assessment.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {filename}")

def main():
    """Main visualization function for relative humidity data."""
    parser = argparse.ArgumentParser(description='Visualize relative humidity data')
    
    parser.add_argument('--rh-dir', default='data/processed/relative_humidity',
                       help='Directory containing RH files')
    parser.add_argument('--output-dir', default='visualizations/output/relative_humidity',
                       help='Output directory for plots')
    parser.add_argument('--years', nargs='+', type=int,
                       help='Specific years to analyze (default: first 5 available)')
    parser.add_argument('--months', nargs='+', type=int, default=[1, 4, 7, 10],
                       help='Months to sample (default: 1 4 7 10)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("RELATIVE HUMIDITY VISUALIZATION")
    print("="*80)
    print(f"Input directory: {args.rh_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Sample months: {args.months}")
    print("="*80)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load data
        rh_ds, file_info = load_rh_data(args.rh_dir, args.years, args.months)
        
        print("\nCreating visualizations...")
        
        # 1. Global climatology maps
        print("1. Global climatology maps...")
        plot_global_rh_climatology(rh_ds, output_dir)
        
        # 2. Temporal patterns
        print("2. Temporal patterns...")
        plot_rh_temporal_patterns(rh_ds, output_dir)
        
        # 3. Regional analysis
        print("3. Regional analysis...")
        plot_rh_regional_analysis(rh_ds, output_dir)
        
        # 4. Quality assessment
        print("4. Data quality assessment...")
        plot_rh_quality_assessment(rh_ds, output_dir)
        
        print("\n" + "="*80)
        print("RELATIVE HUMIDITY VISUALIZATION COMPLETED!")
        print("="*80)
        print(f"Output files saved in: {output_dir}")
        print("\nGenerated plots:")
        for plot_file in output_dir.glob('*.png'):
            print(f"  - {plot_file.name}")
        
        # Close dataset
        rh_ds.close()
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
