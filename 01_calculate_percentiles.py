#!/usr/bin/env python3
"""
Calculate percentiles for heatwave detection using ERA5 data (1980-2000 baseline).
Optimized for high-performance computing with 192 cores and 755GB RAM.
"""

import sys
import argparse
import numpy as np
import xarray as xr
from pathlib import Path
import multiprocessing as mp
from functools import partial
import gc
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_era5_monthly_file(file_path):
    """Load a single ERA5 monthly file."""
    try:
        ds = xr.open_dataset(file_path, chunks={'valid_time': 50})
        return ds
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def daily_extremes_from_subdaily(temp_data):
    """Calculate daily max and min from sub-daily data."""
    # Group by date and calculate daily extremes
    daily_max = temp_data.resample(valid_time='1D').max()
    daily_min = temp_data.resample(valid_time='1D').min()
    return daily_max, daily_min

def calculate_percentile_doy(data, times, percentile=95, window_days=15):
    """
    Calculate percentile for each day of year using a moving window.
    
    Parameters:
    - data: numpy array of shape (time, lat, lon)
    - times: array of datetime objects
    - percentile: percentile to calculate (default 95)
    - window_days: window size around each day of year (default 15)
    """
    # Convert times to pandas datetime
    times_pd = pd.to_datetime(times)
    day_of_years = times_pd.dayofyear.values
    
    n_lat, n_lon = data.shape[1], data.shape[2]
    percentiles = np.full((366, n_lat, n_lon), np.nan)
    
    for doy in range(1, 367):
        # Create window around day of year (handles year boundaries)
        window_doys = []
        for offset in range(-window_days, window_days + 1):
            target_doy = doy + offset
            if target_doy <= 0:
                target_doy += 366
            elif target_doy > 366:
                target_doy -= 366
            window_doys.append(target_doy)
        
        # Find all days in the window across all years
        mask = np.isin(day_of_years, window_doys)
        
        if np.sum(mask) > 0:
            window_data = data[mask, :, :]
            percentiles[doy-1, :, :] = np.nanpercentile(window_data, percentile, axis=0)
    
    return percentiles

def process_spatial_chunk(chunk_info, years, data_dir, percentile=95):
    """Process a spatial chunk for percentile calculation."""
    lat_start, lat_end, lon_start, lon_end = chunk_info
    
    print(f"Processing chunk: lat[{lat_start}:{lat_end}], lon[{lon_start}:{lon_end}]")
    
    # Collect all data for the chunk across all years
    all_tmax_data = []
    all_tmin_data = []
    all_times = []
    lat_coords = None
    lon_coords = None
    
    for year in years:
        for month in range(1, 13):
            file_path = Path(data_dir) / f"era5_daily_{year}_{month:02d}.nc"
            
            if not file_path.exists():
                print(f"Warning: File not found: {file_path}")
                continue
            
            try:
                ds = load_era5_monthly_file(file_path)
                if ds is None:
                    continue
                
                # Subset to chunk
                ds_chunk = ds.isel(
                    latitude=slice(lat_start, lat_end),
                    longitude=slice(lon_start, lon_end)
                )
                
                # Calculate daily extremes
                daily_tmax, daily_tmin = daily_extremes_from_subdaily(ds_chunk.t2m)
                
                # Store coordinates from first file
                if lat_coords is None:
                    lat_coords = daily_tmax.latitude.values
                    lon_coords = daily_tmax.longitude.values
                
                # Collect data
                all_tmax_data.append(daily_tmax.values)
                all_tmin_data.append(daily_tmin.values)
                all_times.extend(daily_tmax.valid_time.values)
                
                # Clean up
                ds.close()
                daily_tmax.close()
                daily_tmin.close()
                del ds, ds_chunk, daily_tmax, daily_tmin
                gc.collect()
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
    
    if not all_tmax_data:
        print(f"No data found for chunk lat[{lat_start}:{lat_end}], lon[{lon_start}:{lon_end}]")
        return None
    
    # Combine all data
    combined_tmax = np.concatenate(all_tmax_data, axis=0)
    combined_tmin = np.concatenate(all_tmin_data, axis=0)
    
    # Calculate percentiles
    tmax_percentiles = calculate_percentile_doy(combined_tmax, all_times, percentile)
    tmin_percentiles = calculate_percentile_doy(combined_tmin, all_times, percentile)
    
    result = {
        'lat_bounds': (lat_start, lat_end),
        'lon_bounds': (lon_start, lon_end),
        'tmax_percentiles': tmax_percentiles,
        'tmin_percentiles': tmin_percentiles,
        'latitude': lat_coords,
        'longitude': lon_coords
    }
    
    # Clean up
    del combined_tmax, combined_tmin, all_tmax_data, all_tmin_data
    gc.collect()
    
    print(f"Completed chunk: lat[{lat_start}:{lat_end}], lon[{lon_start}:{lon_end}]")
    return result

def create_spatial_chunks(n_lat, n_lon, chunk_size_lat=50, chunk_size_lon=100):
    """Create spatial chunks for parallel processing."""
    chunks = []
    for lat_start in range(0, n_lat, chunk_size_lat):
        lat_end = min(lat_start + chunk_size_lat, n_lat)
        for lon_start in range(0, n_lon, chunk_size_lon):
            lon_end = min(lon_start + chunk_size_lon, n_lon)
            chunks.append((lat_start, lat_end, lon_start, lon_end))
    return chunks

def combine_chunks(chunk_results, full_lat, full_lon, output_dir, percentile, years):
    """Combine spatial chunks into full global arrays."""
    n_doy = 366
    tmax_percentiles_full = np.full((n_doy, len(full_lat), len(full_lon)), np.nan)
    tmin_percentiles_full = np.full((n_doy, len(full_lat), len(full_lon)), np.nan)
    
    for result in chunk_results:
        if result is None:
            continue
        
        lat_start, lat_end = result['lat_bounds']
        lon_start, lon_end = result['lon_bounds']
        
        tmax_percentiles_full[:, lat_start:lat_end, lon_start:lon_end] = result['tmax_percentiles']
        tmin_percentiles_full[:, lat_start:lat_end, lon_start:lon_end] = result['tmin_percentiles']
    
    # Create datasets
    doy_coord = np.arange(1, 367)
    
    tmax_ds = xr.Dataset({
        f'tmax_p{percentile}': (['dayofyear', 'latitude', 'longitude'], tmax_percentiles_full)
    }, coords={
        'dayofyear': doy_coord,
        'latitude': full_lat,
        'longitude': full_lon
    })
    
    tmin_ds = xr.Dataset({
        f'tmin_p{percentile}': (['dayofyear', 'latitude', 'longitude'], tmin_percentiles_full)
    }, coords={
        'dayofyear': doy_coord,
        'latitude': full_lat,
        'longitude': full_lon
    })
    
    # Add attributes
    tmax_ds[f'tmax_p{percentile}'].attrs = {
        'long_name': f'{percentile}th percentile of daily maximum temperature',
        'units': 'K',
        'description': f'Calculated from {years[0]}-{years[-1]} using 15-day window around each day of year',
        'baseline_period': f'{years[0]}-{years[-1]}',
        'calculation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    tmin_ds[f'tmin_p{percentile}'].attrs = {
        'long_name': f'{percentile}th percentile of daily minimum temperature',
        'units': 'K', 
        'description': f'Calculated from {years[0]}-{years[-1]} using 15-day window around each day of year',
        'baseline_period': f'{years[0]}-{years[-1]}',
        'calculation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save files
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tmax_file = output_dir / f'tmax_p{percentile}_{years[0]}-{years[-1]}.nc'
    tmin_file = output_dir / f'tmin_p{percentile}_{years[0]}-{years[-1]}.nc'
    
    print("Saving percentile files...")
    tmax_ds.to_netcdf(tmax_file)
    tmin_ds.to_netcdf(tmin_file)
    
    print(f"Percentile files saved:")
    print(f"  - {tmax_file}")
    print(f"  - {tmin_file}")
    
    return tmax_file, tmin_file

def main():
    parser = argparse.ArgumentParser(description='Calculate percentiles for heatwave detection using ERA5 data')
    parser.add_argument('--start-year', type=int, default=1980, 
                       help='Start year for baseline period (default: 1980)')
    parser.add_argument('--end-year', type=int, default=2000,
                       help='End year for baseline period (default: 2000)')
    parser.add_argument('--data-dir', default='/data/climate/disk3/datasets/era5',
                       help='Directory containing ERA5 daily files')
    parser.add_argument('--output-dir', default='data/processed/percentiles',
                       help='Output directory for percentile files')
    parser.add_argument('--percentile', type=float, default=95.0,
                       help='Percentile to calculate (default: 95)')
    parser.add_argument('--n-processes', type=int, default=96,
                       help='Number of processes to use (default: 96)')
    parser.add_argument('--chunk-size-lat', type=int, default=50,
                       help='Latitude chunk size (default: 50)')
    parser.add_argument('--chunk-size-lon', type=int, default=100,
                       help='Longitude chunk size (default: 100)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("ERA5 PERCENTILE CALCULATION")
    print("="*80)
    print(f"Baseline period: {args.start_year}-{args.end_year}")
    print(f"Percentile: {args.percentile}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Processes: {args.n_processes}")
    print(f"Chunk size: {args.chunk_size_lat} x {args.chunk_size_lon}")
    print("="*80)
    
    # Validate data directory
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")
    
    years = list(range(args.start_year, args.end_year + 1))
    
    # Get grid dimensions from first available file
    sample_file = None
    for year in years:
        for month in range(1, 13):
            test_file = data_dir / f"era5_daily_{year}_{month:02d}.nc"
            if test_file.exists():
                sample_file = test_file
                break
        if sample_file:
            break
    
    if sample_file is None:
        raise ValueError("No ERA5 files found in the specified directory and year range")
    
    print(f"Reading grid dimensions from: {sample_file}")
    sample_ds = xr.open_dataset(sample_file)
    full_lat = sample_ds.latitude.values
    full_lon = sample_ds.longitude.values
    sample_ds.close()
    
    print(f"Grid dimensions: {len(full_lat)} x {len(full_lon)}")
    
    # Create spatial chunks
    spatial_chunks = create_spatial_chunks(
        len(full_lat), len(full_lon),
        args.chunk_size_lat, args.chunk_size_lon
    )
    
    print(f"Created {len(spatial_chunks)} spatial chunks")
    
    # Estimate memory usage
    estimated_memory_per_process = 3.0  # GB per process (conservative estimate)
    total_estimated_memory = estimated_memory_per_process * args.n_processes
    
    print(f"Estimated memory usage: {total_estimated_memory:.1f} GB")
    if total_estimated_memory > 600:  # Conservative limit for 755GB system
        print("WARNING: High memory usage estimated. Consider reducing --n-processes if needed.")
    
    # Process chunks in parallel
    print(f"\nStarting parallel processing with {args.n_processes} processes...")
    
    process_func = partial(
        process_spatial_chunk,
        years=years,
        data_dir=args.data_dir,
        percentile=args.percentile
    )
    
    with mp.Pool(args.n_processes) as pool:
        chunk_results = pool.map(process_func, spatial_chunks)
    
    print("\nCombining chunks...")
    combine_chunks(chunk_results, full_lat, full_lon, args.output_dir, 
                  int(args.percentile), years)
    
    print("\n" + "="*80)
    print("PERCENTILE CALCULATION COMPLETED SUCCESSFULLY!")
    print("="*80)

if __name__ == "__main__":
    main()
