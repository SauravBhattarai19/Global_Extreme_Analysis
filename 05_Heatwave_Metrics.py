#!/usr/bin/env python3
"""
OPTIMIZED: Calculate comprehensive heatwave metrics from ERA5 daily temperature data.
Major optimization: Load data once per chunk, not per pixel.

This version loads data at the chunk level and processes all pixels in memory,
dramatically reducing I/O operations and improving CPU utilization.
"""

import sys
import argparse
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
import multiprocessing as mp
from functools import partial
import gc
import warnings
from datetime import datetime, timedelta
import uuid
from collections import defaultdict
warnings.filterwarnings('ignore')

def load_daily_data_chunk_with_padding(year, data_dir, var_name, chunk_bounds, min_duration=3):
    """
    Load daily data for a spatial chunk with temporal padding.
    MAJOR OPTIMIZATION: Load once per chunk, not per pixel.
    
    Returns:
    - data: Combined data array with padding [time, lat_chunk, lon_chunk]
    - dates: Corresponding dates
    - year_slice: (start_idx, end_idx) for the target year
    """
    lat_start, lat_end, lon_start, lon_end = chunk_bounds
    data_dir = Path(data_dir)
    pad = min_duration - 1
    
    def load_year_chunk(year_to_load):
        """Load all months for a specific year and spatial chunk."""
        year_data = []
        year_dates = []
        
        for month in range(1, 13):
            file_path = data_dir / f"era5_daily_{year_to_load}_{month:02d}.nc"
            if file_path.exists():
                try:
                    ds = xr.open_dataset(file_path)
                    # Subset to spatial chunk immediately
                    ds_chunk = ds.isel(
                        latitude=slice(lat_start, lat_end),
                        longitude=slice(lon_start, lon_end)
                    )
                    
                    # Calculate daily extremes
                    if var_name == 'tmax':
                        daily_data = ds_chunk.t2m.resample(valid_time='1D').max()
                    elif var_name == 'tmin':
                        daily_data = ds_chunk.t2m.resample(valid_time='1D').min()
                    else:
                        daily_data = ds_chunk.t2m.resample(valid_time='1D').mean()
                    
                    year_data.append(daily_data.values)
                    year_dates.extend(daily_data.valid_time.values)
                    
                    ds.close()
                    daily_data.close()
                    del ds, ds_chunk, daily_data
                    
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    continue
        
        if year_data:
            combined_data = np.concatenate(year_data, axis=0)
            combined_dates = pd.to_datetime(year_dates)
            return combined_data, combined_dates
        else:
            return None, None
    
    # Load target year
    year_data, year_dates = load_year_chunk(year)
    if year_data is None:
        return None, None, None
    
    # Load padding from previous year (last few days)
    prev_data, prev_dates = None, None
    if pad > 0:
        prev_year_data, prev_year_dates = load_year_chunk(year - 1)
        if prev_year_data is not None and len(prev_year_data) >= pad:
            prev_data = prev_year_data[-pad:]
            prev_dates = prev_year_dates[-pad:]
    
    # Load padding from next year (first few days)
    next_data, next_dates = None, None
    if pad > 0:
        next_year_data, next_year_dates = load_year_chunk(year + 1)
        if next_year_data is not None and len(next_year_data) >= pad:
            next_data = next_year_data[:pad]
            next_dates = next_year_dates[:pad]
    
    # Combine all data
    all_data = []
    all_dates = []
    
    if prev_data is not None:
        all_data.append(prev_data)
        all_dates.extend(prev_dates)
    
    all_data.append(year_data)
    all_dates.extend(year_dates)
    
    if next_data is not None:
        all_data.append(next_data)
        all_dates.extend(next_dates)
    
    combined_data = np.concatenate(all_data, axis=0)
    combined_dates = pd.to_datetime(all_dates)
    
    # Calculate year slice indices
    year_start_idx = len(prev_data) if prev_data is not None else 0
    year_end_idx = year_start_idx + len(year_data) - 1
    
    return combined_data, combined_dates, (year_start_idx, year_end_idx)

def map_thresholds_to_chunk(dates, pct_data, chunk_bounds):
    """
    Map thresholds to all pixels in chunk at once.
    Returns: thresholds array [time, lat_chunk, lon_chunk]
    """
    lat_start, lat_end, lon_start, lon_end = chunk_bounds
    n_time = len(dates)
    n_lat = lat_end - lat_start
    n_lon = lon_end - lon_start
    
    thresholds = np.full((n_time, n_lat, n_lon), np.nan)
    
    for t, date in enumerate(dates):
        doy = date.dayofyear
        # Handle leap year
        if doy == 366 and pct_data.shape[0] == 365:
            doy = 365
        
        thresholds[t, :, :] = pct_data[doy-1, :, :]
    
    return thresholds

def find_runs_vectorized(bool_arr):
    """Vectorized version of find_runs for 3D arrays."""
    if bool_arr.ndim != 3:
        raise ValueError("Expected 3D array")
    
    n_time, n_lat, n_lon = bool_arr.shape
    
    # Pad with False at beginning and end
    padded = np.concatenate([
        np.zeros((1, n_lat, n_lon), dtype=bool),
        bool_arr,
        np.zeros((1, n_lat, n_lon), dtype=bool)
    ], axis=0)
    
    # Find transitions
    diff = np.diff(padded.astype(int), axis=0)
    
    # Find start and end indices for each pixel
    starts_dict = {}
    ends_dict = {}
    
    for i in range(n_lat):
        for j in range(n_lon):
            pixel_diff = diff[:, i, j]
            starts = np.where(pixel_diff == 1)[0]
            ends = np.where(pixel_diff == -1)[0] - 1
            
            if len(starts) > 0 and len(ends) > 0:
                starts_dict[(i, j)] = starts
                ends_dict[(i, j)] = ends
    
    return starts_dict, ends_dict

def process_spatial_chunk_optimized(chunk_info, year, data_dir, pct_file, var_name, min_duration=3):
    """
    OPTIMIZED: Process entire spatial chunk at once.
    Load data once, process all pixels in memory.
    """
    lat_start, lat_end, lon_start, lon_end = chunk_info
    
    print(f"Processing {var_name} chunk: year={year}, lat[{lat_start}:{lat_end}], lon[{lon_start}:{lon_end}]")
    
    # Load percentile data for this chunk
    pct_ds = xr.open_dataset(pct_file)
    if var_name == 'tmax':
        pct_data = pct_ds['tmax_p95'].values[:, lat_start:lat_end, lon_start:lon_end]
    else:  # tmin
        pct_data = pct_ds['tmin_p95'].values[:, lat_start:lat_end, lon_start:lon_end]
    pct_ds.close()
    
    # OPTIMIZATION: Load data once for entire chunk
    temps, dates, year_slice = load_daily_data_chunk_with_padding(
        year, data_dir, var_name, chunk_info, min_duration
    )
    
    if temps is None:
        print(f"No data for chunk lat[{lat_start}:{lat_end}], lon[{lon_start}:{lon_end}]")
        return None
    
    # Map thresholds for entire chunk
    thresholds = map_thresholds_to_chunk(dates, pct_data, chunk_info)
    
    # Find exceedances for entire chunk
    exceed = temps > thresholds
    
    # Find runs for all pixels
    starts_dict, ends_dict = find_runs_vectorized(exceed)
    
    # Process results
    chunk_metrics = []
    chunk_events = []
    chunk_days = []
    
    year_start_idx, year_end_idx = year_slice
    n_lat, n_lon = temps.shape[1], temps.shape[2]
    
    # Process each pixel
    for i in range(n_lat):
        for j in range(n_lon):
            global_i = lat_start + i
            global_j = lon_start + j
            pixel_key = (i, j)
            
            if pixel_key not in starts_dict:
                # No heatwaves for this pixel
                pixel_metrics = {
                    'grid_y': global_i, 'grid_x': global_j,
                    'hwn': 0, 'hwtd': 0, 'hwld': 0,
                    'hwmt': np.nan, 'hdt': np.nan
                }
                chunk_metrics.append(pixel_metrics)
                continue
            
            starts = starts_dict[pixel_key]
            ends = ends_dict[pixel_key]
            
            # Filter by minimum duration
            durations_global = ends - starts + 1
            keep = durations_global >= min_duration
            starts = starts[keep]
            ends = ends[keep]
            durations_global = durations_global[keep]
            
            if len(starts) == 0:
                # No heatwaves meeting duration criteria
                pixel_metrics = {
                    'grid_y': global_i, 'grid_x': global_j,
                    'hwn': 0, 'hwtd': 0, 'hwld': 0,
                    'hwmt': np.nan, 'hdt': np.nan
                }
                chunk_metrics.append(pixel_metrics)
                continue
            
            # Process heatwaves for this pixel
            pixel_temps = temps[:, i, j]
            pixel_thresholds = thresholds[:, i, j]
            
            hwn = 0
            hwtd = 0
            hwld = 0
            all_clipped_temps = []
            
            for k in range(len(starts)):
                g_s = starts[k]
                g_e = ends[k]
                g_len = durations_global[k]
                g_start_date = dates[g_s]
                g_end_date = dates[g_e]
                
                # Clip to target year
                y_s = max(g_s, year_start_idx)
                y_e = min(g_e, year_end_idx)
                
                if y_e < y_s:
                    continue  # No overlap with target year
                
                y_len = y_e - y_s + 1
                
                # Extract clipped data
                temps_clip = pixel_temps[y_s:y_e+1]
                thresh_clip = pixel_thresholds[y_s:y_e+1]
                exceed_clip = temps_clip - thresh_clip
                dates_clip = dates[y_s:y_e+1]
                
                # Update metrics
                hwn += 1
                hwtd += y_len
                hwld = max(hwld, y_len)
                all_clipped_temps.extend(temps_clip)
                
                # Create event record
                event_id = f"{var_name}:{global_i}:{global_j}:{g_start_date.strftime('%Y%m%d')}"
                
                event_record = {
                    'event_id': event_id,
                    'var': var_name,
                    'grid_y': global_i,
                    'grid_x': global_j,
                    'year': year,
                    'global_start': g_start_date,
                    'global_end': g_end_date,
                    'year_start': dates_clip[0],
                    'year_end': dates_clip[-1],
                    'duration_days': y_len,
                    'duration_days_global': g_len,
                    'mean_temp_clipped': np.mean(temps_clip),
                    'max_temp_clipped': np.max(temps_clip),
                    'mean_thresh_clipped': np.mean(thresh_clip),
                    'mean_exceed_clipped': np.mean(exceed_clip),
                    'sum_exceed_clipped': np.sum(exceed_clip)
                }
                chunk_events.append(event_record)
                
                # Create day records
                for idx in range(len(dates_clip)):
                    day_record = {
                        'event_id': event_id,
                        'var': var_name,
                        'grid_y': global_i,
                        'grid_x': global_j,
                        'date': dates_clip[idx],
                        'temp': temps_clip[idx],
                        'thresh': thresh_clip[idx],
                        'exceed': exceed_clip[idx],
                        'year': dates_clip[idx].year,
                        'month': dates_clip[idx].month
                    }
                    chunk_days.append(day_record)
            
            # Finalize pixel metrics
            if len(all_clipped_temps) > 0:
                pixel_metrics = {
                    'grid_y': global_i,
                    'grid_x': global_j,
                    'hwn': hwn,
                    'hwtd': hwtd,
                    'hwld': hwld,
                    'hwmt': np.mean(all_clipped_temps),
                    'hdt': np.max(all_clipped_temps)
                }
            else:
                pixel_metrics = {
                    'grid_y': global_i,
                    'grid_x': global_j,
                    'hwn': 0,
                    'hwtd': 0,
                    'hwld': 0,
                    'hwmt': np.nan,
                    'hdt': np.nan
                }
            
            chunk_metrics.append(pixel_metrics)
    
    # Clean up
    del temps, thresholds, exceed, pct_data
    gc.collect()
    
    print(f"Completed {var_name} chunk: year={year}, lat[{lat_start}:{lat_end}], lon[{lon_start}:{lon_end}]")
    
    return {
        'chunk_bounds': chunk_info,
        'metrics': chunk_metrics,
        'events': chunk_events,
        'days': chunk_days
    }

def create_spatial_chunks(n_lat, n_lon, chunk_size_lat=40, chunk_size_lon=80):
    """Create larger spatial chunks for better efficiency."""
    chunks = []
    for lat_start in range(0, n_lat, chunk_size_lat):
        lat_end = min(lat_start + chunk_size_lat, n_lat)
        for lon_start in range(0, n_lon, chunk_size_lon):
            lon_end = min(lon_start + chunk_size_lon, n_lon)
            chunks.append((lat_start, lat_end, lon_start, lon_end))
    return chunks

def combine_results(chunk_results, full_lat, full_lon, year, var_name, output_dir):
    """Combine chunk results into final outputs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize metric grids
    n_lat, n_lon = len(full_lat), len(full_lon)
    hwn_grid = np.zeros((n_lat, n_lon), dtype=np.float64)
    hwmt_grid = np.full((n_lat, n_lon), np.nan, dtype=np.float64)
    hdt_grid = np.full((n_lat, n_lon), np.nan, dtype=np.float64)
    hwtd_grid = np.zeros((n_lat, n_lon), dtype=np.float64)  # Ensure float, not timedelta
    hwld_grid = np.zeros((n_lat, n_lon), dtype=np.float64)  # Ensure float, not timedelta
    
    all_events = []
    all_days = []
    
    # Combine results from all chunks
    for result in chunk_results:
        if result is None:
            continue
        
        # Combine metrics
        for metric in result['metrics']:
            i, j = metric['grid_y'], metric['grid_x']
            hwn_grid[i, j] = metric['hwn']
            hwmt_grid[i, j] = metric['hwmt']
            hdt_grid[i, j] = metric['hdt']
            hwtd_grid[i, j] = metric['hwtd']
            hwld_grid[i, j] = metric['hwld']
        
        # Combine events and days
        all_events.extend(result['events'])
        all_days.extend(result['days'])
    
    # Save gridded metrics
    metrics_ds = xr.Dataset({
        f'hwn_{var_name}': (['latitude', 'longitude'], hwn_grid),
        f'hwmt_{var_name}': (['latitude', 'longitude'], hwmt_grid),
        f'hdt_{var_name}': (['latitude', 'longitude'], hdt_grid),
        f'hwtd_{var_name}': (['latitude', 'longitude'], hwtd_grid),
        f'hwld_{var_name}': (['latitude', 'longitude'], hwld_grid)
    }, coords={
        'latitude': full_lat,
        'longitude': full_lon
    })
    
    # Add attributes
    metrics_ds[f'hwn_{var_name}'].attrs = {'long_name': f'Number of {var_name} heatwaves', 'units': 'count'}
    metrics_ds[f'hwmt_{var_name}'].attrs = {'long_name': f'Mean temperature during {var_name} heatwaves', 'units': 'K'}
    metrics_ds[f'hdt_{var_name}'].attrs = {'long_name': f'Maximum temperature during {var_name} heatwaves', 'units': 'K'}
    metrics_ds[f'hwtd_{var_name}'].attrs = {'long_name': f'Total duration of {var_name} heatwaves', 'units': 'days'}
    metrics_ds[f'hwld_{var_name}'].attrs = {'long_name': f'Longest {var_name} heatwave duration', 'units': 'days'}
    
    metrics_file = output_dir / f'heatwave_metrics_{var_name}_{year}.nc'
    metrics_ds.to_netcdf(metrics_file)
    metrics_ds.close()
    
    # Save events and days tables
    if all_events:
        events_df = pd.DataFrame(all_events)
        events_file = output_dir / f'heatwave_events_{var_name}_{year}.parquet'
        events_df.to_parquet(events_file)
    
    if all_days:
        days_df = pd.DataFrame(all_days)
        days_file = output_dir / f'heatwave_days_{var_name}_{year}.parquet'
        days_df.to_parquet(days_file)
    
    print(f"Saved {var_name} results for {year}:")
    print(f"  - Metrics: {metrics_file}")
    if all_events:
        print(f"  - Events: {events_file} ({len(all_events)} events)")
    if all_days:
        print(f"  - Days: {days_file} ({len(all_days)} days)")
    
    return metrics_file

def main():
    parser = argparse.ArgumentParser(description='OPTIMIZED: Calculate heatwave metrics from ERA5 data')
    parser.add_argument('--start-year', type=int, default=2001,
                       help='Start year for analysis (default: 2001)')
    parser.add_argument('--end-year', type=int, default=2025,
                       help='End year for analysis (default: 2025)')
    parser.add_argument('--data-dir', default='/data/climate/disk3/datasets/era5',
                       help='Directory containing ERA5 daily files')
    parser.add_argument('--percentile-dir', default='data/processed/percentiles',
                       help='Directory containing percentile files')
    parser.add_argument('--output-dir', default='data/processed/heatwave_metrics',
                       help='Output directory for heatwave metrics')
    parser.add_argument('--min-duration', type=int, default=3,
                       help='Minimum heatwave duration in days (default: 3)')
    parser.add_argument('--n-processes', type=int, default=96,
                       help='Number of processes to use (default: 96)')
    parser.add_argument('--chunk-size-lat', type=int, default=40,
                       help='Latitude chunk size (default: 40)')
    parser.add_argument('--chunk-size-lon', type=int, default=80,
                       help='Longitude chunk size (default: 80)')
    parser.add_argument('--variables', nargs='+', default=['tmax', 'tmin'],
                       help='Variables to process (default: tmax tmin)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("OPTIMIZED ERA5 HEATWAVE METRICS CALCULATION")
    print("="*80)
    print(f"Analysis period: {args.start_year}-{args.end_year}")
    print(f"Variables: {args.variables}")
    print(f"Minimum duration: {args.min_duration} days")
    print(f"Data directory: {args.data_dir}")
    print(f"Percentile directory: {args.percentile_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Processes: {args.n_processes}")
    print(f"Chunk size: {args.chunk_size_lat} x {args.chunk_size_lon}")
    print("="*80)
    
    # Validate directories and files
    data_dir = Path(args.data_dir)
    pct_dir = Path(args.percentile_dir)
    
    if not data_dir.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")
    
    if not pct_dir.exists():
        raise ValueError(f"Percentile directory does not exist: {pct_dir}")
    
    # Find percentile files
    tmax_pct_file = None
    tmin_pct_file = None
    
    for file in pct_dir.glob('*.nc'):
        if 'tmax_p95' in file.name:
            tmax_pct_file = file
        elif 'tmin_p95' in file.name:
            tmin_pct_file = file
    
    if 'tmax' in args.variables and tmax_pct_file is None:
        raise ValueError("No tmax percentile file found")
    
    if 'tmin' in args.variables and tmin_pct_file is None:
        raise ValueError("No tmin percentile file found")
    
    # Get grid dimensions
    sample_pct_file = tmax_pct_file if tmax_pct_file else tmin_pct_file
    sample_ds = xr.open_dataset(sample_pct_file)
    full_lat = sample_ds.latitude.values
    full_lon = sample_ds.longitude.values
    sample_ds.close()
    
    print(f"Grid dimensions: {len(full_lat)} x {len(full_lon)}")
    
    # Create spatial chunks (larger for better efficiency)
    spatial_chunks = create_spatial_chunks(
        len(full_lat), len(full_lon),
        args.chunk_size_lat, args.chunk_size_lon
    )
    
    print(f"Created {len(spatial_chunks)} spatial chunks")
    
    # Estimate memory per chunk
    chunk_pixels = args.chunk_size_lat * args.chunk_size_lon
    days_per_year = 365
    memory_per_chunk_gb = chunk_pixels * days_per_year * 4 * 3 / (1024**3)  # 3 arrays (temp, thresh, exceed)
    total_memory_gb = memory_per_chunk_gb * args.n_processes
    
    print(f"Estimated memory per chunk: {memory_per_chunk_gb:.2f} GB")
    print(f"Total estimated memory: {total_memory_gb:.1f} GB")
    
    if total_memory_gb > 600:
        print("WARNING: High memory usage. Consider reducing chunk size or processes.")
    
    # Process each year and variable
    years = list(range(args.start_year, args.end_year + 1))
    
    for year in years:
        print(f"\n{'='*20} PROCESSING YEAR {year} {'='*20}")
        
        for var_name in args.variables:
            print(f"\nProcessing variable: {var_name}")
            
            # Select appropriate percentile file
            pct_file = tmax_pct_file if var_name == 'tmax' else tmin_pct_file
            
            # Process chunks in parallel
            process_func = partial(
                process_spatial_chunk_optimized,
                year=year,
                data_dir=args.data_dir,
                pct_file=pct_file,
                var_name=var_name,
                min_duration=args.min_duration
            )
            
            print(f"Processing {len(spatial_chunks)} chunks with {args.n_processes} processes...")
            
            with mp.Pool(args.n_processes) as pool:
                chunk_results = pool.map(process_func, spatial_chunks)
            
            # Combine results
            combine_results(chunk_results, full_lat, full_lon, year, var_name, args.output_dir)
    
    print("\n" + "="*80)
    print("OPTIMIZED HEATWAVE METRICS CALCULATION COMPLETED!")
    print("="*80)

if __name__ == "__main__":
    main()
