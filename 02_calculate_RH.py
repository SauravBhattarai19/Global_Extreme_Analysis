#!/usr/bin/env python3
"""
Calculate relative humidity from ERA5 temperature and dewpoint data.
Optimized for high-performance computing with 192 cores and 755GB RAM.

Formula:
RH = 100 * e(Td) / es(T)

Where:
- es(T) = 6.112 * exp(17.62 * T_C / (243.12 + T_C))  [saturation vapor pressure]
- e(Td) = 6.112 * exp(17.62 * Td_C / (243.12 + Td_C))  [actual vapor pressure]
- T_C, Td_C are temperatures in Celsius
"""

import sys
import argparse
import numpy as np
import xarray as xr
from pathlib import Path
import multiprocessing as mp
from functools import partial
import gc
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

def kelvin_to_celsius(temp_k):
    """Convert temperature from Kelvin to Celsius."""
    return temp_k - 273.15

def calculate_vapor_pressure(temp_c):
    """
    Calculate saturation vapor pressure using Magnus/Tetens formula.
    
    Parameters:
    - temp_c: Temperature in Celsius
    
    Returns:
    - Vapor pressure in hPa
    """
    return 6.112 * np.exp(17.62 * temp_c / (243.12 + temp_c))

def calculate_relative_humidity(t2m, d2m):
    """
    Calculate relative humidity from 2m temperature and dewpoint.
    
    Parameters:
    - t2m: 2-meter temperature in Kelvin
    - d2m: 2-meter dewpoint temperature in Kelvin
    
    Returns:
    - Relative humidity in percent (0-100)
    """
    # Convert to Celsius
    t_c = kelvin_to_celsius(t2m)
    td_c = kelvin_to_celsius(d2m)
    
    # Calculate vapor pressures
    es = calculate_vapor_pressure(t_c)  # Saturation vapor pressure
    e = calculate_vapor_pressure(td_c)  # Actual vapor pressure
    
    # Calculate relative humidity
    rh = 100.0 * e / es
    
    # Clip to valid range [0, 100]
    rh = np.clip(rh, 0.0, 100.0)
    
    return rh

def process_monthly_file(file_info, output_dir):
    """Process a single monthly file to calculate relative humidity."""
    file_path, year, month = file_info
    
    print(f"Processing: {file_path.name}")
    
    try:
        # Load data with chunking for memory efficiency
        ds = xr.open_dataset(file_path, chunks={'valid_time': 50})
        
        # Calculate relative humidity
        rh = calculate_relative_humidity(ds.t2m, ds.d2m)
        
        # Create output dataset
        rh_ds = xr.Dataset({
            'rh': (['valid_time', 'latitude', 'longitude'], rh.values)
        }, coords={
            'valid_time': ds.valid_time,
            'latitude': ds.latitude,
            'longitude': ds.longitude
        })
        
        # Add attributes
        rh_ds.rh.attrs = {
            'long_name': 'Relative humidity',
            'units': 'percent',
            'description': 'Calculated from 2m temperature and dewpoint using Magnus/Tetens formula',
            'formula': 'RH = 100 * e(Td) / es(T)',
            'valid_range': [0.0, 100.0],
            'calculation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Add global attributes
        rh_ds.attrs = {
            'title': 'ERA5 Relative Humidity',
            'source': 'Calculated from ERA5 2m temperature and dewpoint',
            'institution': 'Calculated using ERA5 data',
            'history': f'Created on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
            'references': 'Magnus/Tetens formula for vapor pressure'
        }
        
        # Save output file
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"era5_rh_{year}_{month:02d}.nc"
        
        # Use compression for efficient storage
        encoding = {
            'rh': {
                'zlib': True,
                'complevel': 4,
                'dtype': 'float32'
            }
        }
        
        rh_ds.to_netcdf(output_file, encoding=encoding)
        
        # Clean up
        ds.close()
        rh_ds.close()
        del ds, rh_ds, rh
        gc.collect()
        
        print(f"Completed: {output_file.name}")
        return output_file
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def process_year_batch(year_batch, data_dir, output_dir):
    """Process a batch of years sequentially to control memory usage."""
    results = []
    
    for year in year_batch:
        print(f"\nProcessing year: {year}")
        
        for month in range(1, 13):
            file_path = Path(data_dir) / f"era5_daily_{year}_{month:02d}.nc"
            
            if not file_path.exists():
                print(f"Warning: File not found: {file_path}")
                continue
            
            result = process_monthly_file((file_path, year, month), output_dir)
            if result:
                results.append(result)
    
    return results

def get_file_list(data_dir, start_year, end_year):
    """Get list of available ERA5 files in the specified year range."""
    files = []
    data_dir = Path(data_dir)
    
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            file_path = data_dir / f"era5_daily_{year}_{month:02d}.nc"
            if file_path.exists():
                files.append((file_path, year, month))
            else:
                print(f"Warning: File not found: {file_path}")
    
    return files

def main():
    parser = argparse.ArgumentParser(description='Calculate relative humidity from ERA5 temperature and dewpoint data')
    parser.add_argument('--start-year', type=int, default=1980,
                       help='Start year (default: 1980)')
    parser.add_argument('--end-year', type=int, default=2025,
                       help='End year (default: 2025)')
    parser.add_argument('--data-dir', default='/data/climate/disk3/datasets/era5',
                       help='Directory containing ERA5 daily files')
    parser.add_argument('--output-dir', default='data/processed/relative_humidity',
                       help='Output directory for RH files')
    parser.add_argument('--n-processes', type=int, default=48,
                       help='Number of processes to use (default: 48)')
    parser.add_argument('--years-per-batch', type=int, default=2,
                       help='Number of years to process per batch (default: 2)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("ERA5 RELATIVE HUMIDITY CALCULATION")
    print("="*80)
    print(f"Period: {args.start_year}-{args.end_year}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Processes: {args.n_processes}")
    print(f"Years per batch: {args.years_per_batch}")
    print("="*80)
    
    # Validate data directory
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")
    
    # Get list of available files
    file_list = get_file_list(args.data_dir, args.start_year, args.end_year)
    
    if not file_list:
        raise ValueError("No ERA5 files found in the specified directory and year range")
    
    print(f"Found {len(file_list)} files to process")
    
    # Create year batches for parallel processing
    years = list(range(args.start_year, args.end_year + 1))
    year_batches = []
    
    for i in range(0, len(years), args.years_per_batch):
        batch = years[i:i + args.years_per_batch]
        year_batches.append(batch)
    
    print(f"Created {len(year_batches)} year batches")
    
    # Estimate memory usage
    # Each file ~1.3GB, processing 2 years * 12 months * 1.3GB * 2 (input+output) ≈ 62GB per process
    estimated_memory_per_process = args.years_per_batch * 12 * 1.3 * 2  # GB
    total_estimated_memory = estimated_memory_per_process * args.n_processes
    
    print(f"Estimated memory usage: {total_estimated_memory:.1f} GB")
    if total_estimated_memory > 600:  # Conservative limit for 755GB system
        print("WARNING: High memory usage estimated.")
        print("Consider reducing --n-processes or --years-per-batch if needed.")
    
    # Process year batches in parallel
    print(f"\nStarting parallel processing with {args.n_processes} processes...")
    
    process_func = partial(
        process_year_batch,
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    
    with mp.Pool(args.n_processes) as pool:
        batch_results = pool.map(process_func, year_batches)
    
    # Flatten results
    all_results = []
    for batch_result in batch_results:
        if batch_result:
            all_results.extend(batch_result)
    
    print(f"\nProcessed {len(all_results)} files successfully")
    
    # Summary
    print("\n" + "="*80)
    print("RELATIVE HUMIDITY CALCULATION COMPLETED!")
    print("="*80)
    print(f"Output directory: {args.output_dir}")
    print(f"Files created: {len(all_results)}")
    
    if all_results:
        print("\nSample output files:")
        for i, result_file in enumerate(all_results[:5]):
            print(f"  - {result_file.name}")
        if len(all_results) > 5:
            print(f"  ... and {len(all_results) - 5} more files")
    
    print("\nRelative humidity files are ready for analysis!")

if __name__ == "__main__":
    main()
