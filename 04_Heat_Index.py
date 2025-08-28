#!/usr/bin/env python3
"""
Calculate Heat Index from ERA5 temperature and relative humidity data.
Optimized for high-performance computing with 192 cores and 755GB RAM.

Heat Index Formula (Rothfusz 1990):
- Uses Fahrenheit internally for calculation
- Applies adjustments for low humidity and high humidity conditions
- Returns values in specified unit (Celsius or Fahrenheit)
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

def compute_heat_index(T, RH, unit="C"):
    """
    Compute heat index from temperature and relative humidity.
    
    Parameters:
    - T: Temperature (Celsius if unit="C", Fahrenheit if unit="F")
    - RH: Relative humidity (percent, 0-100)
    - unit: "C" for Celsius, "F" for Fahrenheit
    
    Returns:
    - Heat index in the same unit as input temperature
    """
    # Convert to Fahrenheit if needed
    if unit == "C":
        T_F = (9/5) * T + 32
    else:
        T_F = T.copy()
    
    # For temperatures below 80°F or humidity below 40%, heat index = temperature
    low_condition = (T_F < 80) | (RH < 40)
    
    # Base heat index calculation (Rothfusz equation)
    HI = (-42.379 + 2.04901523 * T_F + 10.14333127 * RH
          - 0.22475541 * T_F * RH - 6.83783e-3 * T_F**2
          - 5.481717e-2 * RH**2 + 1.22874e-3 * T_F**2 * RH
          + 8.5282e-4 * T_F * RH**2 - 1.99e-6 * T_F**2 * RH**2)
    
    # Adjustment for low humidity (RH < 13% and 80°F <= T <= 112°F)
    low_humidity_mask = (RH < 13) & (T_F >= 80) & (T_F <= 112)
    if np.any(low_humidity_mask):
        adj = ((13 - RH) / 4) * np.sqrt((17 - np.abs(T_F - 95)) / 17)
        HI = np.where(low_humidity_mask, HI - adj, HI)
    
    # Adjustment for high humidity (RH > 85% and 80°F <= T <= 87°F)
    high_humidity_mask = (RH > 85) & (T_F >= 80) & (T_F <= 87)
    if np.any(high_humidity_mask):
        adj = ((RH - 85) / 10) * ((87 - T_F) / 5)
        HI = np.where(high_humidity_mask, HI + adj, HI)
    
    # For low temperature/humidity conditions, use original temperature
    HI = np.where(low_condition, T_F, HI)
    
    # Convert back to Celsius if needed
    if unit == "C":
        HI = (5/9) * (HI - 32)
    
    return HI

def process_monthly_file_pair(file_info, temp_dir, rh_dir, output_dir):
    """Process a pair of temperature and RH files to calculate heat index."""
    year, month = file_info
    
    temp_file = Path(temp_dir) / f"era5_daily_{year}_{month:02d}.nc"
    rh_file = Path(rh_dir) / f"era5_rh_{year}_{month:02d}.nc"
    
    print(f"Processing: {year}-{month:02d}")
    
    # Check if both files exist
    if not temp_file.exists():
        print(f"Warning: Temperature file not found: {temp_file}")
        return None
    
    if not rh_file.exists():
        print(f"Warning: RH file not found: {rh_file}")
        return None
    
    try:
        # Load data with chunking for memory efficiency
        ds_temp = xr.open_dataset(temp_file, chunks={'valid_time': 50})
        ds_rh = xr.open_dataset(rh_file, chunks={'valid_time': 50})
        
        # Ensure time coordinates match
        if not np.array_equal(ds_temp.valid_time.values, ds_rh.valid_time.values):
            print(f"Warning: Time coordinates don't match for {year}-{month:02d}")
            # Interpolate RH to match temperature time coordinates
            ds_rh = ds_rh.interp(valid_time=ds_temp.valid_time, method='linear')
        
        # Convert temperature from Kelvin to Celsius
        temp_celsius = ds_temp.t2m - 273.15
        
        # Calculate heat index
        heat_index = compute_heat_index(temp_celsius, ds_rh.rh, unit="C")
        
        # Create output dataset
        hi_ds = xr.Dataset({
            'heat_index': (['valid_time', 'latitude', 'longitude'], heat_index)
        }, coords={
            'valid_time': ds_temp.valid_time,
            'latitude': ds_temp.latitude,
            'longitude': ds_temp.longitude
        })
        
        # Add attributes
        hi_ds.heat_index.attrs = {
            'long_name': 'Heat Index',
            'units': 'degrees_Celsius',
            'description': 'Heat index calculated from 2m temperature and relative humidity using Rothfusz equation',
            'formula': 'Rothfusz (1990) heat index with humidity adjustments',
            'references': 'Rothfusz, L.P., 1990: The heat index equation. NWS Technical Attachment SR 90-23',
            'calculation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Add global attributes
        hi_ds.attrs = {
            'title': 'ERA5 Heat Index',
            'source': 'Calculated from ERA5 2m temperature and relative humidity',
            'institution': 'Calculated using ERA5 data',
            'history': f'Created on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
            'references': 'Rothfusz (1990) heat index equation with adjustments'
        }
        
        # Save output file
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"era5_heat_index_{year}_{month:02d}.nc"
        
        # Use compression for efficient storage
        encoding = {
            'heat_index': {
                'zlib': True,
                'complevel': 4,
                'dtype': 'float32'
            }
        }
        
        hi_ds.to_netcdf(output_file, encoding=encoding)
        
        # Clean up
        ds_temp.close()
        ds_rh.close()
        hi_ds.close()
        del ds_temp, ds_rh, hi_ds, heat_index, temp_celsius
        gc.collect()
        
        print(f"Completed: {output_file.name}")
        return output_file
        
    except Exception as e:
        print(f"Error processing {year}-{month:02d}: {e}")
        return None

def process_year_batch(year_batch, temp_dir, rh_dir, output_dir):
    """Process a batch of years sequentially to control memory usage."""
    results = []
    
    for year in year_batch:
        print(f"\nProcessing year: {year}")
        
        for month in range(1, 13):
            result = process_monthly_file_pair((year, month), temp_dir, rh_dir, output_dir)
            if result:
                results.append(result)
    
    return results

def get_available_years(temp_dir, rh_dir, start_year, end_year):
    """Get list of years that have both temperature and RH data available."""
    available_years = []
    temp_dir = Path(temp_dir)
    rh_dir = Path(rh_dir)
    
    for year in range(start_year, end_year + 1):
        # Check if at least one month has both files
        has_data = False
        for month in range(1, 13):
            temp_file = temp_dir / f"era5_daily_{year}_{month:02d}.nc"
            rh_file = rh_dir / f"era5_rh_{year}_{month:02d}.nc"
            
            if temp_file.exists() and rh_file.exists():
                has_data = True
                break
        
        if has_data:
            available_years.append(year)
        else:
            print(f"Warning: No complete data found for year {year}")
    
    return available_years

def main():
    parser = argparse.ArgumentParser(description='Calculate Heat Index from ERA5 temperature and relative humidity data')
    parser.add_argument('--start-year', type=int, default=1980,
                       help='Start year (default: 1980)')
    parser.add_argument('--end-year', type=int, default=2024,
                       help='End year (default: 2025)')
    parser.add_argument('--temp-dir', default='/data/climate/disk3/datasets/era5',
                       help='Directory containing ERA5 temperature files')
    parser.add_argument('--rh-dir', default='data/processed/relative_humidity',
                       help='Directory containing relative humidity files')
    parser.add_argument('--output-dir', default='data/processed/heat_index',
                       help='Output directory for heat index files')
    parser.add_argument('--n-processes', type=int, default=24,
                       help='Number of processes to use (default: 12)')
    parser.add_argument('--years-per-batch', type=int, default=1,
                       help='Number of years to process per batch (default: 1)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("ERA5 HEAT INDEX CALCULATION")
    print("="*80)
    print(f"Period: {args.start_year}-{args.end_year}")
    print(f"Temperature directory: {args.temp_dir}")
    print(f"Relative humidity directory: {args.rh_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Processes: {args.n_processes}")
    print(f"Years per batch: {args.years_per_batch}")
    print("="*80)
    
    # Validate directories
    temp_dir = Path(args.temp_dir)
    rh_dir = Path(args.rh_dir)
    
    if not temp_dir.exists():
        raise ValueError(f"Temperature directory does not exist: {temp_dir}")
    
    if not rh_dir.exists():
        raise ValueError(f"Relative humidity directory does not exist: {rh_dir}")
    
    # Get available years
    available_years = get_available_years(args.temp_dir, args.rh_dir, 
                                        args.start_year, args.end_year)
    
    if not available_years:
        raise ValueError("No years with complete temperature and RH data found")
    
    print(f"Found data for {len(available_years)} years: {available_years[0]}-{available_years[-1]}")
    
    # Create year batches for parallel processing
    year_batches = []
    for i in range(0, len(available_years), args.years_per_batch):
        batch = available_years[i:i + args.years_per_batch]
        year_batches.append(batch)
    
    print(f"Created {len(year_batches)} year batches")
    
    # Estimate memory usage
    estimated_memory_per_process = args.years_per_batch * 12 * 1.3 * 2  # GB (temp + RH input, HI output)
    total_estimated_memory = estimated_memory_per_process * args.n_processes
    
    print(f"Estimated memory usage: {total_estimated_memory:.1f} GB")
    if total_estimated_memory > 600:  # Conservative limit for 755GB system
        print("WARNING: High memory usage estimated.")
        print("Consider reducing --n-processes or --years-per-batch if needed.")
    
    # Process year batches in parallel
    print(f"\nStarting parallel processing with {args.n_processes} processes...")
    
    process_func = partial(
        process_year_batch,
        temp_dir=args.temp_dir,
        rh_dir=args.rh_dir,
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
    print("HEAT INDEX CALCULATION COMPLETED!")
    print("="*80)
    print(f"Output directory: {args.output_dir}")
    print(f"Files created: {len(all_results)}")
    
    if all_results:
        print("\nSample output files:")
        for i, result_file in enumerate(all_results[:5]):
            print(f"  - {result_file.name}")
        if len(all_results) > 5:
            print(f"  ... and {len(all_results) - 5} more files")
    
    print("\nHeat index files are ready for heatwave analysis!")

if __name__ == "__main__":
    main()
