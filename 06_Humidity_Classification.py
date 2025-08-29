#!/usr/bin/env python3
"""
Heatwave Humidity Classification System

Implements comprehensive humidity-based classification of heatwaves into:
- Hot-dry vs hot-humid categories
- Day-level and event-level analysis
- Absolute and percentile-based thresholds
- Annual aggregations by humidity category

Based on the pseudocode specification for studying heatwave humidity characteristics.
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
from collections import defaultdict
import logging

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HumidityClassificationConfig:
    """Configuration parameters for humidity classification."""
    
    def __init__(self, 
                 humidity_mode="absolute",
                 abs_dry=33.0,
                 abs_humid=66.0,
                 dominance_alpha=0.60,
                 min_valid_rh_fraction=0.8,
                 doy_leap_rule="fold_366_to_365"):
        
        self.humidity_mode = humidity_mode
        self.abs_dry = abs_dry
        self.abs_humid = abs_humid
        self.dominance_alpha = dominance_alpha
        self.min_valid_rh_fraction = min_valid_rh_fraction
        self.doy_leap_rule = doy_leap_rule
        
        # Validate parameters
        if humidity_mode not in ["absolute", "percentile"]:
            raise ValueError("humidity_mode must be 'absolute' or 'percentile'")
        if not 0 < dominance_alpha < 1:
            raise ValueError("dominance_alpha must be between 0 and 1")
        if not 0 < min_valid_rh_fraction <= 1:
            raise ValueError("min_valid_rh_fraction must be between 0 and 1")

def doy_of(date):
    """Get day of year (1-366) for a given date."""
    return date.dayofyear

def load_rh_data_chunk(year, rh_dir, chunk_bounds, rh_type="day"):
    """
    Load relative humidity data for a spatial chunk.
    
    Args:
        year: Target year
        rh_dir: Directory containing RH files
        chunk_bounds: (lat_start, lat_end, lon_start, lon_end)
        rh_type: "day" for daytime RH, "night" for nighttime RH (currently both use same data)
        
    Returns:
        rh_data: RH array [time, lat_chunk, lon_chunk]
        dates: Corresponding dates
    """
    lat_start, lat_end, lon_start, lon_end = chunk_bounds
    rh_dir = Path(rh_dir)
    
    # Look for ERA5 RH files with the actual naming pattern
    rh_pattern = f"era5_rh_{year}_*.nc"
    rh_files = sorted(rh_dir.glob(rh_pattern))
    
    if not rh_files:
        logger.warning(f"No RH files found for {year} in {rh_dir}")
        return None, None
    
    rh_data_list = []
    dates_list = []
    
    for rh_file in rh_files:
        try:
            ds = xr.open_dataset(rh_file)
            
            # Subset to spatial chunk
            ds_chunk = ds.isel(
                latitude=slice(lat_start, lat_end),
                longitude=slice(lon_start, lon_end)
            )
            
            # Extract RH data - your files use 'rh' variable
            if 'rh' in ds_chunk.variables:
                rh_var = ds_chunk.rh
            elif 'relative_humidity' in ds_chunk.variables:
                rh_var = ds_chunk.relative_humidity
            elif 'r' in ds_chunk.variables:
                rh_var = ds_chunk.r
            else:
                # Try to find any humidity-related variable
                rh_vars = [v for v in ds_chunk.variables if 'rh' in v.lower() or 'humid' in v.lower()]
                if rh_vars:
                    rh_var = ds_chunk[rh_vars[0]]
                else:
                    raise ValueError(f"No RH variable found in {rh_file}")
            
            # Handle time coordinate naming
            time_coord = None
            if 'valid_time' in rh_var.dims:
                time_coord = 'valid_time'
            elif 'time' in rh_var.dims:
                time_coord = 'time'
            else:
                # Use the first time-like coordinate
                time_coords = [c for c in rh_var.dims if 'time' in c.lower()]
                if time_coords:
                    time_coord = time_coords[0]
            
            if time_coord:
                # For now, use the same RH data for both day and night
                # In future, you could separate by time of day here
                daily_rh = rh_var
                time_values = ds_chunk[time_coord].values
            else:
                raise ValueError(f"No time coordinate found in {rh_file}")
            
            rh_data_list.append(daily_rh.values)
            dates_list.extend(pd.to_datetime(time_values))
            
            ds.close()
            
        except Exception as e:
            logger.error(f"Error loading {rh_file}: {e}")
            continue
    
    if not rh_data_list:
        return None, None
    
    combined_rh = np.concatenate(rh_data_list, axis=0)
    combined_dates = pd.to_datetime(dates_list)
    
    return combined_rh, combined_dates

def load_rh_percentiles(percentile_file, chunk_bounds):
    """
    Load RH percentile climatology for a spatial chunk.
    
    Args:
        percentile_file: Path to percentile file containing RH_P33_DOY and RH_P66_DOY
        chunk_bounds: (lat_start, lat_end, lon_start, lon_end)
        
    Returns:
        rh_p33: P33 climatology [365/366, lat_chunk, lon_chunk]
        rh_p66: P66 climatology [365/366, lat_chunk, lon_chunk]
    """
    lat_start, lat_end, lon_start, lon_end = chunk_bounds
    
    ds = xr.open_dataset(percentile_file)
    
    # Extract percentiles for chunk
    rh_p33 = ds['rh_p33'].values[:, lat_start:lat_end, lon_start:lon_end]
    rh_p66 = ds['rh_p66'].values[:, lat_start:lat_end, lon_start:lon_end]
    
    ds.close()
    
    return rh_p33, rh_p66

def map_percentile_thresholds(dates, rh_p33_doy, rh_p66_doy, config):
    """
    Map percentile thresholds to dates for a chunk.
    
    Args:
        dates: Array of dates
        rh_p33_doy: P33 climatology [365/366, lat_chunk, lon_chunk]
        rh_p66_doy: P66 climatology [365/366, lat_chunk, lon_chunk]
        config: HumidityClassificationConfig
        
    Returns:
        th33: P33 thresholds [time, lat_chunk, lon_chunk]
        th66: P66 thresholds [time, lat_chunk, lon_chunk]
    """
    n_time = len(dates)
    n_lat, n_lon = rh_p33_doy.shape[1], rh_p33_doy.shape[2]
    
    th33 = np.full((n_time, n_lat, n_lon), np.nan)
    th66 = np.full((n_time, n_lat, n_lon), np.nan)
    
    for i, date in enumerate(dates):
        doy = doy_of(date)
        
        # Handle leap year
        if doy == 366 and rh_p33_doy.shape[0] == 365:
            if config.doy_leap_rule == "fold_366_to_365":
                doy = 365
            elif config.doy_leap_rule == "interpolate":
                # Simple interpolation between day 59 and 60
                if rh_p33_doy.shape[0] >= 60:
                    th33[i, :, :] = (rh_p33_doy[58, :, :] + rh_p33_doy[59, :, :]) / 2
                    th66[i, :, :] = (rh_p66_doy[58, :, :] + rh_p66_doy[59, :, :]) / 2
                    continue
                else:
                    doy = 365
        
        th33[i, :, :] = rh_p33_doy[doy-1, :, :]
        th66[i, :, :] = rh_p66_doy[doy-1, :, :]
    
    return th33, th66

def classify_humidity(rh_value, dry_thr, humid_thr):
    """
    Classify humidity level based on thresholds.
    
    Args:
        rh_value: Relative humidity value (%)
        dry_thr: Dry threshold
        humid_thr: Humid threshold
        
    Returns:
        Classification: "missing", "dry", "humid", or "moderate"
    """
    if np.isnan(rh_value):
        return "missing"
    elif rh_value <= dry_thr:
        return "dry"
    elif rh_value >= humid_thr:
        return "humid"
    else:
        return "moderate"

def longest_consecutive(sequence, target_value):
    """
    Find longest consecutive occurrence of target_value in sequence.
    
    Args:
        sequence: List or array of values
        target_value: Value to find consecutive occurrences of
        
    Returns:
        Length of longest consecutive sequence
    """
    if not sequence:
        return 0
    
    max_length = 0
    current_length = 0
    
    for value in sequence:
        if value == target_value:
            current_length += 1
            max_length = max(max_length, current_length)
        else:
            current_length = 0
    
    return max_length

def classify_humidity_day_level(day_table, rh_day, rh_night, config, 
                               rh_p33_doy=None, rh_p66_doy=None, chunk_bounds=None):
    """
    Add humidity classification to day-level heatwave data.
    
    Args:
        day_table: DataFrame with heatwave day records
        rh_day: Daytime RH data [time, lat_chunk, lon_chunk]
        rh_night: Nighttime RH data [time, lat_chunk, lon_chunk]
        config: HumidityClassificationConfig
        rh_p33_doy: P33 climatology (if percentile mode)
        rh_p66_doy: P66 climatology (if percentile mode)
        chunk_bounds: Spatial chunk bounds
        
    Returns:
        Updated day_table with humidity classifications
    """
    lat_start, lat_end, lon_start, lon_end = chunk_bounds
    
    # Create date-to-index mapping for RH data
    rh_dates = pd.to_datetime(day_table['date'].unique())
    date_to_idx = {date: i for i, date in enumerate(rh_dates)}
    
    # Prepare thresholds if percentile mode
    if config.humidity_mode == "percentile":
        th33_day, th66_day = map_percentile_thresholds(rh_dates, rh_p33_doy, rh_p66_doy, config)
        th33_night, th66_night = map_percentile_thresholds(rh_dates, rh_p33_doy, rh_p66_doy, config)
    
    # Add humidity columns
    humidity_cols = ['humidity_class_day', 'RH_day', 'dry_thr_day', 'humid_thr_day',
                    'humidity_class_night', 'RH_night', 'dry_thr_night', 'humid_thr_night']
    
    for col in humidity_cols:
        if col not in day_table.columns:
            day_table[col] = np.nan
    
    # Process each row
    for idx, row in day_table.iterrows():
        date = pd.to_datetime(row['date'])
        grid_y = int(row['grid_y'])
        grid_x = int(row['grid_x'])
        var = row['var']
        
        # Convert global coordinates to chunk coordinates
        chunk_y = grid_y - lat_start
        chunk_x = grid_x - lon_start
        
        # Skip if outside chunk bounds
        if not (0 <= chunk_y < lat_end - lat_start and 0 <= chunk_x < lon_end - lon_start):
            continue
        
        # Get time index
        if date not in date_to_idx:
            continue
        time_idx = date_to_idx[date]
        
        # Process based on variable type
        if var == "tmax":
            # Daytime classification
            rh_value = rh_day[time_idx, chunk_y, chunk_x] if rh_day is not None else np.nan
            
            if config.humidity_mode == "absolute":
                dry_thr = config.abs_dry
                humid_thr = config.abs_humid
            else:  # percentile
                dry_thr = th33_day[time_idx, chunk_y, chunk_x]
                humid_thr = th66_day[time_idx, chunk_y, chunk_x]
            
            humidity_class = classify_humidity(rh_value, dry_thr, humid_thr)
            
            day_table.at[idx, 'humidity_class_day'] = humidity_class
            day_table.at[idx, 'RH_day'] = rh_value
            day_table.at[idx, 'dry_thr_day'] = dry_thr
            day_table.at[idx, 'humid_thr_day'] = humid_thr
            
        elif var == "tmin":
            # Nighttime classification
            rh_value = rh_night[time_idx, chunk_y, chunk_x] if rh_night is not None else np.nan
            
            if config.humidity_mode == "absolute":
                dry_thr = config.abs_dry
                humid_thr = config.abs_humid
            else:  # percentile
                dry_thr = th33_night[time_idx, chunk_y, chunk_x]
                humid_thr = th66_night[time_idx, chunk_y, chunk_x]
            
            humidity_class = classify_humidity(rh_value, dry_thr, humid_thr)
            
            day_table.at[idx, 'humidity_class_night'] = humidity_class
            day_table.at[idx, 'RH_night'] = rh_value
            day_table.at[idx, 'dry_thr_night'] = dry_thr
            day_table.at[idx, 'humid_thr_night'] = humid_thr
    
    return day_table

def summarize_event_humidity(event_table, day_table, config):
    """
    Add event-level humidity summary statistics and labels.
    
    Args:
        event_table: DataFrame with heatwave event records
        day_table: DataFrame with heatwave day records (with humidity classifications)
        config: HumidityClassificationConfig
        
    Returns:
        Updated event_table with humidity statistics
    """
    # Add humidity columns to event table
    humidity_cols = [
        'mean_RH_day', 'median_RH_day', 'f_dry_day', 'f_humid_day',
        'n_dry_day', 'n_humid_day', 'n_moderate_day', 'frac_valid_RH_day',
        'label_day', 'longest_humid_streak_day', 'longest_dry_streak_day',
        'mean_RH_night', 'median_RH_night', 'f_dry_night', 'f_humid_night',
        'n_dry_night', 'n_humid_night', 'n_moderate_night', 'frac_valid_RH_night',
        'label_night', 'longest_humid_streak_night', 'longest_dry_streak_night'
    ]
    
    for col in humidity_cols:
        if col not in event_table.columns:
            event_table[col] = np.nan
    
    # Process each event
    for idx, event in event_table.iterrows():
        event_id = event['event_id']
        var = event['var']
        duration_days = event['duration_days']
        
        # Get all days for this event
        event_days = day_table[day_table['event_id'] == event_id].copy()
        
        if len(event_days) == 0:
            continue
        
        if var == "tmax":
            # Process daytime humidity
            valid_days = event_days[event_days['humidity_class_day'] != "missing"]
            n_valid = len(valid_days)
            frac_valid = n_valid / max(1, duration_days)
            
            if n_valid > 0:
                # Count by category
                n_dry = len(valid_days[valid_days['humidity_class_day'] == "dry"])
                n_humid = len(valid_days[valid_days['humidity_class_day'] == "humid"])
                n_moderate = len(valid_days[valid_days['humidity_class_day'] == "moderate"])
                
                # Calculate fractions
                f_dry = n_dry / max(1, n_valid)
                f_humid = n_humid / max(1, n_valid)
                
                # Calculate statistics
                rh_values = valid_days['RH_day'].dropna()
                mean_rh = rh_values.mean() if len(rh_values) > 0 else np.nan
                median_rh = rh_values.median() if len(rh_values) > 0 else np.nan
                
                # Determine label based on dominance
                if frac_valid >= config.min_valid_rh_fraction:
                    if f_humid >= config.dominance_alpha:
                        label = "humid-hot"
                    elif f_dry >= config.dominance_alpha:
                        label = "dry-hot"
                    else:
                        label = "mixed-hot"
                else:
                    label = "insufficient-RH"
                
                # Calculate longest streaks
                humidity_sequence = event_days['humidity_class_day'].tolist()
                longest_humid = longest_consecutive(humidity_sequence, "humid")
                longest_dry = longest_consecutive(humidity_sequence, "dry")
                
                # Update event record
                event_table.at[idx, 'mean_RH_day'] = mean_rh
                event_table.at[idx, 'median_RH_day'] = median_rh
                event_table.at[idx, 'f_dry_day'] = f_dry
                event_table.at[idx, 'f_humid_day'] = f_humid
                event_table.at[idx, 'n_dry_day'] = n_dry
                event_table.at[idx, 'n_humid_day'] = n_humid
                event_table.at[idx, 'n_moderate_day'] = n_moderate
                event_table.at[idx, 'frac_valid_RH_day'] = frac_valid
                event_table.at[idx, 'label_day'] = label
                event_table.at[idx, 'longest_humid_streak_day'] = longest_humid
                event_table.at[idx, 'longest_dry_streak_day'] = longest_dry
            
        elif var == "tmin":
            # Process nighttime humidity (similar logic)
            valid_days = event_days[event_days['humidity_class_night'] != "missing"]
            n_valid = len(valid_days)
            frac_valid = n_valid / max(1, duration_days)
            
            if n_valid > 0:
                # Count by category
                n_dry = len(valid_days[valid_days['humidity_class_night'] == "dry"])
                n_humid = len(valid_days[valid_days['humidity_class_night'] == "humid"])
                n_moderate = len(valid_days[valid_days['humidity_class_night'] == "moderate"])
                
                # Calculate fractions
                f_dry = n_dry / max(1, n_valid)
                f_humid = n_humid / max(1, n_valid)
                
                # Calculate statistics
                rh_values = valid_days['RH_night'].dropna()
                mean_rh = rh_values.mean() if len(rh_values) > 0 else np.nan
                median_rh = rh_values.median() if len(rh_values) > 0 else np.nan
                
                # Determine label based on dominance
                if frac_valid >= config.min_valid_rh_fraction:
                    if f_humid >= config.dominance_alpha:
                        label = "humid-hot"
                    elif f_dry >= config.dominance_alpha:
                        label = "dry-hot"
                    else:
                        label = "mixed-hot"
                else:
                    label = "insufficient-RH"
                
                # Calculate longest streaks
                humidity_sequence = event_days['humidity_class_night'].tolist()
                longest_humid = longest_consecutive(humidity_sequence, "humid")
                longest_dry = longest_consecutive(humidity_sequence, "dry")
                
                # Update event record
                event_table.at[idx, 'mean_RH_night'] = mean_rh
                event_table.at[idx, 'median_RH_night'] = median_rh
                event_table.at[idx, 'f_dry_night'] = f_dry
                event_table.at[idx, 'f_humid_night'] = f_humid
                event_table.at[idx, 'n_dry_night'] = n_dry
                event_table.at[idx, 'n_humid_night'] = n_humid
                event_table.at[idx, 'n_moderate_night'] = n_moderate
                event_table.at[idx, 'frac_valid_RH_night'] = frac_valid
                event_table.at[idx, 'label_night'] = label
                event_table.at[idx, 'longest_humid_streak_night'] = longest_humid
                event_table.at[idx, 'longest_dry_streak_night'] = longest_dry
    
    return event_table

def calculate_annual_aggregations(event_table, day_table, year, chunk_bounds):
    """
    Calculate annual aggregations by humidity category for each grid cell.
    
    Args:
        event_table: DataFrame with event-level humidity statistics
        day_table: DataFrame with day-level humidity classifications
        year: Target year
        chunk_bounds: (lat_start, lat_end, lon_start, lon_end)
        
    Returns:
        Dictionary with annual aggregation grids
    """
    lat_start, lat_end, lon_start, lon_end = chunk_bounds
    n_lat = lat_end - lat_start
    n_lon = lon_end - lon_start
    
    # Initialize aggregation grids
    aggregations = {
        # Daytime event counts
        'HWN_day_humid': np.zeros((n_lat, n_lon)),
        'HWN_day_dry': np.zeros((n_lat, n_lon)),
        'HWN_day_mixed': np.zeros((n_lat, n_lon)),
        
        # Nighttime event counts
        'HWN_night_humid': np.zeros((n_lat, n_lon)),
        'HWN_night_dry': np.zeros((n_lat, n_lon)),
        'HWN_night_mixed': np.zeros((n_lat, n_lon)),
        
        # Daytime day counts
        'HWTD_day_humid': np.zeros((n_lat, n_lon)),
        'HWTD_day_dry': np.zeros((n_lat, n_lon)),
        'HWTD_day_moderate': np.zeros((n_lat, n_lon)),
        
        # Nighttime day counts
        'HWTD_night_humid': np.zeros((n_lat, n_lon)),
        'HWTD_night_dry': np.zeros((n_lat, n_lon)),
        'HWTD_night_moderate': np.zeros((n_lat, n_lon)),
        
        # Longest durations
        'HWLD_day_humid': np.zeros((n_lat, n_lon)),
        'HWLD_day_dry': np.zeros((n_lat, n_lon)),
        'HWLD_night_humid': np.zeros((n_lat, n_lon)),
        'HWLD_night_dry': np.zeros((n_lat, n_lon))
    }
    
    # Process each grid cell
    for i in range(n_lat):
        for j in range(n_lon):
            global_i = lat_start + i
            global_j = lon_start + j
            
            # Filter events for this grid cell and year
            cell_events_day = event_table[
                (event_table['grid_y'] == global_i) & 
                (event_table['grid_x'] == global_j) & 
                (event_table['year'] == year) & 
                (event_table['var'] == 'tmax')
            ]
            
            cell_events_night = event_table[
                (event_table['grid_y'] == global_i) & 
                (event_table['grid_x'] == global_j) & 
                (event_table['year'] == year) & 
                (event_table['var'] == 'tmin')
            ]
            
            # Count events by category
            if len(cell_events_day) > 0:
                aggregations['HWN_day_humid'][i, j] = len(cell_events_day[cell_events_day['label_day'] == 'humid-hot'])
                aggregations['HWN_day_dry'][i, j] = len(cell_events_day[cell_events_day['label_day'] == 'dry-hot'])
                aggregations['HWN_day_mixed'][i, j] = len(cell_events_day[cell_events_day['label_day'] == 'mixed-hot'])
                
                # Longest durations
                humid_streaks = cell_events_day['longest_humid_streak_day'].dropna()
                dry_streaks = cell_events_day['longest_dry_streak_day'].dropna()
                aggregations['HWLD_day_humid'][i, j] = humid_streaks.max() if len(humid_streaks) > 0 else 0
                aggregations['HWLD_day_dry'][i, j] = dry_streaks.max() if len(dry_streaks) > 0 else 0
            
            if len(cell_events_night) > 0:
                aggregations['HWN_night_humid'][i, j] = len(cell_events_night[cell_events_night['label_night'] == 'humid-hot'])
                aggregations['HWN_night_dry'][i, j] = len(cell_events_night[cell_events_night['label_night'] == 'dry-hot'])
                aggregations['HWN_night_mixed'][i, j] = len(cell_events_night[cell_events_night['label_night'] == 'mixed-hot'])
                
                # Longest durations
                humid_streaks = cell_events_night['longest_humid_streak_night'].dropna()
                dry_streaks = cell_events_night['longest_dry_streak_night'].dropna()
                aggregations['HWLD_night_humid'][i, j] = humid_streaks.max() if len(humid_streaks) > 0 else 0
                aggregations['HWLD_night_dry'][i, j] = dry_streaks.max() if len(dry_streaks) > 0 else 0
            
            # Count days by humidity category
            cell_days_day = day_table[
                (day_table['grid_y'] == global_i) & 
                (day_table['grid_x'] == global_j) & 
                (day_table['date'].dt.year == year) & 
                (day_table['var'] == 'tmax')
            ]
            
            cell_days_night = day_table[
                (day_table['grid_y'] == global_i) & 
                (day_table['grid_x'] == global_j) & 
                (day_table['date'].dt.year == year) & 
                (day_table['var'] == 'tmin')
            ]
            
            if len(cell_days_day) > 0:
                aggregations['HWTD_day_humid'][i, j] = len(cell_days_day[cell_days_day['humidity_class_day'] == 'humid'])
                aggregations['HWTD_day_dry'][i, j] = len(cell_days_day[cell_days_day['humidity_class_day'] == 'dry'])
                aggregations['HWTD_day_moderate'][i, j] = len(cell_days_day[cell_days_day['humidity_class_day'] == 'moderate'])
            
            if len(cell_days_night) > 0:
                aggregations['HWTD_night_humid'][i, j] = len(cell_days_night[cell_days_night['humidity_class_night'] == 'humid'])
                aggregations['HWTD_night_dry'][i, j] = len(cell_days_night[cell_days_night['humidity_class_night'] == 'dry'])
                aggregations['HWTD_night_moderate'][i, j] = len(cell_days_night[cell_days_night['humidity_class_night'] == 'moderate'])
    
    return aggregations

def process_humidity_classification_chunk(chunk_info, year, heatwave_dir, rh_dir, 
                                        percentile_file, config, variables):
    """
    Process humidity classification for a spatial chunk.
    
    Args:
        chunk_info: (lat_start, lat_end, lon_start, lon_end)
        year: Target year
        heatwave_dir: Directory containing heatwave results
        rh_dir: Directory containing RH data
        percentile_file: Path to RH percentile file (if percentile mode)
        config: HumidityClassificationConfig
        variables: List of variables to process
        
    Returns:
        Dictionary with processed results
    """
    lat_start, lat_end, lon_start, lon_end = chunk_info
    
    logger.info(f"Processing humidity chunk: year={year}, lat[{lat_start}:{lat_end}], lon[{lon_start}:{lon_end}]")
    
    try:
        # Load heatwave data
        all_events = []
        all_days = []
        
        for var in variables:
            events_file = Path(heatwave_dir) / f'heatwave_events_{var}_{year}.parquet'
            days_file = Path(heatwave_dir) / f'heatwave_days_{var}_{year}.parquet'
            
            if events_file.exists():
                events_df = pd.read_parquet(events_file)
                # Filter to chunk
                chunk_events = events_df[
                    (events_df['grid_y'] >= lat_start) & 
                    (events_df['grid_y'] < lat_end) &
                    (events_df['grid_x'] >= lon_start) & 
                    (events_df['grid_x'] < lon_end)
                ]
                all_events.append(chunk_events)
            
            if days_file.exists():
                days_df = pd.read_parquet(days_file)
                # Filter to chunk
                chunk_days = days_df[
                    (days_df['grid_y'] >= lat_start) & 
                    (days_df['grid_y'] < lat_end) &
                    (days_df['grid_x'] >= lon_start) & 
                    (days_df['grid_x'] < lon_end)
                ]
                all_days.append(chunk_days)
        
        if not all_events or not all_days:
            logger.warning(f"No heatwave data found for chunk lat[{lat_start}:{lat_end}], lon[{lon_start}:{lon_end}]")
            return None
        
        # Combine events and days
        event_table = pd.concat(all_events, ignore_index=True)
        day_table = pd.concat(all_days, ignore_index=True)
        
        # Ensure date column is datetime
        day_table['date'] = pd.to_datetime(day_table['date'])
        
        # Load RH data
        rh_day, rh_dates_day = load_rh_data_chunk(year, rh_dir, chunk_info, "day")
        rh_night, rh_dates_night = load_rh_data_chunk(year, rh_dir, chunk_info, "night")
        
        # Load percentile thresholds if needed
        rh_p33_doy = None
        rh_p66_doy = None
        if config.humidity_mode == "percentile" and percentile_file:
            rh_p33_doy, rh_p66_doy = load_rh_percentiles(percentile_file, chunk_info)
        
        # Classify humidity at day level
        day_table = classify_humidity_day_level(
            day_table, rh_day, rh_night, config, 
            rh_p33_doy, rh_p66_doy, chunk_info
        )
        
        # Summarize at event level
        event_table = summarize_event_humidity(event_table, day_table, config)
        
        # Calculate annual aggregations
        annual_aggs = calculate_annual_aggregations(event_table, day_table, year, chunk_info)
        
        logger.info(f"Completed humidity chunk: year={year}, lat[{lat_start}:{lat_end}], lon[{lon_start}:{lon_end}]")
        
        return {
            'chunk_bounds': chunk_info,
            'events': event_table,
            'days': day_table,
            'annual_aggregations': annual_aggs
        }
        
    except Exception as e:
        logger.error(f"Error processing humidity chunk lat[{lat_start}:{lat_end}], lon[{lon_start}:{lon_end}]: {e}")
        return None

def combine_humidity_results(chunk_results, full_lat, full_lon, year, output_dir, config):
    """
    Combine chunk results and save final outputs.
    
    Args:
        chunk_results: List of chunk processing results
        full_lat: Full latitude array
        full_lon: Full longitude array
        year: Target year
        output_dir: Output directory
        config: HumidityClassificationConfig
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Combine events and days
    all_events = []
    all_days = []
    
    # Initialize aggregation grids
    n_lat, n_lon = len(full_lat), len(full_lon)
    aggregation_grids = {}
    
    # Define all aggregation variables
    agg_vars = [
        'HWN_day_humid', 'HWN_day_dry', 'HWN_day_mixed',
        'HWN_night_humid', 'HWN_night_dry', 'HWN_night_mixed',
        'HWTD_day_humid', 'HWTD_day_dry', 'HWTD_day_moderate',
        'HWTD_night_humid', 'HWTD_night_dry', 'HWTD_night_moderate',
        'HWLD_day_humid', 'HWLD_day_dry', 'HWLD_night_humid', 'HWLD_night_dry'
    ]
    
    for var in agg_vars:
        aggregation_grids[var] = np.zeros((n_lat, n_lon))
    
    # Combine results from all chunks
    for result in chunk_results:
        if result is None:
            continue
        
        # Combine events and days
        all_events.append(result['events'])
        all_days.append(result['days'])
        
        # Combine aggregations
        lat_start, lat_end, lon_start, lon_end = result['chunk_bounds']
        
        for var in agg_vars:
            if var in result['annual_aggregations']:
                aggregation_grids[var][lat_start:lat_end, lon_start:lon_end] = result['annual_aggregations'][var]
    
    # Save combined tables
    if all_events:
        combined_events = pd.concat(all_events, ignore_index=True)
        events_file = output_dir / f'heatwave_events_humidity_{year}.parquet'
        combined_events.to_parquet(events_file)
        logger.info(f"Saved events with humidity: {events_file} ({len(combined_events)} events)")
    
    if all_days:
        combined_days = pd.concat(all_days, ignore_index=True)
        days_file = output_dir / f'heatwave_days_humidity_{year}.parquet'
        combined_days.to_parquet(days_file)
        logger.info(f"Saved days with humidity: {days_file} ({len(combined_days)} days)")
    
    # Save aggregation grids
    agg_ds = xr.Dataset({
        var: (['latitude', 'longitude'], aggregation_grids[var])
        for var in agg_vars
    }, coords={
        'latitude': full_lat,
        'longitude': full_lon
    })
    
    # Add attributes
    agg_ds.attrs['humidity_mode'] = config.humidity_mode
    agg_ds.attrs['abs_dry_threshold'] = config.abs_dry
    agg_ds.attrs['abs_humid_threshold'] = config.abs_humid
    agg_ds.attrs['dominance_alpha'] = config.dominance_alpha
    agg_ds.attrs['min_valid_rh_fraction'] = config.min_valid_rh_fraction
    
    # Add variable attributes
    for var in agg_vars:
        if 'HWN' in var:
            agg_ds[var].attrs = {'long_name': f'Number of heatwave events ({var})', 'units': 'count'}
        elif 'HWTD' in var:
            agg_ds[var].attrs = {'long_name': f'Total heatwave days ({var})', 'units': 'days'}
        elif 'HWLD' in var:
            agg_ds[var].attrs = {'long_name': f'Longest heatwave duration ({var})', 'units': 'days'}
    
    agg_file = output_dir / f'heatwave_humidity_aggregations_{year}.nc'
    agg_ds.to_netcdf(agg_file)
    agg_ds.close()
    
    logger.info(f"Saved humidity aggregations: {agg_file}")

def main():
    """Main function for humidity classification."""
    parser = argparse.ArgumentParser(description='Heatwave Humidity Classification System')
    
    # Basic parameters
    parser.add_argument('--start-year', type=int, default=1980, help='Start year')
    parser.add_argument('--end-year', type=int, default=2024, help='End year')
    parser.add_argument('--heatwave-dir', default='data/processed/heatwave_metrics',
                       help='Directory containing heatwave results')
    parser.add_argument('--rh-dir', default='data/processed/relative_humidity',
                       help='Directory containing RH data')
    parser.add_argument('--percentile-file', default = 'data/processed/rh_percentiles.nc',
                       help='Path to RH percentile file (for percentile mode)')
    parser.add_argument('--output-dir', default='data/processed/humidity_classification',
                       help='Output directory')
    
    # Humidity classification parameters
    parser.add_argument('--humidity-mode', choices=['absolute', 'percentile'], 
                       default='absolute', help='Humidity classification mode')
    parser.add_argument('--abs-dry', type=float, default=33.0,
                       help='Absolute dry threshold (%)')
    parser.add_argument('--abs-humid', type=float, default=66.0,
                       help='Absolute humid threshold (%)')
    parser.add_argument('--dominance-alpha', type=float, default=0.60,
                       help='Dominance threshold for event labeling')
    parser.add_argument('--min-valid-rh', type=float, default=0.8,
                       help='Minimum fraction of valid RH for event labeling')
    parser.add_argument('--doy-leap-rule', choices=['fold_366_to_365', 'interpolate'],
                       default='fold_366_to_365', help='Leap year handling')
    
    # Processing parameters
    parser.add_argument('--variables', nargs='+', default=['tmax', 'tmin'],
                       help='Variables to process')
    parser.add_argument('--n-processes', type=int, default=16,
                       help='Number of processes')
    parser.add_argument('--chunk-size-lat', type=int, default=50,
                       help='Latitude chunk size')
    parser.add_argument('--chunk-size-lon', type=int, default=100,
                       help='Longitude chunk size')
    
    args = parser.parse_args()
    
    # Create configuration
    config = HumidityClassificationConfig(
        humidity_mode=args.humidity_mode,
        abs_dry=args.abs_dry,
        abs_humid=args.abs_humid,
        dominance_alpha=args.dominance_alpha,
        min_valid_rh_fraction=args.min_valid_rh,
        doy_leap_rule=args.doy_leap_rule
    )
    
    logger.info("="*80)
    logger.info("HEATWAVE HUMIDITY CLASSIFICATION SYSTEM")
    logger.info("="*80)
    logger.info(f"Analysis period: {args.start_year}-{args.end_year}")
    logger.info(f"Variables: {args.variables}")
    logger.info(f"Humidity mode: {config.humidity_mode}")
    if config.humidity_mode == "absolute":
        logger.info(f"Thresholds: dry≤{config.abs_dry}%, humid≥{config.abs_humid}%")
    logger.info(f"Dominance alpha: {config.dominance_alpha}")
    logger.info(f"Min valid RH fraction: {config.min_valid_rh_fraction}")
    logger.info("="*80)
    
    # Validate directories
    heatwave_dir = Path(args.heatwave_dir)
    rh_dir = Path(args.rh_dir)
    
    if not heatwave_dir.exists():
        raise ValueError(f"Heatwave directory does not exist: {heatwave_dir}")
    
    if not rh_dir.exists():
        raise ValueError(f"RH directory does not exist: {rh_dir}")
    
    if config.humidity_mode == "percentile" and not args.percentile_file:
        raise ValueError("Percentile file required for percentile mode")
    
    # Get grid dimensions from sample heatwave file
    sample_files = list(heatwave_dir.glob('heatwave_metrics_*.nc'))
    if not sample_files:
        raise ValueError(f"No heatwave metric files found in {heatwave_dir}")
    
    sample_ds = xr.open_dataset(sample_files[0])
    full_lat = sample_ds.latitude.values
    full_lon = sample_ds.longitude.values
    sample_ds.close()
    
    logger.info(f"Grid dimensions: {len(full_lat)} x {len(full_lon)}")
    
    # Create spatial chunks
    from collections import namedtuple
    spatial_chunks = []
    for lat_start in range(0, len(full_lat), args.chunk_size_lat):
        lat_end = min(lat_start + args.chunk_size_lat, len(full_lat))
        for lon_start in range(0, len(full_lon), args.chunk_size_lon):
            lon_end = min(lon_start + args.chunk_size_lon, len(full_lon))
            spatial_chunks.append((lat_start, lat_end, lon_start, lon_end))
    
    logger.info(f"Created {len(spatial_chunks)} spatial chunks")
    
    # Process each year
    years = list(range(args.start_year, args.end_year + 1))
    
    for year in years:
        logger.info(f"\n{'='*20} PROCESSING YEAR {year} {'='*20}")
        
        # Process chunks in parallel
        process_func = partial(
            process_humidity_classification_chunk,
            year=year,
            heatwave_dir=args.heatwave_dir,
            rh_dir=args.rh_dir,
            percentile_file=args.percentile_file,
            config=config,
            variables=args.variables
        )
        
        logger.info(f"Processing {len(spatial_chunks)} chunks with {args.n_processes} processes...")
        
        with mp.Pool(args.n_processes) as pool:
            chunk_results = pool.map(process_func, spatial_chunks)
        
        # Combine results
        combine_humidity_results(chunk_results, full_lat, full_lon, year, args.output_dir, config)
    
    logger.info("\n" + "="*80)
    logger.info("HUMIDITY CLASSIFICATION COMPLETED!")
    logger.info("="*80)

if __name__ == "__main__":
    main()
