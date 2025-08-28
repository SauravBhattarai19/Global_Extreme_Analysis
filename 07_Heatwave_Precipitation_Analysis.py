#!/usr/bin/env python3
"""
Heatwave-Precipitation Interaction Analysis System

Implements comprehensive analysis of precipitation patterns before, during, and after heatwaves.
Includes compound event detection, recovery patterns, and meteorological classifications.

Based on the heatwave detection and humidity classification systems, this script analyzes:
- Pre-heatwave precipitation conditions
- During-heatwave precipitation patterns  
- Post-heatwave recovery precipitation
- Compound drought-heat events
- Precipitation-humidity interactions

Optimized for high-performance computing with parallel processing.
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

class PrecipitationAnalysisConfig:
    """Configuration parameters for precipitation analysis."""
    
    def __init__(self,
                 pre_heatwave_days=7,
                 post_heatwave_days=[3, 5, 7, 10, 15, 30],
                 min_precip_threshold=1.0,  # mm/day
                 drought_percentile=10,
                 intense_precip_percentile=90,
                 units_conversion_factor=1000.0):  # ERA5 tp is in meters, convert to mm
        
        self.pre_heatwave_days = pre_heatwave_days
        self.post_heatwave_days = post_heatwave_days
        self.min_precip_threshold = min_precip_threshold
        self.drought_percentile = drought_percentile
        self.intense_precip_percentile = intense_precip_percentile
        self.units_conversion_factor = units_conversion_factor
        
        # Validate parameters
        if pre_heatwave_days < 1:
            raise ValueError("pre_heatwave_days must be >= 1")
        if not all(d > 0 for d in post_heatwave_days):
            raise ValueError("All post_heatwave_days must be > 0")
        if min_precip_threshold < 0:
            raise ValueError("min_precip_threshold must be >= 0")

def load_precipitation_data_chunk(year, precip_dir, chunk_bounds):
    """
    Load precipitation data for a spatial chunk and year, including adjacent years as needed.
    
    Args:
        year: Target year
        precip_dir: Directory containing precipitation files
        chunk_bounds: (lat_start, lat_end, lon_start, lon_end)
        
    Returns:
        precip_data: Precipitation array [time, lat_chunk, lon_chunk] in mm
        dates: Corresponding dates
    """
    lat_start, lat_end, lon_start, lon_end = chunk_bounds
    precip_dir = Path(precip_dir)
    
    precip_data_list = []
    dates_list = []
    
    # Load December from previous year (for early January heatwaves)
    prev_year_file = precip_dir / f"era5_daily_{year-1}_12.nc"
    if prev_year_file.exists():
        try:
            ds = xr.open_dataset(prev_year_file, chunks={'valid_time': 50})
            
            # Subset to spatial chunk
            ds_chunk = ds.isel(
                latitude=slice(lat_start, lat_end),
                longitude=slice(lon_start, lon_end)
            )
            
            # Extract precipitation data and convert to mm/day
            if 'tp' in ds_chunk.variables:
                precip_var = ds_chunk.tp * 1000.0  # Convert from m to mm
            else:
                raise ValueError(f"No precipitation variable found in {prev_year_file}")
            
            # Calculate daily precipitation (ERA5 tp might be sub-daily)
            daily_precip = precip_var.resample(valid_time='1D').sum()
            
            precip_data_list.append(daily_precip.values)
            dates_list.extend(pd.to_datetime(daily_precip.valid_time.values))
            
            ds.close()
            logger.info(f"Loaded previous year December data from {prev_year_file}")
            
        except Exception as e:
            logger.warning(f"Could not load previous year December data from {prev_year_file}: {e}")
    
    # Load all months for target year
    for month in range(1, 13):
        file_path = precip_dir / f"era5_daily_{year}_{month:02d}.nc"
        
        if not file_path.exists():
            logger.warning(f"Precipitation file not found: {file_path}")
            continue
        
        try:
            ds = xr.open_dataset(file_path, chunks={'valid_time': 50})
            
            # Subset to spatial chunk
            ds_chunk = ds.isel(
                latitude=slice(lat_start, lat_end),
                longitude=slice(lon_start, lon_end)
            )
            
            # Extract precipitation data and convert to mm/day
            if 'tp' in ds_chunk.variables:
                precip_var = ds_chunk.tp * 1000.0  # Convert from m to mm
            else:
                raise ValueError(f"No precipitation variable found in {file_path}")
            
            # Calculate daily precipitation (ERA5 tp might be sub-daily)
            daily_precip = precip_var.resample(valid_time='1D').sum()
            
            precip_data_list.append(daily_precip.values)
            dates_list.extend(pd.to_datetime(daily_precip.valid_time.values))
            
            ds.close()
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            continue
    
    # Load January from next year (for late December heatwaves)
    next_year_file = precip_dir / f"era5_daily_{year+1}_01.nc"
    if next_year_file.exists():
        try:
            ds = xr.open_dataset(next_year_file, chunks={'valid_time': 50})
            
            # Subset to spatial chunk
            ds_chunk = ds.isel(
                latitude=slice(lat_start, lat_end),
                longitude=slice(lon_start, lon_end)
            )
            
            # Extract precipitation data and convert to mm/day
            if 'tp' in ds_chunk.variables:
                precip_var = ds_chunk.tp * 1000.0  # Convert from m to mm
            else:
                raise ValueError(f"No precipitation variable found in {next_year_file}")
            
            # Calculate daily precipitation (ERA5 tp might be sub-daily)
            daily_precip = precip_var.resample(valid_time='1D').sum()
            
            precip_data_list.append(daily_precip.values)
            dates_list.extend(pd.to_datetime(daily_precip.valid_time.values))
            
            ds.close()
            logger.info(f"Loaded next year January data from {next_year_file}")
            
        except Exception as e:
            logger.warning(f"Could not load next year January data from {next_year_file}: {e}")
    
    if not precip_data_list:
        return None, None
    
    combined_precip = np.concatenate(precip_data_list, axis=0)
    combined_dates = pd.to_datetime(dates_list)
    
    # Sort by date to ensure chronological order
    sort_idx = np.argsort(combined_dates)
    combined_precip = combined_precip[sort_idx]
    combined_dates = combined_dates[sort_idx]
    
    return combined_precip, combined_dates

def load_precipitation_percentiles(percentile_file, chunk_bounds):
    """
    Load precipitation percentile climatology for a spatial chunk.
    
    Args:
        percentile_file: Path to percentile file
        chunk_bounds: (lat_start, lat_end, lon_start, lon_end)
        
    Returns:
        Dictionary with percentile arrays [365/366, lat_chunk, lon_chunk]
    """
    lat_start, lat_end, lon_start, lon_end = chunk_bounds
    
    ds = xr.open_dataset(percentile_file)
    
    percentiles = {}
    for var in ds.data_vars:
        if 'precip_p' in var or 'tp_p' in var:
            percentiles[var] = ds[var].values[:, lat_start:lat_end, lon_start:lon_end]
    
    ds.close()
    
    return percentiles

def map_percentile_thresholds(dates, percentile_doy, config):
    """
    Map percentile thresholds to dates for a chunk.
    
    Args:
        dates: Array of dates
        percentile_doy: Percentile climatology [365/366, lat_chunk, lon_chunk]
        config: PrecipitationAnalysisConfig
        
    Returns:
        thresholds: Threshold array [time, lat_chunk, lon_chunk]
    """
    n_time = len(dates)
    n_lat, n_lon = percentile_doy.shape[1], percentile_doy.shape[2]
    
    thresholds = np.full((n_time, n_lat, n_lon), np.nan)
    
    for i, date in enumerate(dates):
        doy = date.dayofyear
        
        # Handle leap year
        if doy == 366 and percentile_doy.shape[0] == 365:
            doy = 365
        
        thresholds[i, :, :] = percentile_doy[doy-1, :, :]
    
    return thresholds

def classify_precipitation_intensity(precip_value, p10_threshold, p90_threshold, config):
    """
    Classify precipitation intensity based on thresholds.
    
    Args:
        precip_value: Precipitation value (mm)
        p10_threshold: Drought threshold (P10)
        p90_threshold: Intense precipitation threshold (P90)
        config: PrecipitationAnalysisConfig
        
    Returns:
        Classification: "no_precip", "very_light", "moderate", or "intense"
    """
    if precip_value < 0.1:
        return "no_precip"
    elif precip_value <= p10_threshold:
        return "very_light"
    elif precip_value <= p90_threshold:
        return "moderate"
    else:
        return "intense"

def calculate_precipitation_metrics(precip_data, dates, config):
    """
    Calculate comprehensive precipitation metrics for a time period.
    
    Args:
        precip_data: Precipitation array
        dates: Corresponding dates
        config: PrecipitationAnalysisConfig
        
    Returns:
        Dictionary with precipitation metrics
    """
    # Handle empty arrays
    if len(precip_data) == 0:
        return {
            'total_precip': 0.0,
            'max_daily_precip': 0.0,
            'precip_days': 0,
            'dry_days': len(dates),
            'consecutive_dry_days': len(dates),
            'mean_precip_wet_days': 0.0,
            'precip_frequency': 0.0
        }
    
    # Convert to mm if needed
    precip_mm = precip_data * config.units_conversion_factor if np.max(precip_data) < 1 else precip_data
    
    total_precip = np.sum(precip_mm)
    max_daily_precip = np.max(precip_mm)
    
    # Precipitation days (>= threshold)
    precip_days = np.sum(precip_mm >= config.min_precip_threshold)
    dry_days = len(dates) - precip_days
    
    # Consecutive dry days
    is_dry = precip_mm < config.min_precip_threshold
    consecutive_dry_days = longest_consecutive_sequence(is_dry, True)
    
    # Mean precipitation on wet days
    wet_day_precip = precip_mm[precip_mm >= config.min_precip_threshold]
    mean_precip_wet_days = np.mean(wet_day_precip) if len(wet_day_precip) > 0 else 0.0
    
    # Precipitation frequency
    precip_frequency = precip_days / max(1, len(dates))
    
    return {
        'total_precip': total_precip,
        'max_daily_precip': max_daily_precip,
        'precip_days': precip_days,
        'dry_days': dry_days,
        'consecutive_dry_days': consecutive_dry_days,
        'mean_precip_wet_days': mean_precip_wet_days,
        'precip_frequency': precip_frequency
    }

def longest_consecutive_sequence(sequence, target_value):
    """
    Find longest consecutive occurrence of target_value in sequence.
    
    Args:
        sequence: Array of values
        target_value: Value to find consecutive occurrences of
        
    Returns:
        Length of longest consecutive sequence
    """
    if len(sequence) == 0:
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

def get_date_range_extended(start_date, end_date, buffer_days_before=0, buffer_days_after=0):
    """
    Get extended date range with buffers.
    
    Args:
        start_date: Start date
        end_date: End date
        buffer_days_before: Days to extend before start
        buffer_days_after: Days to extend after end
        
    Returns:
        List of dates in extended range
    """
    extended_start = start_date - timedelta(days=buffer_days_before)
    extended_end = end_date + timedelta(days=buffer_days_after)
    
    dates = []
    current_date = extended_start
    while current_date <= extended_end:
        dates.append(current_date)
        current_date += timedelta(days=1)
    
    return dates

def analyze_heatwave_precipitation_interactions(heatwave_events, heatwave_days, humidity_events,
                                              precip_data, precip_dates, precip_percentiles,
                                              chunk_bounds, config):
    """
    Main analysis function for heatwave-precipitation interactions.
    
    Args:
        heatwave_events: DataFrame with heatwave event records
        heatwave_days: DataFrame with heatwave day records
        humidity_events: DataFrame with humidity-classified events
        precip_data: Precipitation data [time, lat_chunk, lon_chunk]
        precip_dates: Corresponding dates
        precip_percentiles: Dictionary with percentile thresholds
        chunk_bounds: Spatial chunk bounds
        config: PrecipitationAnalysisConfig
        
    Returns:
        enhanced_events: List of enhanced event records
        precipitation_analysis: List of detailed precipitation analysis records
    """
    lat_start, lat_end, lon_start, lon_end = chunk_bounds
    
    # Create date-to-index mapping
    date_to_idx = {pd.to_datetime(date).date(): i for i, date in enumerate(precip_dates)}
    
    # Map percentile thresholds if available
    p10_thresholds = None
    p90_thresholds = None
    
    if f'precip_p{config.drought_percentile}' in precip_percentiles:
        p10_thresholds = map_percentile_thresholds(
            precip_dates, precip_percentiles[f'precip_p{config.drought_percentile}'], config
        )
    
    if f'precip_p{config.intense_precip_percentile}' in precip_percentiles:
        p90_thresholds = map_percentile_thresholds(
            precip_dates, precip_percentiles[f'precip_p{config.intense_precip_percentile}'], config
        )
    
    enhanced_events = []
    precipitation_analysis = []
    
    for idx, event in heatwave_events.iterrows():
        event_id = event['event_id']
        grid_y = int(event['grid_y'])
        grid_x = int(event['grid_x'])
        var = event['var']
        
        # Convert global coordinates to chunk coordinates
        chunk_y = grid_y - lat_start
        chunk_x = grid_x - lon_start
        
        # Skip if outside chunk bounds
        if not (0 <= chunk_y < lat_end - lat_start and 0 <= chunk_x < lon_end - lon_start):
            continue
        
        hw_start_date = pd.to_datetime(event['year_start']).date()
        hw_end_date = pd.to_datetime(event['year_end']).date()
        hw_duration = event['duration_days']
        
        # Get humidity classification
        humidity_info = humidity_events[humidity_events['event_id'] == event_id]
        humidity_label = None
        if len(humidity_info) > 0:
            try:
                if var == "tmax" and 'label_day' in humidity_info.columns:
                    humidity_label = humidity_info.iloc[0]['label_day']
                elif var == "tmin" and 'label_night' in humidity_info.columns:
                    humidity_label = humidity_info.iloc[0]['label_night']
                else:
                    humidity_label = "unknown"
            except (KeyError, IndexError):
                humidity_label = "unknown"
        
        try:
            # =====================================================================
            # DATA AVAILABILITY CHECK
            # =====================================================================
            
            # Check if required dates are available in precipitation data
            pre_hw_start = hw_start_date - timedelta(days=config.pre_heatwave_days)
            max_post_hw_days = max(config.post_heatwave_days)
            post_hw_end = hw_end_date + timedelta(days=max_post_hw_days)
            
            # Check data availability
            earliest_precip_date = precip_dates[0].date() if len(precip_dates) > 0 else None
            latest_precip_date = precip_dates[-1].date() if len(precip_dates) > 0 else None
            
            # Flags for data completeness
            pre_hw_data_complete = earliest_precip_date is not None and pre_hw_start >= earliest_precip_date
            during_hw_data_complete = (earliest_precip_date is not None and latest_precip_date is not None and
                                     hw_start_date >= earliest_precip_date and hw_end_date <= latest_precip_date)
            post_hw_data_complete = latest_precip_date is not None and post_hw_end <= latest_precip_date
            
            # Log warnings for incomplete data
            if not pre_hw_data_complete:
                logger.warning(f"Event {event_id}: Pre-heatwave period extends before available data "
                             f"(needs: {pre_hw_start}, available from: {earliest_precip_date})")
            if not during_hw_data_complete:
                logger.warning(f"Event {event_id}: Heatwave period not fully covered by precipitation data "
                             f"(heatwave: {hw_start_date} to {hw_end_date}, precip: {earliest_precip_date} to {latest_precip_date})")
            if not post_hw_data_complete:
                logger.warning(f"Event {event_id}: Post-heatwave period extends beyond available data "
                             f"(needs: {post_hw_end}, available until: {latest_precip_date})")
            
            # =====================================================================
            # PHASE 1: PRE-HEATWAVE PRECIPITATION ANALYSIS
            # =====================================================================
            
            pre_hw_dates = get_date_range_extended(pre_hw_start, hw_start_date - timedelta(days=1))
            
            # Extract precipitation data for pre-heatwave period (only if data is available)
            pre_hw_precip = []
            pre_hw_p10 = []
            pre_hw_p90 = []
            
            if pre_hw_data_complete:
                for date in pre_hw_dates:
                    if date in date_to_idx:
                        time_idx = date_to_idx[date]
                        precip_val = precip_data[time_idx, chunk_y, chunk_x]
                        pre_hw_precip.append(precip_val)
                        
                        if p10_thresholds is not None:
                            pre_hw_p10.append(p10_thresholds[time_idx, chunk_y, chunk_x])
                        if p90_thresholds is not None:
                            pre_hw_p90.append(p90_thresholds[time_idx, chunk_y, chunk_x])
            else:
                logger.debug(f"Skipping pre-heatwave analysis for event {event_id} due to incomplete data")
            
            pre_hw_precip = np.array(pre_hw_precip)
            
            # Debug information for troubleshooting
            if len(pre_hw_precip) == 0 and pre_hw_data_complete:
                logger.warning(f"No precipitation data found for pre-heatwave period for event {event_id}")
                logger.warning(f"Pre-heatwave dates: {pre_hw_dates[:3]}...{pre_hw_dates[-3:]} (total: {len(pre_hw_dates)})")
                logger.warning(f"Available date range in precip data: {precip_dates[0]} to {precip_dates[-1]}")
            
            pre_hw_metrics = calculate_precipitation_metrics(pre_hw_precip, pre_hw_dates, config)
            
            # Drought and intense precipitation day counts
            pre_hw_drought_days = 0
            pre_hw_intense_days = 0
            if len(pre_hw_p10) > 0 and len(pre_hw_precip) > 0:
                pre_hw_drought_days = np.sum(pre_hw_precip <= np.array(pre_hw_p10))
            if len(pre_hw_p90) > 0 and len(pre_hw_precip) > 0:
                pre_hw_intense_days = np.sum(pre_hw_precip >= np.array(pre_hw_p90))
            
            # =====================================================================
            # PHASE 2: DURING-HEATWAVE PRECIPITATION ANALYSIS
            # =====================================================================
            
            during_hw_dates = get_date_range_extended(hw_start_date, hw_end_date)
            
            during_hw_precip = []
            during_hw_p10 = []
            during_hw_p90 = []
            daily_precip_classes = []
            
            if during_hw_data_complete:
                for date in during_hw_dates:
                    if date in date_to_idx:
                        time_idx = date_to_idx[date]
                        precip_val = precip_data[time_idx, chunk_y, chunk_x]
                        during_hw_precip.append(precip_val)
                        
                        p10_val = p10_thresholds[time_idx, chunk_y, chunk_x] if p10_thresholds is not None else 0
                        p90_val = p90_thresholds[time_idx, chunk_y, chunk_x] if p90_thresholds is not None else 100
                        
                        during_hw_p10.append(p10_val)
                        during_hw_p90.append(p90_val)
                        
                        # Classify daily precipitation
                        precip_class = classify_precipitation_intensity(precip_val, p10_val, p90_val, config)
                        daily_precip_classes.append(precip_class)
            else:
                logger.debug(f"Skipping during-heatwave analysis for event {event_id} due to incomplete data")
            
            during_hw_precip = np.array(during_hw_precip)
            
            # Debug information for troubleshooting
            if len(during_hw_precip) == 0 and during_hw_data_complete:
                logger.warning(f"No precipitation data found for during-heatwave period for event {event_id}")
                logger.warning(f"During-heatwave dates: {during_hw_dates}")
                logger.warning(f"Heatwave period: {hw_start_date} to {hw_end_date}")
            
            during_hw_metrics = calculate_precipitation_metrics(during_hw_precip, during_hw_dates, config)
            
            during_hw_drought_days = 0
            during_hw_intense_days = 0
            if len(during_hw_p10) > 0 and len(during_hw_precip) > 0:
                during_hw_drought_days = np.sum(during_hw_precip <= np.array(during_hw_p10))
            if len(during_hw_p90) > 0 and len(during_hw_precip) > 0:
                during_hw_intense_days = np.sum(during_hw_precip >= np.array(during_hw_p90))
            
            # =====================================================================
            # PHASE 3: POST-HEATWAVE PRECIPITATION ANALYSIS
            # =====================================================================
            
            post_hw_analysis = {}
            
            for period_days in config.post_heatwave_days:
                post_hw_start = hw_end_date + timedelta(days=1)
                post_hw_end = hw_end_date + timedelta(days=period_days)
                post_hw_dates = get_date_range_extended(post_hw_start, post_hw_end)
                
                # Check if this post-heatwave period is within available data
                period_data_complete = (latest_precip_date is not None and 
                                      post_hw_end <= latest_precip_date)
                
                post_hw_precip = []
                post_hw_p10 = []
                post_hw_p90 = []
                
                if period_data_complete:
                    for date in post_hw_dates:
                        if date in date_to_idx:
                            time_idx = date_to_idx[date]
                            precip_val = precip_data[time_idx, chunk_y, chunk_x]
                            post_hw_precip.append(precip_val)
                            
                            if p10_thresholds is not None:
                                post_hw_p10.append(p10_thresholds[time_idx, chunk_y, chunk_x])
                            if p90_thresholds is not None:
                                post_hw_p90.append(p90_thresholds[time_idx, chunk_y, chunk_x])
                else:
                    logger.debug(f"Skipping post-heatwave {period_days}d analysis for event {event_id} "
                               f"due to incomplete data (needs until {post_hw_end}, have until {latest_precip_date})")
                
                post_hw_precip = np.array(post_hw_precip)
                
                # Debug information for troubleshooting
                if len(post_hw_precip) == 0 and period_data_complete:
                    logger.warning(f"No precipitation data found for post-heatwave {period_days}d period for event {event_id}")
                
                post_hw_metrics = calculate_precipitation_metrics(post_hw_precip, post_hw_dates, config)
                
                post_hw_drought_days = 0
                post_hw_intense_days = 0
                if len(post_hw_p10) > 0 and len(post_hw_precip) > 0:
                    post_hw_drought_days = np.sum(post_hw_precip <= np.array(post_hw_p10))
                if len(post_hw_p90) > 0 and len(post_hw_precip) > 0:
                    post_hw_intense_days = np.sum(post_hw_precip >= np.array(post_hw_p90))
                
                # First significant precipitation event
                first_precip_day = None
                first_precip_amount = 0
                for i, precip in enumerate(post_hw_precip):
                    if precip >= config.min_precip_threshold:
                        first_precip_day = i + 1
                        first_precip_amount = precip
                        break
                
                post_hw_analysis[f'post_{period_days}d'] = {
                    **post_hw_metrics,
                    'drought_days': post_hw_drought_days,
                    'intense_precip_days': post_hw_intense_days,
                    'first_precip_day': first_precip_day,
                    'first_precip_amount': first_precip_amount,
                    'data_complete': period_data_complete
                }
            
            # =====================================================================
            # PHASE 4: METEOROLOGICAL PATTERN CLASSIFICATION
            # =====================================================================
            
            # Classify precipitation pattern (only if data is complete)
            if pre_hw_data_complete and during_hw_data_complete:
                if (pre_hw_metrics['consecutive_dry_days'] >= config.pre_heatwave_days and
                    during_hw_metrics['consecutive_dry_days'] >= hw_duration):
                    precip_pattern = "compound_drought_heat"
                elif during_hw_metrics['precip_days'] == 0:
                    precip_pattern = "dry_heatwave"
                elif during_hw_intense_days > 0:
                    precip_pattern = "wet_heatwave"
                else:
                    precip_pattern = "mixed_heatwave"
            else:
                precip_pattern = "incomplete_data"
            
            # Recovery pattern classification (only if relevant post-heatwave data is complete)
            recovery_pattern = "incomplete_data"
            if ('post_3d' in post_hw_analysis and post_hw_analysis['post_3d'].get('data_complete', False) and
                post_hw_analysis['post_3d'].get('first_precip_day') and 
                post_hw_analysis['post_3d']['first_precip_day'] <= 3):
                recovery_pattern = "immediate_recovery"
            elif ('post_7d' in post_hw_analysis and post_hw_analysis['post_7d'].get('data_complete', False) and
                  post_hw_analysis['post_7d'].get('first_precip_day') and 
                  post_hw_analysis['post_7d']['first_precip_day'] <= 7):
                recovery_pattern = "short_term_recovery"
            elif ('post_30d' in post_hw_analysis and post_hw_analysis['post_30d'].get('data_complete', False) and
                  post_hw_analysis['post_30d'].get('first_precip_day') and 
                  post_hw_analysis['post_30d']['first_precip_day'] <= 30):
                recovery_pattern = "delayed_recovery"
            elif ('post_30d' in post_hw_analysis and post_hw_analysis['post_30d'].get('data_complete', False)):
                recovery_pattern = "persistent_drought"
            
            # =====================================================================
            # PHASE 5: COMPOUND EVENT ANALYSIS
            # =====================================================================
            
            # Compound event classification (only if data is complete)
            if precip_pattern != "incomplete_data" and humidity_label:
                if humidity_label == "dry-hot" and precip_pattern == "compound_drought_heat":
                    compound_event_type = "extreme_drought_heat"
                elif humidity_label == "humid-hot" and precip_pattern == "wet_heatwave":
                    compound_event_type = "humid_heat_with_precip"
                elif humidity_label == "humid-hot" and precip_pattern == "dry_heatwave":
                    compound_event_type = "humid_heat_no_precip"
                else:
                    compound_event_type = "standard_heatwave"
            else:
                compound_event_type = "incomplete_data"
            
            # =====================================================================
            # PHASE 6: CREATE ENHANCED EVENT RECORD
            # =====================================================================
            
            enhanced_event = {
                # Original event data
                **event.to_dict(),
                
                # Data completeness flags
                'pre_hw_data_complete': pre_hw_data_complete,
                'during_hw_data_complete': during_hw_data_complete,
                'post_hw_data_complete': post_hw_data_complete,
                
                # Humidity information
                'humidity_label': humidity_label,
                
                # Pre-heatwave precipitation
                'pre_hw_total_precip': pre_hw_metrics['total_precip'],
                'pre_hw_drought_days': pre_hw_drought_days,
                'pre_hw_consecutive_dry_days': pre_hw_metrics['consecutive_dry_days'],
                'pre_hw_precip_frequency': pre_hw_metrics['precip_frequency'],
                
                # During-heatwave precipitation
                'during_hw_total_precip': during_hw_metrics['total_precip'],
                'during_hw_max_daily_precip': during_hw_metrics['max_daily_precip'],
                'during_hw_precip_days': during_hw_metrics['precip_days'],
                'during_hw_drought_days': during_hw_drought_days,
                'during_hw_intense_days': during_hw_intense_days,
                'during_hw_consecutive_dry_days': during_hw_metrics['consecutive_dry_days'],
                
                # Post-heatwave precipitation (flatten nested dictionary)
                **{f"{key}_{metric}": value 
                   for key, analysis in post_hw_analysis.items() 
                   for metric, value in analysis.items()},
                
                # Pattern classifications
                'precip_pattern': precip_pattern,
                'recovery_pattern': recovery_pattern,
                'compound_event_type': compound_event_type
            }
            
            enhanced_events.append(enhanced_event)
            
            # =====================================================================
            # PHASE 7: DETAILED PRECIPITATION ANALYSIS RECORD
            # =====================================================================
            
            precip_record = {
                'event_id': event_id,
                'grid_y': grid_y,
                'grid_x': grid_x,
                'year': event['year'],
                'var': var,
                'humidity_label': humidity_label,
                'hw_start_date': hw_start_date,
                'hw_end_date': hw_end_date,
                'hw_duration': hw_duration,
                
                # Time series data
                'pre_hw_precip_series': pre_hw_precip.tolist(),
                'during_hw_precip_series': during_hw_precip.tolist(),
                'daily_precip_classes': daily_precip_classes,
                
                # Summary classifications
                'precip_pattern': precip_pattern,
                'recovery_pattern': recovery_pattern,
                'compound_event_type': compound_event_type
            }
            
            precipitation_analysis.append(precip_record)
            
        except Exception as e:
            logger.error(f"Error processing event {event_id}: {e}")
            continue
    
    return enhanced_events, precipitation_analysis

def calculate_annual_precipitation_heatwave_stats(enhanced_events, year, chunk_bounds):
    """
    Calculate annual aggregations by precipitation-heatwave pattern for each grid cell.
    
    Args:
        enhanced_events: List of enhanced event records
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
        # Heatwave-precipitation pattern counts
        'compound_drought_heat_events': np.zeros((n_lat, n_lon)),
        'dry_heatwave_events': np.zeros((n_lat, n_lon)),
        'wet_heatwave_events': np.zeros((n_lat, n_lon)),
        'mixed_heatwave_events': np.zeros((n_lat, n_lon)),
        
        # Recovery pattern counts
        'immediate_recovery_events': np.zeros((n_lat, n_lon)),
        'short_term_recovery_events': np.zeros((n_lat, n_lon)),
        'delayed_recovery_events': np.zeros((n_lat, n_lon)),
        'persistent_drought_events': np.zeros((n_lat, n_lon)),
        
        # Compound event counts by humidity type
        'extreme_drought_heat_events': np.zeros((n_lat, n_lon)),
        'humid_heat_no_precip_events': np.zeros((n_lat, n_lon)),
        'humid_heat_with_precip_events': np.zeros((n_lat, n_lon)),
        
        # Precipitation statistics during heatwaves
        'mean_precip_during_heatwaves': np.zeros((n_lat, n_lon)),
        'max_precip_during_heatwaves': np.zeros((n_lat, n_lon)),
        'mean_dry_days_during_heatwaves': np.zeros((n_lat, n_lon)),
        
        # Recovery precipitation statistics
        'mean_precip_3d_after_heatwave': np.zeros((n_lat, n_lon)),
        'mean_precip_7d_after_heatwave': np.zeros((n_lat, n_lon)),
        'mean_days_to_first_precip': np.zeros((n_lat, n_lon))
    }
    
    # Process each grid cell
    for i in range(n_lat):
        for j in range(n_lon):
            global_i = lat_start + i
            global_j = lon_start + j
            
            # Filter events for this grid cell and year
            cell_events = [e for e in enhanced_events 
                          if e['grid_y'] == global_i and e['grid_x'] == global_j and e['year'] == year]
            
            if len(cell_events) > 0:
                # Count pattern types
                aggregations['compound_drought_heat_events'][i, j] = len(
                    [e for e in cell_events if e['precip_pattern'] == "compound_drought_heat"]
                )
                aggregations['dry_heatwave_events'][i, j] = len(
                    [e for e in cell_events if e['precip_pattern'] == "dry_heatwave"]
                )
                aggregations['wet_heatwave_events'][i, j] = len(
                    [e for e in cell_events if e['precip_pattern'] == "wet_heatwave"]
                )
                aggregations['mixed_heatwave_events'][i, j] = len(
                    [e for e in cell_events if e['precip_pattern'] == "mixed_heatwave"]
                )
                
                # Count recovery patterns
                aggregations['immediate_recovery_events'][i, j] = len(
                    [e for e in cell_events if e['recovery_pattern'] == "immediate_recovery"]
                )
                aggregations['short_term_recovery_events'][i, j] = len(
                    [e for e in cell_events if e['recovery_pattern'] == "short_term_recovery"]
                )
                aggregations['delayed_recovery_events'][i, j] = len(
                    [e for e in cell_events if e['recovery_pattern'] == "delayed_recovery"]
                )
                aggregations['persistent_drought_events'][i, j] = len(
                    [e for e in cell_events if e['recovery_pattern'] == "persistent_drought"]
                )
                
                # Count compound events
                aggregations['extreme_drought_heat_events'][i, j] = len(
                    [e for e in cell_events if e['compound_event_type'] == "extreme_drought_heat"]
                )
                aggregations['humid_heat_no_precip_events'][i, j] = len(
                    [e for e in cell_events if e['compound_event_type'] == "humid_heat_no_precip"]
                )
                aggregations['humid_heat_with_precip_events'][i, j] = len(
                    [e for e in cell_events if e['compound_event_type'] == "humid_heat_with_precip"]
                )
                
                # Calculate precipitation statistics
                during_precip = [e['during_hw_total_precip'] for e in cell_events if not np.isnan(e['during_hw_total_precip'])]
                max_precip = [e['during_hw_max_daily_precip'] for e in cell_events if not np.isnan(e['during_hw_max_daily_precip'])]
                dry_days = [e['during_hw_drought_days'] for e in cell_events if not np.isnan(e['during_hw_drought_days'])]
                
                aggregations['mean_precip_during_heatwaves'][i, j] = np.mean(during_precip) if during_precip else 0
                aggregations['max_precip_during_heatwaves'][i, j] = np.max(max_precip) if max_precip else 0
                aggregations['mean_dry_days_during_heatwaves'][i, j] = np.mean(dry_days) if dry_days else 0
                
                # Calculate recovery statistics
                precip_3d = [e['post_3d_total_precip'] for e in cell_events if not np.isnan(e['post_3d_total_precip'])]
                precip_7d = [e['post_7d_total_precip'] for e in cell_events if not np.isnan(e['post_7d_total_precip'])]
                days_to_first = [e['post_3d_first_precip_day'] for e in cell_events 
                               if e['post_3d_first_precip_day'] is not None and not np.isnan(e['post_3d_first_precip_day'])]
                
                aggregations['mean_precip_3d_after_heatwave'][i, j] = np.mean(precip_3d) if precip_3d else 0
                aggregations['mean_precip_7d_after_heatwave'][i, j] = np.mean(precip_7d) if precip_7d else 0
                aggregations['mean_days_to_first_precip'][i, j] = np.mean(days_to_first) if days_to_first else np.nan
    
    return aggregations

def process_heatwave_precipitation_chunk(chunk_info, year, heatwave_dir, humidity_dir, 
                                       precip_dir, precip_percentile_file, config, variables):
    """
    Process heatwave-precipitation analysis for a spatial chunk.
    
    Args:
        chunk_info: (lat_start, lat_end, lon_start, lon_end)
        year: Target year
        heatwave_dir: Directory containing heatwave results
        humidity_dir: Directory containing humidity classification results
        precip_dir: Directory containing precipitation data
        precip_percentile_file: Path to precipitation percentile file
        config: PrecipitationAnalysisConfig
        variables: List of variables to process
        
    Returns:
        Dictionary with processed results
    """
    lat_start, lat_end, lon_start, lon_end = chunk_info
    
    logger.info(f"Processing precipitation chunk: year={year}, lat[{lat_start}:{lat_end}], lon[{lon_start}:{lon_end}]")
    
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
        
        if not all_events:
            logger.warning(f"No heatwave data found for chunk lat[{lat_start}:{lat_end}], lon[{lon_start}:{lon_end}]")
            return None
        
        # Combine heatwave data
        heatwave_events = pd.concat(all_events, ignore_index=True)
        heatwave_days = pd.concat(all_days, ignore_index=True)
        
        # Load humidity classification data
        humidity_events_file = Path(humidity_dir) / f'heatwave_events_humidity_{year}.parquet'
        humidity_events = pd.DataFrame()
        
        if humidity_events_file.exists():
            humidity_events = pd.read_parquet(humidity_events_file)
            # Filter to chunk
            humidity_events = humidity_events[
                (humidity_events['grid_y'] >= lat_start) & 
                (humidity_events['grid_y'] < lat_end) &
                (humidity_events['grid_x'] >= lon_start) & 
                (humidity_events['grid_x'] < lon_end)
            ]
        
        # Load precipitation data
        precip_data, precip_dates = load_precipitation_data_chunk(year, precip_dir, chunk_info)
        
        if precip_data is None:
            logger.warning(f"No precipitation data found for chunk lat[{lat_start}:{lat_end}], lon[{lon_start}:{lon_end}]")
            return None
        
        # Load precipitation percentiles
        precip_percentiles = {}
        if precip_percentile_file and Path(precip_percentile_file).exists():
            precip_percentiles = load_precipitation_percentiles(precip_percentile_file, chunk_info)
        
        # Run main analysis
        enhanced_events, precipitation_analysis = analyze_heatwave_precipitation_interactions(
            heatwave_events, heatwave_days, humidity_events,
            precip_data, precip_dates, precip_percentiles,
            chunk_info, config
        )
        
        # Calculate annual aggregations
        annual_aggs = calculate_annual_precipitation_heatwave_stats(enhanced_events, year, chunk_info)
        
        logger.info(f"Completed precipitation chunk: year={year}, lat[{lat_start}:{lat_end}], lon[{lon_start}:{lon_end}]")
        
        return {
            'chunk_bounds': chunk_info,
            'enhanced_events': enhanced_events,
            'precipitation_analysis': precipitation_analysis,
            'annual_aggregations': annual_aggs
        }
        
    except Exception as e:
        logger.error(f"Error processing precipitation chunk lat[{lat_start}:{lat_end}], lon[{lon_start}:{lon_end}]: {e}")
        return None

def combine_precipitation_results(chunk_results, full_lat, full_lon, year, output_dir, config):
    """
    Combine chunk results and save final outputs.
    
    Args:
        chunk_results: List of chunk processing results
        full_lat: Full latitude array
        full_lon: Full longitude array
        year: Target year
        output_dir: Output directory
        config: PrecipitationAnalysisConfig
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Combine enhanced events and precipitation analysis
    all_enhanced_events = []
    all_precipitation_analysis = []
    
    # Initialize aggregation grids
    n_lat, n_lon = len(full_lat), len(full_lon)
    aggregation_grids = {}
    
    # Define all aggregation variables
    agg_vars = [
        'compound_drought_heat_events', 'dry_heatwave_events', 'wet_heatwave_events', 'mixed_heatwave_events',
        'immediate_recovery_events', 'short_term_recovery_events', 'delayed_recovery_events', 'persistent_drought_events',
        'extreme_drought_heat_events', 'humid_heat_no_precip_events', 'humid_heat_with_precip_events',
        'mean_precip_during_heatwaves', 'max_precip_during_heatwaves', 'mean_dry_days_during_heatwaves',
        'mean_precip_3d_after_heatwave', 'mean_precip_7d_after_heatwave', 'mean_days_to_first_precip'
    ]
    
    for var in agg_vars:
        if 'mean' in var or 'max' in var:
            aggregation_grids[var] = np.full((n_lat, n_lon), np.nan)
        else:
            aggregation_grids[var] = np.zeros((n_lat, n_lon))
    
    # Combine results from all chunks
    for result in chunk_results:
        if result is None:
            continue
        
        # Combine events and analysis
        all_enhanced_events.extend(result['enhanced_events'])
        all_precipitation_analysis.extend(result['precipitation_analysis'])
        
        # Combine aggregations
        lat_start, lat_end, lon_start, lon_end = result['chunk_bounds']
        
        for var in agg_vars:
            if var in result['annual_aggregations']:
                aggregation_grids[var][lat_start:lat_end, lon_start:lon_end] = result['annual_aggregations'][var]
    
    # Save enhanced events
    if all_enhanced_events:
        enhanced_events_df = pd.DataFrame(all_enhanced_events)
        events_file = output_dir / f'heatwave_events_precipitation_{year}.parquet'
        enhanced_events_df.to_parquet(events_file)
        logger.info(f"Saved enhanced events: {events_file} ({len(all_enhanced_events)} events)")
    
    # Save detailed precipitation analysis
    if all_precipitation_analysis:
        precip_analysis_df = pd.DataFrame(all_precipitation_analysis)
        analysis_file = output_dir / f'heatwave_precipitation_analysis_{year}.parquet'
        precip_analysis_df.to_parquet(analysis_file)
        logger.info(f"Saved precipitation analysis: {analysis_file} ({len(all_precipitation_analysis)} records)")
    
    # Save aggregation grids
    agg_ds = xr.Dataset({
        var: (['latitude', 'longitude'], aggregation_grids[var])
        for var in agg_vars
    }, coords={
        'latitude': full_lat,
        'longitude': full_lon
    })
    
    # Add attributes
    agg_ds.attrs['pre_heatwave_days'] = config.pre_heatwave_days
    agg_ds.attrs['post_heatwave_days'] = config.post_heatwave_days
    agg_ds.attrs['min_precip_threshold_mm'] = config.min_precip_threshold
    agg_ds.attrs['drought_percentile'] = config.drought_percentile
    agg_ds.attrs['intense_precip_percentile'] = config.intense_precip_percentile
    
    # Add variable attributes
    for var in agg_vars:
        if 'events' in var:
            agg_ds[var].attrs = {'long_name': f'Number of events ({var})', 'units': 'count'}
        elif 'mean_precip' in var or 'max_precip' in var:
            agg_ds[var].attrs = {'long_name': f'Precipitation ({var})', 'units': 'mm'}
        elif 'days' in var:
            agg_ds[var].attrs = {'long_name': f'Days ({var})', 'units': 'days'}
    
    agg_file = output_dir / f'heatwave_precipitation_aggregations_{year}.nc'
    agg_ds.to_netcdf(agg_file)
    agg_ds.close()
    
    logger.info(f"Saved precipitation aggregations: {agg_file}")

def create_spatial_chunks(n_lat, n_lon, chunk_size_lat=50, chunk_size_lon=100):
    """Create spatial chunks for parallel processing."""
    chunks = []
    for lat_start in range(0, n_lat, chunk_size_lat):
        lat_end = min(lat_start + chunk_size_lat, n_lat)
        for lon_start in range(0, n_lon, chunk_size_lon):
            lon_end = min(lon_start + chunk_size_lon, n_lon)
            chunks.append((lat_start, lat_end, lon_start, lon_end))
    return chunks

def main():
    """Main function for heatwave-precipitation analysis."""
    parser = argparse.ArgumentParser(description='Heatwave-Precipitation Interaction Analysis System')
    
    # Basic parameters
    parser.add_argument('--start-year', type=int, default=1980, help='Start year')
    parser.add_argument('--end-year', type=int, default=2024, help='End year')
    parser.add_argument('--heatwave-dir', default='data/processed/heatwave_metrics',
                       help='Directory containing heatwave results')
    parser.add_argument('--humidity-dir', default='data/processed/humidity_classification',
                       help='Directory containing humidity classification results')
    parser.add_argument('--precip-dir', default='/data/climate/disk1/datasets/era5',
                       help='Directory containing precipitation data')
    parser.add_argument('--precip-percentile-file', 
                       help='Path to precipitation percentile file')
    parser.add_argument('--output-dir', default='data/processed/precipitation_analysis',
                       help='Output directory')
    
    # Analysis parameters
    parser.add_argument('--pre-heatwave-days', type=int, default=7,
                       help='Days before heatwave to analyze')
    parser.add_argument('--post-heatwave-days', nargs='+', type=int, 
                       default=[1, 3, 5, 7, 10, 15, 30],
                       help='Post-heatwave analysis periods (days)')
    parser.add_argument('--min-precip-threshold', type=float, default=1.0,
                       help='Minimum precipitation threshold (mm/day)')
    parser.add_argument('--drought-percentile', type=int, default=10,
                       help='Percentile for drought definition')
    parser.add_argument('--intense-precip-percentile', type=int, default=90,
                       help='Percentile for intense precipitation')
    
    # Processing parameters
    parser.add_argument('--variables', nargs='+', default=['tmax', 'tmin'],
                       help='Variables to process')
    parser.add_argument('--n-processes', type=int, default=24,
                       help='Number of processes')
    parser.add_argument('--chunk-size-lat', type=int, default=50,
                       help='Latitude chunk size')
    parser.add_argument('--chunk-size-lon', type=int, default=100,
                       help='Longitude chunk size')
    
    args = parser.parse_args()
    
    # Create configuration
    config = PrecipitationAnalysisConfig(
        pre_heatwave_days=args.pre_heatwave_days,
        post_heatwave_days=args.post_heatwave_days,
        min_precip_threshold=args.min_precip_threshold,
        drought_percentile=args.drought_percentile,
        intense_precip_percentile=args.intense_precip_percentile
    )
    
    logger.info("="*80)
    logger.info("HEATWAVE-PRECIPITATION INTERACTION ANALYSIS SYSTEM")
    logger.info("="*80)
    logger.info(f"Analysis period: {args.start_year}-{args.end_year}")
    logger.info(f"Variables: {args.variables}")
    logger.info(f"Pre-heatwave days: {config.pre_heatwave_days}")
    logger.info(f"Post-heatwave days: {config.post_heatwave_days}")
    logger.info(f"Min precipitation threshold: {config.min_precip_threshold} mm/day")
    logger.info(f"Drought percentile: P{config.drought_percentile}")
    logger.info(f"Intense precipitation percentile: P{config.intense_precip_percentile}")
    logger.info("="*80)
    
    # Validate directories
    heatwave_dir = Path(args.heatwave_dir)
    humidity_dir = Path(args.humidity_dir)
    precip_dir = Path(args.precip_dir)
    
    if not heatwave_dir.exists():
        raise ValueError(f"Heatwave directory does not exist: {heatwave_dir}")
    
    if not humidity_dir.exists():
        logger.warning(f"Humidity directory does not exist: {humidity_dir}")
    
    if not precip_dir.exists():
        raise ValueError(f"Precipitation directory does not exist: {precip_dir}")
    
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
    spatial_chunks = create_spatial_chunks(
        len(full_lat), len(full_lon),
        args.chunk_size_lat, args.chunk_size_lon
    )
    
    logger.info(f"Created {len(spatial_chunks)} spatial chunks")
    
    # Process each year
    years = list(range(args.start_year, args.end_year + 1))
    
    for year in years:
        logger.info(f"\n{'='*20} PROCESSING YEAR {year} {'='*20}")
        
        # Process chunks in parallel
        process_func = partial(
            process_heatwave_precipitation_chunk,
            year=year,
            heatwave_dir=args.heatwave_dir,
            humidity_dir=args.humidity_dir,
            precip_dir=args.precip_dir,
            precip_percentile_file=args.precip_percentile_file,
            config=config,
            variables=args.variables
        )
        
        logger.info(f"Processing {len(spatial_chunks)} chunks with {args.n_processes} processes...")
        
        with mp.Pool(args.n_processes) as pool:
            chunk_results = pool.map(process_func, spatial_chunks)
        
        # Combine results
        combine_precipitation_results(chunk_results, full_lat, full_lon, year, args.output_dir, config)
    
    logger.info("\n" + "="*80)
    logger.info("HEATWAVE-PRECIPITATION ANALYSIS COMPLETED!")
    logger.info("="*80)

if __name__ == "__main__":
    main()