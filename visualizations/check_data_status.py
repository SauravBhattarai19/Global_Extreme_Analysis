#!/usr/bin/env python3
"""
Check Data Status for Humid Heat Analysis

This script checks what data is available and suggests which visualizations can be run.
It provides a quick overview of your analysis pipeline progress.
"""

import sys
from pathlib import Path
import argparse

def check_component_status():
    """Check status of each analysis component."""
    
    print("="*80)
    print("HUMID HEAT ANALYSIS - DATA STATUS CHECK")
    print("="*80)
    
    components = [
        {
            'name': '00. Precipitation Percentiles',
            'script': '00_calculate_precipitation_percentiles.py',
            'data_paths': ['data/processed/precipitation_percentiles.nc'],
            'source_data': '/data/climate/disk1/datasets/era5/era5_daily_*.nc',
            'viz_script': 'viz_00_precipitation_percentiles.py'
        },
        {
            'name': '01. Temperature Percentiles',
            'script': '01_calculate_percentiles.py', 
            'data_paths': ['data/processed/percentiles/tmax_p95_*.nc', 'data/processed/percentiles/tmin_p95_*.nc'],
            'source_data': '/data/climate/disk3/datasets/era5/era5_daily_*.nc',
            'viz_script': 'viz_01_temperature_percentiles.py'
        },
        {
            'name': '02. Relative Humidity',
            'script': '02_calculate_RH.py',
            'data_paths': ['data/processed/relative_humidity/era5_rh_*.nc'],
            'source_data': '/data/climate/disk3/datasets/era5/era5_daily_*.nc (t2m, d2m)',
            'viz_script': 'viz_02_relative_humidity.py'
        },
        {
            'name': '03. RH Percentiles', 
            'script': '03_calculate_RH_percentiles.py',
            'data_paths': ['data/processed/rh_percentiles.nc'],
            'source_data': 'data/processed/relative_humidity/',
            'viz_script': 'viz_03_rh_percentiles.py'
        },
        {
            'name': '04. Heat Index',
            'script': '04_Heat_Index.py',
            'data_paths': ['data/processed/heat_index/era5_heat_index_*.nc'],
            'source_data': 'data/processed/relative_humidity/ + temperature data',
            'viz_script': 'viz_04_heat_index.py'
        },
        {
            'name': '05. Heatwave Metrics',
            'script': '05_Heatwave_Metrics.py',
            'data_paths': [
                'data/processed/heatwave_metrics/heatwave_metrics_*.nc',
                'data/processed/heatwave_metrics/heatwave_events_*.parquet'
            ],
            'source_data': 'data/processed/percentiles/ + temperature data',
            'viz_script': 'viz_05_heatwave_metrics.py'
        },
        {
            'name': '06. Humidity Classification',
            'script': '06_Humidity_Classification.py',
            'data_paths': [
                'data/processed/humidity_classification/heatwave_events_humidity_*.parquet',
                'data/processed/humidity_classification/heatwave_humidity_aggregations_*.nc'
            ],
            'source_data': 'heatwave_metrics/ + relative_humidity/',
            'viz_script': 'viz_06_humidity_classification.py'
        },
        {
            'name': '07. Precipitation Analysis',
            'script': '07_Heatwave_Precipitation_Analysis.py',
            'data_paths': [
                'data/processed/precipitation_analysis/heatwave_events_precipitation_*.parquet',
                'data/processed/precipitation_analysis/heatwave_precipitation_aggregations_*.nc'
            ],
            'source_data': 'heatwave_metrics/ + humidity_classification/ + precipitation data',
            'viz_script': 'viz_07_precipitation_analysis.py'
        }
    ]
    
    ready_for_viz = []
    
    for component in components:
        print(f"\n{component['name']}:")
        print(f"  Script: {component['script']}")
        print(f"  Source: {component['source_data']}")
        
        # Check if output data exists
        data_exists = False
        existing_files = []
        
        for data_path in component['data_paths']:
            files = list(Path('.').glob(data_path))
            if files:
                data_exists = True
                existing_files.extend(files)
        
        if data_exists:
            print(f"  Status: ✅ READY FOR VISUALIZATION")
            print(f"  Data files: {len(existing_files)} found")
            print(f"  Visualization: {component['viz_script']}")
            ready_for_viz.append(component)
        else:
            print(f"  Status: ❌ NO OUTPUT DATA FOUND")
            print(f"  Action: Run {component['script']} first")
    
    # Summary
    print(f"\n{'='*80}")
    print("VISUALIZATION READINESS SUMMARY")
    print(f"{'='*80}")
    print(f"Ready for visualization: {len(ready_for_viz)}/{len(components)} components")
    
    if ready_for_viz:
        print(f"\n🎉 You can visualize these components:")
        for comp in ready_for_viz:
            print(f"  ✅ {comp['name']} → {comp['viz_script']}")
        
        print(f"\n🚀 Quick commands:")
        print(f"# Run all available visualizations:")
        print(f"python visualizations/run_all_visualizations.py --skip-failed")
        
        print(f"\n# Run specific component:")
        if len(ready_for_viz) > 0:
            example_comp = ready_for_viz[0]
            print(f"python visualizations/{example_comp['viz_script']}")
    
    else:
        print(f"\n⏳ No components ready for visualization yet.")
        print(f"Run the analysis scripts first to generate data.")
    
    # Check source data availability
    print(f"\n{'='*40}")
    print("SOURCE DATA CHECK")
    print(f"{'='*40}")
    
    # Temperature data
    temp_files = list(Path('/data/climate/disk3/datasets/era5').glob('era5_daily_*.nc'))
    print(f"Temperature data: {len(temp_files)} files ({'✅ Available' if temp_files else '❌ Missing'})")
    
    # Precipitation data  
    precip_files = list(Path('/data/climate/disk1/datasets/era5').glob('era5_daily_*.nc'))
    print(f"Precipitation data: {len(precip_files)} files ({'✅ Available' if precip_files else '❌ Missing'})")
    
    return len(ready_for_viz)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Check data status for humid heat analysis')
    parser.add_argument('--verbose', action='store_true', help='Show detailed file listings')
    
    args = parser.parse_args()
    
    ready_count = check_component_status()
    
    if ready_count > 0:
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())
