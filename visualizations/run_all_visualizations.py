#!/usr/bin/env python3
"""
Master Visualization Script for Humid Heat Analysis

This script runs all visualization components in the correct order and creates
a comprehensive set of scientific plots for the humid heat analysis pipeline.

It automatically detects available data and creates appropriate visualizations
for each component of the analysis.
"""

import sys
import argparse
import subprocess
from pathlib import Path
import json
from datetime import datetime

def check_data_availability(data_dirs):
    """Check which data components are available."""
    available_components = {}
    
    # 0. Precipitation percentiles
    precip_pct_file = Path(data_dirs.get('precipitation_percentiles', 'data/processed/precipitation_percentiles.nc'))
    available_components['precipitation_percentiles'] = precip_pct_file.exists()
    
    # 1. Temperature percentiles
    percentile_dir = Path(data_dirs.get('percentiles', 'data/processed/percentiles'))
    tmax_files = list(percentile_dir.glob('tmax_p95_*.nc'))
    tmin_files = list(percentile_dir.glob('tmin_p95_*.nc'))
    available_components['temperature_percentiles'] = len(tmax_files) > 0 and len(tmin_files) > 0
    
    # 2. Relative humidity data
    rh_dir = Path(data_dirs.get('rh', 'data/processed/relative_humidity'))
    rh_files = list(rh_dir.glob('era5_rh_*.nc'))
    available_components['relative_humidity'] = len(rh_files) > 0
    
    # 3. RH percentiles
    rh_pct_file = Path(data_dirs.get('rh_percentiles', 'data/processed/rh_percentiles.nc'))
    available_components['rh_percentiles'] = rh_pct_file.exists()
    
    # 4. Heat index
    hi_dir = Path(data_dirs.get('heat_index', 'data/processed/heat_index'))
    hi_files = list(hi_dir.glob('era5_heat_index_*.nc'))
    available_components['heat_index'] = len(hi_files) > 0
    
    # 5. Heatwave metrics
    hw_dir = Path(data_dirs.get('heatwave_metrics', 'data/processed/heatwave_metrics'))
    hw_metric_files = list(hw_dir.glob('heatwave_metrics_*.nc'))
    hw_event_files = list(hw_dir.glob('heatwave_events_*.parquet'))
    available_components['heatwave_metrics'] = len(hw_metric_files) > 0 or len(hw_event_files) > 0
    
    # 6. Humidity classification
    humidity_dir = Path(data_dirs.get('humidity_classification', 'data/processed/humidity_classification'))
    humidity_files = list(humidity_dir.glob('heatwave_events_humidity_*.parquet'))
    available_components['humidity_classification'] = len(humidity_files) > 0
    
    # 7. Precipitation analysis
    precip_analysis_dir = Path(data_dirs.get('precipitation_analysis', 'data/processed/precipitation_analysis'))
    precip_event_files = list(precip_analysis_dir.glob('heatwave_events_precipitation_*.parquet'))
    available_components['precipitation_analysis'] = len(precip_event_files) > 0
    
    return available_components

def run_visualization(script_name, args, output_dir):
    """Run a visualization script with error handling."""
    script_path = Path(__file__).parent / script_name
    
    if not script_path.exists():
        print(f"❌ Script not found: {script_name}")
        return False
    
    cmd = [sys.executable, str(script_path)] + args
    
    try:
        print(f"🔄 Running {script_name}...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
        
        if result.returncode == 0:
            print(f"✅ {script_name} completed successfully")
            return True
        else:
            print(f"❌ {script_name} failed with return code {result.returncode}")
            print(f"Error output: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {script_name} timed out after 30 minutes")
        return False
    except Exception as e:
        print(f"❌ Error running {script_name}: {e}")
        return False

def create_visualization_summary(output_base_dir, results):
    """Create a summary HTML file with links to all visualizations."""
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Humid Heat Analysis - Visualization Summary</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .component {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
        .success {{ background-color: #d4edda; }}
        .failed {{ background-color: #f8d7da; }}
        .skipped {{ background-color: #fff3cd; }}
        h1, h2 {{ color: #333; }}
        ul {{ list-style-type: none; padding: 0; }}
        li {{ margin: 5px 0; }}
        a {{ text-decoration: none; color: #007bff; }}
        a:hover {{ text-decoration: underline; }}
        .timestamp {{ font-style: italic; color: #666; }}
    </style>
</head>
<body>
    <h1>Humid Heat Analysis - Visualization Summary</h1>
    <p class="timestamp">Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <h2>Overview</h2>
    <p>This page provides links to all generated visualizations from the humid heat analysis pipeline.</p>
"""
    
    components = [
        ('01_temperature_percentiles', 'Temperature Percentiles', 'viz_01_temperature_percentiles.py'),
        ('02_relative_humidity', 'Relative Humidity', 'viz_02_relative_humidity.py'),
        ('03_rh_percentiles', 'RH Percentiles', 'viz_03_rh_percentiles.py'),
        ('04_heat_index', 'Heat Index', 'viz_04_heat_index.py'),
        ('05_heatwave_metrics', 'Heatwave Metrics', 'viz_05_heatwave_metrics.py'),
        ('06_humidity_classification', 'Humidity Classification', 'viz_06_humidity_classification.py')
    ]
    
    for comp_id, comp_name, script_name in components:
        status = results.get(script_name, 'skipped')
        status_class = 'success' if status == 'success' else 'failed' if status == 'failed' else 'skipped'
        
        html_content += f"""
    <div class="component {status_class}">
        <h3>{comp_name}</h3>
        <p>Status: <strong>{status.title()}</strong></p>
"""
        
        if status == 'success':
            output_dir = output_base_dir / comp_id
            if output_dir.exists():
                png_files = list(output_dir.glob('*.png'))
                if png_files:
                    html_content += "        <p>Generated plots:</p>\n        <ul>\n"
                    for png_file in sorted(png_files):
                        rel_path = png_file.relative_to(output_base_dir)
                        html_content += f'            <li><a href="{rel_path}" target="_blank">{png_file.name}</a></li>\n'
                    html_content += "        </ul>\n"
        
        html_content += "    </div>\n"
    
    html_content += """
    <h2>Usage Instructions</h2>
    <p>Click on any plot name above to view the visualization in a new tab. 
    All plots are saved as high-resolution PNG files suitable for publication.</p>
    
    <h2>Analysis Components</h2>
    <ul>
        <li><strong>Temperature Percentiles:</strong> Climatological temperature thresholds for heatwave detection</li>
        <li><strong>Relative Humidity:</strong> Humidity patterns and quality assessment</li>
        <li><strong>RH Percentiles:</strong> Humidity thresholds for dry/humid classification</li>
        <li><strong>Heat Index:</strong> Apparent temperature combining heat and humidity</li>
        <li><strong>Heatwave Metrics:</strong> Frequency, intensity, and duration of heatwaves</li>
        <li><strong>Humidity Classification:</strong> Dry vs humid heatwave characterization</li>
    </ul>
</body>
</html>
"""
    
    summary_file = output_base_dir / 'visualization_summary.html'
    with open(summary_file, 'w') as f:
        f.write(html_content)
    
    print(f"📄 Created visualization summary: {summary_file}")

def main():
    """Main function to run all visualizations."""
    parser = argparse.ArgumentParser(description='Run all humid heat visualizations')
    
    # Data directories
    parser.add_argument('--precipitation-percentiles-file', default='data/processed/precipitation_percentiles.nc',
                       help='Precipitation percentiles file')
    parser.add_argument('--percentiles-dir', default='data/processed/percentiles',
                       help='Temperature percentiles directory')
    parser.add_argument('--rh-dir', default='data/processed/relative_humidity',
                       help='Relative humidity directory')
    parser.add_argument('--rh-percentiles-file', default='data/processed/rh_percentiles.nc',
                       help='RH percentiles file')
    parser.add_argument('--heat-index-dir', default='data/processed/heat_index',
                       help='Heat index directory')
    parser.add_argument('--heatwave-dir', default='data/processed/heatwave_metrics',
                       help='Heatwave metrics directory')
    parser.add_argument('--humidity-dir', default='data/processed/humidity_classification',
                       help='Humidity classification directory')
    parser.add_argument('--precipitation-analysis-dir', default='data/processed/precipitation_analysis',
                       help='Precipitation analysis directory')
    
    # Output options
    parser.add_argument('--output-dir', default='visualizations/output',
                       help='Base output directory for all visualizations')
    parser.add_argument('--create-seasonal', action='store_true',
                       help='Create seasonal plots where applicable')
    parser.add_argument('--years', nargs='+', type=int,
                       help='Specific years to analyze')
    parser.add_argument('--variables', nargs='+', default=['tmax', 'tmin'],
                       help='Variables to analyze')
    
    # Processing options
    parser.add_argument('--skip-failed', action='store_true',
                       help='Continue even if some visualizations fail')
    parser.add_argument('--components', nargs='+', 
                       choices=['precipitation_percentiles', 'temperature_percentiles', 'relative_humidity', 
                               'rh_percentiles', 'heat_index', 'heatwave_metrics', 'humidity_classification',
                               'precipitation_analysis'],
                       help='Specific components to visualize (default: all available)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("HUMID HEAT ANALYSIS - MASTER VISUALIZATION")
    print("="*80)
    print(f"Output directory: {args.output_dir}")
    print(f"Variables: {args.variables}")
    if args.years:
        print(f"Years: {args.years}")
    print("="*80)
    
    # Create output directory
    output_base_dir = Path(args.output_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    # Check data availability
    data_dirs = {
        'precipitation_percentiles': args.precipitation_percentiles_file,
        'percentiles': args.percentiles_dir,
        'rh': args.rh_dir,
        'rh_percentiles': args.rh_percentiles_file,
        'heat_index': args.heat_index_dir,
        'heatwave_metrics': args.heatwave_dir,
        'humidity_classification': args.humidity_dir,
        'precipitation_analysis': args.precipitation_analysis_dir
    }
    
    available = check_data_availability(data_dirs)
    
    print("\nData availability check:")
    for component, is_available in available.items():
        status = "✅ Available" if is_available else "❌ Not available"
        print(f"  {component}: {status}")
    
    # Determine which components to run
    if args.components:
        components_to_run = args.components
    else:
        components_to_run = [comp for comp, avail in available.items() if avail]
    
    print(f"\nComponents to visualize: {components_to_run}")
    
    # Define visualization scripts and their arguments
    viz_scripts = {}
    
    viz_scripts['precipitation_percentiles'] = {
        'script': 'viz_00_precipitation_percentiles.py',
        'args': [
            '--percentile-file', args.precipitation_percentiles_file,
            '--output-dir', str(output_base_dir / '00_precipitation_percentiles')
        ]
    }
    
    viz_scripts['temperature_percentiles'] = {
        'script': 'viz_01_temperature_percentiles.py',
        'args': [
            '--percentile-dir', args.percentiles_dir,
            '--output-dir', str(output_base_dir / '01_temperature_percentiles')
        ]
    }
    
    viz_scripts['relative_humidity'] = {
        'script': 'viz_02_relative_humidity.py',
        'args': [
            '--rh-dir', args.rh_dir,
            '--output-dir', str(output_base_dir / '02_relative_humidity')
        ]
    }
    
    viz_scripts['rh_percentiles'] = {
        'script': 'viz_03_rh_percentiles.py',
        'args': [
            '--percentile-file', args.rh_percentiles_file,
            '--output-dir', str(output_base_dir / '03_rh_percentiles')
        ]
    }
    
    viz_scripts['heat_index'] = {
        'script': 'viz_04_heat_index.py',
        'args': [
            '--hi-dir', args.heat_index_dir,
            '--output-dir', str(output_base_dir / '04_heat_index')
        ]
    }
    
    viz_scripts['heatwave_metrics'] = {
        'script': 'viz_05_heatwave_metrics.py',
        'args': [
            '--metrics-dir', args.heatwave_dir,
            '--output-dir', str(output_base_dir / '05_heatwave_metrics'),
            '--variables'
        ] + args.variables
    }
    
    viz_scripts['humidity_classification'] = {
        'script': 'viz_06_humidity_classification.py',
        'args': [
            '--humidity-dir', args.humidity_dir,
            '--output-dir', str(output_base_dir / '06_humidity_classification'),
            '--variables'
        ] + args.variables
    }
    
    viz_scripts['precipitation_analysis'] = {
        'script': 'viz_07_precipitation_analysis.py',
        'args': [
            '--precip-dir', args.precipitation_analysis_dir,
            '--output-dir', str(output_base_dir / '07_precipitation_analysis')
        ]
    }
    
    # Add common arguments
    for component in viz_scripts:
        if args.create_seasonal:
            viz_scripts[component]['args'].append('--create-seasonal')
        if args.years:
            viz_scripts[component]['args'].extend(['--years'] + [str(y) for y in args.years])
    
    # Run visualizations
    results = {}
    success_count = 0
    
    print(f"\n{'='*80}")
    print("RUNNING VISUALIZATIONS")
    print(f"{'='*80}")
    
    for component in components_to_run:
        if component not in viz_scripts:
            print(f"⚠️  Unknown component: {component}")
            continue
        
        script_info = viz_scripts[component]
        script_name = script_info['script']
        script_args = script_info['args']
        
        success = run_visualization(script_name, script_args, output_base_dir)
        
        if success:
            results[script_name] = 'success'
            success_count += 1
        else:
            results[script_name] = 'failed'
            if not args.skip_failed:
                print(f"\n❌ Stopping due to failure in {script_name}")
                print("Use --skip-failed to continue despite failures")
                break
    
    # Create summary
    print(f"\n{'='*80}")
    print("CREATING VISUALIZATION SUMMARY")
    print(f"{'='*80}")
    
    create_visualization_summary(output_base_dir, results)
    
    # Final summary
    print(f"\n{'='*80}")
    print("VISUALIZATION SUMMARY")
    print(f"{'='*80}")
    print(f"Completed: {success_count}/{len(components_to_run)} components")
    print(f"Output directory: {output_base_dir}")
    print(f"Summary file: {output_base_dir / 'visualization_summary.html'}")
    
    failed_components = [comp for comp, result in results.items() if result == 'failed']
    if failed_components:
        print(f"\nFailed components: {failed_components}")
    
    print(f"\n🎉 Visualization pipeline completed!")
    print(f"Open {output_base_dir / 'visualization_summary.html'} to view all results.")
    
    return 0 if success_count > 0 else 1

if __name__ == "__main__":
    sys.exit(main())
