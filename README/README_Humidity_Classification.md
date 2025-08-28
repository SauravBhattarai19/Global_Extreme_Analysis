# Heatwave Humidity Classification System

This system implements comprehensive humidity-based classification of heatwaves into hot-dry vs hot-humid categories, following the scientific framework outlined in the provided pseudocode.

## Overview

The system classifies heatwaves based on their humidity characteristics at both day-level and event-level, providing:

- **Day-level classification**: Each heatwave day is classified as dry, humid, or moderate
- **Event-level summarization**: Heatwave events are labeled based on dominance patterns
- **Annual aggregations**: Summary statistics by humidity category for each grid cell
- **Flexible thresholds**: Support for both absolute and percentile-based classification

## Key Features

### Classification Modes

1. **Absolute Mode** (default)
   - Uses fixed RH thresholds (e.g., 33% dry, 66% humid)
   - Simple interpretation tied to physiological comfort
   - Consistent across regions and seasons

2. **Percentile Mode**
   - Uses climatological percentiles (P33/P66) by day-of-year
   - Normalizes for local climate regimes and seasonality
   - Requires pre-computed percentile climatology

### Scientific Approach

- **Phase-aligned humidity**: Daytime RH for Tmax events, nighttime RH for Tmin events
- **Dominance labeling**: Events labeled based on ≥60% of days meeting criteria
- **Robust statistics**: Preserves within-event variability via fractions and streaks
- **Quality control**: Requires minimum fraction of valid RH data for labeling

## Input Data Requirements

### Required Inputs

1. **Heatwave Data** (from `04_Heatwave_Metrics.py`):
   - `heatwave_events_{var}_{year}.parquet`: Event-level records
   - `heatwave_days_{var}_{year}.parquet`: Day-level records

2. **Relative Humidity Data**:
   - Daytime RH: `rh_day_{year}_*.nc` (aligned to Tmax timing)
   - Nighttime RH: `rh_night_{year}_*.nc` (aligned to Tmin timing)
   - Variables: `rh`, `relative_humidity`, or `r`

3. **Percentile Climatology** (for percentile mode):
   - `rh_percentiles.nc` containing `rh_p33` and `rh_p66` by day-of-year

### Data Structure Expected

```
data/
├── processed/
│   ├── heatwave_metrics/           # Output from 04_Heatwave_Metrics.py
│   │   ├── heatwave_events_tmax_2020.parquet
│   │   ├── heatwave_days_tmax_2020.parquet
│   │   └── ...
│   ├── rh/                         # Relative humidity data
│   │   ├── rh_day_2020_01.nc
│   │   ├── rh_night_2020_01.nc
│   │   └── ...
│   └── rh_percentiles.nc           # Optional: for percentile mode
```

## Usage

### Basic Usage

```bash
# Run with default absolute thresholds (33%/66%)
python 05_Humidity_Classification.py \
    --start-year 2020 \
    --end-year 2022 \
    --heatwave-dir data/processed/heatwave_metrics \
    --rh-dir data/processed/rh \
    --output-dir data/processed/humidity_classification
```

### Advanced Configuration

```bash
# Stricter absolute thresholds with higher dominance requirement
python 05_Humidity_Classification.py \
    --start-year 2020 \
    --end-year 2022 \
    --humidity-mode absolute \
    --abs-dry 25.0 \
    --abs-humid 75.0 \
    --dominance-alpha 0.70 \
    --min-valid-rh 0.85 \
    --heatwave-dir data/processed/heatwave_metrics \
    --rh-dir data/processed/rh \
    --output-dir data/processed/humidity_strict
```

### Percentile-Based Classification

```bash
# Use climatological percentiles
python 05_Humidity_Classification.py \
    --start-year 2020 \
    --end-year 2022 \
    --humidity-mode percentile \
    --percentile-file data/processed/rh_percentiles.nc \
    --doy-leap-rule interpolate \
    --heatwave-dir data/processed/heatwave_metrics \
    --rh-dir data/processed/rh \
    --output-dir data/processed/humidity_percentile
```

### Example Script

Use the provided example script for common configurations:

```bash
python run_humidity_classification.py
```

## Configuration Parameters

### Classification Parameters

- `--humidity-mode`: Classification mode (`absolute` or `percentile`)
- `--abs-dry`: Dry threshold for absolute mode (default: 33.0%)
- `--abs-humid`: Humid threshold for absolute mode (default: 66.0%)
- `--dominance-alpha`: Fraction of days required for event labeling (default: 0.60)
- `--min-valid-rh`: Minimum fraction of valid RH for event labeling (default: 0.8)
- `--doy-leap-rule`: Leap year handling (`fold_366_to_365` or `interpolate`)

### Processing Parameters

- `--n-processes`: Number of parallel processes (default: 48)
- `--chunk-size-lat`: Latitude chunk size (default: 40)
- `--chunk-size-lon`: Longitude chunk size (default: 80)
- `--variables`: Variables to process (default: `tmax tmin`)

## Output Files

### Per Year Outputs

1. **Enhanced Event Table**: `heatwave_events_humidity_{year}.parquet`
   - Original event data plus humidity statistics
   - Event-level labels: `humid-hot`, `dry-hot`, `mixed-hot`, `insufficient-RH`
   - Humidity fractions and longest streaks

2. **Enhanced Day Table**: `heatwave_days_humidity_{year}.parquet`
   - Original day data plus humidity classifications
   - Day-level classes: `humid`, `dry`, `moderate`, `missing`
   - RH values and thresholds used

3. **Annual Aggregations**: `heatwave_humidity_aggregations_{year}.nc`
   - Gridded annual summaries by humidity category
   - Event counts (HWN) and day counts (HWTD) by category
   - Longest durations (HWLD) by category

### Aggregation Variables

- **Event Counts (HWN)**:
  - `HWN_day_humid`, `HWN_day_dry`, `HWN_day_mixed`
  - `HWN_night_humid`, `HWN_night_dry`, `HWN_night_mixed`

- **Day Counts (HWTD)**:
  - `HWTD_day_humid`, `HWTD_day_dry`, `HWTD_day_moderate`
  - `HWTD_night_humid`, `HWTD_night_dry`, `HWTD_night_moderate`

- **Longest Durations (HWLD)**:
  - `HWLD_day_humid`, `HWLD_day_dry`
  - `HWLD_night_humid`, `HWLD_night_dry`

## Scientific Rationale

### Phase-Aligned Humidity

The system uses humidity data aligned to the same temporal phase as the heat stress:
- **Daytime events** (Tmax): Use daytime RH (e.g., at time of maximum temperature)
- **Nighttime events** (Tmin): Use nighttime RH (e.g., at time of minimum temperature)

This ensures humidity measurements reflect conditions during actual heat exposure.

### Threshold Selection

**Absolute Thresholds (33%/66%)**:
- Based on physiological comfort ranges
- 33% represents transition to dry conditions
- 66% represents transition to humid conditions
- Consistent interpretation across regions

**Percentile Thresholds (P33/P66)**:
- Normalizes for local climate regimes
- Accounts for seasonal humidity patterns
- Better for comparative studies across diverse climates

### Event Labeling Logic

Events are labeled using a dominance approach:
- **humid-hot**: ≥60% of event days are humid
- **dry-hot**: ≥60% of event days are dry  
- **mixed-hot**: Neither category dominates
- **insufficient-RH**: <80% of days have valid RH data

This approach prevents single outlier days from changing event classification while preserving ambiguous cases.

## Performance Considerations

### Memory Usage

- Estimated memory per chunk: ~0.5-2 GB depending on chunk size
- Total memory scales with number of processes
- Reduce chunk size or processes if memory constrained

### Processing Time

- Typical processing: ~10-30 minutes per year depending on:
  - Grid resolution and spatial extent
  - Number of heatwave events
  - Chunk size and process count
  - I/O performance

### Optimization Tips

1. **Chunk Size**: Balance memory usage vs I/O efficiency
2. **Process Count**: Match to available CPU cores
3. **Data Location**: Keep input data on fast storage
4. **Memory**: Monitor total memory usage across processes

## Quality Control

The system includes several quality control measures:

1. **Data Validation**: Checks for required input files and variables
2. **Missing Data Handling**: Classifies missing RH as "missing" category
3. **Minimum Data Requirements**: Events need ≥80% valid RH for labeling
4. **Leap Year Handling**: Proper handling of February 29th in percentile mode
5. **Coordinate Mapping**: Ensures proper alignment between datasets

## Example Analysis Workflow

1. **Prepare Data**:
   ```bash
   # Run heatwave detection first
   python 04_Heatwave_Metrics.py --start-year 2020 --end-year 2022
   
   # Ensure RH data is available and properly formatted
   ```

2. **Run Classification**:
   ```bash
   # Test with single year first
   python 05_Humidity_Classification.py --start-year 2020 --end-year 2020
   
   # Run full analysis
   python run_humidity_classification.py
   ```

3. **Analyze Results**:
   ```python
   import pandas as pd
   import xarray as xr
   
   # Load event data
   events = pd.read_parquet('data/processed/humidity_classification/heatwave_events_humidity_2020.parquet')
   
   # Analyze event labels
   print(events['label_day'].value_counts())
   
   # Load aggregations
   aggs = xr.open_dataset('data/processed/humidity_classification/heatwave_humidity_aggregations_2020.nc')
   
   # Plot spatial patterns
   aggs.HWN_day_humid.plot()
   ```

## Troubleshooting

### Common Issues

1. **No RH data found**: Check file naming convention and directory structure
2. **Memory errors**: Reduce chunk size or number of processes
3. **Missing percentile file**: Required for percentile mode
4. **Coordinate mismatch**: Ensure RH and heatwave data use same grid

### Error Messages

- `"No RH files found"`: Check RH directory and file naming
- `"High memory usage warning"`: Reduce chunk size or processes
- `"No heatwave data found"`: Run `04_Heatwave_Metrics.py` first
- `"Percentile file required"`: Provide `--percentile-file` for percentile mode

## Extensions and Customization

The system can be extended for specific research needs:

1. **Custom Thresholds**: Modify threshold values for regional studies
2. **Additional Variables**: Add other humidity metrics (dewpoint, vapor pressure)
3. **Temporal Aggregation**: Extend to seasonal or decadal summaries
4. **Health Applications**: Link to heat-health impact models
5. **Climate Change Analysis**: Compare threshold exceedances across periods

## Citation and References

When using this system, please cite the relevant methodological papers and acknowledge the data sources used for temperature and humidity inputs.

## Support

For questions or issues:
1. Check this documentation first
2. Examine log output for specific error messages  
3. Verify input data format and availability
4. Test with smaller spatial/temporal domains first
