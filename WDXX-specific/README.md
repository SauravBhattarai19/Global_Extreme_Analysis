# Precipitation Concentration and Distribution Indices
## A Comprehensive Framework for Analyzing Precipitation Patterns

**Authors:** Saurav Bhattarai¹'³, Nawa Raj Pradhan², Rocky Talchabhadel¹  
**Affiliations:**  
¹ Department of Civil and Environmental Engineering, Jackson State University, Jackson, MS, USA  
² U.S. Army Corps of Engineers, ERDC, Vicksburg, MS, USA  
³ Corresponding author: J01013381@students.jsums.edu

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Index Descriptions](#index-descriptions)
3. [Code Structure](#code-structure)
4. [Data Requirements](#data-requirements)
5. [Installation & Dependencies](#installation--dependencies)
6. [Usage Examples](#usage-examples)
7. [Output Files](#output-files)
8. [Citation](#citation)
9. [Contact](#contact)

---

## 🌧️ Overview

This repository contains a comprehensive suite of precipitation indices designed to quantify the temporal concentration and distribution patterns of precipitation. The framework includes:

- **WDXX/WDXXR Indices**: Window Duration for XX% of annual rainfall
- **ETCCDI Indices**: Expert Team on Climate Change Detection and Indices
- **Concentration Indices**: Gini coefficient, Lorenz curve, entropy measures
- **Precipitation Percentiles**: Climatological reference thresholds

### Scientific Motivation

Understanding precipitation distribution patterns is crucial for:
- Water resource management
- Agricultural planning
- Flood risk assessment
- Climate change impact studies
- Drought monitoring

Traditional precipitation statistics (e.g., annual total) fail to capture the temporal distribution of precipitation. Our indices quantify:
- **When precipitation occurs** during the year
- **How concentrated** precipitation is in time
- **Efficiency** of precipitation delivery
- **Extreme precipitation** characteristics

---

## 📊 Index Descriptions

### 1. WDXX/WDXXR Indices (Window Duration)

**File:** `10_WD50R_Chronological_Precipitation_Index.py`

#### Core Concept
WDXX represents the **minimum number of consecutive days** required to accumulate XX% of annual precipitation.

#### Key Indices

##### WD50R (Primary Index)
- **Definition**: Minimum consecutive days for 50% of annual rainfall
- **Range**: 1 to 365 days
- **Interpretation**:
  - Low WD50R (< 90 days): Highly concentrated precipitation
  - Moderate WD50R (90-180 days): Moderately distributed
  - High WD50R (> 180 days): Well-distributed precipitation

##### Additional Percentiles
- **WD25R**: Days for 25% of annual rainfall
- **WD75R**: Days for 75% of annual rainfall
- **WD90R**: Days for 90% of annual rainfall

#### Regime Classification

Based on WD50R and annual total precipitation:

| Regime | WD50R Range | Annual Total | Characteristics |
|--------|-------------|--------------|-----------------|
| **Highly Concentrated** | < 60 days | > 800 mm | Monsoon-dominated |
| **Concentrated** | 60-120 days | > 500 mm | Seasonal rainfall |
| **Moderate** | 120-180 days | > 300 mm | Bi-modal patterns |
| **Distributed** | > 180 days | Variable | Year-round rainfall |
| **Arid** | Any | < 200 mm | Low total precipitation |

#### Novel Metrics

1. **Window Efficiency (WE50)**
   ```
   WE50 = (P_in_window / WD50R) / (P_annual / 365)
   ```
   - Measures precipitation intensity concentration
   - Values > 1: More efficient delivery in window period

2. **Concentration Ratio (CR50)**
   ```
   CR50 = WD50R / 365
   ```
   - Fraction of year for 50% of rainfall
   - Range: 0 to 1

3. **Timing Metrics**
   - **Start Date**: When WD50R window begins
   - **End Date**: When WD50R window ends
   - **Peak Month**: Month with maximum precipitation
   - **Centroid DOY**: Precipitation-weighted center of year

---

### 2. ETCCDI Precipitation Indices

**File:** `08_ETCCDI_Precipitation_Indices.py`

Based on Expert Team on Climate Change Detection and Indices guidelines.

#### Indices Calculated

| Index | Name | Description | Units |
|-------|------|-------------|-------|
| **PRCPTOT** | Total Precipitation | Annual total on wet days (≥1mm) | mm |
| **SDII** | Simple Daily Intensity | Average precipitation on wet days | mm/day |
| **CWD** | Consecutive Wet Days | Maximum consecutive days ≥1mm | days |
| **CDD** | Consecutive Dry Days | Maximum consecutive days <1mm | days |
| **R10mm** | Heavy Precipitation Days | Days with ≥10mm | days |
| **R20mm** | Very Heavy Precipitation | Days with ≥20mm | days |
| **R95p** | Very Wet Days | Precipitation from days >95th percentile | mm |
| **R95pTOT** | R95p Fraction | R95p / PRCPTOT * 100 | % |

#### Applications
- Climate change detection
- Extreme precipitation analysis
- Standardized reporting
- International comparisons

---

### 3. Enhanced Concentration Indices

**File:** `09_enhanced_precipitation_concentration_indices.py`

Statistical measures of precipitation concentration from inequality/concentration literature.

#### Gini Coefficient (G)
```
G = (2 * Σ(i * P_i)) / (n * Σ P_i) - (n + 1) / n
```
- **Range**: 0 to 1
- **Interpretation**:
  - G = 0: Perfectly uniform daily precipitation
  - G = 1: All precipitation on single day
- **Application**: Quantify temporal inequality

#### Lorenz Curve Analysis
- **Lorenz Maximum Deviation (LMD)**: Maximum distance from equality line
- **Area Under Lorenz Curve (AULC)**: Measures concentration intensity

#### Shannon Entropy (H)
```
H = -Σ(p_i * log(p_i))
```
- **High H**: More uniform distribution
- **Low H**: More concentrated distribution

#### Precipitation Concentration Factor (PCF)
```
PCF = (PRCPTOT / WD50) * regime_multiplier
```
- Combines amount and timing
- Regime-adjusted for context

#### Precipitation Regime Classification

Based on PRCPTOT, R95pTOT, and WD50:

1. **Arid**: PRCPTOT < 200 mm
2. **Semi-Arid**: 200-500 mm, low WD50
3. **Sub-humid**: 500-1000 mm, moderate WD50
4. **Humid**: 1000-2000 mm
5. **Very Humid**: > 2000 mm
6. **Extreme-Concentrated**: High R95pTOT (>50%), low WD50
7. **Distributed**: WD50 > 180, moderate R95pTOT

---

### 4. Precipitation Percentiles

**File:** `00_calculate_precipitation_percentiles.py`

Climatological reference thresholds for drought/extreme precipitation classification.

#### Percentiles Calculated
- **P10**: 10th percentile (drought threshold)
- **P25**: 25th percentile (dry conditions)
- **P50**: 50th percentile (median)
- **P75**: 75th percentile (wet conditions)
- **P90**: 90th percentile (very wet)
- **P95**: 95th percentile (extreme precipitation)

#### Methodology
- Day-of-year climatology
- 15-day moving window
- Based on 1980-2000 baseline period
- Handles leap years appropriately

---

## 🗂️ Code Structure

```
publication_precipitation_indices/
├── code/
│   ├── 00_calculate_precipitation_percentiles.py
│   ├── 08_ETCCDI_Precipitation_Indices.py
│   ├── 09_enhanced_precipitation_concentration_indices.py
│   └── 10_WD50R_Chronological_Precipitation_Index.py
├── results/
│   ├── percentiles/           # Precipitation percentile climatology
│   ├── wd50r_indices/          # WDXX/WDXXR results
│   ├── etccdi_indices/         # ETCCDI results
│   └── concentration_indices/  # Gini, Lorenz, entropy results
├── docs/
│   └── figures/                # Publication-quality figures
└── README.md                   # This file
```

### Code Dependencies

All scripts follow a consistent pattern:
1. **Parallel processing**: Multiprocessing for efficiency
2. **Chunked computation**: Memory-efficient spatial processing
3. **NetCDF output**: CF-compliant gridded results
4. **Parquet tables**: Event/pixel-level details

---

## 💾 Data Requirements

### Input Data

#### ERA5 Precipitation Data
- **Variable**: Total precipitation (`tp`)
- **Units**: meters (automatically converted to mm)
- **Temporal Resolution**: Sub-daily (aggregated to daily)
- **Spatial Resolution**: 0.25° × 0.25°
- **Format**: NetCDF files organized by year and month
- **Naming Convention**: `era5_daily_{YYYY}_{MM:02d}.nc`

#### File Structure Expected
```
/data/climate/disk1/datasets/era5/
├── era5_daily_1980_01.nc
├── era5_daily_1980_02.nc
├── ...
└── era5_daily_2024_12.nc
```

### Output Data

#### Gridded NetCDF Files
- Annual indices per grid cell
- CF-compliant metadata
- Compressed (zlib level 4)
- Float32 precision

#### Tabular Parquet Files
- Pixel-level statistics
- Event/regime classifications
- Efficient for analysis

---

## 🔧 Installation & Dependencies

### Python Environment

```bash
# Create conda environment
conda create -n precip_indices python=3.9
conda activate precip_indices

# Install dependencies
pip install numpy pandas xarray netCDF4 scipy
pip install pyarrow  # For parquet files
```

### Required Packages

```python
numpy >= 1.21.0      # Numerical operations
pandas >= 1.3.0      # Data manipulation
xarray >= 0.19.0     # NetCDF handling
scipy >= 1.7.0       # Statistical functions
pyarrow >= 5.0.0     # Parquet I/O
```

### System Requirements

- **Memory**: Minimum 32 GB RAM (64 GB recommended)
- **Processors**: Multi-core (24+ cores recommended)
- **Storage**: ~500 GB for intermediate files
- **OS**: Linux (tested on Ubuntu 20.04+)

---

## 🚀 Usage Examples

### 1. Calculate Precipitation Percentiles (Baseline)

```bash
# Calculate climatological percentiles (1980-2000)
python code/00_calculate_precipitation_percentiles.py \
    --precip-dir /data/climate/disk1/datasets/era5 \
    --start-year 1980 \
    --end-year 2000 \
    --output-file results/percentiles/precipitation_percentiles_1980-2000.nc \
    --percentiles 10 25 50 75 90 95 \
    --n-processes 48 \
    --chunk-size-lat 50 \
    --chunk-size-lon 100
```

**Output:**
- `results/percentiles/precipitation_percentiles_1980-2000.nc`
  - Variables: precip_p10, precip_p25, precip_p50, precip_p75, precip_p90, precip_p95
  - Dimensions: [dayofyear=366, latitude, longitude]

---

### 2. Calculate WD50R Indices (Single Year)

```bash
# Calculate WD50R for 2020
python code/10_WD50R_Chronological_Precipitation_Index.py \
    --year 2020 \
    --precip-dir /data/climate/disk1/datasets/era5 \
    --output-dir results/wd50r_indices \
    --percentiles 25 50 75 90 \
    --wet-day-threshold 0.1 \
    --min-annual-precip 50 \
    --n-processes 24 \
    --chunk-size-lat 50 \
    --chunk-size-lon 100
```

**Output:**
- `results/wd50r_indices/wd50r_indices_2020.nc`
  - WD25R, WD50R, WD75R, WD90R grids
  - Window start/end dates
  - Efficiency metrics
  - Regime classifications

- `results/wd50r_indices/wd50r_regimes_2020.txt`
  - Detailed pixel-level statistics
  - Timing information
  - Classification labels

---

### 3. Calculate WD50R for Multiple Years

```bash
# Calculate WD50R for 1991-2020 period
python code/10_WD50R_Chronological_Precipitation_Index.py \
    --start-year 1991 \
    --end-year 2020 \
    --precip-dir /data/climate/disk1/datasets/era5 \
    --output-dir results/wd50r_indices \
    --percentiles 50 \
    --n-processes 24
```

**Output:**
- One file per year: `wd50r_indices_{year}.nc`
- Suitable for trend analysis

---

### 4. Calculate ETCCDI Indices

```bash
# Calculate ETCCDI indices for 2020
python code/08_ETCCDI_Precipitation_Indices.py \
    --year 2020 \
    --precip-dir /data/climate/disk1/datasets/era5 \
    --percentile-file results/percentiles/precipitation_percentiles_1980-2000.nc \
    --output-dir results/etccdi_indices \
    --wet-day-threshold 1.0 \
    --n-processes 24
```

**Output:**
- `results/etccdi_indices/etccdi_indices_2020.nc`
  - PRCPTOT, SDII, CWD, CDD, R10mm, R20mm, R95p, R95pTOT
  - Dimensions: [latitude, longitude]

---

### 5. Calculate Enhanced Concentration Indices

```bash
# Calculate Gini, Lorenz, Entropy for 2020
python code/09_enhanced_precipitation_concentration_indices.py \
    --year 2020 \
    --precip-dir /data/climate/disk1/datasets/era5 \
    --output-dir results/concentration_indices \
    --method all \
    --wet-day-threshold 1.0 \
    --n-processes 24
```

**Method options:**
- `all`: All concentration indices
- `gini`: Gini coefficient only
- `lorenz`: Lorenz curve metrics
- `entropy`: Shannon entropy

**Output:**
- `results/concentration_indices/concentration_indices_all_2020.nc`
  - Gini coefficient
  - Lorenz maximum deviation
  - Shannon entropy
  - Precipitation concentration factor
  - Regime classifications

---

### 6. Complete Analysis Pipeline

```bash
#!/bin/bash
# Complete analysis for 1991-2020

PRECIP_DIR="/data/climate/disk1/datasets/era5"
OUTPUT_BASE="results"
N_PROC=24

# Step 1: Calculate percentiles (once)
python code/00_calculate_precipitation_percentiles.py \
    --precip-dir $PRECIP_DIR \
    --start-year 1980 --end-year 2000 \
    --output-file $OUTPUT_BASE/percentiles/precip_percentiles.nc \
    --n-processes $N_PROC

# Step 2: Loop through years
for YEAR in {1991..2020}; do
    echo "Processing year $YEAR"
    
    # WD50R indices
    python code/10_WD50R_Chronological_Precipitation_Index.py \
        --year $YEAR \
        --precip-dir $PRECIP_DIR \
        --output-dir $OUTPUT_BASE/wd50r_indices \
        --n-processes $N_PROC
    
    # ETCCDI indices
    python code/08_ETCCDI_Precipitation_Indices.py \
        --year $YEAR \
        --precip-dir $PRECIP_DIR \
        --percentile-file $OUTPUT_BASE/percentiles/precip_percentiles.nc \
        --output-dir $OUTPUT_BASE/etccdi_indices \
        --n-processes $N_PROC
    
    # Concentration indices
    python code/09_enhanced_precipitation_concentration_indices.py \
        --year $YEAR \
        --precip-dir $PRECIP_DIR \
        --output-dir $OUTPUT_BASE/concentration_indices \
        --n-processes $N_PROC
done

echo "Analysis complete!"
```

---

## 📁 Output Files

### 1. WD50R Output Structure

#### NetCDF File (`wd50r_indices_{year}.nc`)

```python
# Example structure
dimensions:
    latitude = 721
    longitude = 1440

variables:
    # Core WDXXR indices
    WD25R(latitude, longitude): Days for 25% of rainfall
    WD50R(latitude, longitude): Days for 50% of rainfall
    WD75R(latitude, longitude): Days for 75% of rainfall
    WD90R(latitude, longitude): Days for 90% of rainfall
    
    # Timing information
    WD50R_start_doy(latitude, longitude): Window start day-of-year
    WD50R_end_doy(latitude, longitude): Window end day-of-year
    peak_month(latitude, longitude): Month with maximum precipitation
    
    # Efficiency metrics
    window_efficiency_50(latitude, longitude): WE50 ratio
    concentration_ratio_50(latitude, longitude): CR50
    
    # Basic precipitation
    annual_total_precip(latitude, longitude): PRCPTOT in mm
    
    # Regime classification
    precip_regime(latitude, longitude): Categorical (1-7)

attributes:
    year = 2020
    creation_date = "2025-10-28"
    percentiles_calculated = [25, 50, 75, 90]
    wet_day_threshold_mm = 0.1
    min_annual_precip_mm = 50.0
```

#### Text File (`wd50r_regimes_{year}.txt`)

Tab-separated values with columns:
- lat, lon: Coordinates
- annual_precip: Total precipitation (mm)
- WD25R, WD50R, WD75R, WD90R: Index values (days)
- WD50R_start, WD50R_end: Window boundaries
- regime: Classification label
- window_efficiency: WE50
- concentration_ratio: CR50
- peak_month: Peak precipitation month

---

### 2. ETCCDI Output Structure

#### NetCDF File (`etccdi_indices_{year}.nc`)

```python
dimensions:
    latitude = 721
    longitude = 1440

variables:
    PRCPTOT(latitude, longitude): Total precipitation on wet days
    SDII(latitude, longitude): Simple daily intensity index
    CWD(latitude, longitude): Consecutive wet days maximum
    CDD(latitude, longitude): Consecutive dry days maximum
    R10mm(latitude, longitude): Heavy precipitation days count
    R20mm(latitude, longitude): Very heavy precipitation days
    R95p(latitude, longitude): Precipitation from >95th percentile
    R95pTOT(latitude, longitude): R95p as percentage of PRCPTOT

attributes:
    year = 2020
    wet_day_threshold_mm = 1.0
    baseline_period = "1980-2000"
```

---

### 3. Concentration Indices Output

#### NetCDF File (`concentration_indices_all_{year}.nc`)

```python
dimensions:
    latitude = 721
    longitude = 1440

variables:
    gini_coefficient(latitude, longitude): Gini G
    lorenz_max_deviation(latitude, longitude): LMD
    lorenz_area(latitude, longitude): Area under Lorenz curve
    shannon_entropy(latitude, longitude): H
    precipitation_concentration_factor(latitude, longitude): PCF
    
    # Supporting data
    PRCPTOT(latitude, longitude): Annual total
    R95pTOT(latitude, longitude): Extreme fraction
    WD50(latitude, longitude): Wet days for 50%
    
    # Regime classification
    precip_regime(latitude, longitude): Categorical
    regime_label(latitude, longitude): String labels

attributes:
    method = "all"
    year = 2020
```

---

## 📈 Interpretation Guide

### WD50R Values

| WD50R Range | Interpretation | Example Regions |
|-------------|----------------|-----------------|
| < 60 days | Highly concentrated monsoon | South Asia monsoon, West Africa |
| 60-90 days | Concentrated wet season | Mediterranean, California |
| 90-150 days | Moderate bi-modal | Tropical regions with two seasons |
| 150-220 days | Distributed rainfall | Mid-latitude oceanic climates |
| > 220 days | Year-round precipitation | Marine west coast, tropical rainforest |

### Gini Coefficient

| Gini Value | Distribution | Implications |
|------------|-------------|---------------|
| 0.0-0.3 | Very uniform | Low flood/drought risk |
| 0.3-0.5 | Moderate concentration | Typical for many climates |
| 0.5-0.7 | Concentrated | Seasonal water management needs |
| 0.7-0.9 | Highly concentrated | High variability, extreme events |
| 0.9-1.0 | Extreme concentration | Single-event dominance |

### ETCCDI Thresholds

**Precipitation Regime Classification:**
- **R95pTOT > 50%**: Extreme-event dominated
- **CDD > 100 days**: Prone to dry spells
- **CWD > 30 days**: Long wet periods
- **SDII > 15 mm/day**: Intense rainfall regime

---

## 🔬 Scientific Applications

### 1. Climate Change Studies
- **Temporal trend analysis** of WD50R
- **Extreme precipitation** frequency changes
- **Seasonality shifts** via window timing

### 2. Water Resource Management
- **Storage requirements**: Inversely related to WD50R
- **Irrigation planning**: Use window timing
- **Reservoir operations**: Peak precipitation months

### 3. Agricultural Planning
- **Crop selection**: Match to precipitation regime
- **Planting windows**: Use WD50R timing
- **Drought risk**: WD50R + CDD

### 4. Flood Risk Assessment
- **Concentration indices**: Higher Gini → higher flash flood risk
- **CWD + SDII**: Sustained heavy precipitation events
- **R95pTOT**: Extreme event contribution

### 5. Drought Monitoring
- **Percentile departures**: Current vs. climatology
- **CDD tracking**: Dry spell intensity
- **WD50R anomalies**: Distribution shifts

---

## 📊 Example Analyses

### Trend Analysis (1991-2020)

```python
import xarray as xr
import numpy as np
from scipy import stats

# Load all years
years = range(1991, 2021)
wd50r_list = []

for year in years:
    ds = xr.open_dataset(f'results/wd50r_indices/wd50r_indices_{year}.nc')
    wd50r_list.append(ds.WD50R)

# Stack into time series
wd50r_timeseries = xr.concat(wd50r_list, dim='year')

# Calculate trend at each grid point
def calculate_trend(data):
    years_array = np.arange(len(data))
    slope, intercept, r_value, p_value, std_err = stats.linregress(years_array, data)
    return slope, p_value

# Apply to each pixel
trends = xr.apply_ufunc(
    calculate_trend,
    wd50r_timeseries,
    input_core_dims=[['year']],
    output_core_dims=[[], []],
    vectorize=True
)

print(f"Significant trends: {(trends[1] < 0.05).sum().values} pixels")
```

### Spatial Correlation

```python
# Correlate WD50R with annual precipitation
ds = xr.open_dataset('results/wd50r_indices/wd50r_indices_2020.nc')

correlation = xr.corr(ds.WD50R, ds.annual_total_precip, dim=['latitude', 'longitude'])
print(f"WD50R vs Precip correlation: {correlation.values:.3f}")
```

### Regime Change Detection

```python
# Compare two periods
period1 = slice(1991, 2005)
period2 = slice(2006, 2020)

# Load and average
wd50r_p1 = xr.open_mfdataset('results/wd50r_indices/wd50r_indices_*.nc', 
                              concat_dim='year').sel(year=period1).WD50R.mean('year')
wd50r_p2 = xr.open_mfdataset('results/wd50r_indices/wd50r_indices_*.nc',
                              concat_dim='year').sel(year=period2).WD50R.mean('year')

# Calculate change
change = wd50r_p2 - wd50r_p1

# Find regions with >20 day shifts
significant_shift = np.abs(change) > 20
print(f"Pixels with >20 day shift: {significant_shift.sum().values}")
```

---

## 📖 Citation

If you use these indices in your research, please cite:

```bibtex
@article{bhattarai2025precipitation,
  title={Do the Wettest Days Occur Together? A Global Analysis on Disentangling 
         Precipitation Intensity from Seasonal Timing},
  author={Bhattarai, Saurav and Pradhan, Nawa Raj and Talchabhadel, Rocky},
  journal={[In Review]},
  year={2025}
}
```

### Key References

1. **ETCCDI Indices:**
   - Zhang, X., et al. (2011). Indices for monitoring changes in extremes. WMO-TD No. 1500.

2. **Gini Coefficient Application:**
   - Masaki, Y., et al. (2014). Global-scale analysis on future changes in flow regimes. Hydrol. Earth Syst. Sci., 18, 171-188.

3. **Precipitation Concentration:**
   - Martin-Vide, J. (2004). Spatial distribution of a daily precipitation concentration index in peninsular Spain. Int. J. Climatol., 24, 959-971.

4. **ERA5 Reanalysis:**
   - Hersbach, H., et al. (2020). The ERA5 global reanalysis. Q. J. R. Meteorol. Soc., 146, 1999-2049.

---

## 🛠️ Troubleshooting

### Common Issues

#### 1. Memory Errors
**Problem:** Out of memory during processing

**Solution:**
```bash
# Reduce chunk sizes
--chunk-size-lat 25 --chunk-size-lon 50

# Reduce number of processes
--n-processes 12
```

#### 2. Missing Data
**Problem:** NaN values in output

**Causes:**
- Insufficient data for year
- Very low precipitation (< min_annual_precip)
- Missing input files

**Check:**
```python
import xarray as xr
ds = xr.open_dataset('results/wd50r_indices/wd50r_indices_2020.nc')
print(f"NaN pixels: {ds.WD50R.isnull().sum().values}")
```

#### 3. Slow Processing
**Problem:** Taking too long

**Solutions:**
- Increase `--n-processes` (up to number of CPU cores)
- Use SSD for input/output
- Process years in parallel (separate jobs)

#### 4. Percentile File Not Found
**Problem:** ETCCDI script can't find percentile file

**Solution:**
```bash
# Always run Step 0 first
python code/00_calculate_precipitation_percentiles.py \
    --output-file results/percentiles/precip_percentiles.nc

# Then use exact path in subsequent scripts
--percentile-file results/percentiles/precip_percentiles.nc
```

---

## 🔄 Updates and Versions

### Current Version: 1.0 (October 2025)

**Features:**
- WD50R and related percentile indices
- ETCCDI standard indices
- Gini, Lorenz, Entropy concentration measures
- Parallel processing with chunking
- CF-compliant NetCDF output

**Coming Soon:**
- Interactive visualization tools
- Regional aggregation utilities
- Statistical significance testing
- Python API for easier integration

---

## 👥 Contributing

We welcome contributions! Areas for improvement:
- Additional precipitation indices
- Visualization scripts
- Statistical testing frameworks
- Documentation enhancements
- Bug fixes

Please contact the author for collaboration opportunities.

---

## 📧 Contact

**Saurav Bhattarai** (Corresponding Author)  
Email: J01013381@students.jsums.edu  
GitHub: SauravBhattarai19  
Institution: Jackson State University

For questions, bug reports, or collaboration inquiries, please reach out via email or create an issue in the repository.

---

## 📜 License

This code is provided for research and educational purposes. Please cite appropriately if used in publications.

---

## 🙏 Acknowledgments

- ERA5 reanalysis data provided by ECMWF
- ETCCDI guidelines from CCl/CLIVAR/JCOMM Expert Team
- Computing resources provided by Jackson State University

---

**Last Updated:** October 28, 2025  
**Documentation Version:** 1.0

