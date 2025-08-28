# Humid Heat Analysis - Visualization Suite

This directory contains comprehensive scientific visualization tools for the humid heat analysis pipeline. Each component creates publication-quality plots with scientific color schemes, proper projections, and detailed statistical analysis.

## 📊 **Visualization Components**

### **0. Precipitation Percentiles** (`viz_00_precipitation_percentiles.py`)
**Input:** Precipitation percentile files from `00_calculate_precipitation_percentiles.py`
- **Global maps** of P10/P25/P50/P75/P90 precipitation thresholds
- **Drought/flood analysis** with P10 and P90 thresholds
- **Seasonal cycles** by latitude bands
- **Climate zone classification** based on precipitation patterns

**Key Plots:**
- `precipitation_percentiles_global_annual.png` - Global precipitation percentile maps
- `precipitation_drought_flood_analysis.png` - Drought and flood threshold analysis
- `precipitation_percentiles_seasonal_cycle.png` - Seasonal patterns
- `precipitation_percentiles_statistics.png` - Statistical overview

### **1. Temperature Percentiles** (`viz_01_temperature_percentiles.py`)
**Input:** Temperature percentile files from `01_calculate_percentiles.py`
- **Global maps** of P95 temperature thresholds
- **Seasonal variations** by latitude bands
- **Latitudinal gradients** and diurnal temperature range
- **Statistical summaries** and distributions

**Key Plots:**
- `temperature_percentiles_global_annual.png` - Global P95 maps
- `temperature_percentiles_seasonal_cycle.png` - Seasonal patterns
- `temperature_percentiles_latitudinal_gradients.png` - Zonal analysis
- `temperature_percentiles_statistics.png` - Statistical overview

### **2. Relative Humidity** (`viz_02_relative_humidity.py`)
**Input:** RH files from `02_calculate_RH.py`
- **Global climatology** maps by season
- **Temporal patterns** and diurnal cycles
- **Regional analysis** for key climate zones
- **Data quality assessment** and validation

**Key Plots:**
- `relative_humidity_climatology_*.png` - Seasonal RH maps
- `relative_humidity_temporal_patterns.png` - Time series analysis
- `relative_humidity_regional_analysis.png` - Regional comparisons
- `relative_humidity_quality_assessment.png` - Data validation

### **3. RH Percentiles** (`viz_03_rh_percentiles.py`)
**Input:** RH percentile file from `03_calculate_RH_percentiles.py`
- **P33/P66 threshold maps** for dry/humid classification
- **Seasonal cycles** of humidity thresholds
- **Comparison** with absolute thresholds (33%/66%)
- **Climate zone analysis** based on humidity patterns

**Key Plots:**
- `rh_percentiles_global_annual.png` - P33/P66 threshold maps
- `rh_percentiles_seasonal_cycle.png` - Temporal variations
- `rh_percentiles_threshold_comparison.png` - Absolute vs percentile
- `rh_percentiles_climate_zones.png` - Humidity-based climate zones

### **4. Heat Index** (`viz_04_heat_index.py`)
**Input:** Heat index files from `04_Heat_Index.py`
- **Global heat index climatology** with danger zones
- **Category analysis** by heat stress levels
- **Temporal patterns** and seasonal distributions
- **Regional comparisons** for heat stress hotspots

**Key Plots:**
- `heat_index_climatology_*.png` - Seasonal heat index maps
- `heat_index_categories.png` - Danger level analysis
- `heat_index_relationships.png` - Temporal and spatial patterns
- `heat_index_regional_comparison.png` - Regional heat stress

### **5. Heatwave Metrics** (`viz_05_heatwave_metrics.py`)
**Input:** Heatwave files from `05_Heatwave_Metrics.py`
- **Global heatwave climatology** (frequency, intensity, duration)
- **Temporal trends** and variability analysis
- **Hotspot identification** and extreme event mapping
- **Event-level analysis** and Tmax vs Tmin comparison

**Key Plots:**
- `heatwave_climatology_*.png` - HWN, HWMT, HWTD, HWLD maps
- `heatwave_trends.png` - Multi-year trends
- `heatwave_hotspots_*.png` - Extreme event regions
- `heatwave_event_analysis.png` - Individual event statistics
- `heatwave_tmax_tmin_comparison.png` - Variable comparison

### **6. Humidity Classification** (`viz_06_humidity_classification.py`)
**Input:** Humidity classification files from `06_Humidity_Classification.py`
- **Humid vs dry heatwave patterns** globally
- **Event-level humidity analysis** and distributions
- **Temporal trends** in humidity categories
- **Day-level classification** patterns

**Key Plots:**
- `humidity_patterns_*.png` - Dry/humid heatwave maps
- `humidity_event_analysis.png` - Event-level statistics
- `humidity_trends.png` - Multi-year humidity trends
- `humidity_regional_analysis.png` - Regional patterns
- `humidity_day_analysis.png` - Daily classification analysis

### **7. Precipitation Analysis** (`viz_07_precipitation_analysis.py`)
**Input:** Heatwave-precipitation analysis files from `07_Heatwave_Precipitation_Analysis.py`
- **Compound event patterns** (drought-heat, wet heatwaves)
- **Recovery analysis** after heatwaves
- **Event-level precipitation** characteristics
- **Temporal trends** in compound events

**Key Plots:**
- `compound_event_patterns.png` - Global compound event maps
- `precipitation_recovery_analysis.png` - Post-heatwave recovery patterns
- `event_precipitation_analysis.png` - Individual event analysis
- `precipitation_temporal_trends.png` - Multi-year trends
- `regional_compound_analysis.png` - Regional compound event patterns

## 🚀 **Quick Start**

### **Run All Visualizations**
```bash
# Run complete visualization suite
python visualizations/run_all_visualizations.py \
    --output-dir visualizations/output \
    --create-seasonal \
    --skip-failed
```

### **Run Individual Components**
```bash
# Temperature percentiles
python visualizations/viz_01_temperature_percentiles.py \
    --percentile-dir data/processed/percentiles \
    --output-dir visualizations/output/temperature_percentiles \
    --create-seasonal

# Heatwave metrics
python visualizations/viz_05_heatwave_metrics.py \
    --metrics-dir data/processed/heatwave_metrics \
    --output-dir visualizations/output/heatwave_metrics \
    --years 2020 2021 2022 \
    --variables tmax tmin
```

## 📋 **Command Line Options**

### **Common Parameters**
- `--output-dir`: Output directory for plots
- `--years`: Specific years to analyze
- `--variables`: Variables to process (`tmax`, `tmin`)
- `--create-seasonal`: Generate seasonal plots (DJF, MAM, JJA, SON)

### **Master Script Options** (`run_all_visualizations.py`)
- `--skip-failed`: Continue if individual components fail
- `--components`: Run specific components only
- `--percentiles-dir`: Temperature percentiles directory
- `--rh-dir`: Relative humidity directory
- `--heatwave-dir`: Heatwave metrics directory
- `--humidity-dir`: Humidity classification directory

## 📁 **Output Structure**

```
visualizations/output/
├── visualization_summary.html          # Master summary page
├── 01_temperature_percentiles/
│   ├── temperature_percentiles_global_annual.png
│   ├── temperature_percentiles_seasonal_cycle.png
│   └── ...
├── 02_relative_humidity/
│   ├── relative_humidity_climatology_annual.png
│   ├── relative_humidity_temporal_patterns.png
│   └── ...
├── 03_rh_percentiles/
├── 04_heat_index/
├── 05_heatwave_metrics/
└── 06_humidity_classification/
```

## 🎨 **Scientific Features**

### **Publication-Quality Graphics**
- **High-resolution** PNG output (300 DPI)
- **Professional color schemes** with accessibility considerations
- **Proper map projections** (Robinson, PlateCarree)
- **Statistical significance** indicators where appropriate

### **Scientific Color Schemes**
- **Temperature:** Red-Yellow-Blue (RdYlBu_r)
- **Humidity:** Brown-Green (BrBG)
- **Heat Index:** Custom danger-level colors
- **Heatwaves:** Hot/Plasma scales for intensity
- **Categories:** Distinct qualitative colors

### **Comprehensive Analysis**
- **Spatial patterns** and global/regional distributions
- **Temporal trends** and seasonal cycles
- **Statistical summaries** and validation metrics
- **Comparative analysis** between methods/variables
- **Quality assessment** and data validation

## 🔧 **Dependencies**

```bash
pip install matplotlib cartopy xarray pandas seaborn numpy pathlib
```

**Required packages:**
- `matplotlib` ≥ 3.5.0 (plotting)
- `cartopy` ≥ 0.20.0 (map projections)
- `xarray` ≥ 0.20.0 (netCDF handling)
- `pandas` ≥ 1.4.0 (data analysis)
- `seaborn` ≥ 0.11.0 (statistical plots)

## 📖 **Usage Examples**

### **Complete Analysis Workflow**
```bash
# 1. Run all components with seasonal analysis
python visualizations/run_all_visualizations.py \
    --output-dir results/visualizations \
    --create-seasonal \
    --years 2020 2021 2022 2023

# 2. View results
open results/visualizations/visualization_summary.html
```

### **Specific Component Analysis**
```bash
# Focus on heatwave patterns for recent years
python visualizations/viz_05_heatwave_metrics.py \
    --metrics-dir data/processed/heatwave_metrics \
    --output-dir results/heatwaves_2020s \
    --years 2020 2021 2022 2023 2024

# Compare humidity classification methods
python visualizations/viz_06_humidity_classification.py \
    --humidity-dir data/processed/humidity_classification \
    --output-dir results/humidity_comparison
```

### **Regional Focus**
```bash
# All visualizations with custom data paths
python visualizations/run_all_visualizations.py \
    --percentiles-dir /path/to/percentiles \
    --heatwave-dir /path/to/heatwaves \
    --humidity-dir /path/to/humidity \
    --output-dir results/regional_analysis \
    --years 2015 2016 2017 2018 2019 2020
```

## 🏆 **Best Practices**

### **For Publication**
1. **Use high-resolution output** (300 DPI default)
2. **Include error bars** and confidence intervals where available
3. **Verify color accessibility** for colorblind readers
4. **Add proper citations** and data sources
5. **Include statistical significance** tests

### **For Analysis**
1. **Start with overview plots** (global patterns)
2. **Focus on specific regions** or time periods
3. **Compare multiple variables** (tmax vs tmin)
4. **Validate data quality** before interpretation
5. **Document methodology** and assumptions

### **Performance Tips**
1. **Limit years** for large datasets (`--years` option)
2. **Use chunking** for memory-intensive plots
3. **Skip seasonal plots** for faster processing
4. **Run components separately** for debugging

## 🔍 **Troubleshooting**

### **Common Issues**

**Missing Data:**
```bash
# Check data availability first
python visualizations/run_all_visualizations.py --components temperature_percentiles
```

**Memory Issues:**
```bash
# Reduce years or use smaller chunks
python visualizations/viz_02_relative_humidity.py --years 2020 2021
```

**Map Projection Errors:**
```bash
# Install cartopy with proper dependencies
conda install cartopy
# or
pip install cartopy --no-binary cartopy
```

### **Debug Mode**
```bash
# Run individual scripts with Python -u for unbuffered output
python -u visualizations/viz_05_heatwave_metrics.py [options]
```

## 📚 **Scientific Background**

This visualization suite implements state-of-the-art methods for analyzing compound heat-humidity extremes:

- **Percentile-based thresholds** for robust extreme event detection
- **Humidity classification** using both absolute and relative approaches  
- **Multi-variable analysis** of temperature and moisture interactions
- **Spatiotemporal patterns** across different climate zones
- **Event-based metrics** for comprehensive characterization

The plots follow best practices for scientific visualization and are designed for use in peer-reviewed publications, reports, and presentations.

---

**Created for the Humid Heat Analysis Pipeline**  
*For questions or issues, please refer to the main project documentation.*
