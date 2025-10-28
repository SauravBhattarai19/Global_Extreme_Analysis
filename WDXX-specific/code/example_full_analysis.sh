#!/bin/bash
################################################################################
# Complete Precipitation Index Analysis Pipeline
# 
# This script runs the full analysis workflow for precipitation indices
# 
# Author: Saurav Bhattarai
# Date: October 28, 2025
################################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable

################################################################################
# CONFIGURATION
################################################################################

# Directories
PRECIP_DIR="/data/climate/disk1/datasets/era5"
BASE_OUTPUT="results"
CODE_DIR="code"

# Analysis period
BASELINE_START=1980
BASELINE_END=2000
ANALYSIS_START=1991
ANALYSIS_END=2020

# Processing parameters
N_PROCESSES=24
CHUNK_LAT=50
CHUNK_LON=100

# Thresholds
WET_DAY_THRESHOLD=0.1  # mm/day for WD50R
WET_DAY_ETCCDI=1.0     # mm/day for ETCCDI
MIN_ANNUAL_PRECIP=50   # mm/year

# Percentiles for WD50R
PERCENTILES="25 50 75 90"

################################################################################
# SETUP
################################################################################

echo "=============================================================================="
echo "PRECIPITATION INDEX ANALYSIS PIPELINE"
echo "=============================================================================="
echo "Baseline period: ${BASELINE_START}-${BASELINE_END}"
echo "Analysis period: ${ANALYSIS_START}-${ANALYSIS_END}"
echo "Processes: ${N_PROCESSES}"
echo "=============================================================================="
echo ""

# Create output directories
mkdir -p ${BASE_OUTPUT}/{percentiles,wd50r_indices,etccdi_indices,concentration_indices}
mkdir -p logs

# Get start time
START_TIME=$(date +%s)

################################################################################
# STEP 1: CALCULATE PRECIPITATION PERCENTILES (Baseline)
################################################################################

echo "STEP 1: Calculating precipitation percentiles..."
echo "  Period: ${BASELINE_START}-${BASELINE_END}"
echo "  Output: ${BASE_OUTPUT}/percentiles/precip_percentiles.nc"

PERCENTILE_FILE="${BASE_OUTPUT}/percentiles/precip_percentiles_${BASELINE_START}-${BASELINE_END}.nc"

if [ -f "$PERCENTILE_FILE" ]; then
    echo "  ⚠️  Percentile file already exists. Skipping..."
else
    python ${CODE_DIR}/00_calculate_precipitation_percentiles.py \
        --precip-dir ${PRECIP_DIR} \
        --start-year ${BASELINE_START} \
        --end-year ${BASELINE_END} \
        --output-file ${PERCENTILE_FILE} \
        --percentiles 10 25 50 75 90 95 \
        --n-processes ${N_PROCESSES} \
        --chunk-size-lat ${CHUNK_LAT} \
        --chunk-size-lon ${CHUNK_LON} \
        2>&1 | tee logs/00_percentiles.log
    
    echo "  ✅ Percentiles calculated successfully!"
fi

echo ""

################################################################################
# STEP 2: CALCULATE WD50R INDICES (Analysis Period)
################################################################################

echo "STEP 2: Calculating WD50R indices..."
echo "  Period: ${ANALYSIS_START}-${ANALYSIS_END}"
echo "  Percentiles: ${PERCENTILES}"
echo "  Output: ${BASE_OUTPUT}/wd50r_indices/"

WD50R_COUNT=0
WD50R_SKIP=0

for YEAR in $(seq ${ANALYSIS_START} ${ANALYSIS_END}); do
    OUTPUT_FILE="${BASE_OUTPUT}/wd50r_indices/wd50r_indices_${YEAR}.nc"
    
    if [ -f "$OUTPUT_FILE" ]; then
        echo "  ⚠️  Year ${YEAR} already processed. Skipping..."
        WD50R_SKIP=$((WD50R_SKIP + 1))
    else
        echo "  Processing year ${YEAR}..."
        
        python ${CODE_DIR}/10_WD50R_Chronological_Precipitation_Index.py \
            --year ${YEAR} \
            --precip-dir ${PRECIP_DIR} \
            --output-dir ${BASE_OUTPUT}/wd50r_indices \
            --percentiles ${PERCENTILES} \
            --wet-day-threshold ${WET_DAY_THRESHOLD} \
            --min-annual-precip ${MIN_ANNUAL_PRECIP} \
            --n-processes ${N_PROCESSES} \
            --chunk-size-lat ${CHUNK_LAT} \
            --chunk-size-lon ${CHUNK_LON} \
            2>&1 | tee logs/10_wd50r_${YEAR}.log
        
        echo "  ✅ Year ${YEAR} completed!"
        WD50R_COUNT=$((WD50R_COUNT + 1))
    fi
done

echo "  Summary: ${WD50R_COUNT} years processed, ${WD50R_SKIP} years skipped"
echo ""

################################################################################
# STEP 3: CALCULATE ETCCDI INDICES (Analysis Period)
################################################################################

echo "STEP 3: Calculating ETCCDI indices..."
echo "  Period: ${ANALYSIS_START}-${ANALYSIS_END}"
echo "  Output: ${BASE_OUTPUT}/etccdi_indices/"

ETCCDI_COUNT=0
ETCCDI_SKIP=0

for YEAR in $(seq ${ANALYSIS_START} ${ANALYSIS_END}); do
    OUTPUT_FILE="${BASE_OUTPUT}/etccdi_indices/etccdi_indices_${YEAR}.nc"
    
    if [ -f "$OUTPUT_FILE" ]; then
        echo "  ⚠️  Year ${YEAR} already processed. Skipping..."
        ETCCDI_SKIP=$((ETCCDI_SKIP + 1))
    else
        echo "  Processing year ${YEAR}..."
        
        python ${CODE_DIR}/08_ETCCDI_Precipitation_Indices.py \
            --year ${YEAR} \
            --precip-dir ${PRECIP_DIR} \
            --percentile-file ${PERCENTILE_FILE} \
            --output-dir ${BASE_OUTPUT}/etccdi_indices \
            --wet-day-threshold ${WET_DAY_ETCCDI} \
            --n-processes ${N_PROCESSES} \
            --chunk-size-lat ${CHUNK_LAT} \
            --chunk-size-lon ${CHUNK_LON} \
            2>&1 | tee logs/08_etccdi_${YEAR}.log
        
        echo "  ✅ Year ${YEAR} completed!"
        ETCCDI_COUNT=$((ETCCDI_COUNT + 1))
    fi
done

echo "  Summary: ${ETCCDI_COUNT} years processed, ${ETCCDI_SKIP} years skipped"
echo ""

################################################################################
# STEP 4: CALCULATE CONCENTRATION INDICES (Analysis Period)
################################################################################

echo "STEP 4: Calculating concentration indices (Gini, Lorenz, Entropy)..."
echo "  Period: ${ANALYSIS_START}-${ANALYSIS_END}"
echo "  Output: ${BASE_OUTPUT}/concentration_indices/"

CONC_COUNT=0
CONC_SKIP=0

for YEAR in $(seq ${ANALYSIS_START} ${ANALYSIS_END}); do
    OUTPUT_FILE="${BASE_OUTPUT}/concentration_indices/concentration_indices_all_${YEAR}.nc"
    
    if [ -f "$OUTPUT_FILE" ]; then
        echo "  ⚠️  Year ${YEAR} already processed. Skipping..."
        CONC_SKIP=$((CONC_SKIP + 1))
    else
        echo "  Processing year ${YEAR}..."
        
        python ${CODE_DIR}/09_enhanced_precipitation_concentration_indices.py \
            --year ${YEAR} \
            --precip-dir ${PRECIP_DIR} \
            --output-dir ${BASE_OUTPUT}/concentration_indices \
            --method all \
            --wet-day-threshold ${WET_DAY_ETCCDI} \
            --n-processes ${N_PROCESSES} \
            --chunk-size-lat ${CHUNK_LAT} \
            --chunk-size-lon ${CHUNK_LON} \
            2>&1 | tee logs/09_concentration_${YEAR}.log
        
        echo "  ✅ Year ${YEAR} completed!"
        CONC_COUNT=$((CONC_COUNT + 1))
    fi
done

echo "  Summary: ${CONC_COUNT} years processed, ${CONC_SKIP} years skipped"
echo ""

################################################################################
# SUMMARY
################################################################################

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))

echo "=============================================================================="
echo "ANALYSIS PIPELINE COMPLETED!"
echo "=============================================================================="
echo "Total runtime: ${HOURS}h ${MINUTES}m"
echo ""
echo "Output Summary:"
echo "  Percentiles:   ${BASE_OUTPUT}/percentiles/"
echo "  WD50R:         ${BASE_OUTPUT}/wd50r_indices/ (${WD50R_COUNT} years)"
echo "  ETCCDI:        ${BASE_OUTPUT}/etccdi_indices/ (${ETCCDI_COUNT} years)"
echo "  Concentration: ${BASE_OUTPUT}/concentration_indices/ (${CONC_COUNT} years)"
echo ""
echo "Logs saved in: logs/"
echo "=============================================================================="
echo ""

# Count output files
echo "File counts:"
echo "  WD50R NetCDF:        $(ls ${BASE_OUTPUT}/wd50r_indices/*.nc 2>/dev/null | wc -l)"
echo "  WD50R text tables:   $(ls ${BASE_OUTPUT}/wd50r_indices/*.txt 2>/dev/null | wc -l)"
echo "  ETCCDI NetCDF:       $(ls ${BASE_OUTPUT}/etccdi_indices/*.nc 2>/dev/null | wc -l)"
echo "  Concentration NetCDF: $(ls ${BASE_OUTPUT}/concentration_indices/*.nc 2>/dev/null | wc -l)"
echo ""

# Estimate total size
echo "Storage usage:"
du -sh ${BASE_OUTPUT}/wd50r_indices/ 2>/dev/null || echo "  WD50R: N/A"
du -sh ${BASE_OUTPUT}/etccdi_indices/ 2>/dev/null || echo "  ETCCDI: N/A"
du -sh ${BASE_OUTPUT}/concentration_indices/ 2>/dev/null || echo "  Concentration: N/A"
echo ""

echo "✅ All indices calculated successfully!"
echo ""
echo "Next steps:"
echo "  1. Check logs in logs/ for any errors"
echo "  2. Verify output files are complete"
echo "  3. Run analysis scripts to generate figures"
echo "  4. See docs/QUICK_START.md for visualization examples"
echo ""

