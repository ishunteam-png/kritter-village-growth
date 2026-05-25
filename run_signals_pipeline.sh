#!/bin/bash
# Run Sentinel-2 NDBI → SAR → downstream scoring pipeline
# Usage: nohup bash run_signals_pipeline.sh > /data/satalite/kritter/logs/pipeline_rerun.log 2>&1 &

set -e
cd /home/ubuntu/kritter
CONDA_RUN="conda run -n insar python3 -u"
LOG_DIR="/data/satalite/kritter/logs"
mkdir -p "$LOG_DIR"

echo "========================================="
echo "Pipeline start: $(date)"
echo "========================================="

echo ""
echo "--- Step 01c: Sentinel-2 NDBI ---"
$CONDA_RUN src/01c_sentinel2_ndbi.py 2>&1
echo "01c complete: $(date)"

echo ""
echo "--- Step 01d: Sentinel-1 SAR ---"
$CONDA_RUN src/01d_sentinel1_sar.py 2>&1
echo "01d complete: $(date)"

echo ""
echo "--- Step 03: Village stats (merge signals) ---"
$CONDA_RUN src/03_village_stats.py 2>&1
echo "03 complete: $(date)"

echo ""
echo "--- Step 04: Score + rank ---"
$CONDA_RUN src/04_score_rank.py 2>&1
echo "04 complete: $(date)"

echo ""
echo "--- Step 07: Validate (Morans I, bootstrap) ---"
$CONDA_RUN src/07_validate.py 2>&1
echo "07 complete: $(date)"

echo ""
echo "--- Step 12: SHAP archetypes ---"
$CONDA_RUN src/12_shap_archetypes.py 2>&1
echo "12 complete: $(date)"

echo ""
echo "--- Step 05: Visualize ---"
$CONDA_RUN src/05_visualize.py 2>&1
echo "05 complete: $(date)"

echo ""
echo "--- Step 06: Upload to S3 ---"
$CONDA_RUN src/06_upload_s3.py 2>&1
echo "06 complete: $(date)"

echo ""
echo "========================================="
echo "Pipeline COMPLETE: $(date)"
echo "========================================="
