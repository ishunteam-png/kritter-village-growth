#!/bin/bash
# One-time EC2 setup script.
# Run after SSH-ing into the EC2 instance and cloning the repo.
#
# Usage:
#   chmod +x src/00_setup_ec2.sh
#   bash src/00_setup_ec2.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
echo "Repo: $REPO_DIR"

# ── 1. Create conda environment from environment.yml ─────────────────────────
if conda env list | grep -q "^insar "; then
    echo "Conda env 'insar' already exists — skipping create"
else
    echo "Creating conda env 'insar' from environment.yml..."
    conda env create -f "$REPO_DIR/environment.yml"
    echo "Conda env created."
fi

# Activate for remaining steps in this script
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate insar

# ── 2. Install any pip-only packages not in environment.yml ──────────────────
pip install \
    earthaccess \
    h5py \
    rasterstats \
    rioxarray \
    boto3 \
    folium \
    plotly \
    kaleido \
    branca \
    tqdm \
    pystac-client \
    --quiet

# ── 3. Verify core imports ────────────────────────────────────────────────────
python -c "
import rasterio, geopandas, rasterstats, earthaccess
import boto3, folium, plotly, pystac_client, sklearn, joblib
print('All packages OK')
"

# ── 4. Create working directories on the EBS volume (mounted at /data) ───────
# Pipeline writes to /data/satellite/kritter/ — must match config.yaml paths.
mkdir -p /data/satellite/kritter/viirs/{2019,2020,2021,2022,2023,2024}
mkdir -p /data/satellite/kritter/{shrug,processed,output,logs}
mkdir -p /data/satellite/kritter/processed/{s2_cells,s1_cells,ghsl_raw}
echo "Data directories created under /data/satellite/kritter/"

# ── 5. Download Natural Earth admin-1 shapefile (India polygon filter) ────────
NE_SHP="/tmp/ne_10m_admin_1_states_provinces.shp"
if [ ! -f "$NE_SHP" ]; then
    echo "Downloading Natural Earth admin-1 shapefile..."
    curl -sL "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_1_states_provinces.zip" \
        -o /tmp/ne10m.zip
    unzip -q /tmp/ne10m.zip -d /tmp/
    echo "  Extracted to /tmp/"
else
    echo "Natural Earth shapefile already present."
fi

echo ""
echo "EC2 setup complete."
echo "Next: conda activate insar && bash run_pipeline.sh"
