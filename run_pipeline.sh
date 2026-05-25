#!/usr/bin/env bash
# Full pipeline — run on EC2 (conda activate insar first)
#
# Usage:
#   bash run_pipeline.sh                        full run (phase A + B + C + D)
#   bash run_pipeline.sh --skip-viirs           skip VIIRS re-download
#   bash run_pipeline.sh --skip-viirs --skip-s2 skip VIIRS + Sentinel-2
#   bash run_pipeline.sh --phase-c-only         scoring/viz/validate/upload only
#
# Sentinel-2 NDBI (01c) and Sentinel-1 SAR (01d) run in BACKGROUND.
# After they finish, re-run with --phase-c-only to incorporate new signals.

set -euo pipefail
PY="$(/opt/miniconda3/envs/insar/bin/python -c 'import sys; print(sys.executable)')"
SRC="/home/ubuntu/kritter/src"
LOGS="/data/satellite/kritter/logs"
mkdir -p "$LOGS"

SKIP_VIIRS=false; SKIP_S2=false; SKIP_S1=false; PHASE_C_ONLY=false
for arg in "$@"; do
  [[ "$arg" == "--skip-viirs"   ]] && SKIP_VIIRS=true
  [[ "$arg" == "--skip-s2"      ]] && SKIP_S2=true
  [[ "$arg" == "--skip-s1"      ]] && SKIP_S1=true
  [[ "$arg" == "--phase-c-only" ]] && PHASE_C_ONLY=true
done

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "================================================================"
log "  Village Economic Growth Pipeline"
log "================================================================"

if [[ "$PHASE_C_ONLY" == false ]]; then

# ── Phase A: Data download ────────────────────────────────────────────────────

  if [[ "$SKIP_VIIRS" == false ]]; then
    log "Step 00b: OSM village centroids"
    $PY "$SRC/00b_download_villages_osm.py"

    log "Step 01: VIIRS NTL download (2019-2024)"
    $PY "$SRC/01_download_viirs.py"
  fi

  log "Step 01b: WorldPop population"
  $PY "$SRC/01b_worldpop.py"

  if [[ "$SKIP_S2" == false ]]; then
    log "Step 01c: Sentinel-2 NDBI/NDVI (BACKGROUND ~6-10 h)"
    nohup $PY "$SRC/01c_sentinel2_ndbi.py" > "$LOGS/sentinel2.log" 2>&1 &
    log "  Sentinel-2 PID: $!  |  tail -f $LOGS/sentinel2.log"
  fi

  if [[ "$SKIP_S1" == false ]]; then
    log "Step 01d: Sentinel-1 SAR (BACKGROUND ~3-5 h)"
    nohup $PY "$SRC/01d_sentinel1_sar.py" > "$LOGS/sentinel1.log" 2>&1 &
    log "  Sentinel-1 PID: $!  |  tail -f $LOGS/sentinel1.log"
  fi

  log "Step 01e: GHSL built-up 2015/2020"
  $PY "$SRC/01e_ghsl_builtup.py"

  log "Step 01f: OSM mobile tower density"
  $PY "$SRC/01f_osm_towers.py"

  log "Step 01g: Distance to nearest city / highway"
  $PY "$SRC/01g_city_distance.py"

  log "Step 02: WorldCover built-up"
  $PY "$SRC/02_worldcover_builtup.py"

  # Natural Earth shapefile for India polygon filter
  NE_STATES="/tmp/ne_10m_admin_1_states_provinces.shp"
  if [[ ! -f "$NE_STATES" ]]; then
    log "Downloading Natural Earth admin-1 shapefile"
    curl -sL "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_1_states_provinces.zip" \
      -o /tmp/ne10m.zip && unzip -q /tmp/ne10m.zip -d /tmp/
  fi

# ── Phase B: Village statistics ───────────────────────────────────────────────

  log "Step 03: Village statistics (all signals)"
  $PY "$SRC/03_village_stats.py"

fi  # end PHASE_C_ONLY block

# ── Signal completeness check ────────────────────────────────────────────────
# Warns when optional signals are all-NaN (e.g. Sentinel-2/S1 stalled).
# Does NOT abort — composite weights auto-redistribute. Exit code 0 always.
log "Signal completeness check"
$PY - <<'PYEOF'
import pathlib, sys
import pandas as pd

stats_csv = pathlib.Path("/data/satellite/kritter/village_all_stats.csv")
if not stats_csv.exists():
    print("  [SKIP] village_all_stats.csv not found yet", flush=True)
    sys.exit(0)

df = pd.read_csv(stats_csv, nrows=5000)
signals = {
    "VIIRS NTL 2024":      "ntl_2024",
    "WorldCover built-up": "worldcover_builtup_frac",
    "GHSL built-up":       "ghsl_builtup_frac_2020",
    "Sentinel-2 NDBI":     "ndbi_mean",
    "Sentinel-1 SAR VV":   "s1_vv_mean",
    "WorldPop pop 2020":   "worldpop_2020",
    "Tower density":       "tower_density",
}
ok, warn = [], []
for name, col in signals.items():
    if col not in df.columns:
        warn.append(f"  [MISSING ]  {name} ({col})")
    elif df[col].isna().mean() > 0.95:
        warn.append(f"  [ALL-NaN ]  {name} ({col}) — {df[col].isna().mean()*100:.0f}% NaN")
    else:
        ok.append(f"  [OK      ]  {name}  ({df[col].notna().mean()*100:.0f}% coverage)")

for line in ok:   print(line, flush=True)
for line in warn: print(line, flush=True)
if warn:
    print(f"\n  ⚠  {len(warn)} signal(s) missing/all-NaN — composite weights will auto-redistribute.", flush=True)
    print(  "     Re-run --phase-c-only after stalled signals complete to use full composite.", flush=True)
PYEOF

# ── Phase C: Score / Validate / Visualise ────────────────────────────────────

log "Step 04: Scoring (ML + composite + dark track)"
$PY "$SRC/04_score_rank.py"

log "Step 05: Visualisations"
$PY "$SRC/05_visualize.py"

log "Step 07: Validation (Moran's I + bootstrap + correlation)"
$PY "$SRC/07_validate.py"

log "Step 08: Forecast"
$PY "$SRC/08_forecast.py"

log "Step 09: Report cards"
$PY "$SRC/09_report_cards.py"

log "Step 10: Animation"
$PY "$SRC/10_animate.py"

log "Step 12: SHAP explainability + growth archetypes"
$PY "$SRC/12_shap_archetypes.py"

log "Step 13: SECC 2011 ground-truth cross-validation"
$PY "$SRC/13_secc_validation.py"

# ── Phase D: S3 upload ────────────────────────────────────────────────────────

log "Step 06: S3 upload"
$PY "$SRC/06_upload_s3.py"

log "================================================================"
log "  Pipeline complete."
log "  Sentinel-2 / Sentinel-1 may still be running in background."
log "  After they finish: bash run_pipeline.sh --phase-c-only"
log "================================================================"
