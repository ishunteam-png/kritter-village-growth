"""
Step 1d: Annual Sentinel-1 SAR VV backscatter composites for India (2019–2024).

Sentinel-1 uses radar — it sees through monsoon clouds year-round, filling the
gap left by Sentinel-2 (which is cloud-blocked June–September over India).
VV backscatter increases with urban/built-up surfaces; change 2019→2024 is a
cloud-independent proxy for infrastructure expansion.

Metric: delta_vv = median_vv_2024 - median_vv_2019  (dB, positive = more built-up)

Uses Element84 earth-search STAC (no auth required).
Collection search order:
  1. sentinel-1-rtc  (Radiometric Terrain Corrected, linear power)
  2. sentinel-1-grd  (Ground Range Detected, fallback)

Search is split into 4 quarterly windows per year to maximise scene count,
since the quarterly limit is applied per search and India receives 12-day
revisits (~30 scenes/year/cell).

Output: <data_root>/processed/s1_vv_{year}.tif  (dB scale)
Runtime: ~3–5 h on t3.xlarge.
"""

import warnings
import traceback
import yaml
import numpy as np
import rasterio
import rasterio.warp
from rasterio.merge import merge
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from rasterio.enums import Resampling
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pystac_client

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
_CFG_PATH = Path(__file__).parent.parent / "config.yaml"
_CFG      = yaml.safe_load(_CFG_PATH.read_text()) if _CFG_PATH.exists() else {}
_data     = _CFG.get("paths", {}).get("data_root", "/data/satellite/kritter")

PROC_DIR  = Path(_data) / "processed"          # was hardcoded with typo "satalite"
YEARS     = [2019, 2024]
STAC_URL  = "https://earth-search.aws.element84.com/v1"
MAX_ITEMS = 10      # per quarterly window; 4 windows × 10 = up to 40 scenes/year
MIN_SCENES = 3
OUT_RES   = 0.0009   # ~100 m
CELL_DEG  = 3.0
N_WORKERS = 4

INDIA_W, INDIA_E = 68.0, 97.5
INDIA_S, INDIA_N =  7.5, 37.5

# S1 works year-round (radar); quarterly windows maximise scene discovery
# given Element84 per-search limits.
QUARTERLY_WINDOWS = [
    ("{year}-01-01", "{year}-03-31"),
    ("{year}-04-01", "{year}-06-30"),
    ("{year}-07-01", "{year}-09-30"),
    ("{year}-10-01", "{year}-12-31"),
]

# Collection search order: RTC preferred (already terrain-corrected + calibrated);
# GRD as fallback if RTC unavailable for this region/year.
S1_COLLECTIONS = ["sentinel-1-rtc", "sentinel-1-grd"]


def make_grid():
    cells = []
    lat = INDIA_S
    while lat < INDIA_N:
        lon = INDIA_W
        while lon < INDIA_E:
            cells.append([round(lon, 4), round(lat, 4),
                          round(min(lon + CELL_DEG, INDIA_E), 4),
                          round(min(lat + CELL_DEG, INDIA_N), 4)])
            lon += CELL_DEG
        lat += CELL_DEG
    return cells


def search_s1_scenes(catalog, bbox, year):
    """
    Search across all quarterly windows and both S1 collections.
    Returns deduplicated list of STAC items with a 'vv' or equivalent asset.
    """
    seen, items = set(), []
    for collection in S1_COLLECTIONS:
        for start_tmpl, end_tmpl in QUARTERLY_WINDOWS:
            start = start_tmpl.format(year=year)
            end   = end_tmpl.format(year=year)
            try:
                results = list(catalog.search(
                    collections=[collection],
                    bbox=bbox,
                    datetime=f"{start}/{end}",
                    max_items=MAX_ITEMS,
                ).items())
                for item in results:
                    if item.id not in seen:
                        # Confirm item has a VV band asset before accepting
                        vv_key = next(
                            (k for k in ("vv", "VV", "hh", "HH")
                             if k in item.assets), None)
                        if vv_key:
                            seen.add(item.id)
                            items.append(item)
            except Exception as e:
                # Individual quarter/collection failures are non-fatal
                pass
        if len(items) >= MIN_SCENES:
            # Enough scenes found from this collection — skip GRD fallback
            break
    return items


def process_cell(args):
    bbox, year, cell_dir_str = args
    cell_dir = Path(cell_dir_str)
    cell_id  = f"{bbox[1]:.0f}N{bbox[0]:.0f}E"
    out_path = cell_dir / f"s1vv_{year}_{cell_id}.tif"

    if out_path.exists():
        return str(out_path)

    try:
        catalog = pystac_client.Client.open(STAC_URL)
        items   = search_s1_scenes(catalog, bbox, year)

        if len(items) < MIN_SCENES:
            return None

        nrows = max(2, round((bbox[3] - bbox[1]) / OUT_RES))
        ncols = max(2, round((bbox[2] - bbox[0]) / OUT_RES))
        transform = from_bounds(*bbox, ncols, nrows)

        vv_stack = []
        for item in items:
            try:
                vv_key = next(
                    (k for k in ("vv", "VV", "hh", "HH")
                     if k in item.assets), None)
                if not vv_key:
                    continue

                arr = np.zeros((nrows, ncols), dtype="float32")
                with rasterio.open(item.assets[vv_key].href) as src:
                    rasterio.warp.reproject(
                        source=rasterio.band(src, 1),
                        destination=arr,
                        dst_transform=transform,
                        dst_crs="EPSG:4326",
                        resampling=Resampling.bilinear,
                        src_nodata=0,
                        dst_nodata=0,
                    )

                # Convert linear power to dB; mask ocean/invalid
                # Land VV: typically -20 to -3 dB; ocean: -25 to -35 dB
                arr_db = np.where(arr > 0, 10 * np.log10(arr + 1e-10), np.nan)
                arr_db = np.where((arr_db > -35) & (arr_db < 5), arr_db, np.nan)
                if np.isnan(arr_db).mean() > 0.95:
                    continue   # skip scenes that are mostly NaN after masking
                vv_stack.append(arr_db)

            except Exception:
                continue

        if not vv_stack:
            return None

        median_vv = np.nanmedian(np.stack(vv_stack), axis=0).astype("float32")
        median_vv[np.isnan(median_vv)] = -9999.0

        with rasterio.open(out_path, "w", driver="GTiff",
                           height=nrows, width=ncols, count=1, dtype="float32",
                           crs=CRS.from_epsg(4326), transform=transform,
                           nodata=-9999.0, compress="lzw") as dst:
            dst.write(median_vv, 1)
        return str(out_path)

    except Exception:
        traceback.print_exc()
        return None


def mosaic_and_save(cell_paths, out_path):
    valid = [p for p in cell_paths if p and Path(p).exists()]
    if not valid:
        print(f"  ⚠  No valid cells — mosaic skipped for {out_path.name}")
        return
    srcs = [rasterio.open(p) for p in valid]
    mosaic, transform = merge(srcs, nodata=-9999.0)
    meta = srcs[0].meta.copy()
    meta.update(height=mosaic.shape[1], width=mosaic.shape[2],
                transform=transform, compress="lzw", nodata=-9999.0)
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(mosaic)
    for s in srcs:
        s.close()
    print(f"  Saved: {out_path.name}  ({len(valid)} cells)")


def main():
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    cell_dir = PROC_DIR / "s1_cells"
    cell_dir.mkdir(exist_ok=True)

    grid = make_grid()
    print(f"Sentinel-1 SAR: {len(grid)} cells  |  {OUT_RES*111000:.0f} m resolution")
    print(f"Collections: {S1_COLLECTIONS}  |  4 quarterly windows per year")
    print(f"Output dir: {PROC_DIR}")

    for year in YEARS:
        mosaic_path = PROC_DIR / f"s1_vv_{year}.tif"
        if mosaic_path.exists():
            print(f"{year}: already done")
            continue

        print(f"\n── Year {year} ────────────────────────────────────────────────")
        paths = []
        with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
            futures = {
                ex.submit(process_cell, (bbox, year, str(cell_dir))): bbox
                for bbox in grid
            }
            done, succeeded = 0, 0
            for fut in as_completed(futures):
                bbox = futures[fut]
                p = fut.result()
                paths.append(p)
                done += 1
                if p:
                    succeeded += 1
                status = "✓" if p else "✗"
                print(f"  {status} {bbox[1]:.0f}°N {bbox[0]:.0f}°E  "
                      f"[{done}/{len(grid)}]  ({succeeded} ok so far)")

        good = sum(1 for p in paths if p)
        print(f"\n  {good}/{len(grid)} cells succeeded")
        if good == 0:
            print("  ⚠  All cells failed — check STAC connectivity and PROC_DIR path")
        mosaic_and_save(paths, mosaic_path)

    print("\nSentinel-1 SAR complete.")


if __name__ == "__main__":
    main()
