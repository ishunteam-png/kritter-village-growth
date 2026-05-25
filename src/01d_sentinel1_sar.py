"""
Step 1d: Annual Sentinel-1 SAR VV backscatter composites for India (2019–2024).

Sentinel-1 uses radar — it sees through monsoon clouds year-round, filling the
gap left by Sentinel-2 (which is cloud-blocked June–September over India).
VV backscatter increases with urban/built-up surfaces; change 2019→2024 is a
cloud-independent proxy for infrastructure expansion.

Metric: delta_vv = median_vv_2024 - median_vv_2019  (dB, positive = more built-up)

Uses Element84 earth-search STAC (no auth required).
Collection: sentinel-1-rtc  (Radiometric Terrain Corrected, linear power units)

Output: /data/satalite/kritter/processed/s1_vv_{year}.tif  (dB scale)
Runtime: ~3–5 h on t3.xlarge.
"""

import warnings, traceback
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

PROC_DIR  = Path("/data/satalite/kritter/processed")
YEARS     = [2019, 2020, 2021, 2022, 2023, 2024]
STAC_URL  = "https://earth-search.aws.element84.com/v1"
MAX_ITEMS = 20
OUT_RES   = 0.0009   # ~100 m
CELL_DEG  = 3.0
N_WORKERS = 4

INDIA_W, INDIA_E = 68.0, 97.5
INDIA_S, INDIA_N =  7.5, 37.5


def make_grid():
    cells = []
    lat = INDIA_S
    while lat < INDIA_N:
        lon = INDIA_W
        while lon < INDIA_E:
            cells.append([round(lon,4), round(lat,4),
                          round(min(lon+CELL_DEG, INDIA_E),4),
                          round(min(lat+CELL_DEG, INDIA_N),4)])
            lon += CELL_DEG
        lat += CELL_DEG
    return cells


def process_cell(args):
    bbox, year, cell_dir_str = args
    cell_dir = Path(cell_dir_str)
    cell_id  = f"{bbox[1]:.0f}N{bbox[0]:.0f}E"
    out_path = cell_dir / f"s1vv_{year}_{cell_id}.tif"

    if out_path.exists():
        return str(out_path)

    try:
        catalog = pystac_client.Client.open(STAC_URL)
        items = list(catalog.search(
            collections=["sentinel-1-rtc"],
            bbox=bbox,
            datetime=f"{year}-01-01/{year}-12-31",
            max_items=MAX_ITEMS,
        ).items())

        # Try GRD collection as fallback
        if len(items) < 3:
            items = list(catalog.search(
                collections=["sentinel-1-grd"],
                bbox=bbox,
                datetime=f"{year}-01-01/{year}-12-31",
                max_items=MAX_ITEMS,
            ).items())

        if len(items) < 3:
            return None

        nrows = max(2, round((bbox[3]-bbox[1]) / OUT_RES))
        ncols = max(2, round((bbox[2]-bbox[0]) / OUT_RES))
        transform = from_bounds(*bbox, ncols, nrows)

        vv_stack = []
        for item in items:
            try:
                vv_key = next((k for k in ("vv","VV","hh","HH") if k in item.assets), None)
                if not vv_key:
                    continue
                with rasterio.open(item.assets[vv_key].href) as src:
                    arr = np.zeros((nrows, ncols), dtype="float32")
                    rasterio.warp.reproject(
                        source=rasterio.band(src, 1),
                        destination=arr,
                        dst_transform=transform,
                        dst_crs="EPSG:4326",
                        resampling=Resampling.bilinear,
                        src_nodata=0, dst_nodata=0,
                    )
                # Convert linear power to dB
                arr = np.where(arr > 0, 10 * np.log10(arr), np.nan)
                # Mask ocean/invalid (VV over ocean ≈ -25 to -35 dB, land -15 to -5 dB)
                arr = np.where((arr > -40) & (arr < 10), arr, np.nan)
                vv_stack.append(arr)
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
        return
    srcs = [rasterio.open(p) for p in valid]
    mosaic, transform = merge(srcs, nodata=-9999.0)
    meta = srcs[0].meta.copy()
    meta.update(height=mosaic.shape[1], width=mosaic.shape[2],
                transform=transform, compress="lzw", nodata=-9999.0)
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(mosaic)
    for s in srcs: s.close()
    print(f"  Saved: {out_path.name}  ({len(valid)} cells)")


def main():
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    cell_dir = PROC_DIR / "s1_cells"
    cell_dir.mkdir(exist_ok=True)

    grid = make_grid()
    print(f"Sentinel-1 SAR: {len(grid)} cells  |  {OUT_RES*111000:.0f} m resolution")

    for year in YEARS:
        mosaic_path = PROC_DIR / f"s1_vv_{year}.tif"
        if mosaic_path.exists():
            print(f"{year}: already done"); continue

        print(f"\n── Year {year} ────────────────────────")
        paths = []
        with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
            futures = {ex.submit(process_cell, (bbox, year, str(cell_dir))): bbox
                       for bbox in grid}
            done = 0
            for fut in as_completed(futures):
                bbox = futures[fut]
                p = fut.result()
                paths.append(p)
                done += 1
                print(f"  {'✓' if p else '✗'} {bbox[1]:.0f}°N {bbox[0]:.0f}°E  [{done}/{len(grid)}]")

        mosaic_and_save(paths, mosaic_path)

    print("\nSentinel-1 SAR complete.")


if __name__ == "__main__":
    main()
