"""
Step 1c: Annual Sentinel-2 NDBI & NDVI composites for India (2019–2024).

Uses Element84 earth-search STAC (no authentication required).
Processes India in 3°×3° grid cells with 4 parallel workers,
computes cloud-filtered median composites, and mosaics into single-file
annual GeoTIFFs matching the VIIRS pipeline convention.

  NDBI = (B11 - B08) / (B11 + B08)   built-up proxy (SWIR vs NIR)
  NDVI = (B08 - B04) / (B08 + B04)   vegetation / agricultural proxy

Output resolution: ~100 m (0.0009°) — good trade-off for India-wide mosaic.
Runtime: ~6–10 h on t3.xlarge (4 vCPU).

Install:
    pip install pystac-client rioxarray

Output:
    /data/satellite/kritter/processed/sentinel2_ndbi_{year}.tif
    /data/satellite/kritter/processed/sentinel2_ndvi_{year}.tif
"""

import warnings
import traceback
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from rasterio.enums import Resampling
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import pystac_client

warnings.filterwarnings("ignore")

PROC_DIR  = Path("/data/satellite/kritter/processed")
YEARS     = [2019, 2020, 2021, 2022, 2023, 2024]
STAC_URL   = "https://earth-search.aws.element84.com/v1"
CLOUD_MAX  = 50      # raised from 25: India monsoon (Jun-Sep) blocks low-cloud scenes entirely
MAX_SCENES = 120     # more scenes to compensate for cloud variance
OUT_RES    = 0.0009  # ~100 m in degrees
CELL_DEG   = 3.0     # 3° × 3° processing grid
N_WORKERS  = 4       # parallel workers (= EC2 vCPUs)
# Dry-season month filter: query Oct–May only to avoid Jun–Sep monsoon blackout.
# This is the primary fix for the all-NaN NDBI/SAR issue from the first run.
DRY_SEASON = [(f"{y}-10-01", f"{y+1}-05-31") for y in range(2018, 2025)]

INDIA_W, INDIA_E = 68.0, 97.5
INDIA_S, INDIA_N = 7.5, 37.5


def make_grid():
    cells = []
    lat = INDIA_S
    while lat < INDIA_N:
        lon = INDIA_W
        while lon < INDIA_E:
            cells.append([
                round(lon, 4), round(lat, 4),
                round(min(lon + CELL_DEG, INDIA_E), 4),
                round(min(lat + CELL_DEG, INDIA_N), 4),
            ])
            lon += CELL_DEG
        lat += CELL_DEG
    return cells


def _read_band_window(href: str, bbox: list, nrows: int, ncols: int) -> np.ndarray:
    with rasterio.open(href) as src:
        win = rasterio.windows.from_bounds(*bbox, src.transform)
        arr = src.read(
            1, window=win,
            out_shape=(nrows, ncols),
            resampling=Resampling.bilinear,
            fill_value=0,
        ).astype("float32")
    return arr


def process_cell(args):
    bbox, year, cell_dir_str = args
    cell_dir = Path(cell_dir_str)
    cell_id  = f"{bbox[1]:.0f}N{bbox[0]:.0f}E"

    out_ndbi = cell_dir / f"ndbi_{year}_{cell_id}.tif"
    out_ndvi = cell_dir / f"ndvi_{year}_{cell_id}.tif"

    if out_ndbi.exists() and out_ndvi.exists():
        return (str(out_ndbi), str(out_ndvi))

    try:
        catalog = pystac_client.Client.open(STAC_URL)
        # Dry-season search: Oct(year-1)–May(year) avoids the Jun–Sep monsoon
        # cloud blackout that produced all-NaN composites in the prior run.
        dry_start = f"{year - 1}-10-01"
        dry_end   = f"{year}-05-31"
        items = list(catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=f"{dry_start}/{dry_end}",
            query={"eo:cloud_cover": {"lt": CLOUD_MAX}},
            max_items=MAX_SCENES,
        ).items())

        if len(items) < 3:
            return (None, None)

        nrows = max(2, round((bbox[3] - bbox[1]) / OUT_RES))
        ncols = max(2, round((bbox[2] - bbox[0]) / OUT_RES))
        transform = from_bounds(*bbox, ncols, nrows)

        ndbi_stack, ndvi_stack = [], []

        for item in items:
            try:
                assets = item.assets
                # Band asset names vary between element84 versions
                b04_key = next((k for k in ("B04", "red")    if k in assets), None)
                b08_key = next((k for k in ("B08", "nir")    if k in assets), None)
                b11_key = next((k for k in ("B11", "swir16") if k in assets), None)
                if not (b04_key and b08_key and b11_key):
                    continue

                b04 = _read_band_window(assets[b04_key].href, bbox, nrows, ncols) / 10000.0
                b08 = _read_band_window(assets[b08_key].href, bbox, nrows, ncols) / 10000.0
                b11 = _read_band_window(assets[b11_key].href, bbox, nrows, ncols) / 10000.0

                valid = (b04 > 0.01) & (b08 > 0.01) & (b11 > 0.01) & \
                        (b04 < 1.5)  & (b08 < 1.5)  & (b11 < 1.5)

                denom_ndbi = b11 + b08 + 1e-10
                denom_ndvi = b08 + b04 + 1e-10
                ndbi_stack.append(np.where(valid, (b11 - b08) / denom_ndbi, np.nan))
                ndvi_stack.append(np.where(valid, (b08 - b04) / denom_ndvi, np.nan))

            except Exception:
                continue

        if not ndbi_stack:
            return (None, None)

        def _save(arr, path):
            arr = arr.astype("float32")
            arr[np.isnan(arr)] = -9999.0
            with rasterio.open(path, "w",
                driver="GTiff", height=nrows, width=ncols,
                count=1, dtype="float32",
                crs=CRS.from_epsg(4326), transform=transform,
                nodata=-9999.0, compress="lzw",
            ) as dst:
                dst.write(arr, 1)

        _save(np.nanmedian(np.stack(ndbi_stack), axis=0), out_ndbi)
        _save(np.nanmedian(np.stack(ndvi_stack), axis=0), out_ndvi)
        return (str(out_ndbi), str(out_ndvi))

    except Exception:
        traceback.print_exc()
        return (None, None)


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
    for s in srcs:
        s.close()
    print(f"  Saved: {out_path.name}  ({len(valid)} cells merged)")


def main():
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    cell_dir = PROC_DIR / "s2_cells"
    cell_dir.mkdir(exist_ok=True)

    grid = make_grid()
    print(f"Grid: {len(grid)} cells  |  cloud < {CLOUD_MAX}%  |  {OUT_RES*111000:.0f} m resolution")

    for year in YEARS:
        ndbi_mosaic = PROC_DIR / f"sentinel2_ndbi_{year}.tif"
        ndvi_mosaic = PROC_DIR / f"sentinel2_ndvi_{year}.tif"
        if ndbi_mosaic.exists() and ndvi_mosaic.exists():
            print(f"{year}: already complete, skipping")
            continue

        print(f"\n── Year {year} ──────────────────────────")
        ndbi_paths, ndvi_paths = [], []

        with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
            futures = {
                ex.submit(process_cell, (bbox, year, str(cell_dir))): bbox
                for bbox in grid
            }
            done = 0
            for fut in as_completed(futures):
                bbox = futures[fut]
                nb, nv = fut.result()
                ndbi_paths.append(nb)
                ndvi_paths.append(nv)
                done += 1
                status = "✓" if nb else "✗"
                print(f"  {status} {bbox[1]:.0f}°N {bbox[0]:.0f}°E  [{done}/{len(grid)}]")

        good = sum(1 for p in ndbi_paths if p)
        print(f"  {good}/{len(grid)} cells succeeded")
        mosaic_and_save(ndbi_paths, ndbi_mosaic)
        mosaic_and_save(ndvi_paths, ndvi_mosaic)

    print("\nSentinel-2 NDBI/NDVI complete.")


if __name__ == "__main__":
    main()
