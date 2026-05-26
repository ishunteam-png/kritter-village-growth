"""
Step 1c: Annual Sentinel-2 NDBI & NDVI composites for India (2019–2024).

Uses Element84 earth-search STAC (no authentication required).
Processes India in 3°×3° grid cells with 4 parallel workers,
computes cloud-filtered median composites, and mosaics into single-file
annual GeoTIFFs matching the VIIRS pipeline convention.

  NDBI = (B11 - B08) / (B11 + B08)   built-up proxy (SWIR vs NIR)
  NDVI = (B08 - B04) / (B08 + B04)   vegetation / agricultural proxy

DRY-SEASON FILTER (fix for monsoon-era NaN):
  India's monsoon (June–September) blocks >90% of Sentinel-2 scenes even
  with CLOUD_MAX=25.  We query two windows per calendar year:
    • Jan–May   (post-winter dry season)
    • Oct–Dec   (post-monsoon dry season)
  Combined, this gives 7 cloud-free months per year and produces valid
  composites for cells that had all-NaN output in full-year queries.

Output resolution: ~100 m (0.0009°) — good trade-off for India-wide mosaic.
Runtime: ~6–10 h on t3.xlarge (4 vCPU).

Output:
    <data_root>/processed/sentinel2_ndbi_{year}.tif
    <data_root>/processed/sentinel2_ndvi_{year}.tif
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
CLOUD_MAX = 25      # max cloud cover % to accept a scene
MAX_SCENES = 15     # raised from 10 — more candidates before filtering
MIN_SCENES = 2      # lowered from 3 — accepts cells with 2 valid dry-season scenes
OUT_RES   = 0.0009  # ~100 m in degrees
CELL_DEG  = 3.0     # 3° × 3° processing grid
N_WORKERS = 4       # parallel workers (= EC2 vCPUs)

INDIA_W, INDIA_E = 68.0, 97.5
INDIA_S, INDIA_N =  7.5, 37.5

# Dry-season date windows: Jan–May + Oct–Dec for each year.
# Avoids the June–Sep monsoon that blocks >90% of S2 scenes over India.
DRY_WINDOWS = [
    ("{year}-01-01", "{year}-05-31"),   # Jan → May
    ("{year}-10-01", "{year}-12-31"),   # Oct → Dec
]


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


def search_dry_season(catalog, bbox, year):
    """
    Query STAC across two dry-season windows and deduplicate by item id.
    Returns a list of STAC items (may span Jan-May and Oct-Dec of `year`).
    """
    seen, items = set(), []
    for start_tmpl, end_tmpl in DRY_WINDOWS:
        start = start_tmpl.format(year=year)
        end   = end_tmpl.format(year=year)
        try:
            results = list(catalog.search(
                collections=["sentinel-2-l2a"],
                bbox=bbox,
                datetime=f"{start}/{end}",
                query={"eo:cloud_cover": {"lt": CLOUD_MAX}},
                max_items=MAX_SCENES,
            ).items())
            for item in results:
                if item.id not in seen:
                    seen.add(item.id)
                    items.append(item)
        except Exception as e:
            print(f"    STAC search error ({start}/{end}): {e}")
    return items


def _read_band_window(href: str, bbox: list, nrows: int, ncols: int) -> np.ndarray:
    """Read COG band, reprojecting S2 UTM tiles into the WGS84 output grid."""
    from rasterio.transform import from_bounds as _fb
    dst_transform = _fb(*bbox, ncols, nrows)
    dst = np.zeros((nrows, ncols), dtype="float32")
    with rasterio.open(href) as src:
        rasterio.warp.reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            dst_transform=dst_transform,
            dst_crs="EPSG:4326",
            resampling=Resampling.bilinear,
            src_nodata=0,
            dst_nodata=0,
        )
    return dst


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
        items   = search_dry_season(catalog, bbox, year)

        if len(items) < MIN_SCENES:
            return (None, None)

        nrows = max(2, round((bbox[3] - bbox[1]) / OUT_RES))
        ncols = max(2, round((bbox[2] - bbox[0]) / OUT_RES))
        transform = from_bounds(*bbox, ncols, nrows)

        ndbi_stack, ndvi_stack = [], []

        for item in items:
            try:
                assets = item.assets
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
        print(f"  ⚠  No valid cells for {out_path.name} — mosaic skipped")
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
    print(f"Dry-season filter: Jan–May + Oct–Dec (skipping Jun–Sep monsoon)")
    print(f"Output dir: {PROC_DIR}")

    for year in YEARS:
        ndbi_mosaic = PROC_DIR / f"sentinel2_ndbi_{year}.tif"
        ndvi_mosaic = PROC_DIR / f"sentinel2_ndvi_{year}.tif"
        if ndbi_mosaic.exists() and ndvi_mosaic.exists():
            print(f"{year}: already complete, skipping")
            continue

        print(f"\n── Year {year} ──────────────────────────────────────────────")
        ndbi_paths, ndvi_paths = [], []

        with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
            futures = {
                ex.submit(process_cell, (bbox, year, str(cell_dir))): bbox
                for bbox in grid
            }
            done, succeeded = 0, 0
            for fut in as_completed(futures):
                bbox = futures[fut]
                nb, nv = fut.result()
                ndbi_paths.append(nb)
                ndvi_paths.append(nv)
                done += 1
                if nb:
                    succeeded += 1
                status = "✓" if nb else "✗"
                print(f"  {status} {bbox[1]:.0f}°N {bbox[0]:.0f}°E  [{done}/{len(grid)}]  "
                      f"({succeeded} ok so far)")

        good = sum(1 for p in ndbi_paths if p)
        print(f"\n  {good}/{len(grid)} cells succeeded")
        if good == 0:
            print(f"  ⚠  All cells failed for {year} — check STAC connectivity and PROC_DIR path")
        mosaic_and_save(ndbi_paths, ndbi_mosaic)
        mosaic_and_save(ndvi_paths, ndvi_mosaic)

    print("\nSentinel-2 NDBI/NDVI complete.")


if __name__ == "__main__":
    main()
