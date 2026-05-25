"""
Step 1: Download NASA VIIRS Black Marble annual NTL composites for India.

Product: VNP46A4 v2 (Annual, 500m, MODIS sinusoidal grid)
Band used: AllAngle_Composite_Snow_Free (nW/cm²/sr)
Years: 2019-2024 (6 years for trend analysis)
Output: /data/satellite/kritter/viirs/{year}/*.h5
        /data/satellite/kritter/processed/viirs_ntl_{year}.tif  (India mosaic, WGS84)

Requires NASA Earthdata credentials in ~/.netrc or env vars.

Run on EC2:
    conda activate insar
    python 01_download_viirs.py
"""

import os
import re
import h5py
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.merge import merge
from rasterio.transform import from_bounds
from pathlib import Path
from tqdm import tqdm
import earthaccess

YEARS      = [2019, 2020, 2021, 2022, 2023, 2024]
INDIA_BBOX = (68.0, 8.0, 97.5, 37.0)   # lon_min, lat_min, lon_max, lat_max
BASE_DIR   = Path("/data/satellite/kritter")
VIIRS_DIR  = BASE_DIR / "viirs"
PROC_DIR   = BASE_DIR / "processed"

DATASET    = "HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields/AllAngle_Composite_Snow_Free"
FILL_VALUE = -999.9

WGS84 = CRS.from_epsg(4326)


def login():
    user = os.environ.get("EARTHDATA_USERNAME")
    pwd  = os.environ.get("EARTHDATA_PASSWORD")
    if user and pwd:
        earthaccess.login(strategy="environment")
    else:
        earthaccess.login(strategy="netrc")
    print("NASA Earthdata authenticated")


def download_year(year: int) -> list[Path]:
    out_dir = VIIRS_DIR / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)

    # CMR temporal offset: to get year X data, search year X+1
    search_year = year + 1
    results = earthaccess.search_data(
        short_name="VNP46A4",
        version="2",
        temporal=(f"{search_year}-01-01", f"{search_year}-12-31"),
        bounding_box=INDIA_BBOX,
    )

    # Filter to tiles with A{year}001 in filename (annual composite for that year)
    results = [
        r for r in results
        if f"A{year}001" in r.data_links()[0].split("/")[-1]
    ]
    print(f"  {year}: {len(results)} tiles")

    # Skip tiles already downloaded
    to_download = []
    existing = []
    for r in results:
        fname = r.data_links()[0].split("/")[-1]
        dest  = out_dir / fname
        if dest.exists():
            existing.append(dest)
        else:
            to_download.append(r)

    if existing:
        print(f"    {len(existing)} already downloaded, skipping")

    new_files = []
    if to_download:
        new_files = earthaccess.download(to_download, str(out_dir))
        new_files = [Path(f) for f in new_files]

    return existing + new_files


def h5_to_tif(h5_path: Path) -> Path:
    """Convert VNP46A4 HDF5 tile to WGS84 GeoTIFF via h5py (bypasses gdal HDF5 driver).

    VNP46A4 StructMetadata.0 stores bounding box in microdegrees (degrees × 1e6),
    so we write directly in EPSG:4326 — no gdalwarp needed.
    """
    tif_path = h5_path.with_suffix(".tif")
    if tif_path.exists():
        return tif_path

    with h5py.File(h5_path, "r") as f:
        data = f[DATASET][()].astype("float32")

        meta_bytes = f["HDFEOS INFORMATION"]["StructMetadata.0"][()]
        meta = meta_bytes.decode() if isinstance(meta_bytes, (bytes, bytearray)) else str(meta_bytes)

        ul = re.search(r"UpperLeftPointMtrs=\(([^)]+)\)", meta)
        lr = re.search(r"LowerRightMtrs=\(([^)]+)\)", meta)
        if not ul or not lr:
            raise ValueError(f"Could not parse geotransform from {h5_path}")

        # Coordinates are in microdegrees — convert to decimal degrees
        xmin, ymax = [float(v) / 1e6 for v in ul.group(1).split(",")]
        xmax, ymin = [float(v) / 1e6 for v in lr.group(1).split(",")]

    data[data <= FILL_VALUE] = np.nan
    nrows, ncols = data.shape

    transform = from_bounds(xmin, ymin, xmax, ymax, ncols, nrows)
    with rasterio.open(
        tif_path, "w", driver="GTiff",
        height=nrows, width=ncols, count=1,
        dtype="float32", crs=WGS84,
        transform=transform, nodata=np.nan,
        compress="lzw",
    ) as dst:
        dst.write(data, 1)

    return tif_path


def mosaic_year(year: int, tif_paths: list[Path]) -> Path:
    """Merge all reprojected tiles into a single India mosaic."""
    out_path = PROC_DIR / f"viirs_ntl_{year}.tif"
    if out_path.exists():
        print(f"  Mosaic exists: {out_path}")
        return out_path

    valid = [p for p in tif_paths if p.exists()]
    if not valid:
        raise FileNotFoundError(f"No valid TIFs to mosaic for {year}")

    datasets  = [rasterio.open(p) for p in valid]
    mosaic, transform = merge(datasets, nodata=np.nan)
    profile = datasets[0].profile.copy()
    for ds in datasets:
        ds.close()

    profile.update(
        height=mosaic.shape[1], width=mosaic.shape[2],
        transform=transform, count=1,
        dtype="float32", nodata=np.nan, compress="lzw",
    )
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mosaic[0], 1)

    print(f"  Saved mosaic → {out_path}")
    return out_path


def main():
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    login()

    for year in YEARS:
        mosaic_path = PROC_DIR / f"viirs_ntl_{year}.tif"
        if mosaic_path.exists():
            print(f"\n{year}: mosaic already exists, skipping")
            continue

        print(f"\nYear {year}...")
        h5_files = download_year(year)

        tif_files = []
        for h5 in tqdm(h5_files, desc=f"  Converting {year}"):
            try:
                tif_files.append(h5_to_tif(h5))
            except Exception as e:
                print(f"  WARNING: {h5.name}: {e}")

        mosaic_year(year, tif_files)

    print("\nAll VIIRS mosaics ready →", PROC_DIR)


if __name__ == "__main__":
    main()
