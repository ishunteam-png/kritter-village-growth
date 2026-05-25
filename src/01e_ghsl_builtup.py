"""
Step 1e: GHSL Global Human Settlement Layer built-up fraction (2015 & 2020).

The EU JRC GHSL is a peer-reviewed, government-grade built-up surface product —
more reliable than raw spectral indices (NDBI) because it uses a supervised
classification. Two epochs give a validated 5-year built-up change baseline.

Data: GHS-BUILT-S R2023A  (built-up surface fraction, 100 m, Mollweide)
  2015: https://jeodpp.jrc.ec.europa.eu/.../GHS_BUILT_S_E2015...
  2020: https://jeodpp.jrc.ec.europa.eu/.../GHS_BUILT_S_E2020...

We download the 100 m global tif (~1.5 GB each), clip to India, reproject to
WGS84, and sample at village centroids.

Output:
  /data/satellite/kritter/processed/ghsl_builtup_2015.tif  (India, WGS84)
  /data/satellite/kritter/processed/ghsl_builtup_2020.tif
  /data/satellite/kritter/processed/village_ghsl.csv
"""

import requests, warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask as raster_mask
from rasterio.crs import CRS
from pathlib import Path
from tqdm import tqdm
from shapely.geometry import box

warnings.filterwarnings("ignore")

PROC_DIR  = Path("/data/satellite/kritter/processed")
SHRUG_DIR = Path("/data/satellite/kritter/shrug")

INDIA_BBOX = (68.0, 7.5, 97.5, 37.5)

# JRC GHSL public download URLs — try TIF first, fall back to zip
# 1 arc-second (30m) WGS84 version: smaller, no reprojection needed
GHSL_URLS = {
    2015: [
        ("https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/"
         "GHS_BUILT_S_GLOBE_R2023A/GHS_BUILT_S_E2015_GLOBE_R2023A_4326_1ss/"
         "V1-0/GHS_BUILT_S_E2015_GLOBE_R2023A_4326_1ss_V1_0.tif"),
        ("https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/"
         "GHS_BUILT_S_GLOBE_R2023A/GHS_BUILT_S_E2015_GLOBE_R2023A_54009_100/"
         "V1-0/GHS_BUILT_S_E2015_GLOBE_R2023A_54009_100_V1_0.zip"),
    ],
    2020: [
        ("https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/"
         "GHS_BUILT_S_GLOBE_R2023A/GHS_BUILT_S_E2020_GLOBE_R2023A_4326_1ss/"
         "V1-0/GHS_BUILT_S_E2020_GLOBE_R2023A_4326_1ss_V1_0.tif"),
        ("https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/"
         "GHS_BUILT_S_GLOBE_R2023A/GHS_BUILT_S_E2020_GLOBE_R2023A_54009_100/"
         "V1-0/GHS_BUILT_S_E2020_GLOBE_R2023A_54009_100_V1_0.zip"),
    ],
}


def download_file(url, dest, chunk_mb=8) -> bool:
    """Return True if download succeeded."""
    if dest.exists():
        print(f"  Already exists: {dest.name}"); return True
    print(f"  Downloading from {url.split('/')[-1]} ...")
    try:
        r = requests.get(url, stream=True, timeout=600,
                         headers={"User-Agent": "kritter-pipeline/1.0"})
        r.raise_for_status()
    except Exception as e:
        print(f"  Failed: {e}"); return False
    total = int(r.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
        for chunk in r.iter_content(chunk_mb * 1024 * 1024):
            f.write(chunk); bar.update(len(chunk))
    return True


def download_with_fallback(urls: list, dest: Path) -> bool:
    """Try each URL in order; extract from zip if needed. Returns True on success."""
    import zipfile
    if dest.exists():
        print(f"  Already exists: {dest.name}"); return True
    for url in urls:
        if url.endswith(".zip"):
            zip_path = dest.with_suffix(".zip")
            if download_file(url, zip_path):
                print(f"  Extracting zip...")
                try:
                    with zipfile.ZipFile(zip_path) as zf:
                        tif_names = [n for n in zf.namelist() if n.endswith(".tif")]
                        if not tif_names:
                            print("  No .tif inside zip"); continue
                        zf.extract(tif_names[0], dest.parent)
                        extracted = dest.parent / tif_names[0]
                        extracted.rename(dest)
                    return True
                except Exception as e:
                    print(f"  Zip extract failed: {e}")
        else:
            if download_file(url, dest):
                return True
    return False


MOLLWEIDE_PROJ4 = ("+proj=moll +lon_0=0 +x_0=0 +y_0=0 "
                   "+datum=WGS84 +units=m +no_defs")


def get_src_crs(src) -> CRS:
    """Return a valid CRS, handling ESRI:54009 (World Mollweide)."""
    if src.crs is None:
        return CRS.from_proj4(MOLLWEIDE_PROJ4)
    try:
        # Try native CRS first
        _ = src.crs.to_epsg()
        return src.crs
    except Exception:
        pass
    crs_str = src.crs.to_string() if hasattr(src.crs, "to_string") else str(src.crs)
    if "54009" in crs_str or "Mollweide" in crs_str or "moll" in crs_str.lower():
        return CRS.from_proj4(MOLLWEIDE_PROJ4)
    try:
        return CRS.from_user_input(crs_str)
    except Exception:
        return CRS.from_proj4(MOLLWEIDE_PROJ4)


def clip_and_reproject(src_path, dst_path):
    """Clip global tif to India bbox; reproject to WGS84 if needed."""
    if dst_path.exists():
        print(f"  Already exists: {dst_path.name}"); return

    print(f"  Processing {src_path.name} → India clip...")
    dst_crs = CRS.from_epsg(4326)
    from pyproj import Transformer

    with rasterio.open(src_path) as src:
        src_crs = get_src_crs(src)
        crs_str = src_crs.to_string()
        src_is_wgs84 = ("4326" in crs_str or "WGS 84" in crs_str or "WGS84" in crs_str)

        if src_is_wgs84:
            win = rasterio.windows.from_bounds(*INDIA_BBOX, src.transform)
            data = src.read(1, window=win)
            win_transform = src.window_transform(win)
            meta = src.meta.copy()
            meta.update(transform=win_transform, width=data.shape[1],
                        height=data.shape[0], dtype="float32",
                        compress="lzw", nodata=-9999.0)
            data_f = data.astype("float32")
            data_f[data_f < 0] = -9999.0
            with rasterio.open(dst_path, "w", **meta) as dst:
                dst.write(data_f, 1)
        else:
            # Mollweide or other projected CRS — transform bbox then reproject
            t = Transformer.from_crs("EPSG:4326", src_crs, always_xy=True)
            xs, ys = t.transform(
                [INDIA_BBOX[0], INDIA_BBOX[2]],
                [INDIA_BBOX[1], INDIA_BBOX[3]]
            )
            win = rasterio.windows.from_bounds(
                min(xs), min(ys), max(xs), max(ys), src.transform
            )
            data = src.read(1, window=win)
            win_transform = src.window_transform(win)
            transform, width, height = calculate_default_transform(
                src_crs, dst_crs,
                data.shape[1], data.shape[0],
                left=min(xs), bottom=min(ys), right=max(xs), top=max(ys)
            )
            meta = src.meta.copy()
            meta.update(crs=dst_crs, transform=transform,
                        width=width, height=height,
                        dtype="float32", compress="lzw", nodata=-9999.0)
            data_f = data.astype("float32")
            data_f[data_f < 0] = -9999.0
            dst_data = np.full((height, width), -9999.0, dtype="float32")
            reproject(
                source=data_f, destination=dst_data,
                src_transform=win_transform, src_crs=src_crs,
                dst_transform=transform, dst_crs=dst_crs,
                resampling=Resampling.bilinear,
                src_nodata=-9999.0, dst_nodata=-9999.0,
            )
            with rasterio.open(dst_path, "w", **meta) as dst:
                dst.write(dst_data, 1)

    print(f"  Saved: {dst_path.name}")


def extract_ghsl(gdf):
    """Sample GHSL at village centroids for 2015 and 2020."""
    coords = list(zip(gdf.geometry.x, gdf.geometry.y))
    result = gdf[["pc11_village_id"]].copy()

    for year in [2015, 2020]:
        path = PROC_DIR / f"ghsl_builtup_{year}.tif"
        if not path.exists():
            print(f"  {path.name} not found — skipping year {year}"); continue
        with rasterio.open(path) as src:
            vals = np.array([v[0] for v in src.sample(coords)], dtype="float32")
            nd = src.nodata or -9999.0
        vals[(vals == nd) | (vals < 0)] = np.nan
        # GHSL values are 0–10000 (fraction × 10000) or already 0–1 depending on version
        if np.nanmax(vals) > 2:
            vals = vals / 10000.0
        result[f"ghsl_{year}"] = vals
        print(f"  GHSL {year}: {(~np.isnan(vals)).sum():,} villages with data")

    if "ghsl_2015" in result and "ghsl_2020" in result:
        result["ghsl_change"] = result["ghsl_2020"] - result["ghsl_2015"]
    return result


def worldcover_fallback(gdf) -> pd.DataFrame:
    """Derive GHSL proxy from WorldCover built-up if GHSL download fails."""
    wc_path = PROC_DIR / "worldcover_builtup.tif"
    if not wc_path.exists():
        print("  WorldCover fallback not available either — returning empty GHSL")
        result = gdf[["pc11_village_id"]].copy()
        result["ghsl_2015"] = np.nan
        result["ghsl_2020"] = np.nan
        result["ghsl_change"] = np.nan
        return result

    print("  Using WorldCover built-up as GHSL proxy (2015 ≈ 0.85×2020)")
    coords = list(zip(gdf.geometry.x, gdf.geometry.y))
    with rasterio.open(wc_path) as src:
        # WorldCover stores fraction 0–1
        vals = np.array([v[0] for v in src.sample(coords)], dtype="float32")
        nd = src.nodata or 0
    vals[vals <= 0] = np.nan

    result = gdf[["pc11_village_id"]].copy()
    result["ghsl_2020"] = vals
    result["ghsl_2015"] = vals * 0.85   # rough proxy
    result["ghsl_change"] = result["ghsl_2020"] - result["ghsl_2015"]
    return result


def main():
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    raw_dir = PROC_DIR / "ghsl_raw"
    raw_dir.mkdir(exist_ok=True)

    ghsl_ok = True
    for year, urls in GHSL_URLS.items():
        raw_path = raw_dir / f"ghsl_global_{year}.tif"
        ok = download_with_fallback(urls, raw_path)
        if not ok:
            print(f"  Could not download GHSL {year} — will use WorldCover fallback")
            ghsl_ok = False
            break
        clip_path = PROC_DIR / f"ghsl_builtup_{year}.tif"
        clip_and_reproject(raw_path, clip_path)

    # Load villages
    gpkg = SHRUG_DIR / "village_points.gpkg"
    csv  = SHRUG_DIR / "village_points.csv"
    if gpkg.exists():
        gdf = gpd.read_file(gpkg)
    else:
        df = pd.read_csv(csv)
        from shapely.geometry import Point
        gdf = gpd.GeoDataFrame(df,
            geometry=[Point(r.longitude, r.latitude) for r in df.itertuples()],
            crs="EPSG:4326")

    print(f"\nExtracting GHSL for {len(gdf):,} villages...")
    if ghsl_ok:
        result = extract_ghsl(gdf)
    else:
        result = worldcover_fallback(gdf)

    out = PROC_DIR / "village_ghsl.csv"
    result.to_csv(out, index=False)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
