"""
Step 2: Per-village built-up fraction from ESA WorldCover (HTTPS COGs).

ESA WorldCover is a 10m global land cover map. Class 50 = Built-up.
  v100/2020: https://esa-worldcover.s3.eu-central-1.amazonaws.com/v100/2020/map/
  v200/2021: https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/

Modes:
  polygon  — shapefile with village polygons → zonal-stats fraction
  point    — village centroids (OSM) → pixel value at centroid (1=built-up, 0=not)

Output: /data/satellite/kritter/processed/village_builtup.csv

Run on EC2:
    conda activate insar
    python 02_worldcover_builtup.py                         # point mode (OSM)
    python 02_worldcover_builtup.py --shapefile /path/to/village.shp  # polygon mode
"""

import argparse
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import rasterstats
from pathlib import Path
from shapely.geometry import Point
from tqdm import tqdm

warnings.filterwarnings("ignore")

PROC_DIR   = Path("/data/satellite/kritter/processed")
SHRUG_DIR  = Path("/data/satellite/kritter/shrug")

HTTPS_TMPL = {
    2020: "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v100/2020/map/ESA_WorldCover_10m_2020_v100_{tile}_Map.tif",
    2021: "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{tile}_Map.tif",
}
BUILTUP_CLASS = 50
INDIA_BBOX    = (68.0, 8.0, 97.5, 37.0)


def worldcover_tile_name(lat: float, lon: float) -> str:
    lat_int = int(np.floor(lat / 3) * 3)
    lon_int = int(np.floor(lon / 3) * 3)
    lat_str = f"N{abs(lat_int):02d}" if lat_int >= 0 else f"S{abs(lat_int):02d}"
    lon_str = f"E{abs(lon_int):03d}" if lon_int >= 0 else f"W{abs(lon_int):03d}"
    return f"{lat_str}{lon_str}"


def india_tile_names() -> list[str]:
    tiles = set()
    lon_min, lat_min, lon_max, lat_max = INDIA_BBOX
    lat = lat_min
    while lat <= lat_max:
        lon = lon_min
        while lon <= lon_max:
            tiles.add(worldcover_tile_name(lat, lon))
            lon += 3
        lat += 3
    return sorted(tiles)


def load_villages(shapefile: str | None) -> tuple[gpd.GeoDataFrame, str]:
    if shapefile and Path(shapefile).exists():
        gdf = gpd.read_file(shapefile).to_crs("EPSG:4326")
        return gdf, "polygon"

    gpkg = SHRUG_DIR / "village_points.gpkg"
    csv  = SHRUG_DIR / "village_points.csv"
    if gpkg.exists():
        gdf = gpd.read_file(gpkg)
    elif csv.exists():
        df = pd.read_csv(csv)
        gdf = gpd.GeoDataFrame(
            df,
            geometry=[Point(r.longitude, r.latitude) for r in df.itertuples()],
            crs="EPSG:4326",
        )
    else:
        raise FileNotFoundError("No village data. Run 00b_download_villages_osm.py first.")

    return gdf, "point"


def process_tile_point(villages: gpd.GeoDataFrame, tile: str, year: int) -> pd.DataFrame:
    """Point sampling: extract WorldCover class at each village centroid."""
    url = HTTPS_TMPL[year].format(tile=tile)
    col = f"built_{year}"
    try:
        with rasterio.open(url) as src:
            bounds = src.bounds
            # Filter to villages within this tile
            mask = (
                (villages.geometry.x >= bounds.left) &
                (villages.geometry.x <= bounds.right) &
                (villages.geometry.y >= bounds.bottom) &
                (villages.geometry.y <= bounds.top)
            )
            subset = villages[mask]
            if subset.empty:
                return pd.DataFrame()

            coords = list(zip(subset.geometry.x, subset.geometry.y))
            vals = list(src.sample(coords))

        result = subset[["pc11_village_id"]].copy().reset_index(drop=True)
        result[col] = [float(v[0] == BUILTUP_CLASS) if v[0] != 0 else np.nan for v in vals]
        return result
    except Exception:
        return pd.DataFrame()


def process_tile_polygon(villages: gpd.GeoDataFrame, tile: str, year: int) -> pd.DataFrame:
    """Zonal stats: built-up fraction for polygon villages."""
    url = HTTPS_TMPL[year].format(tile=tile)
    col = f"built_{year}"
    try:
        with rasterio.open(url) as src:
            bounds = src.bounds

        subset = villages.cx[bounds.left:bounds.right, bounds.bottom:bounds.top].copy()
        if subset.empty:
            return pd.DataFrame()

        def builtup_frac(values, nodata):
            arr = np.array(values)
            valid = arr[arr != nodata] if nodata is not None else arr
            if len(valid) == 0:
                return np.nan
            return float((valid == BUILTUP_CLASS).sum()) / len(valid)

        stats = rasterstats.zonal_stats(
            subset, url, stats=[], add_stats={"f": builtup_frac}, nodata=0
        )
        result = subset[["pc11_village_id"]].copy().reset_index(drop=True)
        result[col] = [s.get("f", np.nan) for s in stats]
        return result
    except Exception:
        return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapefile", default=None)
    args = parser.parse_args()

    PROC_DIR.mkdir(parents=True, exist_ok=True)

    out = PROC_DIR / "village_builtup.csv"
    if out.exists():
        print(f"Already exists: {out}")
        return

    villages, mode = load_villages(args.shapefile)
    print(f"  {len(villages):,} villages loaded (mode: {mode})")

    process_tile = process_tile_point if mode == "point" else process_tile_polygon

    tiles = india_tile_names()
    print(f"Processing {len(tiles)} WorldCover tiles for 2020 and 2021...")

    all_results = {}
    for year in [2020, 2021]:
        print(f"\n  Year {year}:")
        year_dfs = []
        for tile in tqdm(tiles):
            df = process_tile(villages, tile, year)
            if not df.empty:
                year_dfs.append(df)
        if year_dfs:
            combined = pd.concat(year_dfs, ignore_index=True).drop_duplicates("pc11_village_id")
            all_results[year] = combined
        print(f"  Built-up stats: {len(all_results.get(year, [])):,} villages")

    if 2020 in all_results and 2021 in all_results:
        merged = all_results[2020].merge(all_results[2021], on="pc11_village_id", how="outer")
        merged["builtup_change"] = merged["built_2021"] - merged["built_2020"]
        merged.to_csv(out, index=False)
        print(f"\nSaved → {out}")
        print(merged["builtup_change"].describe())
    else:
        print("ERROR: missing data for one or both years")


if __name__ == "__main__":
    main()
