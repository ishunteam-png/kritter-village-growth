"""
Step 3: Per-village statistics — VIIRS NTL + WorldCover + WorldPop + Sentinel-2.

Supports two modes:
  - polygon mode: shapefile with village polygons → zonal stats (requires SHRUG)
  - point mode:   GeoPackage/CSV with village centroids → point extraction (default, uses OSM)

Columns produced per village:
  ntl_{year}          annual mean NTL radiance nW/cm²/sr (2019-2024)
  ntl_trend_slope     linear regression slope over 2019-2024
  built_2020          built-up fraction (WorldCover class 50, 2020)
  built_2021          built-up fraction (WorldCover class 50, 2021)
  builtup_change      built_2021 - built_2020
  pop_2019            WorldPop population estimate 2019
  pop_2020            WorldPop population estimate 2020
  pop_growth_rate     (pop_2020 - pop_2019) / pop_2019  [fraction]
  ndbi_{year}         Sentinel-2 NDBI (2019-2024), -1 to 1
  ndbi_trend_slope    linear regression slope of NDBI over 2019-2024
  ndvi_{year}         Sentinel-2 NDVI (2019-2024), -1 to 1
  ndvi_trend_slope    linear regression slope of NDVI over 2019-2024

Output: /data/satellite/kritter/processed/village_all_stats.csv
"""

import argparse
import warnings
import yaml
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import rasterstats
from pathlib import Path
from scipy.stats import linregress
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice")
warnings.filterwarnings("ignore", category=UserWarning, module="rasterio")
warnings.filterwarnings("ignore", category=UserWarning, module="geopandas")

_CFG_PATH = Path(__file__).parent.parent / "config.yaml"
_CFG      = yaml.safe_load(_CFG_PATH.read_text()) if _CFG_PATH.exists() else {}
_data     = _CFG.get("paths", {}).get("data_root", "/data/satellite/kritter")

PROC_DIR  = Path(_data) / "processed"
SHRUG_DIR = Path(_data) / "shrug"
YEARS     = [2019, 2020, 2021, 2022, 2023, 2024]


# ── Village loading ────────────────────────────────────────────────────────────

def load_villages(shapefile: str | None):
    if shapefile and Path(shapefile).exists():
        print(f"Loading polygon shapefile: {shapefile}")
        gdf = gpd.read_file(shapefile).to_crs("EPSG:4326")
        return gdf, "polygon"

    gpkg = SHRUG_DIR / "village_points.gpkg"
    csv  = SHRUG_DIR / "village_points.csv"
    if gpkg.exists():
        gdf = gpd.read_file(gpkg)
    elif csv.exists():
        df = pd.read_csv(csv)
        from shapely.geometry import Point
        gdf = gpd.GeoDataFrame(
            df,
            geometry=[Point(r.longitude, r.latitude) for r in df.itertuples()],
            crs="EPSG:4326",
        )
    else:
        raise FileNotFoundError(
            "No village data found. Run 00b_download_villages_osm.py first."
        )
    print(f"  {len(gdf):,} villages loaded (point mode)")
    return gdf, "point"


# ── VIIRS NTL ─────────────────────────────────────────────────────────────────

def extract_ntl_points(gdf, rasters):
    coords = list(zip(gdf.geometry.x, gdf.geometry.y))
    result = gdf[["pc11_village_id"]].copy()
    for year, path in rasters.items():
        with rasterio.open(path) as src:
            vals = [v[0] for v in src.sample(coords)]
        result[f"ntl_{year}"] = vals
    return result


def extract_ntl_polygons(gdf, rasters):
    id_col  = "pc11_village_id"
    all_dfs = [gdf[[id_col]].copy()]
    for year, path in rasters.items():
        print(f"  NTL {year} (zonal stats)...")
        stats = rasterstats.zonal_stats(gdf, path, stats=["mean"], nodata=np.nan)
        all_dfs.append(pd.Series([s.get("mean", np.nan) for s in stats],
                                  name=f"ntl_{year}"))
    return pd.concat(all_dfs, axis=1)


def add_trend(df, years, prefix):
    """Vectorised OLS slope over `years` columns — O(n) NumPy, no per-row Python."""
    x = np.array(years, dtype=float)
    cols = [f"{prefix}_{yr}" for yr in years]
    Y = np.column_stack([
        df[c].values if c in df.columns else np.full(len(df), np.nan)
        for c in cols
    ]).astype(float)                          # (n_villages, n_years)

    valid_count = np.sum(~np.isnan(Y), axis=1)
    n_years = len(years)

    slopes = np.full(len(df), np.nan)
    mask_enough = valid_count >= 3

    if mask_enough.any():
        Y_sub = Y[mask_enough]                # (m, n_years)
        x_rep = np.tile(x, (Y_sub.shape[0], 1))
        nan_m  = np.isnan(Y_sub)
        Y_sub  = np.where(nan_m, 0.0, Y_sub)
        x_rep  = np.where(nan_m, 0.0, x_rep)
        n_ok   = (~nan_m).sum(axis=1).astype(float)
        x_mean = x_rep.sum(axis=1) / n_ok
        y_mean = Y_sub.sum(axis=1) / n_ok
        xd = x_rep - x_mean[:, None]
        xd = np.where(nan_m, 0.0, xd)
        yd = Y_sub - y_mean[:, None]
        yd = np.where(nan_m, 0.0, yd)
        denom = (xd ** 2).sum(axis=1)
        denom = np.where(denom == 0, np.nan, denom)
        slopes[mask_enough] = (xd * yd).sum(axis=1) / denom

    df[f"{prefix}_trend_slope"] = slopes
    return df


# ── Sentinel-2 (NDBI / NDVI) ──────────────────────────────────────────────────

def extract_s2_signals(gdf):
    coords = list(zip(gdf.geometry.x, gdf.geometry.y))
    result = gdf[["pc11_village_id"]].copy()

    for signal in ("ndbi", "ndvi"):
        year_data = {}
        for year in YEARS:
            path = PROC_DIR / f"sentinel2_{signal}_{year}.tif"
            if not path.exists():
                continue
            with rasterio.open(path) as src:
                vals = np.array([v[0] for v in src.sample(coords)], dtype="float32")
            nodata = src.nodata or -9999.0
            vals[vals == nodata] = np.nan
            year_data[year] = vals

        if not year_data:
            print(f"  NOTE: no sentinel2_{signal}_*.tif found — skipping {signal.upper()}")
            for yr in YEARS:
                result[f"{signal}_{yr}"] = np.nan
            result[f"{signal}_trend_slope"] = np.nan
            continue

        for yr, vals in year_data.items():
            result[f"{signal}_{yr}"] = vals

        # Fill missing years with NaN
        for yr in YEARS:
            if yr not in year_data:
                result[f"{signal}_{yr}"] = np.nan

        # Vectorised OLS slope — same approach as add_trend(), no per-row loop
        x_arr = np.array(YEARS, dtype=float)
        arr   = np.stack([
            result[f"{signal}_{yr}"].values if f"{signal}_{yr}" in result
            else np.full(len(result), np.nan)
            for yr in YEARS
        ], axis=1).astype(float)
        nan_m   = np.isnan(arr)
        n_ok    = (~nan_m).sum(axis=1).astype(float)
        mask_ok = n_ok >= 3
        slopes  = np.full(len(result), np.nan)
        if mask_ok.any():
            a   = arr[mask_ok]
            xr  = np.tile(x_arr, (a.shape[0], 1))
            nm  = np.isnan(a)
            a   = np.where(nm, 0.0, a)
            xr  = np.where(nm, 0.0, xr)
            nv  = n_ok[mask_ok]
            xm  = xr.sum(axis=1) / nv
            ym  = a.sum(axis=1) / nv
            xd  = np.where(nm, 0.0, xr - xm[:, None])
            yd  = np.where(nm, 0.0, a  - ym[:, None])
            dv  = (xd ** 2).sum(axis=1)
            dv  = np.where(dv == 0, np.nan, dv)
            slopes[mask_ok] = (xd * yd).sum(axis=1) / dv
        result[f"{signal}_trend_slope"] = slopes
        print(f"  {signal.upper()}: {(~np.isnan(result[f'{signal}_2019'])).sum():,} villages with 2019 data")

    return result


# ── WorldPop ──────────────────────────────────────────────────────────────────

def extract_worldpop(gdf):
    coords = list(zip(gdf.geometry.x, gdf.geometry.y))
    result = gdf[["pc11_village_id"]].copy()

    for year in [2019, 2020]:
        path = PROC_DIR / f"worldpop_india_{year}.tif"
        if not path.exists():
            print(f"  NOTE: worldpop_india_{year}.tif not found — run 01b_worldpop.py")
            result[f"pop_{year}"] = np.nan
            continue
        with rasterio.open(path) as src:
            vals = np.array([v[0] for v in src.sample(coords)], dtype="float32")
            nd = src.nodata or -9999.0
        vals[(vals == nd) | (vals < 0)] = np.nan
        result[f"pop_{year}"] = vals
        print(f"  WorldPop {year}: {(~np.isnan(vals)).sum():,} non-NaN villages")

    if "pop_2019" in result.columns and "pop_2020" in result.columns:
        p19 = result["pop_2019"]
        p20 = result["pop_2020"]
        result["pop_growth_rate"] = np.where(
            p19 > 0, (p20 - p19) / p19, np.nan
        )
    else:
        result["pop_growth_rate"] = np.nan

    return result


# ── WorldCover ────────────────────────────────────────────────────────────────

def merge_worldcover(merged, id_col):
    builtup_csv = PROC_DIR / "village_builtup.csv"
    if not builtup_csv.exists():
        print("  NOTE: village_builtup.csv not found — run 02_worldcover_builtup.py")
        for col in ("built_2020", "built_2021", "builtup_change"):
            merged[col] = np.nan
        return merged

    builtup = pd.read_csv(builtup_csv)
    builtup[id_col] = builtup[id_col].astype(str)
    merged[id_col]  = merged[id_col].astype(str)
    return merged.merge(builtup, on=id_col, how="left")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapefile", default=None,
                        help="Path to SHRUG polygon shapefile (optional)")
    args = parser.parse_args()
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    gdf, mode = load_villages(args.shapefile)

    # Check VIIRS rasters
    rasters = {}
    for year in YEARS:
        p = PROC_DIR / f"viirs_ntl_{year}.tif"
        if not p.exists():
            raise FileNotFoundError(f"Missing VIIRS mosaic: {p}. Run 01_download_viirs.py first.")
        rasters[year] = str(p)

    print("Extracting VIIRS NTL statistics...")
    ntl_df = extract_ntl_points(gdf, rasters) if mode == "point" else extract_ntl_polygons(gdf, rasters)
    ntl_df = add_trend(ntl_df, YEARS, "ntl")

    # Build base
    id_col = "pc11_village_id"
    keep = [c for c in ["pc11_village_id", "village_name", "state_name",
                         "district_name", "subdistrict_name", "latitude", "longitude"]
            if c in gdf.columns]
    merged = gdf[keep].copy().merge(ntl_df, on=id_col, how="left")

    # WorldCover
    merged = merge_worldcover(merged, id_col)

    # WorldPop
    print("Extracting WorldPop population...")
    pop_df = extract_worldpop(gdf)
    pop_df[id_col] = pop_df[id_col].astype(str)
    merged[id_col] = merged[id_col].astype(str)
    merged = merged.merge(pop_df.drop(columns=[id_col]), left_index=True, right_index=True)

    # Sentinel-2
    print("Extracting Sentinel-2 NDBI / NDVI...")
    s2_df = extract_s2_signals(gdf)
    s2_df[id_col] = s2_df[id_col].astype(str)
    merged = merged.merge(s2_df.drop(columns=[id_col]), left_index=True, right_index=True)

    out = PROC_DIR / "village_all_stats.csv"
    merged.to_csv(out, index=False)
    print(f"\nSaved {len(merged):,} rows → {out}")
    print(merged[["ntl_2019", "ntl_2024", "ntl_trend_slope",
                  "ndbi_2019", "ndbi_2024", "pop_growth_rate"]].describe())

    # ── Output boundary checks ───────────────────────────────────────────────
    # Fail loudly if this output is likely to produce garbage scores in step 04.
    errors = []
    REQUIRED_COLS = ["pc11_village_id", "latitude", "longitude",
                     "ntl_2019", "ntl_2024", "ntl_trend_slope"]
    for col in REQUIRED_COLS:
        if col not in merged.columns:
            errors.append(f"MISSING required column: {col}")
    if len(merged) < 1000:
        errors.append(f"Row count suspiciously low: {len(merged):,} (expected 200k+)")
    ntl_ok = merged["ntl_2019"].notna().sum()
    if ntl_ok < 0.5 * len(merged):
        errors.append(f"ntl_2019 coverage only {ntl_ok/len(merged):.1%} — VIIRS extraction likely failed")
    ntl_range = merged["ntl_2019"].max()
    if ntl_range > 5000:
        errors.append(f"ntl_2019 max={ntl_range:.1f} — unit likely wrong (expected nW/cm²/sr ~0-500)")
    if errors:
        for e in errors:
            print(f"  ❌ VALIDATION ERROR: {e}")
        raise ValueError(f"Output validation failed ({len(errors)} errors) — fix before running step 04")
    print(f"\n  ✓ Output validation passed ({len(merged):,} rows, {ntl_ok:,} NTL values)")


if __name__ == "__main__":
    main()
