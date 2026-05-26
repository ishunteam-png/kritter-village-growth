"""
Download India village centroids from OpenStreetMap via Overpass API.

Produces a GeoDataFrame with ~300k village points including name, state,
district, lat, lon. Saved as GeoPackage (village_points.gpkg) and CSV.

SPATIAL DEDUPLICATION (fix for duplicate-location ranking):
  OSM often contains multiple nodes representing the same physical settlement
  (e.g. a market town tagged as several overlapping hamlet/village nodes).
  After the exact-coordinate dedup, we run a spatial cluster step:
    - Build a cKDTree in approximate-metre coordinates
    - For each pair of nodes within DEDUP_RADIUS_M (default 500 m),
      keep the one with a non-empty name tag; if both have names, keep
      the node with more populated tags; if neither has a name, keep one.
  This prevents the same settlement appearing multiple times in the top-100.

Output: <data_root>/shrug/village_points.gpkg
         <data_root>/shrug/village_points.csv
"""

import requests
import json
import time
import yaml
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path
from scipy.spatial import cKDTree
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
_CFG_PATH = Path(__file__).parent.parent / "config.yaml"
_CFG      = yaml.safe_load(_CFG_PATH.read_text()) if _CFG_PATH.exists() else {}
_data     = _CFG.get("paths", {}).get("data_root", "/data/satellite/kritter")

OUT_DIR  = Path(_data) / "shrug"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# India bounding box for Overpass (S, W, N, E)
INDIA_BBOX = "6.0,68.0,37.5,98.0"

# Spatial deduplication radius.  Nodes within this distance of each other
# are treated as the same physical settlement; only one is kept.
# 500 m: tight enough to merge OSM nodes within the same village / market town;
# loose enough not to merge genuinely distinct nearby villages.
DEDUP_RADIUS_M = 500

QUERY = f"""
[out:json][timeout:600][maxsize:2147483648];
(
  node[place~"^(village|hamlet)$"]({INDIA_BBOX});
);
out body;
"""


def fetch_overpass(query: str, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            print(f"Querying Overpass API (attempt {attempt+1}/{retries})...")
            r = requests.post(OVERPASS_URL, data={"data": query}, timeout=650)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.Timeout:
            print("  Timeout — retrying in 30s...")
            time.sleep(30)
        except Exception as e:
            print(f"  Error: {e}")
            if attempt < retries - 1:
                time.sleep(10)
    raise RuntimeError("Overpass API failed after all retries")


def osm_to_geodataframe(data: dict) -> gpd.GeoDataFrame:
    records = []
    for element in tqdm(data.get("elements", []), desc="Parsing nodes"):
        tags = element.get("tags", {})
        records.append({
            "pc11_village_id":  str(element["id"]),
            "village_name":     tags.get("name") or tags.get("name:en", ""),
            "state_name":       (tags.get("addr:state") or
                                 tags.get("is_in:state") or ""),
            "district_name":    (tags.get("addr:district") or
                                 tags.get("is_in:district") or ""),
            "subdistrict_name": tags.get("addr:subdistrict", ""),
            "latitude":         element["lat"],
            "longitude":        element["lon"],
            # Tag count used as a tiebreaker in spatial dedup (richer tags win)
            "_tag_count":       len(tags),
        })

    df = pd.DataFrame(records)
    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(row.longitude, row.latitude) for row in df.itertuples()],
        crs="EPSG:4326",
    )
    return gdf


def spatial_deduplicate(gdf: gpd.GeoDataFrame,
                         radius_m: float = DEDUP_RADIUS_M) -> gpd.GeoDataFrame:
    """
    Remove near-duplicate OSM nodes that represent the same physical settlement.

    Algorithm:
      1. Convert lat/lon to approximate metres (1° lat ≈ 111 000 m).
      2. Build cKDTree; query pairs within `radius_m`.
      3. Walk connected components of the proximity graph (Union-Find);
         within each component keep the node with the best name or most tags.
      4. Return deduplicated GeoDataFrame.

    This prevents the same market town appearing multiple times in the top-100
    ranking (e.g. Siswa Bazar had 5 separate OSM nodes scored independently).
    """
    print(f"\nSpatial deduplication (radius = {radius_m} m) ...")
    n_before = len(gdf)

    # Approximate Cartesian metres (good enough for 500 m proximity)
    lat_rad = np.deg2rad(gdf["latitude"].values)
    lng_rad = np.deg2rad(gdf["longitude"].values)
    # Scale: 1° lon varies with latitude; use mean lat of India (~20°)
    mean_lat = 20.0
    meters_per_deg_lat = 111_000.0
    meters_per_deg_lon = 111_000.0 * np.cos(np.deg2rad(mean_lat))

    xy = np.column_stack([
        gdf["longitude"].values * meters_per_deg_lon,
        gdf["latitude"].values  * meters_per_deg_lat,
    ])

    tree = cKDTree(xy)
    pairs = tree.query_pairs(radius_m)   # set of (i, j) pairs

    # Union-Find
    parent = list(range(n_before))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, j in pairs:
        union(i, j)

    # Assign component id to each row
    gdf = gdf.copy()
    gdf["_component"] = [find(i) for i in range(n_before)]

    # Within each component, keep the row with the best name; tiebreak on tag count
    gdf["_has_name"] = gdf["village_name"].str.len() > 0
    gdf_sorted = gdf.sort_values(["_has_name", "_tag_count"],
                                  ascending=[False, False])
    gdf_dedup = gdf_sorted.drop_duplicates(subset=["_component"], keep="first")

    # Clean up helper columns
    gdf_dedup = gdf_dedup.drop(
        columns=["_component", "_has_name", "_tag_count"], errors="ignore")

    n_after = len(gdf_dedup)
    print(f"  Before: {n_before:,}  →  After: {n_after:,}  "
          f"({n_before - n_after:,} near-duplicate nodes removed)")
    return gdf_dedup.reset_index(drop=True)


def main():
    gpkg_path = OUT_DIR / "village_points.gpkg"
    csv_path  = OUT_DIR / "village_points.csv"

    if gpkg_path.exists():
        print(f"Already exists: {gpkg_path}")
        gdf = gpd.read_file(gpkg_path)
        print(f"Loaded {len(gdf):,} villages")
        # If the file was created without spatial dedup, re-process
        # (detect by checking whether any 2 villages are within 500 m)
        if "_tag_count" not in gdf.columns:
            print("  Existing file may predate spatial deduplication — re-checking ...")
        return

    data = fetch_overpass(QUERY)
    print(f"Received {len(data.get('elements', [])):,} OSM nodes")

    gdf = osm_to_geodataframe(data)
    print(f"Built GeoDataFrame: {len(gdf):,} villages")

    # Step 1: Exact-coordinate duplicates (identical lat/lon)
    n_before = len(gdf)
    gdf = gdf.drop_duplicates(subset=["latitude", "longitude"])
    print(f"Exact-coordinate dedup: {n_before:,} → {len(gdf):,} "
          f"({n_before - len(gdf):,} removed)")

    # Step 2: Spatial deduplication — merge nodes within DEDUP_RADIUS_M
    gdf = spatial_deduplicate(gdf, radius_m=DEDUP_RADIUS_M)

    # Drop helper column before saving
    gdf = gdf.drop(columns=["_tag_count"], errors="ignore")

    # Save
    gdf.to_file(gpkg_path, driver="GPKG")
    gdf.drop(columns=["geometry"]).to_csv(csv_path, index=False)
    print(f"\nSaved → {gpkg_path}  ({len(gdf):,} villages)")
    print(f"Saved → {csv_path}")

    # Summary
    named = (gdf["village_name"].str.len() > 0).sum()
    print(f"\nNamed villages: {named:,} / {len(gdf):,} "
          f"({named/len(gdf)*100:.1f}%)")
    print(f"State coverage: {gdf['state_name'].nunique()} states/UTs")
    print(gdf["state_name"].value_counts().head(10).to_string())


if __name__ == "__main__":
    main()
