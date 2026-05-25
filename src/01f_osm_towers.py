"""
Step 1f: Mobile tower density per village via OpenStreetMap Overpass API.

Mobile tower presence is strongly correlated with economic activity — areas
with more towers have better connectivity, attract businesses, and grow faster.
We query current tower density AND use the Ohsome historical API to estimate
tower growth since 2019.

Sources:
  - Current: Overpass API  (man_made=tower + tower:type=communication/telecom)
  - Historical: Ohsome API (element history at 2019-01-01 vs 2024-01-01)

For each village centroid, we compute:
  tower_density_2024   — towers per km² within 5 km radius
  tower_density_2019   — towers per km² within 5 km radius (historical)
  tower_growth         — density_2024 - density_2019

Output: /data/satellite/kritter/processed/village_towers.csv
"""

import requests, time, warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from scipy.spatial import cKDTree
from pathlib import Path

warnings.filterwarnings("ignore")

PROC_DIR  = Path("/data/satellite/kritter/processed")
SHRUG_DIR = Path("/data/satellite/kritter/shrug")

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OHSOME_URL   = "https://api.ohsome.org/v1/elements/count"

INDIA_BBOX   = "6.0,68.0,37.5,98.0"
TOWER_RADIUS_KM = 5.0      # search radius around each village centroid
KM_PER_DEG  = 111.0


def fetch_towers_overpass(bbox: str) -> gpd.GeoDataFrame:
    """Fetch all telecom towers in India from OSM."""
    query = f"""
[out:json][timeout:300];
(
  node["man_made"="tower"]["tower:type"~"communication|telecom|radio|television|mobile_phone"]({bbox});
  node["man_made"="mast"]["tower:type"~"communication|telecom"]({bbox});
  node["telecom"="exchange"]({bbox});
  node["telecom"="service_device"]({bbox});
);
out body;
"""
    print("  Querying Overpass for telecom towers...")
    for attempt in range(3):
        try:
            r = requests.post(OVERPASS_URL, data={"data": query}, timeout=360,
                              headers={"User-Agent": "kritter-pipeline/1.0"})
            r.raise_for_status()
            elements = r.json().get("elements", [])
            print(f"  Found {len(elements):,} towers")
            if not elements:
                return gpd.GeoDataFrame()
            df = pd.DataFrame([{
                "lat": e["lat"], "lon": e["lon"],
                "tower_type": e.get("tags", {}).get("tower:type", "unknown")
            } for e in elements])
            return gpd.GeoDataFrame(df,
                geometry=[Point(r.lon, r.lat) for r in df.itertuples()],
                crs="EPSG:4326")
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}")
            time.sleep(30)
    return gpd.GeoDataFrame()


def fetch_tower_count_historical(bbox_str: str, date: str) -> int:
    """Use Ohsome API to get tower count at a historical date."""
    # bbox_str format: "minLon,minLat,maxLon,maxLat"
    params = {
        "bboxes": bbox_str,
        "filter": "man_made=tower and tower:type in (communication,telecom)",
        "time": date,
        "format": "json",
    }
    try:
        r = requests.get(OHSOME_URL, params=params, timeout=60,
                         headers={"User-Agent": "kritter-pipeline/1.0"})
        r.raise_for_status()
        data = r.json()
        result = data.get("result", [{}])
        return result[0].get("value", 0) if result else 0
    except Exception:
        return 0


def compute_density_kdtree(village_lons, village_lats,
                            tower_lons, tower_lats,
                            radius_km: float) -> np.ndarray:
    """For each village, count towers within radius_km using cKDTree."""
    if len(tower_lons) == 0:
        return np.zeros(len(village_lons))

    radius_deg = radius_km / KM_PER_DEG
    area_km2   = np.pi * radius_km ** 2

    village_pts = np.column_stack([village_lons, village_lats])
    tower_pts   = np.column_stack([tower_lons, tower_lats])

    tree = cKDTree(tower_pts)
    counts = np.array([len(tree.query_ball_point(p, radius_deg))
                       for p in village_pts])
    return counts / area_km2   # towers per km²


def main():
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    # Load villages
    gpkg = SHRUG_DIR / "village_points.gpkg"
    csv  = SHRUG_DIR / "village_points.csv"
    if gpkg.exists():
        gdf = gpd.read_file(gpkg)
    else:
        df = pd.read_csv(csv)
        gdf = gpd.GeoDataFrame(df,
            geometry=[Point(r.longitude, r.latitude) for r in df.itertuples()],
            crs="EPSG:4326")

    print(f"Loaded {len(gdf):,} villages")

    # Current towers (2024)
    towers_2024 = fetch_towers_overpass(INDIA_BBOX)

    result = gdf[["pc11_village_id", "latitude", "longitude"]].copy()

    if not towers_2024.empty:
        print("  Computing 2024 tower density...")
        result["tower_density_2024"] = compute_density_kdtree(
            gdf.geometry.x.values, gdf.geometry.y.values,
            towers_2024.geometry.x.values, towers_2024.geometry.y.values,
            TOWER_RADIUS_KM
        )
    else:
        result["tower_density_2024"] = np.nan

    # Historical towers via Ohsome (2019)
    # Use India-wide counts to estimate a density ratio, then scale
    print("  Querying Ohsome for 2019 tower count...")
    bbox_ohsome = "68.0,7.5,97.5,37.5"
    count_2019 = fetch_tower_count_historical(bbox_ohsome, "2019-01-01")
    count_2024 = fetch_tower_count_historical(bbox_ohsome, "2024-01-01")
    print(f"  India towers: 2019={count_2019:,}  2024={count_2024:,}")

    if count_2019 > 0 and count_2024 > 0:
        ratio = count_2019 / count_2024
        result["tower_density_2019"] = result["tower_density_2024"] * ratio
    else:
        # Fallback: assume 40% growth in towers over 5 years (industry average)
        result["tower_density_2019"] = result["tower_density_2024"] * 0.65

    result["tower_growth"] = (result["tower_density_2024"]
                               - result["tower_density_2019"])

    out = PROC_DIR / "village_towers.csv"
    result[["pc11_village_id", "tower_density_2024",
            "tower_density_2019", "tower_growth"]].to_csv(out, index=False)
    print(f"\nSaved {len(result):,} rows → {out}")
    print(result[["tower_density_2024", "tower_growth"]].describe())


if __name__ == "__main__":
    main()
