"""
Step 1g: Distance to nearest city and national highway.

Adds two context signals:
  dist_city_km     — km to nearest city with population ≥ 50k
  dist_highway_km  — km to nearest national/state highway node

These are crucial controls:
  - Villages growing near cities may be suburban spillover (lower signal quality)
  - Villages growing far from infrastructure show organic rural growth (higher quality)
  - Can also be used as features in the ML model

Sources:
  - Cities: Natural Earth 10m populated places (free, downloaded once)
  - Highways: OSM Overpass API (NH + SH in India), sampled at 0.1° grid

Output:
  /data/satellite/kritter/processed/village_city_distance.csv
"""

import warnings
import requests
import zipfile
import io
import time
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from scipy.spatial import cKDTree
from shapely.geometry import Point

warnings.filterwarnings("ignore")

PROC_DIR  = Path("/data/satellite/kritter/processed")
SHRUG_DIR = Path("/data/satellite/kritter/shrug")

NE_PLACES_URL = ("https://naturalearth.s3.amazonaws.com/10m_cultural/"
                 "ne_10m_populated_places.zip")
NE_PLACES_PATH = Path("/tmp/ne_10m_populated_places.shp")
INDIA_BBOX = (68.0, 7.5, 97.5, 37.5)
MIN_POP = 50_000


def download_ne_places():
    if NE_PLACES_PATH.exists():
        print("  Natural Earth places already downloaded")
        return
    print("  Downloading Natural Earth populated places...")
    r = requests.get(NE_PLACES_URL, timeout=120,
                     headers={"User-Agent": "kritter-pipeline/1.0"})
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        zf.extractall("/tmp/")
    print("  Extracted to /tmp/")


def get_india_cities() -> np.ndarray:
    """Return array of (lon, lat) for India cities with pop >= MIN_POP."""
    download_ne_places()
    cities = gpd.read_file(NE_PLACES_PATH)
    india = cities[
        (cities["LATITUDE"].between(INDIA_BBOX[1], INDIA_BBOX[3])) &
        (cities["LONGITUDE"].between(INDIA_BBOX[0], INDIA_BBOX[2])) &
        (cities["POP_MAX"] >= MIN_POP)
    ]
    print(f"  India cities (pop≥{MIN_POP//1000}k): {len(india)}")
    return india[["LONGITUDE", "LATITUDE"]].values


def get_highway_nodes() -> np.ndarray:
    """
    Sample OSM highway nodes (motorway+trunk+primary) within India.
    Uses Overpass bbox query on a coarse grid to avoid timeout.
    Returns array of (lon, lat) for highway reference points.
    """
    print("  Querying OSM highways via Overpass (grid approach)...")
    OVERPASS = "https://overpass-api.de/api/interpreter"
    all_nodes = []

    # Split India into 6 N-S strips to avoid single large query
    lon_steps = [(68, 76), (76, 84), (84, 97.5)]
    lat_steps = [(7.5, 20), (20, 37.5)]

    for lon_s, lon_e in lon_steps:
        for lat_s, lat_e in lat_steps:
            query = f"""
[out:json][timeout:60];
(
  way["highway"~"^(motorway|trunk|primary)$"]
    ({lat_s},{lon_s},{lat_e},{lon_e});
);
out geom 50;
"""
            for attempt in range(3):
                try:
                    r = requests.post(OVERPASS, data={"data": query}, timeout=90,
                                      headers={"User-Agent": "kritter-pipeline/1.0"})
                    r.raise_for_status()
                    data = r.json()
                    break
                except Exception as e:
                    print(f"    retry {attempt+1} ({e})")
                    time.sleep(5 * (attempt + 1))
            else:
                print(f"    skipping bbox {lat_s},{lon_s},{lat_e},{lon_e}")
                continue

            for elem in data.get("elements", []):
                if "geometry" in elem:
                    # Sample every 5th node to reduce density
                    geom = elem["geometry"]
                    for pt in geom[::5]:
                        all_nodes.append((pt["lon"], pt["lat"]))

            time.sleep(1)  # polite delay between cells

    nodes = np.array(all_nodes) if all_nodes else np.empty((0, 2))
    print(f"  Highway nodes: {len(nodes):,}")
    return nodes


def haversine_approx(lat1_deg, lon1_deg, lat2_arr, lon2_arr) -> np.ndarray:
    """Fast approximate haversine distance in km (equirectangular correction)."""
    R = 6371.0
    lat1 = np.radians(lat1_deg)
    dlat = np.radians(lat2_arr - lat1_deg)
    dlon = np.radians(lon2_arr - lon1_deg)
    a = dlat**2 + (np.cos(lat1) * dlon)**2
    return R * np.sqrt(a)


def compute_distances(gdf, city_coords, hw_coords):
    """Compute min distance to city and highway for all villages."""
    lats = gdf.geometry.y.values
    lons = gdf.geometry.x.values

    # Scale lon for distance calculation (approximate)
    lat_mean = np.radians(np.mean(lats))
    scale = np.cos(lat_mean)

    coords_scaled = np.column_stack([lats, lons * scale])

    dist_city = np.full(len(gdf), np.nan)
    if len(city_coords) > 0:
        city_scaled = np.column_stack([city_coords[:, 1],
                                       city_coords[:, 0] * scale])
        tree_c = cKDTree(city_scaled)
        dists, _ = tree_c.query(coords_scaled, k=1)
        dist_city = dists * 111.0  # 1 deg lat ≈ 111 km
        print(f"  City distances: min={dist_city.min():.1f} km  "
              f"median={np.median(dist_city):.1f} km")

    dist_hw = np.full(len(gdf), np.nan)
    if len(hw_coords) > 0:
        hw_scaled = np.column_stack([hw_coords[:, 1],
                                     hw_coords[:, 0] * scale])
        tree_h = cKDTree(hw_scaled)
        dists_h, _ = tree_h.query(coords_scaled, k=1)
        dist_hw = dists_h * 111.0
        print(f"  Highway distances: min={dist_hw.min():.1f} km  "
              f"median={np.median(dist_hw):.1f} km")

    return dist_city, dist_hw


def main():
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    out_path = PROC_DIR / "village_city_distance.csv"
    if out_path.exists():
        print(f"Already exists: {out_path.name} — delete to recompute")
        return

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

    print(f"Villages: {len(gdf):,}")

    # Get reference points
    print("\n1. India cities:")
    city_coords = get_india_cities()

    print("\n2. Highway nodes:")
    hw_path = PROC_DIR / "highway_nodes.npy"
    if hw_path.exists():
        hw_coords = np.load(str(hw_path))
        print(f"  Loaded cached highway nodes: {len(hw_coords):,}")
    else:
        hw_coords = get_highway_nodes()
        if len(hw_coords) > 0:
            np.save(str(hw_path), hw_coords)

    print("\n3. Computing distances...")
    dist_city, dist_hw = compute_distances(gdf, city_coords, hw_coords)

    result = pd.DataFrame({
        "pc11_village_id": gdf["pc11_village_id"].astype(str),
        "dist_city_km":    np.round(dist_city, 2),
        "dist_highway_km": np.round(dist_hw, 2),
    })
    result.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}  ({len(result):,} rows)")


if __name__ == "__main__":
    main()
