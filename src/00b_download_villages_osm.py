"""
Download India village centroids from OpenStreetMap via Overpass API.

Produces a GeoDataFrame with ~300k village points including name, state,
district, lat, lon. Saved as GeoPackage (village_points.gpkg) and CSV.

This replaces the need for the SHRUG polygon shapefile for the point-based
extraction pipeline. VIIRS and WorldCover are both extracted at the centroid
point (or within a 500m buffer).

Why OSM:
- Free, no registration required
- ~300k villages in India with coordinates
- Regularly updated by the community

Limitation vs SHRUG:
- OSM admin tags (state/district) are less complete than Census data
- No polygon boundaries — only centroids

Output: /data/satellite/kritter/shrug/village_points.gpkg
         /data/satellite/kritter/shrug/village_points.csv

Run on EC2:
    conda activate insar
    python 00b_download_villages_osm.py
"""

import requests
import json
import time
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path
from tqdm import tqdm

OUT_DIR  = Path("/data/satellite/kritter/shrug")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# India bounding box for Overpass (S, W, N, E)
INDIA_BBOX = "6.0,68.0,37.5,98.0"

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
            "pc11_village_id": str(element["id"]),  # OSM node ID as village ID
            "village_name":    tags.get("name") or tags.get("name:en", ""),
            "state_name":      (tags.get("addr:state") or
                                tags.get("is_in:state") or ""),
            "district_name":   (tags.get("addr:district") or
                                tags.get("is_in:district") or ""),
            "subdistrict_name": tags.get("addr:subdistrict", ""),
            "latitude":        element["lat"],
            "longitude":       element["lon"],
        })

    df = pd.DataFrame(records)
    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(row.longitude, row.latitude) for row in df.itertuples()],
        crs="EPSG:4326",
    )
    return gdf


def main():
    gpkg_path = OUT_DIR / "village_points.gpkg"
    csv_path  = OUT_DIR / "village_points.csv"

    if gpkg_path.exists():
        print(f"Already exists: {gpkg_path}")
        gdf = gpd.read_file(gpkg_path)
        print(f"Loaded {len(gdf):,} villages")
        return

    data = fetch_overpass(QUERY)
    print(f"Received {len(data.get('elements', [])):,} OSM nodes")

    gdf = osm_to_geodataframe(data)
    print(f"Built GeoDataFrame: {len(gdf):,} villages")

    # Remove duplicates (same lat/lon)
    gdf = gdf.drop_duplicates(subset=["latitude", "longitude"])

    # Save
    gdf.to_file(gpkg_path, driver="GPKG")
    gdf.drop(columns=["geometry"]).to_csv(csv_path, index=False)
    print(f"Saved → {gpkg_path}  ({len(gdf):,} villages)")
    print(f"Saved → {csv_path}")

    # Summary
    print(f"\nState coverage: {gdf['state_name'].nunique()} states/UTs")
    print(gdf["state_name"].value_counts().head(10))


if __name__ == "__main__":
    main()
