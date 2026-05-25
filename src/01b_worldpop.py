"""
Step 1b: Download WorldPop annual population rasters for India (2019–2020).

Population change is a demographic signal: economically growing villages
attract labour migration and show rising population density.

Source: WorldPop Global 100m unconstrained estimates (worldpop.org)
  2019 and 2020 are the most recent years with full India coverage.

Output: /data/satellite/kritter/processed/worldpop_india_{year}.tif
"""

import requests
import numpy as np
import rasterio
from pathlib import Path
from tqdm import tqdm

PROC_DIR = Path("/data/satellite/kritter/processed")
YEARS    = [2019, 2020]

# WorldPop unconstrained individual countries, 100m
WORLDPOP_URLS = {
    2019: "https://data.worldpop.org/GIS/Population/Global_2000_2020/2019/IND/ind_ppp_2019.tif",
    2020: "https://data.worldpop.org/GIS/Population/Global_2000_2020/2020/IND/ind_ppp_2020.tif",
}


def download_file(url: str, dest: Path, chunk_mb: int = 4) -> None:
    if dest.exists():
        print(f"  Already exists: {dest.name}")
        return
    print(f"  Downloading {dest.name} ...")
    r = requests.get(url, stream=True, timeout=600,
                     headers={"User-Agent": "kritter-pipeline/1.0"})
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    chunk = chunk_mb * 1024 * 1024
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
        for data in r.iter_content(chunk):
            f.write(data)
            bar.update(len(data))


def main():
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    for year, url in WORLDPOP_URLS.items():
        dest = PROC_DIR / f"worldpop_india_{year}.tif"
        download_file(url, dest)
        with rasterio.open(dest) as src:
            print(f"  {year}: {src.width}×{src.height} px  CRS={src.crs.to_epsg()}")

    print("\nWorldPop download complete.")


if __name__ == "__main__":
    main()
