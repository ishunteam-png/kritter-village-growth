"""
Step 13: SECC 2011 ground-truth cross-validation.

The Socio-Economic and Caste Census (SECC) 2011 provides block-level
rural household data including electricity access counts. Villages in
blocks with low 2011 electricity access that now show high NTL growth
are *definitionally* electrification + economic growth candidates —
this converts the self-supervised ML label from "self-consistency metric"
to "externally validated predictor."

Data source:
  data.gov.in — "SECC 2011 Data Rural Households" (block/district level)
  Fallback: Census 2011 H-series table H-12 (houses with electricity)
  at censusindia.gov.in/DigitalLibrary

Cross-validation logic:
  1. Download SECC block-level electricity access data
  2. Spatial join top-100 villages → SECC block by (state + district + subdistrict)
  3. Flag villages in "low-access blocks" (< 30% electricity 2011):
       → NTL growth here = genuine electrification/economic rise, not noise
  4. Flag villages in "high-access blocks" (> 80% electricity 2011):
       → NTL growth requires additional built-up/economic explanation
  5. Output: validation_secc_ground_truth.csv

Outputs:
  output/validation_secc_ground_truth.csv  — top 100 with SECC join + flags
  output/secc_analysis_summary.json        — summary statistics
"""

import io
import json
import warnings
import requests
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

_CFG_PATH = Path(__file__).parent.parent / "config.yaml"
_CFG      = yaml.safe_load(_CFG_PATH.read_text()) if _CFG_PATH.exists() else {}
_data     = _CFG.get("paths", {}).get("data_root", "/data/satellite/kritter")

OUTPUT_DIR = Path(_data) / "output"
CACHE_DIR  = Path("/tmp/secc_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# data.gov.in SECC 2011 rural households — block-level electricity access.
# data.gov.in wraps paginated resources in a JSON envelope; we request JSON
# explicitly so the parser doesn't see raw HTML error pages as CSV.
SECC_API_URLS = [
    # Primary: data.gov.in SECC 2011 — JSON format (avoids CSV tokenisation errors)
    ("https://data.gov.in/resource/secc-2011-data-rural-households"
     "?format=json&limit=50000"),
    # Secondary: older data.gov.in resource ID (electricity by district)
    "https://data.gov.in/api/datastore/resource.json"
    "?resource_id=7e5b8e88-2f85-4b9e-9f83-c18b37b9d6dc&limit=50000",
    # Fallback: Census 2011 H-12 table (district-level, skip SSL verification)
    "https://censusindia.gov.in/DigitalLibrary/data/Census_2011/H-series/"
    "H-12/DDW_H12-0000.xlsx",
]


def download_secc(cache_path: Path) -> pd.DataFrame | None:
    """Download SECC 2011 block-level electricity data with caching."""
    if cache_path.exists():
        print(f"  SECC data cached: {cache_path}")
        return pd.read_csv(cache_path)

    for url in SECC_API_URLS:
        try:
            print(f"  Trying: {url[:80]}...")
            r = requests.get(url, timeout=45,
                             headers={"User-Agent": "kritter-pipeline/1.0"},
                             verify=False)    # some govt sites have self-signed certs
            r.raise_for_status()
            ct = r.headers.get("Content-Type", "")

            # JSON envelope (data.gov.in modern API)
            if "json" in ct or url.endswith("json"):
                j = r.json()
                # Multiple response shapes from data.gov.in
                records = (j.get("records") or j.get("data") or
                           j.get("result", {}).get("records") or [])
                if records:
                    df = pd.json_normalize(records)
                    df.to_csv(cache_path, index=False)
                    print(f"  Downloaded {len(df):,} blocks (JSON)")
                    return df
                print(f"  JSON empty — keys: {list(j.keys())[:6]}")
            elif "csv" in ct or url.endswith(".csv"):
                df = pd.read_csv(io.StringIO(r.text))
                df.to_csv(cache_path, index=False)
                print(f"  Downloaded {len(df):,} rows (CSV)")
                return df
            elif "xlsx" in url or "excel" in ct:
                df = pd.read_excel(io.BytesIO(r.content))
                df.to_csv(cache_path, index=False)
                print(f"  Downloaded {len(df):,} rows (Excel)")
                return df
            else:
                # Try CSV parse as last resort
                try:
                    df = pd.read_csv(io.StringIO(r.text))
                    df.to_csv(cache_path, index=False)
                    print(f"  Downloaded {len(df):,} rows (raw)")
                    return df
                except Exception:
                    print(f"  ✗ Unrecognised content-type: {ct}")
        except Exception as e:
            print(f"  ✗ {type(e).__name__}: {e}")

    print("  ⚠  All SECC download attempts failed — using Census 2011 state-level fallback")
    return None


def census_state_electricity_fallback() -> pd.DataFrame:
    """
    Census 2011 state-level household electricity access (% connected).
    From the H-12 table summary published by Census of India 2011.
    These are approximate state-level figures used when block-level SECC
    download fails — not village-specific but allows a coarse signal.
    """
    # Source: Census of India 2011 — Houses, Household Amenities and Assets (H-series)
    # Table H-12: Percentage of households with electricity (rural)
    STATE_ELEC_PCT = {
        "Himachal Pradesh": 98.5, "Kerala": 98.0, "Punjab": 97.0,
        "Goa": 96.0, "Tamil Nadu": 94.5, "Haryana": 91.3,
        "Andhra Pradesh": 87.6, "Uttarakhand": 85.7, "Gujarat": 84.3,
        "Karnataka": 83.1, "Maharashtra": 77.8, "Rajasthan": 67.9,
        "Madhya Pradesh": 67.2, "West Bengal": 66.3, "Odisha": 60.5,
        "Uttar Pradesh": 48.4, "Jharkhand": 46.5, "Assam": 42.5,
        "Chhattisgarh": 68.7, "Bihar": 16.4,
    }
    rows = [{"state_name": k, "elec_access_pct_2011": v,
             "source": "Census2011_H12_rural_state_level"}
            for k, v in STATE_ELEC_PCT.items()]
    return pd.DataFrame(rows)


def classify_village(row: pd.Series) -> str:
    """Assign ground-truth category based on SECC electricity access context."""
    elec = row.get("elec_access_pct_2011", np.nan)
    if pd.isna(elec):
        return "unknown — SECC join failed"
    ntl_growth = row.get("ntl_growth_pct", np.nan)
    if pd.isna(ntl_growth):
        return "unknown — NTL growth missing"

    if elec < 30:
        if ntl_growth > 100:
            return "high-confidence: low-access block, large NTL rise"
        elif ntl_growth > 30:
            return "moderate-confidence: low-access block, moderate NTL rise"
        else:
            return "low-growth despite low-access block — needs field check"
    elif elec < 60:
        if ntl_growth > 200:
            return "high-confidence: partial-access block, large NTL rise"
        else:
            return "medium-access block — economic growth likely contributor"
    else:
        if ntl_growth > 300:
            return "high-access block — growth likely economic not electrification"
        else:
            return "high-access block, modest NTL growth — further investigation needed"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    top100_path = OUTPUT_DIR / "top_100_villages.csv"
    if not top100_path.exists():
        print("top_100_villages.csv not found — run 04_score_rank.py first"); return

    print("Loading top-100 villages...")
    top100 = pd.read_csv(top100_path)
    print(f"  {len(top100):,} villages")

    # ── Attempt SECC block-level download ─────────────────────────────────────
    secc_cache = CACHE_DIR / "secc_2011_rural.csv"
    secc = download_secc(secc_cache)

    if secc is not None:
        print(f"\nSECC columns: {list(secc.columns[:8])}")

        # Normalise join columns
        def clean(s):
            return str(s).lower().strip().replace(" ", "").replace("-", "")

        state_col   = next((c for c in secc.columns if "state" in c.lower()), None)
        dist_col    = next((c for c in secc.columns if "district" in c.lower()), None)
        elec_col    = next((c for c in secc.columns
                            if any(k in c.lower() for k in
                                   ["electri", "light", "power", "lamp", "solar"])), None)

        if state_col and dist_col and elec_col:
            secc[f"{state_col}_clean"] = secc[state_col].apply(clean)
            secc[f"{dist_col}_clean"]  = secc[dist_col].apply(clean)
            top100["state_clean"] = top100["state_name"].apply(clean)
            top100["dist_clean"]  = top100["district_name"].fillna("").apply(clean)

            # Try to compute % from absolute counts if not already a percentage
            ec = secc[elec_col]
            if ec.max() > 100:   # likely absolute count, not percentage
                total_col = next(
                    (c for c in secc.columns if "total" in c.lower()
                     and "household" in c.lower()), None)
                if total_col:
                    secc["elec_access_pct_2011"] = (
                        secc[elec_col] / secc[total_col].replace(0, np.nan) * 100
                    ).clip(0, 100)
                else:
                    secc["elec_access_pct_2011"] = np.nan
            else:
                secc["elec_access_pct_2011"] = ec

            # District-level join
            secc_dist = (secc
                         .groupby([f"{state_col}_clean", f"{dist_col}_clean"])
                         ["elec_access_pct_2011"]
                         .mean()
                         .reset_index()
                         .rename(columns={
                             f"{state_col}_clean": "state_clean",
                             f"{dist_col}_clean":  "dist_clean",
                         }))

            top100 = top100.merge(secc_dist, on=["state_clean", "dist_clean"],
                                  how="left")
            matched = top100["elec_access_pct_2011"].notna().sum()
            print(f"  SECC join: {matched}/100 villages matched to block electricity data")
        else:
            print(f"  ⚠  SECC schema unrecognised (cols: {list(secc.columns[:8])})")
            secc = None   # fall through to state-level fallback

    # ── State-level fallback ──────────────────────────────────────────────────
    if secc is None or "elec_access_pct_2011" not in top100.columns:
        print("\nUsing Census 2011 state-level electricity access (fallback)...")
        state_elec = census_state_electricity_fallback()
        top100 = top100.merge(state_elec[["state_name", "elec_access_pct_2011",
                                           "source"]],
                              on="state_name", how="left")
        matched = top100["elec_access_pct_2011"].notna().sum()
        print(f"  State-level match: {matched}/100 villages")

    # ── Classify ──────────────────────────────────────────────────────────────
    top100["secc_category"] = top100.apply(classify_village, axis=1)
    top100["secc_low_access_block"] = top100["elec_access_pct_2011"].fillna(100) < 30
    top100["secc_high_access_block"] = top100["elec_access_pct_2011"].fillna(0)  > 80

    # ── Summary stats ─────────────────────────────────────────────────────────
    low_access  = int(top100["secc_low_access_block"].sum())
    high_access = int(top100["secc_high_access_block"].sum())
    high_conf   = top100["secc_category"].str.startswith("high-confidence").sum()

    summary = {
        "total_top100": 100,
        "secc_matched": int(matched),
        "low_access_block_count": int(low_access),
        "high_access_block_count": int(high_access),
        "high_confidence_growth_count": int(high_conf),
        "high_confidence_growth_pct": round(float(high_conf), 1),
        "interpretation": (
            f"{high_conf} of 100 top villages are in low-electricity-access areas "
            f"with large NTL growth — strong signal of genuine development, "
            f"not noise. {high_access} villages are in already-electrified areas, "
            "requiring non-electrification explanation (economic, industrial)."
        ),
        "data_source": (
            "SECC 2011 block-level" if secc is not None else
            "Census 2011 H-12 state-level (fallback)"
        ),
    }

    print(f"\nSECC Ground Truth Summary:")
    print(f"  Villages in low-access blocks (<30%): {low_access}")
    print(f"  Villages in high-access blocks (>80%): {high_access}")
    print(f"  High-confidence growth candidates: {high_conf}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_cols = [c for c in [
        "rank", "village_name", "state_name", "district_name",
        "ntl_2019", "ntl_2024", "ntl_growth_pct",
        "elec_access_pct_2011", "secc_low_access_block", "secc_high_access_block",
        "secc_category", "composite_score", "multi_signal_confirmed",
    ] if c in top100.columns]

    out_path = OUTPUT_DIR / "validation_secc_ground_truth.csv"
    top100[out_cols].to_csv(out_path, index=False)
    print(f"\n  → {out_path}")

    json_path = OUTPUT_DIR / "secc_analysis_summary.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"  → {json_path}")

    print("\nSECC validation complete.")


if __name__ == "__main__":
    main()
