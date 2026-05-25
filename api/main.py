"""
Village Economic Growth API

FastAPI service exposing the pipeline outputs as a REST API.

Endpoints:
  GET /                      health check + summary stats
  GET /villages              paginated village list with filters
  GET /villages/top100       top 100 villages (pre-computed)
  GET /villages/{id}         single village by pc11_village_id
  GET /states                list states with village counts
  GET /archetypes            growth archetype distribution
  GET /forecast/{id}         NTL forecast for a village
  GET /rising-stars          top villages outside top-100 with steepest slope

Run on EC2:
  conda activate insar
  uvicorn api.main:app --host 0.0.0.0 --port 8502 --reload

Or locally (reads from S3 fallback):
  pip install fastapi uvicorn pandas
  uvicorn api.main:app --reload
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── Config ────────────────────────────────────────────────────────────────────

S3_BASE  = "https://kritter-village-growth-335451551223.s3.ap-south-1.amazonaws.com/"
DATA_DIR = Path(__file__).parent.parent / "output"

app = FastAPI(
    title="Village Economic Growth API",
    description="India village economic growth 2019–2024 — Kritter Software Technologies",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_csv(name: str) -> pd.DataFrame:
    local = DATA_DIR / name
    if local.exists():
        return pd.read_csv(local, low_memory=False)
    try:
        return pd.read_csv(S3_BASE + name)
    except Exception:
        return pd.DataFrame()


@lru_cache(maxsize=8)
def get_top100() -> pd.DataFrame:
    df = _load_csv("top_100_villages.csv")
    if "ntl_growth_pct" in df.columns:
        df["ntl_growth_pct"] = df["ntl_growth_pct"].round(2)
    return df


@lru_cache(maxsize=2)
def get_forecast() -> pd.DataFrame:
    return _load_csv("top_100_forecast.csv")


@lru_cache(maxsize=2)
def get_archetypes() -> pd.DataFrame:
    return _load_csv("village_archetypes.csv")


@lru_cache(maxsize=2)
def get_rising_stars() -> pd.DataFrame:
    return _load_csv("rising_stars.csv")


def _safe_val(v):
    """Return None for NaN/inf floats, pass everything else through."""
    import math
    if isinstance(v, float) and not math.isfinite(v):
        return None
    # numpy scalar types need coercion too
    try:
        if np.isnan(v) or np.isinf(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def _df_to_records(df: pd.DataFrame) -> list[dict]:
    """Convert DataFrame to JSON-safe records (replace NaN/inf with None)."""
    records = df.to_dict(orient="records")
    return [{k: _safe_val(v) for k, v in rec.items()} for rec in records]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    top100 = get_top100()
    return {
        "status": "ok",
        "service": "Village Economic Growth API",
        "version": "2.0.0",
        "data": {
            "top_100_villages": len(top100),
            "top_village": top100.iloc[0]["village_name"] if not top100.empty else None,
            "top_state":   top100.iloc[0]["state_name"]   if not top100.empty else None,
            "s3_bucket":   S3_BASE,
        },
        "endpoints": {
            "top100":       "/villages/top100",
            "filters":      "/villages?state=Karnataka&min_score=80",
            "single":       "/villages/{pc11_village_id}",
            "states":       "/states",
            "archetypes":   "/archetypes",
            "forecast":     "/forecast/{pc11_village_id}",
            "rising_stars": "/rising-stars",
            "docs":         "/docs",
        }
    }


@app.get("/villages/top100", tags=["Villages"])
def top_100_villages(
    state: Optional[str] = Query(None, description="Filter by state name"),
    archetype: Optional[str] = Query(None, description="Filter by growth archetype"),
):
    df = get_top100().copy()
    if state:
        df = df[df["state_name"].str.lower().str.contains(state.lower(), na=False)]
    if archetype:
        if "archetype_name" in df.columns:
            df = df[df["archetype_name"].str.lower().str.contains(
                archetype.lower(), na=False)]
    cols = [c for c in [
        "rank", "village_name", "state_name", "district_name",
        "latitude", "longitude", "composite_score", "ml_growth_prob",
        "ntl_growth_pct", "ntl_2019", "ntl_2024", "ntl_trend_slope",
        "ndbi_growth", "ghsl_change", "confidence_score",
        "multi_signal_confirmed", "archetype_name",
        "shap_top1", "shap_top2", "dist_city_km",
    ] if c in df.columns]
    return {"count": len(df), "data": _df_to_records(df[cols])}


@app.get("/villages", tags=["Villages"])
def list_villages(
    state: Optional[str] = Query(None),
    district: Optional[str] = Query(None),
    min_score: float = Query(0, ge=0, le=100),
    max_rank: Optional[int] = Query(None, ge=1),
    confirmed_only: bool = Query(False),
    archetype: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    df = get_top100().copy()
    if state:
        df = df[df["state_name"].str.lower().str.contains(state.lower(), na=False)]
    if district:
        df = df[df["district_name"].str.lower().str.contains(district.lower(), na=False)]
    if "composite_score" in df.columns:
        df = df[df["composite_score"] >= min_score]
    if max_rank and "rank" in df.columns:
        df = df[df["rank"] <= max_rank]
    if confirmed_only and "multi_signal_confirmed" in df.columns:
        df = df[df["multi_signal_confirmed"] == True]
    if archetype and "archetype_name" in df.columns:
        df = df[df["archetype_name"].str.lower().str.contains(
            archetype.lower(), na=False)]

    total = len(df)
    df = df.iloc[offset: offset + limit]
    return {"total": total, "offset": offset, "limit": limit,
            "villages": _df_to_records(df)}


@app.get("/villages/{village_id}", tags=["Villages"])
def get_village(village_id: str):
    df = get_top100()
    if "pc11_village_id" not in df.columns:
        raise HTTPException(404, "Village index not available")
    row = df[df["pc11_village_id"].astype(str) == str(village_id)]
    if row.empty:
        raise HTTPException(404, f"Village {village_id} not in top-100")
    record = _df_to_records(row)[0]

    # Enrich with forecast if available
    fc = get_forecast()
    if not fc.empty and "pc11_village_id" in fc.columns:
        fc_row = fc[fc["pc11_village_id"].astype(str) == str(village_id)]
        if not fc_row.empty:
            for yr in [2025, 2026, 2027]:
                col = f"ntl_forecast_{yr}"
                if col in fc_row.columns:
                    record[col] = fc_row.iloc[0][col]

    return record


@app.get("/states", tags=["Geography"])
def list_states():
    df = get_top100()
    if df.empty or "state_name" not in df.columns:
        return {"states": []}
    counts = df.groupby("state_name").agg(
        village_count=("rank", "count"),
        avg_score=("composite_score", "mean"),
        top_village=("village_name", "first"),
    ).reset_index().sort_values("village_count", ascending=False)
    counts["avg_score"] = counts["avg_score"].round(2)
    return {"states": _df_to_records(counts)}


@app.get("/archetypes", tags=["Analytics"])
def archetype_summary():
    arch = get_archetypes()
    top  = get_top100()

    if arch.empty:
        if "archetype_name" in top.columns:
            arch = top[["archetype_name", "composite_score"]].copy()
        else:
            return {"archetypes": [], "note": "Run 12_shap_archetypes.py first"}

    summary = arch.groupby("archetype_name").agg(
        count=("archetype_name", "size"),
        avg_score=("composite_score", "mean"),
    ).reset_index().sort_values("count", ascending=False)
    summary["avg_score"] = summary["avg_score"].round(2)
    return {"total_villages": len(arch), "archetypes": _df_to_records(summary)}


@app.get("/forecast/{village_id}", tags=["Analytics"])
def village_forecast(village_id: str):
    fc = get_forecast()
    if fc.empty:
        raise HTTPException(503, "Forecast data not available — run 08_forecast.py")
    row = fc[fc["pc11_village_id"].astype(str) == str(village_id)]
    if row.empty:
        raise HTTPException(404, f"Village {village_id} not in forecast data")
    r = _df_to_records(row)[0]
    fc_years = {str(yr): r.get(f"ntl_forecast_{yr}") for yr in [2025, 2026, 2027]}
    return {
        "pc11_village_id": village_id,
        "village_name": r.get("village_name"),
        "state_name":   r.get("state_name"),
        "ntl_2024":     r.get("ntl_2024"),
        "forecast":     fc_years,
        "method": "damped Holt-Winters + linear ensemble (phi=0.9)",
    }


@app.get("/rising-stars", tags=["Analytics"])
def rising_stars(limit: int = Query(20, ge=1, le=100)):
    df = get_rising_stars()
    if df.empty:
        raise HTTPException(503, "Rising stars not available — run 08_forecast.py")
    cols = [c for c in ["rank_among_stars", "village_name", "state_name",
                         "ntl_trend_slope", "ntl_2024", "composite_score",
                         "archetype_name"] if c in df.columns]
    return {"count": min(limit, len(df)),
            "rising_stars": _df_to_records(df[cols].head(limit))}
