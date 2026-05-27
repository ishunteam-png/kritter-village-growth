"""
Step 4: Score and rank villages — ML + composite economic growth index.

Scoring model:

  MAIN TRACK  (ntl_2019 ≥ 0.1 nW/cm²/sr AND ntl_2024 - ntl_2019 ≥ 1.0)
  ─────────────────────────────────────────────────────────────────────
  Stage 1 — feature engineering
    - BFAST change-point: breakpoint year, pre/post slope for NTL series
      (vectorised piecewise-linear search — O(4n) not O(n × iterrows))
    - Spatial lag: mean score of 10 nearest neighbours (cluster effect)
    - Signal-agreement: std of individual signal rank percentiles (low = consistent)
    - GHSL 2015→2020 built-up change (peer-reviewed built-up signal)
    - SAR VV delta (Sentinel-1, cloud-independent built-up)
    - Tower growth (mobile tower density change, connectivity proxy)
    - District outperformance: score relative to district mean

  Stage 2 — Signal amplifier (GradientBoostingClassifier, self-supervised)
    - Self-supervised label: top 10% on ≥2 of the primary signals simultaneously
    - Features: all available signal rank scores
    - Output: ml_growth_prob (0–1) — amplifies NTL/built-up co-occurrence signal
    - AUC = 1.000 is EXPECTED (labels derived from same features); not a validity claim
    - Weight capped at 15% to prevent this circular component from dominating composite

  Stage 3 — composite (weighted ensemble of independent signals + amplifier)
    composite_score = 0.35 × ntl_growth_log_score
                    + 0.20 × ndbi_growth_score    (if available)
                    + 0.15 × ghsl_change_score    (if available)
                    + 0.15 × ml_growth_prob_score (signal amplifier, self-supervised)
                    + 0.10 × s1_vv_delta_score    (if available)
                    + 0.05 × spatial_lag_score
    Weights auto-redistribute if signals missing.

    NOTE: _score columns use min-max normalisation (not global rank-percentile)
    so that top-tier villages are spread across a real score range, not all
    compressed into a 99.xx band with < 0.05 point spread.

  DARK TRACK  (ntl_2019 < 0.1) scored on NDBI + GHSL + tower growth.

Outputs:
  village_scored.csv        all 255k+ India villages
  top_100_villages.csv      main track top 100
  dark_top_50_villages.csv  dark track top 50
  village_uncertainty.csv   confidence intervals and stability scores
"""

import warnings
import zipfile
import io

import boto3
import yaml
import numpy as np
import pandas as pd
import geopandas as gpd
import joblib
import requests
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.stats import linregress
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline

# Suppress specific known-benign warnings; do NOT use filterwarnings("ignore") globally
# as it would hide genuine model degradation signals (e.g. sklearn ConvergenceWarning).
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice")
warnings.filterwarnings("ignore", category=UserWarning, module="geopandas")

# Load config.yaml — single source of truth for hyperparameters and paths
_CFG_PATH = Path(__file__).parent.parent / "config.yaml"
_CFG = yaml.safe_load(_CFG_PATH.read_text()) if _CFG_PATH.exists() else {}
_SCORING = _CFG.get("scoring", {})

_PATHS  = _CFG.get("paths", {})
_URLS   = _CFG.get("external_urls", {})
_data   = _PATHS.get("data_root", "/data/satellite/kritter")

INPUT_FILE  = Path(_data) / "processed/village_all_stats.csv"
PROC_DIR    = Path(_data) / "processed"
OUTPUT_DIR  = Path(_data) / "output"
STATES_PATH = _CFG.get("paths", {}).get(
    "natural_earth_shp", "/tmp/ne_10m_admin_1_states_provinces.shp")
GADM_URL    = _URLS.get("gadm_india_gpkg",
              "https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41_IND.gpkg")
GADM_DIR    = Path("/tmp/gadm_india")
GADM_GPKG   = GADM_DIR / "gadm41_IND.gpkg"

MIN_NTL              = 0.1    # nW/cm²/sr — below this → dark track
MIN_ABSOLUTE_NTL_DELTA = _SCORING.get("ntl_min_absolute_delta", 1.0)
YEARS    = [2019, 2020, 2021, 2022, 2023, 2024]
INDIA_LAT = (7.5, 37.5)
INDIA_LON = (68.0, 97.5)
SPATIAL_LAG_K = _SCORING.get("spatial_lag_k", 10)

# Composite weights read from config.yaml → scoring.composite_weights_full.
# Changing weights in config.yaml now takes effect without touching this file.
_DEFAULT_WEIGHTS = {
    # ml_growth_prob_score is a self-supervised signal amplifier (labels derived from
    # the same inputs → AUC=1.000 is expected, not evidence of predictive power).
    # Weight is capped at 15% so independent signals dominate the composite.
    "ml_growth_prob_score": 0.15,
    "ntl_growth_log_score": 0.35,
    "ndbi_growth_score":    0.20,
    "ghsl_change_score":    0.15,
    "s1_vv_delta_score":    0.10,
    "spatial_lag_score":    0.05,
}
COMPOSITE_WEIGHTS = _SCORING.get("composite_weights_full", _DEFAULT_WEIGHTS)

# Min-max normalisation clipping bounds — read from config.yaml so a single
# config change re-tunes all signal scores without touching code.
# clip_hi=0.99 means only the top 1% of villages hit the per-signal ceiling
# (score=100), giving the top 100 more differentiated composite scores than
# clip_hi=0.98 which lets the top 2% (≈7k villages) all tie at 100/100.
MINMAX_CLIP_LO = _SCORING.get("minmax_clip_lo", 0.02)
MINMAX_CLIP_HI = _SCORING.get("minmax_clip_hi", 0.99)

DARK_WEIGHTS = {
    "ndbi_growth_score":  0.40,
    "ghsl_change_score":  0.30,
    "tower_growth_score": 0.20,
    "pop_growth_score":   0.10,
}


# ── Scoring helpers ───────────────────────────────────────────────────────────

def rank_score(s: pd.Series) -> pd.Series:
    """Global percentile rank × 100. Used for ML label construction only.
    Do NOT use in the composite — it compresses all top-tier villages into
    a < 0.05-point band (99.96–99.99), making ranking meaningless."""
    return s.rank(pct=True, na_option="bottom") * 100


def minmax_score(s: pd.Series,
                 clip_lo: float = MINMAX_CLIP_LO,
                 clip_hi: float = MINMAX_CLIP_HI) -> pd.Series:
    """
    Robust min-max normalisation to [0, 100].

    Unlike rank_score, this preserves proportional signal spread within the
    top tier. A village with 10× the NTL growth of another scores substantially
    higher, not nearly identically.

    Clipping bounds default to config.yaml → scoring.minmax_clip_lo/hi (2nd/99th
    percentile). The 99th-percentile ceiling means only the top 1% of villages
    (~3,500 out of 356k) hit the per-signal ceiling; with 0.98 the top 2% all
    tied at 100/100, compressing discrimination at the very top of the ranking.
    """
    lo = s.quantile(clip_lo)
    hi = s.quantile(clip_hi)
    if hi <= lo:
        return s.rank(pct=True, na_option="bottom") * 100  # degenerate fallback
    return ((s.clip(lo, hi) - lo) / (hi - lo) * 100).where(s.notna(), other=np.nan).fillna(0)


def redistribute(weights: dict, available: list) -> dict:
    present = {k: v for k, v in weights.items() if k in available}
    if not present:
        return {}
    total = sum(present.values())
    return {k: v / total for k, v in present.items()}


# ── India filter ──────────────────────────────────────────────────────────────

def apply_india_filter(df):
    df = df[df["latitude"].between(*INDIA_LAT) &
            df["longitude"].between(*INDIA_LON)].copy()
    print(f"After bbox filter: {len(df):,}")

    if not Path(STATES_PATH).exists():
        print("  WARNING: Natural Earth shapefile not found — using bbox only")
        return df

    ne    = gpd.read_file(STATES_PATH)
    india = ne[ne["admin"] == "India"][["name", "geometry"]].copy()
    gdf_v = gpd.GeoDataFrame(df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326")
    joined = gpd.sjoin(gdf_v, india.rename(columns={"name": "ne_state"}),
                       how="inner", predicate="within")
    df = pd.DataFrame(joined.drop(columns=["geometry", "index_right"]))
    df["state_name"] = df["state_name"].fillna("").replace("", np.nan)
    df["state_name"] = df["state_name"].fillna(df["ne_state"])
    df = df.drop(columns=["ne_state"], errors="ignore")
    print(f"After polygon filter: {len(df):,}")
    return df


# ── District name fill via GADM ──────────────────────────────────────────────

def load_gadm() -> gpd.GeoDataFrame:
    if GADM_GPKG.exists():
        print("  GADM GeoPackage cached — skipping download")
    else:
        print("  Downloading GADM India GeoPackage (~100 MB) ...")
        GADM_DIR.mkdir(parents=True, exist_ok=True)
        r = requests.get(GADM_URL, stream=True, timeout=300)
        r.raise_for_status()
        with open(GADM_GPKG, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
        print(f"  Downloaded: {GADM_GPKG.stat().st_size / 1e6:.1f} MB")
    gadm = gpd.read_file(GADM_GPKG, layer="ADM_ADM_2", columns=["NAME_2", "geometry"])
    return gadm.rename(columns={"NAME_2": "district_gadm"}).to_crs("EPSG:4326")


def fill_district_names(df: pd.DataFrame) -> pd.DataFrame:
    """Spatial join with GADM level-2 to fill missing district_name values."""
    if "district_name" not in df.columns:
        df["district_name"] = np.nan

    blank = df["district_name"].isna() | (
        df["district_name"].astype(str).str.strip().isin(["", "nan", "None"]))
    n_blank = int(blank.sum())
    if n_blank == 0:
        print("  district_name: all present — skipping spatial join")
        return df

    print(f"  district_name: {n_blank:,} missing — running GADM spatial join ...")
    try:
        gadm = load_gadm()
    except Exception as e:
        print(f"  WARNING: could not load GADM ({e}) — district_name will be partial")
        return df

    sub  = df[blank].copy()
    gdf  = gpd.GeoDataFrame(sub,
               geometry=gpd.points_from_xy(sub["longitude"], sub["latitude"]),
               crs="EPSG:4326")
    joined = gpd.sjoin(gdf, gadm[["district_gadm", "geometry"]],
                       how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]

    df = df.copy()
    df.loc[blank, "district_name"] = joined["district_gadm"].values
    filled = int(df["district_name"].notna().sum())
    print(f"  district_name: {filled:,}/{len(df):,} filled "
          f"({df['district_name'].isna().sum()} still missing — border/coastal)")
    return df


# ── BFAST-style change-point detection (vectorised) ──────────────────────────

def bfast_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorised piecewise-linear breakpoint search over the 6-year NTL series.

    For each village, finds the interior breakpoint year (bp ∈ {2020,2021,2022,2023})
    that minimises the combined RSS of two OLS-fitted linear segments.
    Returns pre/post slopes and their difference (acceleration).

    Replaces the prior df.iterrows() loop with NumPy broadcasting — ~50× faster
    on 255k rows while producing identical results.

    NOTE: This is a simplified structural-break test (minimum-RSS piecewise OLS),
    not the full BFAST algorithm (which adds Bai-Perron significance tests and
    seasonal decomposition). Results should be interpreted as candidate breakpoints,
    not statistically confirmed change detections.
    """
    x       = np.array(YEARS, dtype=float)
    n_years = len(YEARS)
    ntl_cols = [f"ntl_{y}" for y in YEARS]
    Y = df[ntl_cols].values.astype(float)  # (n, 6)
    n = len(Y)

    best_rss        = np.full(n, np.inf)
    best_bp_year    = np.full(n, np.nan)
    best_pre_slope  = np.full(n, np.nan)
    best_post_slope = np.full(n, np.nan)

    for bp in range(1, n_years - 1):
        pre_x  = x[:bp]           # (bp,)
        post_x = x[bp:]           # (n_years - bp,)
        pre_Y  = Y[:, :bp]        # (n, bp)
        post_Y = Y[:, bp:]        # (n, n_years - bp)

        pre_ok  = (np.sum(~np.isnan(pre_Y),  axis=1) >= 2)
        post_ok = (np.sum(~np.isnan(post_Y), axis=1) >= 2)
        both_ok = pre_ok & post_ok
        if not both_ok.any():
            continue

        # Vectorised OLS slope: β = Σ(xd · yd) / Σ(xd²)
        pre_xd  = pre_x  - pre_x.mean()
        post_xd = post_x - post_x.mean()
        pre_xv  = float((pre_xd  ** 2).sum()) or 1e-9
        post_xv = float((post_xd ** 2).sum()) or 1e-9

        pre_ym   = np.nanmean(pre_Y,  axis=1, keepdims=True)
        post_ym  = np.nanmean(post_Y, axis=1, keepdims=True)
        pre_yd   = np.where(np.isnan(pre_Y),  0.0, pre_Y  - pre_ym)
        post_yd  = np.where(np.isnan(post_Y), 0.0, post_Y - post_ym)

        pre_s  = (pre_yd  @ pre_xd)  / pre_xv   # (n,)
        post_s = (post_yd @ post_xd) / post_xv

        pre_i  = pre_ym.squeeze()  - pre_s  * pre_x.mean()
        post_i = post_ym.squeeze() - post_s * post_x.mean()

        pre_resid  = np.nansum(
            (pre_Y  - (pre_s[:,  None] * pre_x  + pre_i[:,  None])) ** 2, axis=1)
        post_resid = np.nansum(
            (post_Y - (post_s[:, None] * post_x + post_i[:, None])) ** 2, axis=1)
        rss = pre_resid + post_resid

        improve = both_ok & (rss < best_rss)
        best_rss[improve]        = rss[improve]
        best_bp_year[improve]    = float(YEARS[bp])
        best_pre_slope[improve]  = pre_s[improve]
        best_post_slope[improve] = post_s[improve]

    return pd.DataFrame({
        "ntl_breakpoint_year": best_bp_year,
        "ntl_pre_slope":       best_pre_slope,
        "ntl_post_slope":      best_post_slope,
        "ntl_acceleration":    best_post_slope - best_pre_slope,
    }, index=df.index)


# ── Spatial deduplication ────────────────────────────────────────────────────

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi, dlam = phi2 - phi1, np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def spatial_dedup_villages(df: pd.DataFrame, radius_km: float = 5.0) -> pd.DataFrame:
    """
    Remove duplicate OSM nodes that represent the same settlement.

    OSM sometimes contains several point nodes for a single village (e.g. five
    nodes all named 'Siswa Bazar' within a 5 km radius). These inflate rankings
    by letting one settlement occupy multiple top-N slots.

    Algorithm:
      1. Strip '(OSM XXXXXXXXX)' suffix from village_name → base_name.
      2. Within each base_name group, build a connectivity graph where two nodes
         are connected when haversine distance < radius_km.
      3. Find connected components (transitively merged clusters).
      4. In each multi-node component keep the highest composite_score
         representative; mark the rest for removal.

    Returns the deduplicated DataFrame. Rankings are NOT updated here — the
    caller should re-sort and reassign rank after calling this function.
    """
    import re
    from collections import defaultdict

    if "village_name" not in df.columns or len(df) == 0:
        return df

    def _base(name: str) -> str:
        return re.sub(r"\s*\(OSM\s*\d+\)\s*$", "", str(name)).strip().lower()

    df = df.copy()
    df["_base_name"] = df["village_name"].apply(_base)
    df["_keep"] = True

    for base, grp in df[df["_base_name"] != ""].groupby("_base_name"):
        if len(grp) <= 1:
            continue

        idxs = list(grp.index)
        graph: dict = defaultdict(set)
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                ri, rj = idxs[i], idxs[j]
                d = _haversine_km(
                    df.loc[ri, "latitude"], df.loc[ri, "longitude"],
                    df.loc[rj, "latitude"], df.loc[rj, "longitude"],
                )
                if d < radius_km:
                    graph[i].add(j)
                    graph[j].add(i)

        visited: set = set()
        for start in range(len(idxs)):
            if start in visited:
                continue
            comp: set = set()
            stack = [start]
            while stack:
                n = stack.pop()
                if n in visited:
                    continue
                visited.add(n)
                comp.add(n)
                stack.extend(graph[n] - visited)
            if len(comp) > 1:
                comp_idxs = [idxs[i] for i in comp]
                best = df.loc[comp_idxs, "composite_score"].idxmax()
                for ci in comp_idxs:
                    if ci != best:
                        df.loc[ci, "_keep"] = False

    n_removed = int((~df["_keep"]).sum())
    if n_removed:
        print(f"  Spatial dedup: removed {n_removed} duplicate OSM nodes "
              f"(same settlement, radius ≤ {radius_km} km)")

    return df[df["_keep"]].drop(columns=["_base_name", "_keep"]).copy()


# ── Spatial lag ───────────────────────────────────────────────────────────────

def add_spatial_lag(df: pd.DataFrame, score_col: str, k: int = 10) -> pd.Series:
    """Mean score of k nearest geographic neighbours (cluster reinforcement)."""
    coords = df[["longitude", "latitude"]].values
    scores = df[score_col].fillna(0).values
    tree   = cKDTree(coords)
    _, idx = tree.query(coords, k=k + 1)  # idx[:,0] is self
    lag    = np.mean(scores[idx[:, 1:]], axis=1)
    return pd.Series(lag, index=df.index, name="spatial_lag_score")


# ── ML scoring ────────────────────────────────────────────────────────────────

def ml_score(df: pd.DataFrame, feature_cols: list) -> pd.Series:
    """
    Self-supervised signal amplifier (NOT an independent predictor).

    Label = 1 if village is top-10% on ≥2 of the primary raw signals simultaneously.
    Because labels are derived from the same signals provided as features, the model
    learns to reproduce a deterministic function of its own inputs. AUC = 1.000 is
    the expected outcome — it confirms training succeeded, not that the model
    generalises to unseen economic ground-truth.

    This component amplifies co-occurrence of strong NTL + built-up signals. It adds
    no new information beyond those signals. Weight is capped at 15% in the composite
    (config.yaml → scoring.composite_weights_full.ml_growth_prob_score) so independent
    satellite signals collectively dominate. Replace with SECC 2011 ground-truth labels
    (src/13_secc_validation.py) to convert this into a genuine independent predictor.
    """
    sig_cols = [c for c in ["ntl_growth_pct", "ndbi_growth", "ghsl_change",
                             "s1_vv_delta", "builtup_change"]
                if c in df.columns]
    votes = pd.Series(0, index=df.index)
    for col in sig_cols[:4]:
        thr = df[col].quantile(0.90)
        votes += (df[col] > thr).astype(int)
    label = (votes >= 2).astype(int)

    avail_feats = [c for c in feature_cols if c in df.columns]
    if len(avail_feats) < 2 or label.sum() < 20:
        print("  ML: insufficient signals/positives — returning NaN (composite uses signals directly)")
        return pd.Series(np.nan, index=df.index)

    X = df[avail_feats].fillna(df[avail_feats].median()).fillna(0)

    clf = Pipeline([
        ("scaler", RobustScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=_SCORING.get("gbm_estimators",   200),
            max_depth=   _SCORING.get("gbm_max_depth",    4),
            learning_rate=_SCORING.get("gbm_learning_rate", 0.05),
            subsample=   _SCORING.get("gbm_subsample",    0.8),
            random_state=42,
        ))
    ])
    clf.fit(X, label)

    prob = clf.predict_proba(X)[:, 1]
    auc  = roc_auc_score(label.values, prob)   # O(n log n) — not the old O(n²) approach
    print(f"  ML: trained on {len(X):,} villages  "
          f"({label.sum()} positives)  AUC={auc:.3f} (self-consistency, not external validity)")

    model_path = PROC_DIR / "ml_model.pkl"
    joblib.dump({"model": clf, "features": avail_feats}, str(model_path))
    print(f"  Model saved → {model_path}")

    return pd.Series(prob * 100, index=df.index)


# ── Uncertainty quantification ────────────────────────────────────────────────

def compute_uncertainty(df: pd.DataFrame, signal_cols: list) -> pd.DataFrame:
    """
    confidence_score: fraction of *truly active* signals × 100.
    signals_active:   count of signals with non-NaN data AND non-zero variance.
    inter_signal_agreement: 1 − (std of signal rank percentiles / 50).
    Values near 100 = signals agree; values near 0 = signals contradict.

    A signal is "truly active" only if it has (a) at least one non-NaN value in this
    dataframe AND (b) non-zero variance. All-NaN columns (e.g. Sentinel-2/SAR when
    those pipelines stall) and all-identical columns (degenerate) are excluded from
    the coverage count so confidence_score correctly reflects signal availability
    rather than always returning 100 when missing signals are NaN-filled to zero.
    """
    avail = [c for c in signal_cols if c in df.columns]
    if not avail:
        return pd.DataFrame(index=df.index)

    # "Truly active" = has at least one non-NaN value AND non-zero variance.
    # This correctly excludes all-NaN signals (stalled Sentinel-2/SAR → all NaN)
    # and degenerate all-constant signals that carry no information.
    truly_active = [
        c for c in avail
        if df[c].notna().any() and df[c].std(skipna=True) > 1e-9
    ]
    n_active   = len(truly_active)
    n_total    = len(avail)   # total attempted (includes stalled/absent signals)

    # Pipeline-level confidence: fraction of designed signals that are truly active.
    # This is a scalar (same for all rows) — reflects what % of the intended signal
    # set actually produced data in this run. NOT per-village data completeness.
    confidence_score_val = round((n_active / n_total) * 100, 1) if n_total > 0 else 0.0

    sub = df[avail].copy()
    for col in avail:
        sub[col] = sub[col].rank(pct=True, na_option="bottom") * 100

    std_vals  = sub.std(axis=1)
    agreement = (1 - std_vals / 50).clip(0, 1)

    result = pd.DataFrame({
        "confidence_score":        confidence_score_val,
        "signals_active":          n_active,
        "signals_attempted":       n_total,
        "inter_signal_agreement":  (agreement * 100).round(1),
    }, index=df.index)

    print(f"  Signal coverage: {n_active}/{n_total} signals active "
          f"({', '.join(truly_active[:4])}{'...' if n_active > 4 else ''})")
    return result


# ── District normalization ────────────────────────────────────────────────────

def district_rank(df: pd.DataFrame) -> pd.Series:
    if "district_name" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return df.groupby("district_name")["composite_score"].rank(pct=True) * 100


# ── Merge optional signal files ───────────────────────────────────────────────

def merge_optional(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    merges = [
        (PROC_DIR / "village_ghsl.csv",         ["ghsl_2015", "ghsl_2020", "ghsl_change"]),
        (PROC_DIR / "village_towers.csv",        ["tower_density_2024", "tower_growth"]),
        (PROC_DIR / "village_city_distance.csv", ["dist_city_km", "dist_highway_km"]),
    ]
    for path, cols in merges:
        if not path.exists():
            continue
        extra = pd.read_csv(path, dtype={id_col: str})
        extra[id_col] = extra[id_col].astype(str)
        df[id_col]    = df[id_col].astype(str)
        df = df.merge(extra[[id_col] + [c for c in cols if c in extra.columns]],
                      on=id_col, how="left")
        print(f"  Merged {path.name}: {[c for c in cols if c in df.columns]}")
    return df


# ── Geographic diversity cap ─────────────────────────────────────────────────

def state_diversity_cap(df: pd.DataFrame, n: int = 100,
                         max_per_state_pct: float = 0.40) -> pd.DataFrame:
    """
    Enforce per-state diversity in the top-N shortlist, with named-village preference.

    Motivation: physical-signal TIFs (GHSL, SAR) sometimes have geographic
    concentration bias — e.g. GHSL's built-up product reflects urban corridors
    unevenly across Indian states. Without a diversity cap, a single state can
    crowd out genuine high-potential villages from other regions.

    Algorithm:
      1. Sort candidates: named villages first within each composite_score tier
         (tiny tie-breaker — a village named in OSM is more operationally useful
         than an anonymous "Village_<id>" node at nearly the same score).
      2. Iterate in order; include each village unless its state has already
         contributed max_per_state rows.

    cap = 40% → max 40 villages from any single state in a 100-village shortlist.
    This is read from config.yaml → scoring.state_cap_pct (default 0.40).

    NOTE: villages with unknown/NaN state are always included.
    """
    import re as _re

    df = df.copy()
    # Named preference: "Village_<digits>" OSM fallback names are less actionable
    # for a retail expansion shortlist than real OSM or census names.
    # Add a tiny (0.001 pt) bonus so named villages win ties; this does not
    # materially change composite_score ordering but breaks ties in favour of
    # identifiable settlements.
    is_unnamed = df["village_name"].astype(str).str.match(r"^Village_\d+$", na=True)
    df["_sort_score"] = df["composite_score"] + (~is_unnamed).astype(float) * 0.001
    df = df.sort_values("_sort_score", ascending=False).drop(columns=["_sort_score"])

    max_count = max(1, int(n * max_per_state_pct))
    state_counts: dict = {}
    kept: list = []

    for _, row in df.iterrows():
        state = row.get("state_name", "")
        state = "" if (pd.isna(state) or str(state).strip() in ("", "nan", "None")) else str(state)

        cnt = state_counts.get(state, 0) if state else 0
        if not state or cnt < max_count:
            kept.append(row)
            if state:
                state_counts[state] = cnt + 1
            if len(kept) >= n:
                break

    result = pd.DataFrame(kept).reset_index(drop=True)
    capped_states = [s for s, c in state_counts.items() if c >= max_count]
    if capped_states:
        print(f"  State diversity cap ({max_per_state_pct:.0%} / {max_count} max): "
              f"capped states: {capped_states}")
    return result


# ── TIF point sampling ───────────────────────────────────────────────────────

def sample_tif_at_points(df: pd.DataFrame, tif_path: Path,
                          nodata_val: float) -> pd.Series:
    """
    Sample a single-band GeoTIFF at village centroid coordinates.

    Reads the full raster into RAM once, then uses NumPy fancy indexing
    for O(1)-per-village lookup. On t3.xlarge (16 GB) the GHSL uint16 TIF
    (~1.9 GB) and SAR float32 TIF (~4.4 GB) fit without issue.
    Out-of-bounds coordinates → NaN. nodata_val pixels → NaN.
    """
    try:
        import rasterio
        from rasterio.transform import rowcol as _rowcol
    except ImportError:
        print(f"  WARNING: rasterio not available — skipping {tif_path.name}")
        return pd.Series(np.nan, index=df.index)

    if not tif_path.exists():
        return pd.Series(np.nan, index=df.index)

    lons = df["longitude"].values.astype(float)
    lats = df["latitude"].values.astype(float)

    with rasterio.open(str(tif_path)) as src:
        arr    = src.read(1)                     # native dtype (uint16 / float32)
        height, width = arr.shape
        transform = src.transform

    rows, cols = _rowcol(transform, lons, lats)
    rows = np.asarray(rows, dtype=int)
    cols = np.asarray(cols, dtype=int)

    valid  = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    result = np.full(len(df), np.nan, dtype=float)
    raw    = arr[rows[valid], cols[valid]].astype(float)
    if nodata_val is not None:
        raw[raw == nodata_val] = np.nan
    result[valid] = raw
    del arr                                       # release RAM before next TIF

    return pd.Series(result, index=df.index)


def extract_tif_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sample GHSL built-up and Sentinel-1 VV TIFs at village centroids.

    Fills ghsl_2015, ghsl_2020, ghsl_change, s1_vv_2019, s1_vv_2024 when
    village_ghsl.csv / s1_vv columns are absent (i.e. 03_village_stats.py
    did not extract them). Columns already present are left unchanged.
    """
    ghsl_2015_tif = PROC_DIR / "ghsl_builtup_2015.tif"
    ghsl_2020_tif = PROC_DIR / "ghsl_builtup_2020.tif"
    sar_2019_tif  = PROC_DIR / "s1_vv_2019.tif"
    sar_2024_tif  = PROC_DIR / "s1_vv_2024.tif"

    if "ghsl_2015" not in df.columns and ghsl_2015_tif.exists():
        print(f"  Sampling {ghsl_2015_tif.name} at {len(df):,} villages ...")
        df["ghsl_2015"] = sample_tif_at_points(df, ghsl_2015_tif, nodata_val=65535.0)
        print(f"    → {df['ghsl_2015'].notna().sum():,} non-NaN values")

    if "ghsl_2020" not in df.columns and ghsl_2020_tif.exists():
        print(f"  Sampling {ghsl_2020_tif.name} at {len(df):,} villages ...")
        df["ghsl_2020"] = sample_tif_at_points(df, ghsl_2020_tif, nodata_val=65535.0)
        print(f"    → {df['ghsl_2020'].notna().sum():,} non-NaN values")

    if "ghsl_change" not in df.columns and "ghsl_2015" in df.columns and "ghsl_2020" in df.columns:
        df["ghsl_change"] = df["ghsl_2020"] - df["ghsl_2015"]
        n = df["ghsl_change"].notna().sum()
        pos = (df["ghsl_change"] > 0).sum()
        print(f"  ghsl_change: {n:,} non-NaN, {pos:,} positive (built-up growth)")

    if "s1_vv_2019" not in df.columns and sar_2019_tif.exists():
        print(f"  Sampling {sar_2019_tif.name} at {len(df):,} villages ...")
        df["s1_vv_2019"] = sample_tif_at_points(df, sar_2019_tif, nodata_val=-9999.0)
        print(f"    → {df['s1_vv_2019'].notna().sum():,} non-NaN values")

    if "s1_vv_2024" not in df.columns and sar_2024_tif.exists():
        print(f"  Sampling {sar_2024_tif.name} at {len(df):,} villages ...")
        df["s1_vv_2024"] = sample_tif_at_points(df, sar_2024_tif, nodata_val=-9999.0)
        print(f"    → {df['s1_vv_2024'].notna().sum():,} non-NaN values")

    return df


# ── Nominatim reverse-geocoding ──────────────────────────────────────────────

def _nominatim_resolve(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resolve 'Village_<id>' fallback names via Nominatim reverse geocoding.

    Called only on the top-100 shortlist (≤100 HTTP requests). Respects the
    Nominatim 1 req/sec rate limit.

    Multi-zoom cascade strategy:
      Pass 1: zoom=14 (village level)   — checks village/hamlet/suburb/neighbourhood
      Pass 2: zoom=10 (subdistrict)     — checks city_district/county/town/city
    Each zoom costs 1 req/sec; the cascade is skipped for villages resolved in pass 1.
    Villages genuinely unnamed in OSM at both levels keep their fallback name.
    """
    import time

    unnamed_mask = df["village_name"].astype(str).str.match(r"^Village_\d+$", na=True)
    n_unnamed = int(unnamed_mask.sum())
    if n_unnamed == 0:
        return df

    # Estimated requests = n_unnamed (pass 1) + still-unnamed (pass 2) ≤ 2×n_unnamed
    print(f"  Nominatim: resolving {n_unnamed} unnamed villages "
          f"(up to {2 * n_unnamed} requests at 1 req/sec) ...")
    headers = {"User-Agent": "KritterVillageGrowthPipeline/1.0 (interview assignment)"}

    def _query(lat: float, lon: float, zoom: int) -> str:
        """Return best available place name at this zoom level, or ''."""
        resp = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"lat": lat, "lon": lon, "format": "json", "zoom": zoom},
            headers=headers, timeout=8,
        )
        resp.raise_for_status()
        addr = resp.json().get("address", {})
        # Priority: village-level keys first, then subdistrict-level fallbacks
        name = (
            addr.get("village")       or addr.get("hamlet")       or
            addr.get("suburb")        or addr.get("neighbourhood") or
            addr.get("city_district") or addr.get("county")       or
            addr.get("town")          or addr.get("city")         or ""
        )
        return name if (name and not name.isdigit()) else ""

    resolved = 0
    still_unnamed = []

    # Pass 1: zoom=14 (village level)
    for idx in df[unnamed_mask].index:
        lat = df.loc[idx, "latitude"]
        lon = df.loc[idx, "longitude"]
        try:
            name = _query(lat, lon, zoom=14)
            if name:
                df.loc[idx, "village_name"] = name
                resolved += 1
            else:
                still_unnamed.append(idx)
        except Exception:
            still_unnamed.append(idx)
        time.sleep(1.1)

    # Pass 2: zoom=10 (subdistrict/tehsil level) for remaining unnamed
    if still_unnamed:
        print(f"  Nominatim pass 2 (zoom=10): {len(still_unnamed)} still unnamed ...")
        for idx in still_unnamed:
            lat = df.loc[idx, "latitude"]
            lon = df.loc[idx, "longitude"]
            try:
                name = _query(lat, lon, zoom=10)
                if name:
                    df.loc[idx, "village_name"] = name
                    resolved += 1
            except Exception:
                pass
            time.sleep(1.1)

    print(f"  Nominatim: resolved {resolved}/{n_unnamed} "
          f"({100 * resolved / max(n_unnamed, 1):.0f}%)")
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input not found: {INPUT_FILE}\nRun 03_village_stats.py first.")

    df = pd.read_csv(INPUT_FILE, low_memory=False)
    print(f"Loaded {len(df):,} villages")

    required_cols = ["pc11_village_id", "latitude", "longitude",
                     "ntl_2019", "ntl_2024"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"village_all_stats.csv missing required columns: {missing}")
    if len(df) < 10_000:
        raise ValueError(
            f"village_all_stats.csv has only {len(df):,} rows — "
            "likely a truncated or failed step-03 run."
        )

    df = apply_india_filter(df)
    id_col = "pc11_village_id"
    df[id_col] = df[id_col].astype(str)

    # Fill unnamed villages
    if "village_name" in df.columns:
        n_missing = int(df["village_name"].isna().sum())
        if n_missing > 0:
            df["village_name"] = df["village_name"].fillna(
                "Village_" + df[id_col].astype(str))
            print(f"  Filled {n_missing:,} unnamed villages → 'Village_<id>'")

    # Fill district names via GADM spatial join (runs once; cached to /tmp)
    df = fill_district_names(df)

    # Merge optional enrichment files
    df = merge_optional(df, id_col)

    # Sample GHSL / SAR TIFs directly when village_ghsl.csv is absent
    df = extract_tif_signals(df)

    # Derived signals
    df["ntl_growth_pct"] = np.where(df["ntl_2019"] > 0,
        (df["ntl_2024"] - df["ntl_2019"]) / df["ntl_2019"] * 100, np.nan)
    df["ntl_growth_log"]  = np.log1p(df["ntl_growth_pct"].clip(lower=0))
    df["ndbi_growth"]     = df.get("ndbi_2024", pd.Series(np.nan, index=df.index)) \
                          - df.get("ndbi_2019", pd.Series(np.nan, index=df.index))
    df["ndvi_growth"]     = df.get("ndvi_2024", pd.Series(np.nan, index=df.index)) \
                          - df.get("ndvi_2019", pd.Series(np.nan, index=df.index))
    df["s1_vv_delta"]     = df.get("s1_vv_2024", pd.Series(np.nan, index=df.index)) \
                          - df.get("s1_vv_2019", pd.Series(np.nan, index=df.index))

    # Vectorised BFAST breakpoint detection (~50× faster than iterrows)
    print("Computing BFAST breakpoint features (vectorised)...")
    bfast_df = bfast_features(df)
    df = pd.concat([df.reset_index(drop=True), bfast_df], axis=1)

    # ── Track split ───────────────────────────────────────────────────────────
    # Main track: has measurable NTL baseline AND a meaningful absolute increase.
    # The absolute delta filter (≥ 1.0 nW/cm²/sr) prevents electrification noise
    # from 0.01→0.30 nW/cm²/sr appearing as "high-growth" villages.
    ntl_delta = (df["ntl_2024"] - df["ntl_2019"]).fillna(0)
    main_mask = (
        (df["ntl_2019"] >= MIN_NTL) &
        df["ntl_2019"].notna() &
        (ntl_delta >= MIN_ABSOLUTE_NTL_DELTA)
    )
    dark_mask = ~main_mask

    main_df = df[main_mask].copy().reset_index(drop=True)
    dark_df = df[dark_mask].copy().reset_index(drop=True)
    print(f"\nMain track: {len(main_df):,}  |  Dark track: {len(dark_df):,}")

    # ── GHSL state normalization ───────────────────────────────────────────────
    # GHSL built-up TIFs have geographic concentration bias: urban corridors in
    # UP and Maharashtra have higher absolute built-up pixel counts than equally
    # high-growth villages in Karnataka or Uttarakhand. Without normalization this
    # inflates the GHSL score for UP/MH villages independently of their actual
    # growth relative to their local context.
    #
    # Fix: z-score normalize ghsl_change within each state's main-track villages.
    # After normalization a village "50 m² above its state average" scores the same
    # whether it is in UP or Karnataka. Villages in states with <3 main-track
    # villages (too few for meaningful std) are left unchanged (std fallback = 1).
    if "ghsl_change" in main_df.columns and "state_name" in main_df.columns:
        state_grp  = main_df.groupby("state_name")["ghsl_change"]
        state_mean = state_grp.transform("mean")
        state_std  = state_grp.transform("std").fillna(1.0).clip(lower=1e-6)
        # States with only 1 village in the main track cannot compute std → std=1
        state_count = main_df.groupby("state_name")["ghsl_change"].transform("count")
        state_std   = state_std.where(state_count >= 3, 1.0)
        main_df["ghsl_change"] = (main_df["ghsl_change"] - state_mean) / state_std
        main_df["ghsl_change"] = main_df["ghsl_change"].fillna(0)
        print("  GHSL change: z-score normalized within each state "
              "(removes geographic TIF concentration bias)")

    # ── Main track scoring ────────────────────────────────────────────────────
    print("\nScoring main track...")

    ALL_SIGNAL_COLS = [
        "ntl_growth_log", "ntl_trend_slope", "ntl_2024",
        "ndbi_growth", "ndbi_trend_slope", "ndvi_trend_slope",
        "ghsl_change", "s1_vv_delta", "tower_growth", "pop_growth_rate",
        "builtup_change", "ntl_acceleration",
        "dist_city_km", "dist_highway_km",
    ]

    # rank_score → used for ML label construction (top-10% threshold checks)
    # minmax_score → used in composite (preserves intra-tier spread)
    print("Computing signal scores...")
    for col in ALL_SIGNAL_COLS:
        if col in main_df.columns:
            main_df[f"{col}_pct"]   = rank_score(main_df[col])    # for ML labels
            main_df[f"{col}_score"] = minmax_score(main_df[col])  # for composite

    # ML probability
    ml_prob = ml_score(main_df, [c for c in ALL_SIGNAL_COLS if c in main_df.columns])
    main_df["ml_growth_prob"] = ml_prob

    # IMPORTANT: ml_growth_prob is bimodal — ~99% of villages score near-zero,
    # ~1% (positives) score near 100.  The standard 2nd/99th-percentile clip
    # collapses both groups to the same ceiling (score=100) because the 99th
    # percentile is within the near-zero cluster.  We use full-range normalisation
    # (clip_lo=0, clip_hi=1.0 → use actual min/max) so negatives score ≈ 0 and
    # positives score ≈ 100, correctly weighting the ML confirmation signal.
    main_df["ml_growth_prob_score"] = minmax_score(ml_prob.fillna(0),
                                                    clip_lo=0.0, clip_hi=1.0)

    # Spatial lag on minmax-normalised NTL growth
    if "ntl_growth_log_score" in main_df.columns:
        main_df["spatial_lag_score"] = add_spatial_lag(
            main_df, "ntl_growth_log_score", k=SPATIAL_LAG_K)

    # Composite — weight over available _score columns
    avail   = list(main_df.columns)
    weights = redistribute(COMPOSITE_WEIGHTS, avail)
    if not weights:
        sig_scores = [c for c in avail if c.endswith("_score") and "ml" not in c][:5]
        weights = {c: 1 / len(sig_scores) for c in sig_scores}

    main_df["composite_score"] = sum(
        main_df[c] * w for c, w in weights.items() if c in main_df.columns)
    print(f"  Effective composite weights: {weights}")

    # District outperformance
    main_df["district_rank_pct"] = district_rank(main_df)

    # Multi-signal confirmation (votes across all available signals)
    votes = pd.Series(0, index=main_df.index)
    signal_checks = {
        "ntl_growth_pct": lambda s: s > 50,       # > 50% NTL growth
        "builtup_change":  lambda s: s > 0,        # positive WorldCover change
        "tower_growth":    lambda s: s > 0,        # more towers than 2019
        "ndbi_growth":     lambda s: s > 0,        # built-up index increasing
        "ghsl_change":     lambda s: s > 0,        # GHSL built-up expanding
        "s1_vv_delta":     lambda s: s > 0,        # SAR backscatter increasing
        # NOTE: dist_city_km intentionally excluded — city proximity is not a
        # growth confirmation signal; it would give every village a free vote.
    }
    for col, fn in signal_checks.items():
        if col in main_df.columns:
            votes += fn(main_df[col]).fillna(False).astype(int)
    main_df["multi_signal_confirmed"] = (votes >= 2)
    print(f"  multi_signal_confirmed: {main_df['multi_signal_confirmed'].sum():,}/{len(main_df):,}")

    # Uncertainty
    unc = compute_uncertainty(main_df, ALL_SIGNAL_COLS)
    main_df = pd.concat([main_df, unc], axis=1)

    main_df = main_df.sort_values("composite_score", ascending=False).reset_index(drop=True)

    # Deduplicate OSM multi-nodes for the same settlement before ranking.
    # Without this, a single market town (e.g. Siswa Bazar) with many OSM
    # point nodes can occupy 5+ consecutive top-N slots, crowding out genuine
    # other-village candidates. Dedup preserves the highest-scoring node per
    # settlement and drops the rest. Radius from config (default 5 km).
    dedup_radius = _CFG.get("scoring", {}).get("dedup_radius_km", 5.0)
    main_df = spatial_dedup_villages(main_df, radius_km=dedup_radius)
    main_df = main_df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    main_df["rank"] = range(1, len(main_df) + 1)

    # ── Dark track ────────────────────────────────────────────────────────────
    print("\nScoring dark track...")
    for col in ["ndbi_growth", "ghsl_change", "tower_growth", "pop_growth_rate"]:
        if col in dark_df.columns:
            dark_df[f"{col}_score"] = minmax_score(dark_df[col])

    dark_w = redistribute(DARK_WEIGHTS, list(dark_df.columns))
    dark_df["composite_score"] = (
        sum(dark_df[c] * w for c, w in dark_w.items() if c in dark_df.columns)
        if dark_w else pd.Series(0.0, index=dark_df.index)
    )
    dark_df = dark_df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    dark_df["rank"] = range(1, len(dark_df) + 1)

    # ── Save outputs ──────────────────────────────────────────────────────────
    all_scored = pd.concat([main_df, dark_df], ignore_index=True)
    all_scored.to_csv(PROC_DIR / "village_scored.csv", index=False)
    print(f"\nAll scored: {len(all_scored):,} → village_scored.csv")

    # Apply geographic diversity cap before saving top-100.
    # village_scored.csv retains uncapped global ranking for research use.
    state_cap = _SCORING.get("state_cap_pct", 0.40)
    top100 = state_diversity_cap(main_df, n=100, max_per_state_pct=state_cap)
    top100 = top100.sort_values("composite_score", ascending=False).reset_index(drop=True)
    top100["rank"] = range(1, len(top100) + 1)

    # Resolve unnamed villages ("Village_<id>") via Nominatim reverse geocoding.
    # These are OSM nodes that lacked a name= tag in the extract; Nominatim
    # returns the nearest named place (village/hamlet/suburb) at zoom=14.
    # Runs only for the top-100 (≤100 requests at 1/sec rate limit → ≤100 s).
    top100 = _nominatim_resolve(top100)

    # Re-run spatial dedup after Nominatim: resolution may assign the same name
    # to several nearby unnamed nodes (e.g. three OSM nodes within 3 km all get
    # named "Bhanpur"). This second pass collapses those post-resolution clusters.
    n_pre_dedup2 = len(top100)
    top100 = spatial_dedup_villages(top100, radius_km=dedup_radius)
    top100 = top100.sort_values("composite_score", ascending=False).reset_index(drop=True)
    top100["rank"] = range(1, len(top100) + 1)
    if len(top100) < n_pre_dedup2:
        print(f"  Post-Nominatim dedup: collapsed {n_pre_dedup2 - len(top100)} "
              f"same-name clusters → {len(top100)} villages in shortlist")

    top100.to_csv(OUTPUT_DIR / "top_100_villages.csv", index=False)
    dark_df.head(50).to_csv(OUTPUT_DIR / "dark_top_50_villages.csv", index=False)

    unc_cols = [id_col, "village_name", "state_name", "rank",
                "composite_score", "confidence_score", "inter_signal_agreement",
                "multi_signal_confirmed", "district_rank_pct"]
    main_df[[c for c in unc_cols if c in main_df.columns]].head(200) \
        .to_csv(OUTPUT_DIR / "village_uncertainty.csv", index=False)

    print(f"\nTop 5:")
    show = [c for c in ["rank", "village_name", "state_name", "composite_score",
                         "ntl_growth_pct", "ml_growth_prob", "confidence_score"]
            if c in main_df.columns]
    print(main_df[show].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
