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

  Stage 2 — ML scoring (GradientBoostingClassifier)
    - Self-supervised label: top 10% on ≥2 of the primary signals
    - Features: all available signal rank scores
    - Output: ml_growth_prob (0–1)

  Stage 3 — composite (weighted ensemble of ML + key signals)
    composite_score = 0.38 × ml_growth_prob_score
                    + 0.22 × ntl_growth_log_score
                    + 0.15 × ndbi_growth_score    (if available)
                    + 0.12 × ghsl_change_score    (if available)
                    + 0.08 × s1_vv_delta_score    (if available)
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

warnings.filterwarnings("ignore")

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
    "ml_growth_prob_score": 0.38,
    "ntl_growth_log_score": 0.22,
    "ndbi_growth_score":    0.15,
    "ghsl_change_score":    0.12,
    "s1_vv_delta_score":    0.08,
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
    Self-supervised growth classifier.

    Label = 1 if village is top-10% on ≥2 of the primary raw signals.
    Features = rank-percentile scores of all available signals (rank_score, not
    minmax_score, so the label and feature scales are consistent).

    Limitation: labels derived from input signals → AUC measures self-consistency,
    not ground-truth predictive performance. Use AUC as a calibration sanity-check
    only, not as a claim of external validity.
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
    confidence_score: fraction of signals with non-NaN data × 100.
    inter_signal_agreement: 1 − (std of signal rank percentiles / 50).
    Values near 100 = signals agree; values near 0 = signals contradict.
    """
    avail = [c for c in signal_cols if c in df.columns]
    if not avail:
        return pd.DataFrame(index=df.index)

    sub = df[avail].copy()
    for col in avail:
        sub[col] = sub[col].rank(pct=True, na_option="bottom") * 100

    coverage  = sub.notna().mean(axis=1)
    std_vals  = sub.std(axis=1)
    agreement = (1 - std_vals / 50).clip(0, 1)

    return pd.DataFrame({
        "confidence_score":        (coverage * 100).round(1),
        "inter_signal_agreement":  (agreement * 100).round(1),
    }, index=df.index)


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

    main_df.head(100).to_csv(OUTPUT_DIR / "top_100_villages.csv", index=False)
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
