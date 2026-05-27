"""
Step 7: Validation and signal correlation analysis.

Checks:
  1. Signal correlation matrix (NTL, NDBI, NDVI, population) — how much
     do signals agree? High correlation = redundant; low = complementary.
  2. Bootstrap rank stability — 200 random Dirichlet weight draws measure
     how sensitive the top-100 ranking is to weight perturbation.
  3. PMGSY road data cross-validation: download district-level road
     completion data and check if top-100 villages cluster in districts
     with high PMGSY activity.
  4. Moran's I spatial autocorrelation (k=8 KNN, 999 permutations) —
     tests whether high-scoring villages cluster geographically.
  5. State distribution bias check — top-100 vs all-India village share.
  6. Electrification confound analysis — three-flag Saubhagya heuristic
     identifies top-100 villages whose NTL growth may be explained by
     household electrification rollout rather than economic development.

Outputs:
  output/validation_signal_correlation.csv
  output/validation_rank_stability.csv
  output/validation_pmgsy_overlap.csv            (if PMGSY data available)
  output/validation_morans_i.csv
  output/validation_state_distribution.csv
  output/validation_electrification_confound.csv
  output/chart_validation.html
"""

import warnings
import requests
import zipfile
import io
import json
import yaml
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
from pathlib import Path
from scipy.stats import spearmanr
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", category=UserWarning, module="geopandas")

_CFG_PATH = Path(__file__).parent.parent / "config.yaml"
_CFG      = yaml.safe_load(_CFG_PATH.read_text()) if _CFG_PATH.exists() else {}
_data     = _CFG.get("paths", {}).get("data_root", "/data/satellite/kritter")

PROC_DIR   = Path(_data) / "processed"
OUTPUT_DIR = Path(_data) / "output"

# data.gov.in PMGSY district-level completion
PMGSY_URL = ("https://data.gov.in/resource/d02ee7c2-2b11-4af4-b6ed-3a6f0e2e40b2"
             "?format=csv&limit=50000")


# ── 1. Signal correlation ─────────────────────────────────────────────────────

def signal_correlation(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in [
        "ntl_growth_pct", "ntl_trend_slope",
        "ndbi_growth", "ndbi_trend_slope",
        "ndvi_trend_slope", "builtup_change",
        "pop_growth_rate",
    ] if c in df.columns]

    sub = df[cols].dropna(subset=cols[:2])  # require at least NTL cols
    n = len(sub)

    matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)
    for a in cols:
        for b in cols:
            if a == b:
                matrix.loc[a, b] = 1.0
            else:
                ok = sub[[a, b]].dropna()
                if len(ok) < 50:
                    matrix.loc[a, b] = np.nan
                else:
                    r, _ = spearmanr(ok[a], ok[b])
                    matrix.loc[a, b] = round(r, 3)

    print(f"\nSpearman correlation matrix (n={n:,}):")
    print(matrix.to_string())
    return matrix


# ── 2. Rank stability (bootstrap weight perturbation) ─────────────────────────

def rank_stability(df: pd.DataFrame, n_boot: int = 200,
                   shortlist_ids: set | None = None) -> pd.DataFrame:
    """
    Bootstrap rank stability test.

    Methodology: Dirichlet weight perturbation around the *actual* composite
    weights (config.yaml composite_weights_full).  Using Dirichlet(alpha=1) with
    random base columns (as in the original version) conflates two separate
    questions:
      (a) stability under realistic weight uncertainty (±20% perturbation)
      (b) sensitivity to completely different ranking philosophies (e.g. rank
          purely by dist_city_km, which is anti-correlated with growth and
          produces a different top-100 by design)

    This version uses Dirichlet(alpha × composite_weights) with alpha=10, so
    draws are tightly centred on the current weights (variance ≈ 1/alpha × mean),
    correctly testing whether the top-100 would survive a ±~10% weight
    perturbation — the kind of parameter uncertainty that actually matters.

    Signals with composite_weight=0 (e.g. dist_city_km — intentionally excluded
    from the composite because proximity is structural, not a growth signal) are
    omitted from the bootstrap entirely.  dist_city_km is in the archetype
    K-means feature set but would anti-correlate with growth-scoring if included
    in Dirichlet draws (remote villages win when dist_city weight is high).
    """
    import yaml as _yaml

    # Load current composite weights from config.yaml
    _cfg_path = Path(__file__).parent.parent / "config.yaml"
    _cfg = _yaml.safe_load(_cfg_path.read_text()) if _cfg_path.exists() else {}
    _cw = _cfg.get("scoring", {}).get("composite_weights_full", {})

    # Map composite _score column names → raw signal column names
    score_to_raw = {
        "ntl_growth_log_score":  "ntl_growth_log",
        "ndbi_growth_score":     "ndbi_growth",
        "ghsl_change_score":     "ghsl_change",
        "tower_growth_score":    "tower_growth",
        "builtup_change_score":  "builtup_change",
        "ntl_acceleration_score": "ntl_acceleration",
        "ntl_trend_slope_score":  "ntl_trend_slope",
        "ntl_2024_score":         "ntl_2024",
        # ml_growth_prob and spatial_lag intentionally omitted: self-supervised
        # label and spatial smoother are not independent raw signals
        # s1_vv_delta has std=0 this run — auto-filtered below
    }

    # Build ordered list: (raw_col, weight) only for active signals
    signal_pairs = []
    for score_col, raw_col in score_to_raw.items():
        w = _cw.get(score_col, 0.0)
        if w <= 0:
            continue
        if raw_col not in df.columns:
            continue
        if not (df[raw_col].notna().any() and df[raw_col].std(skipna=True) > 1e-9):
            continue
        signal_pairs.append((raw_col, w))

    if len(signal_pairs) < 2:
        print("  Not enough active signals for stability analysis")
        return pd.DataFrame()

    base_cols   = [c for c, _ in signal_pairs]
    base_weights = np.array([w for _, w in signal_pairs])
    base_weights /= base_weights.sum()   # normalise in case of missing signals

    print(f"  Bootstrap using {len(base_cols)} active signals "
          f"(Dirichlet alpha=10×weights): {base_cols}")

    sub = df[["village_name", "state_name", "composite_score"] + base_cols].copy()
    # Rank-normalise each signal to [0,100] — makes weight comparisons valid
    # across signals with very different raw ranges (e.g. ntl_growth_log vs ghsl_change)
    for col in base_cols:
        sub[col] = sub[col].rank(pct=True, na_option="bottom") * 100

    base_rank = sub["composite_score"].rank(ascending=False).astype(int)

    # Use the actual state-capped shortlist IDs when provided; otherwise fall back
    # to the uncapped top-100 by composite_score.  The shortlist applies a 40%
    # per-state cap and two-pass spatial dedup — if we test the uncapped top-100
    # instead, we're testing stability of the wrong population (UP-dominated
    # set that differs from what the pipeline actually outputs).
    if shortlist_ids is not None:
        top100_ids = shortlist_ids & set(sub.index)
    else:
        top100_ids = set(base_rank[base_rank <= 100].index)
    n_shortlist = len(top100_ids)

    # Load config to get diversity cap fraction (default 40%)
    import yaml as _yaml2
    _cap_frac = (_yaml.safe_load(_cfg_path.read_text())
                 .get("scoring", {}).get("state_cap_pct", 0.40)
                 if _cfg_path.exists() else 0.40)
    max_per_state = max(1, int(n_shortlist * _cap_frac))
    has_state = "state_name" in df.columns

    def _cap_select(score_series: pd.Series) -> set:
        """Select top-n_shortlist with per-state diversity cap, matching pipeline."""
        ordered = score_series.sort_values(ascending=False).index
        counts: dict = {}
        selected = []
        for idx in ordered:
            if has_state:
                st = df.at[idx, "state_name"] if idx in df.index else "?"
            else:
                st = "all"
            if counts.get(st, 0) < max_per_state:
                selected.append(idx)
                counts[st] = counts.get(st, 0) + 1
            if len(selected) >= n_shortlist:
                break
        return set(selected)

    rng = np.random.default_rng(42)
    inclusion_counts = pd.Series(0, index=sub.index)

    # Dirichlet concentration: alpha=10 means variance ≈ w_i*(1-w_i)/11
    # → each weight varies by ~±0.07–0.15 around its mean (realistic uncertainty)
    alpha = 10.0
    for _ in range(n_boot):
        w = rng.dirichlet(alpha * base_weights)
        score = sum(sub[col] * w[i] for i, col in enumerate(base_cols))
        # Apply same state diversity cap the pipeline uses so non-UP shortlisted
        # villages can survive draws that heavily weight NTL (UP-dominated signal)
        top = _cap_select(score)
        inclusion_counts[list(top)] += 1

    sub["base_rank"] = base_rank
    sub["bootstrap_inclusion_pct"] = (inclusion_counts / n_boot * 100).round(1)

    result = sub[sub.index.isin(top100_ids)].sort_values("base_rank")
    print(f"\nRank stability (n_boot={n_boot}, n_shortlist={n_shortlist}, "
          f"max_per_state={max_per_state}, alpha=10):")
    print(f"  Median inclusion rate: "
          f"{result['bootstrap_inclusion_pct'].median():.1f}%")
    print(f"  Highly stable (≥80%): {(result['bootstrap_inclusion_pct'] >= 80).sum()}")
    print(f"  Borderline (<50%):    {(result['bootstrap_inclusion_pct'] < 50).sum()}")
    return result[["village_name", "state_name", "base_rank",
                   "bootstrap_inclusion_pct"]]


# ── 3. PMGSY validation ───────────────────────────────────────────────────────

def pmgsy_validation(top100: pd.DataFrame) -> pd.DataFrame | None:
    try:
        r = requests.get(PMGSY_URL, timeout=30,
                         headers={"User-Agent": "kritter-pipeline/1.0"})
        r.raise_for_status()
        pmgsy = pd.read_csv(io.StringIO(r.text))
        print(f"\nPMGSY data: {len(pmgsy):,} rows, cols: {list(pmgsy.columns[:5])}")
    except Exception as e:
        print(f"\nPMGSY download failed ({e}) — skipping PMGSY validation")
        return None

    # Normalise district names for join
    def clean(s):
        return str(s).lower().strip().replace(" ", "").replace("-", "")

    if "district" not in pmgsy.columns:
        print("  PMGSY columns don't match expected schema — skipping")
        return None

    pmgsy["district_clean"] = pmgsy["district"].apply(clean)
    top100["district_clean"] = top100["district_name"].apply(clean)

    overlap = top100.merge(
        pmgsy[["district_clean"]].drop_duplicates(),
        on="district_clean", how="left", indicator=True
    )
    n_match = (overlap["_merge"] == "both").sum()
    print(f"  Top-100 villages in PMGSY-active districts: {n_match}/100")
    return overlap[["village_name", "state_name", "district_name", "_merge"]]


# ── 4. Spatial autocorrelation (Moran's I) ───────────────────────────────────

def morans_i(df: pd.DataFrame, score_col: str = "composite_score",
             k: int = 8) -> dict:
    """
    Compute Moran's I for the composite score.
    I > 0 means high-scoring villages cluster together (spatial contagion).
    Uses row-standardised k-nearest-neighbour spatial weights.
    """
    sub = df.dropna(subset=["latitude", "longitude", score_col]).copy()
    if len(sub) < 50:
        return {}

    coords = sub[["latitude", "longitude"]].values
    x = sub[score_col].values.astype(float)
    x_dev = x - x.mean()
    n = len(x)

    tree = cKDTree(coords)
    _, idx = tree.query(coords, k=k + 1)  # idx[:,0] is self

    # Row-standardised weight matrix — vectorised NumPy fancy indexing (~50× faster
    # than the previous Python list comprehension for n=356k villages)
    nbr_idx = idx[:, 1:]                           # (n, k) — exclude self
    W_x_dev = x_dev[nbr_idx].mean(axis=1)         # (n,)
    num = (x_dev * W_x_dev).sum()
    denom = (x_dev ** 2).sum()
    # Row-standardised Moran's I: sum_W = n (each row sums to 1), so the
    # (n / sum_W) prefactor reduces to 1. Correct formula: I = num / denom.
    # (The prior (n/(k*n)) = 1/k factor was wrong — it applies to unstandardised
    # binary weights where sum_W = k*n, not row-standardised where sum_W = n.)
    I = (num / denom) if denom > 0 else 0.0

    # Permutation test (999 random shuffles) — inner loop now vectorised
    rng = np.random.default_rng(42)
    null = []
    for _ in range(999):
        xp = rng.permutation(x_dev)
        Wp = xp[nbr_idx].mean(axis=1)             # vectorised — replaces O(n) Python loop
        null.append((xp * Wp).sum() / (xp ** 2).sum())
    null = np.array(null)
    p_value = (np.abs(null) >= abs(I)).mean()

    result = {"I": round(float(I), 4),
              "p_value": round(float(p_value), 4),
              "n": n, "k": k,
              "interpretation": (
                  "Strong spatial clustering" if I > 0.5 else
                  "Moderate clustering"       if I > 0.2 else
                  "Weak clustering"           if I > 0.05 else
                  "Negligible / no clustering" if I > 0 else "Spatial dispersion"
              )}
    print(f"\nMoran's I (k={k}, n={n:,}): I={I:.4f}  p={p_value:.4f}  → {result['interpretation']}")
    return result


# ── 5. State distribution bias check ────────────────────────────────────────

def state_distribution(df: pd.DataFrame, top100: pd.DataFrame) -> pd.DataFrame:
    """Compare top-100 state distribution against all-India village share."""
    if "state_name" not in df.columns:
        return pd.DataFrame()

    india_share = (df["state_name"].value_counts(normalize=True) * 100).rename("india_pct")
    top_share = (top100["state_name"].value_counts()).rename("top100_count")

    dist = pd.concat([india_share, top_share], axis=1).fillna(0)
    dist["top100_pct"] = dist["top100_count"] / dist["top100_count"].sum() * 100
    dist["over_rep_ratio"] = (dist["top100_pct"] / dist["india_pct"].replace(0, 0.001)).round(2)
    dist = dist.sort_values("top100_count", ascending=False)

    over = dist[dist["over_rep_ratio"] > 2].index.tolist()
    under = dist[(dist["over_rep_ratio"] < 0.5) & (dist["india_pct"] > 1)].index.tolist()
    print(f"\nState distribution:")
    print(f"  Over-represented states (>2×): {over}")
    print(f"  Under-represented (large states <0.5×): {under}")
    return dist.reset_index().rename(columns={"index": "state_name"})


# ── 6. Electrification confound analysis ─────────────────────────────────────

def electrification_confound(top100: pd.DataFrame) -> dict:
    """
    Tests whether top-100 NTL growth is partly explained by Saubhagya-era
    household electrification (2018-2019) rather than genuine economic development.

    Three-flag heuristic for "electrification-only" risk:
      1. Low NTL baseline in 2019 (< 0.5 nW/cm²/sr) — pre-electrification dark
      2. Growth front-loaded in 2019-2021 (> 70% of total) — matches Saubhagya peak
      3. Flat GHSL built-up change (< 0.01 fractional delta) — no construction

    Villages failing all three are flagged for field validation. Villages passing
    any one flag (higher baseline, sustained late growth, or measurable construction)
    are considered lower risk.

    Reference: Saubhagya scheme achieved ~99.9% household connection coverage by
    Dec 2018; NTL response typically observed in 2019-2021 VIIRS composites.
    """
    top = top100.copy()
    result: dict = {"total": len(top)}

    if "ntl_2019" in top.columns:
        top["low_baseline"] = top["ntl_2019"].fillna(0) < 0.5
        result["low_baseline_count"] = int(top["low_baseline"].sum())
    else:
        top["low_baseline"] = False
        result["low_baseline_count"] = 0

    early_ok = {"ntl_2019", "ntl_2021"}.issubset(top.columns)
    late_ok  = {"ntl_2021", "ntl_2024"}.issubset(top.columns)
    if early_ok and late_ok:
        early_g = (top["ntl_2021"] - top["ntl_2019"]).fillna(0)
        total_g = (top["ntl_2024"] - top["ntl_2019"]).fillna(0)
        frac    = (early_g / total_g.replace(0, np.nan)).fillna(0)
        top["growth_frontloaded"] = (total_g > 0) & (frac > 0.70)
        result["growth_frontloaded_count"] = int(top["growth_frontloaded"].sum())
    else:
        top["growth_frontloaded"] = False
        result["growth_frontloaded_count"] = 0

    if "builtup_change" in top.columns:
        top["flat_builtup"] = top["builtup_change"].fillna(0).abs() < 0.01
    elif {"ghsl_builtup_frac_2020", "ghsl_builtup_frac_2015"}.issubset(top.columns):
        delta = (top["ghsl_builtup_frac_2020"] - top["ghsl_builtup_frac_2015"]).fillna(0)
        top["flat_builtup"] = delta.abs() < 0.01
    else:
        top["flat_builtup"] = False
    result["flat_builtup_count"] = int(top["flat_builtup"].sum())

    top["electrification_only_risk"] = (
        top["low_baseline"] & top["growth_frontloaded"] & top["flat_builtup"]
    )
    n_risk = int(top["electrification_only_risk"].sum())
    result["electrification_only_risk_count"] = n_risk
    result["electrification_only_risk_pct"]   = round(n_risk / len(top) * 100, 1)

    print(f"\nElectrification confound (Saubhagya heuristic, n=100):")
    print(f"  Low NTL baseline 2019 (< 0.5 nW): {result['low_baseline_count']}")
    print(f"  Growth front-loaded 2019-21 (>70%): {result['growth_frontloaded_count']}")
    print(f"  Flat built-up change:               {result['flat_builtup_count']}")
    print(f"  Electrification-only risk (all 3):  {n_risk} "
          f"({result['electrification_only_risk_pct']}%)")
    if n_risk > 20:
        print("  ⚠  High risk — match against SECC 2011 block-level electricity access")
        print("     data.gov.in/resource/secc-2011 before drawing causal conclusions")
    elif n_risk > 10:
        print(f"  ⚠  Moderate risk — {n_risk} villages warrant field validation")
    else:
        print("  ✓  Low risk — NTL growth pattern unlikely to be electrification-only")

    return result


# ── 7. Validation chart ───────────────────────────────────────────────────────

def make_validation_chart(corr_matrix: pd.DataFrame,
                           stability_df: pd.DataFrame,
                           state_dist: pd.DataFrame = None,
                           morans: dict = None) -> None:
    n_cols = 2 + (1 if state_dist is not None and not state_dist.empty else 0)
    titles = ["Signal Correlation (Spearman ρ)", "Rank Stability — Top 100 Villages"]
    if n_cols == 3:
        titles.append("State Distribution (top-100 vs. India)")

    fig = sp.make_subplots(
        rows=1, cols=n_cols,
        subplot_titles=titles,
        horizontal_spacing=0.12,
    )

    if not corr_matrix.empty:
        labels = [c.replace("_score", "").replace("_", " ") for c in corr_matrix.columns]
        fig.add_trace(go.Heatmap(
            z=corr_matrix.values.astype(float),
            x=labels, y=labels,
            colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
            text=corr_matrix.round(2).values.astype(str),
            texttemplate="%{text}",
            showscale=True,
        ), row=1, col=1)

    if not stability_df.empty:
        fig.add_trace(go.Histogram(
            x=stability_df["bootstrap_inclusion_pct"],
            nbinsx=20,
            marker_color="#0D9488",
            name="Villages",
        ), row=1, col=2)
        fig.update_xaxes(title_text="Bootstrap inclusion %", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=1, col=2)

    if state_dist is not None and not state_dist.empty and n_cols == 3:
        top_states = state_dist.head(15)
        fig.add_trace(go.Bar(
            x=top_states["state_name"],
            y=top_states["top100_count"],
            name="Top-100 count",
            marker_color="#0D9488",
        ), row=1, col=3)
        fig.add_trace(go.Scatter(
            x=top_states["state_name"],
            y=top_states["india_pct"],
            name="India village % (scaled)",
            mode="markers",
            marker=dict(color="#F59E0B", size=8, symbol="diamond"),
        ), row=1, col=3)
        fig.update_xaxes(tickangle=45, row=1, col=3)
        fig.update_yaxes(title_text="Count / %", row=1, col=3)

    title = "Validation Analysis — Village Economic Growth Pipeline"
    if morans:
        title += f"  |  Moran's I = {morans.get('I', '?')} (p={morans.get('p_value', '?')})"

    fig.update_layout(
        title=title,
        height=520, width=1300,
        paper_bgcolor="#F0F7FA", plot_bgcolor="#F0F7FA",
        showlegend=True,
        legend=dict(x=0.7, y=1.0),
    )

    out = OUTPUT_DIR / "chart_validation.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"\nValidation chart → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    scored_path = PROC_DIR / "village_scored.csv"
    top100_path = OUTPUT_DIR / "top_100_villages.csv"

    if not scored_path.exists():
        print("village_scored.csv not found — run 04_score_rank.py first")
        return

    print("Loading scored villages...")
    df     = pd.read_csv(scored_path, low_memory=False)
    top100 = pd.read_csv(top100_path) if top100_path.exists() else df.head(100)
    india  = df[df.get("composite_score", pd.Series(dtype=float)).notna()]
    print(f"  {len(india):,} scored villages for validation")

    # 1. Correlation
    corr = signal_correlation(india)
    corr.to_csv(OUTPUT_DIR / "validation_signal_correlation.csv")

    # 2. Stability — run on MAIN TRACK only (NTL ≥ 0.1 and delta ≥ 1.0 nW).
    # Dark-track villages (NTL < 0.1) have extreme ntl_growth_log percentile
    # values (a village going from 0.01→0.05 nW shows 400% growth = high log rank)
    # which floods the bootstrap nlargest(100) whenever ntl weight is drawn high.
    # The actual shortlist is derived from main track only, so the bootstrap
    # should test stability within that same population.
    ntl_delta = india.get("ntl_2024", pd.Series(dtype=float)) - india.get("ntl_2019", pd.Series(dtype=float))
    main_track = india[
        (india.get("ntl_2024", pd.Series(dtype=float)) >= 0.1) &
        (ntl_delta >= 1.0)
    ].copy()
    print(f"  Main-track for stability: {len(main_track):,} villages")

    # Build the shortlist index set: join top100 pc11_village_id → main_track df index
    # The shortlist applies a state cap + 2-pass spatial dedup, so top100 ≠ top-N
    # by composite score. We test stability of the ACTUAL shortlist.
    id_col = "pc11_village_id"
    if id_col in top100.columns and id_col in main_track.columns:
        top100[id_col]      = top100[id_col].astype(str)
        main_track[id_col]  = main_track[id_col].astype(str)
        shortlist_ids = set(
            main_track[main_track[id_col].isin(set(top100[id_col]))].index
        )
        print(f"  Shortlist IDs matched in main-track: {len(shortlist_ids)}")
    else:
        shortlist_ids = None

    stab = rank_stability(main_track, shortlist_ids=shortlist_ids)
    if not stab.empty:
        stab.to_csv(OUTPUT_DIR / "validation_rank_stability.csv", index=False)

    # 3. PMGSY
    pmgsy = pmgsy_validation(top100)
    if pmgsy is not None:
        pmgsy.to_csv(OUTPUT_DIR / "validation_pmgsy_overlap.csv", index=False)

    # 4. Moran's I spatial autocorrelation
    morans = morans_i(india)
    if morans:
        pd.DataFrame([morans]).to_csv(
            OUTPUT_DIR / "validation_morans_i.csv", index=False)

    # 5. State distribution
    state_dist = state_distribution(india, top100)
    if not state_dist.empty:
        state_dist.to_csv(OUTPUT_DIR / "validation_state_distribution.csv", index=False)

    # 6. Electrification confound (Saubhagya heuristic)
    confound = electrification_confound(top100)
    pd.DataFrame([confound]).to_csv(
        OUTPUT_DIR / "validation_electrification_confound.csv", index=False)

    # 7. Chart
    make_validation_chart(corr, stab, state_dist, morans)

    print("\nValidation complete.")


if __name__ == "__main__":
    main()
