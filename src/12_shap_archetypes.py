"""
Step 12: SHAP explainability + village growth archetypes.

SHAP (SHapley Additive exPlanations):
  - Loads the trained GradientBoosting model saved by 04_score_rank.py
  - Computes SHAP values for every village in the top-100
  - Identifies the top-3 driving signals per village
  - Adds shap_top1_signal, shap_top2_signal to top_100_villages.csv
  - Outputs: chart_shap.html (global importance + top-10 waterfall)

Growth archetypes (K-means, k=6):
  - Clusters all scored villages on standardised multi-signal features
  - Assigns each cluster a descriptive name based on centroid profile
  - Outputs: village_archetypes.csv, chart_archetypes.html

Run after: 04_score_rank.py
"""

import warnings
import importlib
import yaml
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
from pathlib import Path

warnings.filterwarnings("ignore")

# Load config.yaml — single source of truth for paths and hyperparameters
_CFG_PATH = Path(__file__).parent.parent / "config.yaml"
_CFG      = yaml.safe_load(_CFG_PATH.read_text()) if _CFG_PATH.exists() else {}
_data     = _CFG.get("paths", {}).get("data_root", "/data/satellite/kritter")

OUTPUT_DIR = Path(_data) / "output"
PROC_DIR   = Path(_data) / "processed"
MODEL_PATH = PROC_DIR / "ml_model.pkl"

TEAL  = "#0D9488"
NAVY  = "#021B2C"
LIGHT = "#F0F7FA"

FEATURE_DISPLAY = {
    "ntl_growth_log":     "NTL Growth (log)",
    "ntl_trend_slope":    "NTL Trend Slope",
    "ntl_2024":           "NTL Level 2024",
    "ndbi_growth":        "NDBI Growth (built-up index)",
    "ndbi_trend_slope":   "NDBI Trend Slope",
    "ghsl_change":        "GHSL Built-up Change",
    "s1_vv_delta":        "SAR VV Δ (Sentinel-1)",
    "tower_growth":       "Mobile Tower Growth",
    "pop_growth_rate":    "Population Growth Rate",
    "builtup_change":     "WorldCover Built-up Δ",
    "ntl_acceleration":   "NTL Acceleration",
    "dist_city_km":       "Distance to City (km)",
    "dist_highway_km":    "Distance to Highway (km)",
}

# Archetype definitions (assigned after clustering based on centroid profile)
ARCHETYPE_DEFS = [
    ("Urban Fringe Surge",   "#F97316", "High NTL + NDBI near cities — peri-urban expansion"),
    ("Rural Electrification","#3B82F6", "High NTL growth, low built-up — power connection boom"),
    ("Construction Boom",    "#10B981", "High NDBI + GHSL + population — new settlement"),
    ("Industrial Corridor",  "#8B5CF6", "Near highways, high NTL + built-up — logistics/industrial"),
    ("Agricultural Shift",   "#F59E0B", "NDVI decline + NDBI rise — cropland conversion"),
    ("Steady Grower",        "#6B7280", "Moderate gains across all signals — organic slow growth"),
]


# ── SHAP ──────────────────────────────────────────────────────────────────────

def run_shap(model, X: pd.DataFrame, feature_names: list) -> np.ndarray:
    try:
        import shap
        explainer = shap.TreeExplainer(model["clf"])
        X_scaled  = model["scaler"].transform(X)
        X_df      = pd.DataFrame(X_scaled, columns=feature_names)
        sv        = explainer.shap_values(X_df)
        # For GBM binary, shap_values returns array of shape (n, features)
        if isinstance(sv, list):
            sv = sv[1]
        return sv
    except ImportError:
        print("  shap not installed — run: pip install shap")
        return None
    except Exception as e:
        print(f"  SHAP failed: {e}")
        return None


def shap_chart(shap_vals: np.ndarray, feature_names: list,
               top100: pd.DataFrame) -> go.Figure:
    """Global SHAP importance bar chart + per-signal waterfall for top-5."""
    mean_abs = np.abs(shap_vals).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]

    labels = [FEATURE_DISPLAY.get(feature_names[i], feature_names[i])
              for i in order]
    values = mean_abs[order]

    fig = sp.make_subplots(
        rows=1, cols=2,
        subplot_titles=["Global Feature Importance (mean |SHAP|)",
                        "Top Village — Signal Contributions"],
        horizontal_spacing=0.12,
    )

    # Global importance
    fig.add_trace(go.Bar(
        y=labels[:12], x=values[:12],
        orientation="h",
        marker_color=[TEAL if v > np.median(values) else "#94A3B8"
                      for v in values[:12]],
        showlegend=False,
    ), row=1, col=1)
    fig.update_xaxes(title_text="Mean |SHAP value|", row=1, col=1)

    # Waterfall for rank-1 village
    sv_top = shap_vals[0]
    top_idx = np.argsort(np.abs(sv_top))[::-1][:8]
    wf_labels = [FEATURE_DISPLAY.get(feature_names[i], feature_names[i])
                 for i in top_idx]
    wf_vals = sv_top[top_idx]

    fig.add_trace(go.Bar(
        y=wf_labels, x=wf_vals,
        orientation="h",
        marker_color=[TEAL if v > 0 else "#EF4444" for v in wf_vals],
        showlegend=False,
        name="SHAP contribution",
    ), row=1, col=2)
    fig.add_vline(x=0, line_color="#475569", line_width=1, row=1, col=2)

    vname = top100.iloc[0].get("village_name", "Rank-1 village")
    fig.update_xaxes(title_text="SHAP contribution (→ growth probability)",
                     row=1, col=2)
    fig.update_layout(
        title=f"SHAP Explainability — {vname} (Rank #1)",
        height=450, width=1200,
        paper_bgcolor=LIGHT, plot_bgcolor=LIGHT,
        font_color="#1E293B",
    )
    return fig


# ── Archetypes ────────────────────────────────────────────────────────────────

ARCHETYPE_FEATURES = [
    "ntl_growth_pct", "ndbi_growth", "ghsl_change",
    "s1_vv_delta", "tower_growth", "pop_growth_rate",
    "dist_city_km", "dist_highway_km",
]


def rule_based_archetype(row: pd.Series) -> str:
    """
    Assign an archetype using explicit signal thresholds instead of K-means.

    Used when K-means clustering is degenerate (few features, collapsed clusters).
    Uses physical signals (NTL, NDBI, GHSL, WorldCover) that are always available
    in a full pipeline run, with geometric signals (dist_city, dist_highway) used
    only when present.

    Priority order (first matching rule wins):
      1. NTL Breakout             — sudden NTL acceleration post-2021
      2. Construction Boom        — NDBI + GHSL + WorldCover all positive
      3. Urban Fringe Surge       — strong NTL with any built-up signal
      4. Rural Electrification    — high NTL growth, no physical construction evidence
      5. Market Corridor Growth   — near road infrastructure (if dist_highway available)
      6. Urban Fringe Expansion   — within 35 km of a city (if dist_city available)
      7. Connectivity-Led Growth  — mobile tower density expanded
      8. Remote Village Emergence — far from city (if dist_city available)
      9. Steady Grower            — default (moderate, consistent growth)
    """
    import math

    def _f(key, default=0.0):
        v = row.get(key, default)
        try:
            fv = float(v)
            return default if math.isnan(fv) else fv
        except (TypeError, ValueError):
            return default

    accel   = _f("ntl_acceleration")
    bp      = _f("ntl_breakpoint_year")
    ntl24   = _f("ntl_2024")
    ntl_pct = _f("ntl_growth_pct")
    ndbi    = _f("ndbi_growth")
    ghsl    = _f("ghsl_change")
    builtu  = _f("builtup_change")
    tower   = _f("tower_growth")

    # Geometric signals: skip if absent (NaN / not in row)
    def _geo(key):
        v = row.get(key, None)
        if v is None:
            return None
        try:
            fv = float(v)
            return None if math.isnan(fv) else fv
        except (TypeError, ValueError):
            return None

    dcity = _geo("dist_city_km")
    dhwy  = _geo("dist_highway_km")

    if accel > 3.5 and bp >= 2022:
        return "NTL Breakout"
    if ndbi > 0 and ghsl > 0 and builtu > 0:
        return "Construction Boom"
    if ntl24 > 10 and (ndbi > 0 or ghsl > 200):
        return "Urban Fringe Surge"
    if ntl_pct > 80 and ndbi <= 0 and ghsl <= 0:
        return "Rural Electrification"
    if dhwy is not None and ntl24 > 10 and dhwy < 200:
        return "Market Corridor Growth"
    if dcity is not None and dcity < 35:
        return "Urban Fringe Expansion"
    if tower > 0.1:
        return "Connectivity-Led Growth"
    if dcity is not None and dcity > 90:
        return "Remote Village Emergence"
    return "Steady Grower"


def assign_archetype_names(km, feature_names: list) -> list:
    """
    Auto-label K-means clusters by comparing centroid profiles RELATIVE to each other.

    Design principle: all comparisons are cluster-vs-cluster (above/below mean across
    clusters), so the logic is scale-invariant and works regardless of how RobustScaler
    transforms the feature space.  Absolute thresholds in scaled space are fragile
    (centroid values depend on data distribution) and are intentionally avoided.

    Assignment order:
      Clusters are processed highest-composite-growth first.  For each cluster,
      the first matching condition claims a semantic label (greedy; once claimed
      a label is locked so no two clusters share the same name).

      Conditions (all relative to cross-cluster mean):
        "Urban Fringe Surge"   : above-avg NTL+NDBI  AND  below-avg dist_city
        "Construction Boom"    : above-avg NTL+NDBI+GHSL  (physical build activity)
        "Industrial Corridor"  : below-avg dist_highway  AND  above-avg NTL
        "Rural Electrification": above-avg NTL  AND  below-avg NDBI  (lights, no build)
        "Agricultural Shift"   : above-avg NDBI  AND  below-avg NTL
        "Connectivity-Led Growth": above-avg tower  AND  not above-avg NTL or NDBI

      Unmatched clusters fall to a rank-based pool:
        "Active Growth" → "Emerging Growth" → "Latent Potential" → …

    Returns list of archetype names in cluster-index order (length = k).
    """
    centers = km.cluster_centers_   # (k, n_features) in RobustScaler space
    k = len(centers)

    def _v(c, key):
        i = feature_names.index(key) if key in feature_names else -1
        return float(c[i]) if i >= 0 else 0.0

    ntl_v   = [_v(c, "ntl_growth_pct")  for c in centers]
    ndbi_v  = [_v(c, "ndbi_growth")     for c in centers]
    ghsl_v  = [_v(c, "ghsl_change")     for c in centers]
    dcity_v = [_v(c, "dist_city_km")    for c in centers]
    dhwy_v  = [_v(c, "dist_highway_km") for c in centers]
    tower_v = [_v(c, "tower_growth")    for c in centers]

    # Cross-cluster means (reference for "above/below average")
    mu_ntl   = np.mean(ntl_v)
    mu_ndbi  = np.mean(ndbi_v)
    mu_ghsl  = np.mean(ghsl_v)
    mu_dcity = np.mean(dcity_v)
    mu_dhwy  = np.mean(dhwy_v)
    mu_tower = np.mean(tower_v)

    has_dcity = "dist_city_km"    in feature_names
    has_dhwy  = "dist_highway_km" in feature_names
    has_tower = "tower_growth"    in feature_names
    has_ghsl  = "ghsl_change"     in feature_names

    # Process highest-composite-growth cluster first
    growth = [
        ntl_v[i] * 0.4 + ndbi_v[i] * 0.3 + ghsl_v[i] * 0.2 + tower_v[i] * 0.1
        for i in range(k)
    ]
    growth_order = sorted(range(k), key=lambda i: growth[i], reverse=True)

    names: list = [""] * k
    used:  set  = set()

    def _claim(ci: int, label: str) -> bool:
        if label not in used:
            names[ci] = label
            used.add(label)
            return True
        return False

    for ci in growth_order:
        above_ntl   = ntl_v[ci]   > mu_ntl
        above_ndbi  = ndbi_v[ci]  > mu_ndbi
        above_ghsl  = has_ghsl  and ghsl_v[ci]   > mu_ghsl
        near_city   = has_dcity and dcity_v[ci]  < mu_dcity   # lower = nearer
        near_hwy    = has_dhwy  and dhwy_v[ci]   < mu_dhwy
        above_tower = has_tower and tower_v[ci]  > mu_tower

        if above_ntl and above_ndbi and near_city:
            if _claim(ci, "Urban Fringe Surge"):
                continue
        if above_ntl and above_ndbi and above_ghsl:
            if _claim(ci, "Construction Boom"):
                continue
        if above_ntl and above_ndbi:
            # NTL+NDBI without GHSL confirmation → still construction-led
            if _claim(ci, "Construction Boom"):
                continue
        if near_hwy and above_ntl:
            if _claim(ci, "Industrial Corridor"):
                continue
        if above_ntl and not above_ndbi:
            if _claim(ci, "Rural Electrification"):
                continue
        if above_ndbi and not above_ntl:
            if _claim(ci, "Agricultural Shift"):
                continue
        if above_tower and not above_ntl and not above_ndbi:
            if _claim(ci, "Connectivity-Led Growth"):
                continue

    # Fallback pool (growth-rank order) for any still-unassigned clusters
    pool = [
        "Active Growth", "Emerging Growth", "Latent Potential",
        "Stable Base", "Low Activity",
    ]
    for ci in growth_order:
        if not names[ci]:
            for label in pool:
                if _claim(ci, label):
                    break

    # Ensure uniqueness: append suffix if same name used twice
    seen_n: dict = {}
    unique = []
    for nm in names:
        seen_n[nm] = seen_n.get(nm, 0) + 1
        unique.append(f"{nm} {seen_n[nm]}" if seen_n[nm] > 1 else nm)
    return unique


def run_archetypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign growth archetypes to all villages in df.

    Two paths:
      A. K-means path (full signals ≥ 4): clusters on scaled multi-signal
         features, selects k ∈ {3…6} by silhouette score, labels clusters
         via assign_archetype_names(). Requires ndbi_growth to be meaningful
         (all-NaN ndbi is excluded from avail, reducing feature count).

      B. Rule-based path: used when fewer than 4 features have real variance
         (e.g. NDBI/SAR stalled) OR when best K-means silhouette < 0.10
         (degenerate clustering — all villages collapse to ≤ 2 clusters
         because there is insufficient feature spread to discriminate).
         Applies rule_based_archetype() row-wise on NTL + geometric signals.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import RobustScaler
    from sklearn.metrics import silhouette_score

    id_col = "pc11_village_id" if "pc11_village_id" in df.columns else df.columns[0]
    meta   = [id_col, "village_name", "state_name", "composite_score", "rank"]
    meta   = [c for c in meta if c in df.columns]

    # Only count features that have real variance (not all-NaN, not constant).
    avail = [f for f in ARCHETYPE_FEATURES
             if f in df.columns
             and df[f].notna().any()
             and df[f].std(skipna=True) > 1e-9]

    use_rule_based = len(avail) < 4  # too few signals for meaningful K-means

    if not use_rule_based:
        X  = df[avail].copy().fillna(df[avail].median()).fillna(0)
        Xs = RobustScaler().fit_transform(X)

        best_k, best_sil, best_km = 3, -1.0, None
        for k in range(3, 7):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(Xs)
            try:
                sil = silhouette_score(Xs, labels,
                                       sample_size=min(5000, len(Xs)))
            except Exception:
                sil = 0.0
            if sil > best_sil:
                best_k, best_sil, best_km = k, sil, km

        print(f"  K-means: best k={best_k}  silhouette={best_sil:.3f}")
        if best_sil < 0.10:
            print(f"  Silhouette {best_sil:.3f} < 0.10 — K-means is degenerate; "
                  "switching to rule-based archetypes")
            use_rule_based = True

    result = df[meta + [c for c in avail if c in df.columns]].copy()

    if use_rule_based:
        reason = (f"{len(avail)} active features" if len(avail) < 4
                  else f"silhouette={best_sil:.3f}")
        print(f"  Rule-based archetypes ({reason})")
        result["archetype_id"]   = -1   # sentinel: rule-based, not cluster ID
        result["archetype_name"] = df.apply(rule_based_archetype, axis=1).values
    else:
        labels = best_km.labels_
        names  = assign_archetype_names(best_km, avail)
        print(f"  Archetypes: {names}")
        result["archetype_id"]   = labels
        result["archetype_name"] = [names[l] for l in labels]

    for name, grp in result.groupby("archetype_name"):
        print(f"    [{name}]: {len(grp):,} villages  "
              f"avg score={grp['composite_score'].mean():.1f}")

    return result


def archetypes_chart(arch_df: pd.DataFrame) -> go.Figure:
    color_map = {name: color for name, color, _ in ARCHETYPE_DEFS}
    desc_map  = {name: desc  for name, _, desc  in ARCHETYPE_DEFS}

    fig = sp.make_subplots(
        rows=1, cols=2,
        subplot_titles=["Village Count per Archetype",
                        "Score Distribution by Archetype"],
        horizontal_spacing=0.12,
    )

    counts = arch_df["archetype_name"].value_counts()
    colors = [color_map.get(n, "#64748B") for n in counts.index]

    fig.add_trace(go.Bar(
        x=counts.index, y=counts.values,
        marker_color=colors, showlegend=False,
        text=counts.values, textposition="outside",
    ), row=1, col=1)
    fig.update_xaxes(tickangle=30, row=1, col=1)
    fig.update_yaxes(title_text="Villages", row=1, col=1)

    for arch_name, grp in arch_df.groupby("archetype_name"):
        fig.add_trace(go.Box(
            y=grp["composite_score"],
            name=arch_name,
            marker_color=color_map.get(arch_name, "#64748B"),
            showlegend=True,
            boxmean=True,
        ), row=1, col=2)
    fig.update_yaxes(title_text="Composite Score (0–100)", row=1, col=2)

    fig.update_layout(
        title="Village Growth Archetypes — K-means Clustering",
        height=480, width=1200,
        paper_bgcolor=LIGHT, plot_bgcolor=LIGHT,
        font_color="#1E293B",
        showlegend=True,
        legend=dict(x=0.55, y=1.0, font_size=10),
    )
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    top100_path = OUTPUT_DIR / "top_100_villages.csv"
    scored_path = PROC_DIR  / "village_scored.csv"

    if not top100_path.exists():
        print("top_100_villages.csv not found — run 04_score_rank.py first")
        return

    top100 = pd.read_csv(top100_path)
    print(f"Loading data...  top100: {len(top100)}")

    # village_scored.csv is a >1 GB EC2 artifact — not committed to the repo.
    # When it is absent (local review, GitHub Actions), archetypes are assigned
    # directly from top_100_villages.csv using rule_based_archetype().
    # When it is present, K-means clustering runs on the full village corpus
    # and the result is merged back to top_100_villages.csv.
    if scored_path.exists():
        scored = pd.read_csv(scored_path, low_memory=False)
        print(f"  scored: {len(scored):,}")
    else:
        scored = None
        print("  village_scored.csv not found — "
              "rule-based archetypes will be applied to top_100 directly")

    # ── SHAP ──────────────────────────────────────────────────────────────────
    if MODEL_PATH.exists() and not top100.empty:
        print("\n[SHAP] Loading trained model...")
        try:
            import joblib
            bundle = joblib.load(MODEL_PATH)
            model  = bundle["model"]
            feats  = bundle["features"]

            avail = [f for f in feats if f in top100.columns]
            if len(avail) == len(feats):
                X = top100[feats].fillna(top100[feats].median()).fillna(0)
                print(f"  Computing SHAP for {len(X)} villages, {len(feats)} features...")
                sv = run_shap(model.named_steps, X, feats)

                if sv is not None:
                    # Top-3 signals per village
                    top_idx = np.argsort(np.abs(sv), axis=1)[:, ::-1]
                    for rank_i, col_name in enumerate(["shap_top1", "shap_top2", "shap_top3"], 1):
                        top100[col_name] = [feats[top_idx[i, rank_i-1]]
                                            for i in range(len(top100))]

                    top100.to_csv(top100_path, index=False)
                    print("  Updated top_100_villages.csv with SHAP columns")

                    fig_shap = shap_chart(sv, feats, top100)
                    shap_out = OUTPUT_DIR / "chart_shap.html"
                    fig_shap.write_html(str(shap_out), include_plotlyjs="cdn")
                    print(f"  → {shap_out}")
            else:
                print(f"  Feature mismatch: {set(feats) - set(top100.columns)}")
        except Exception as e:
            print(f"  SHAP skipped: {e}")
    else:
        print("\n[SHAP] Model not found or top100 empty — skipping")
        print(f"  MODEL_PATH={MODEL_PATH}  exists={MODEL_PATH.exists()}")

    # ── Archetypes ────────────────────────────────────────────────────────────
    print("\n[Archetypes]")

    if scored is not None:
        # Full K-means path: cluster on all scored main-track villages then
        # merge the result back to top100.
        main_df = scored[scored["ntl_2019"].notna() &
                         (scored["ntl_2019"] >= 0.1)].copy()
        print(f"  Main track from village_scored.csv: {len(main_df):,}")
        arch_df = run_archetypes(main_df)
    else:
        # Fallback: run directly on the top_100 DataFrame.
        # rule_based_archetype() only needs NTL + geometric columns which
        # are always present in top_100_villages.csv.
        print("  Running archetypes on top_100_villages.csv (rule-based)")
        arch_df = run_archetypes(top100)

    if not arch_df.empty:
        arch_out = OUTPUT_DIR / "village_archetypes.csv"
        arch_df.to_csv(arch_out, index=False)
        print(f"  → {arch_out}")

        fig_arch = archetypes_chart(arch_df)
        arch_chart_out = OUTPUT_DIR / "chart_archetypes.html"
        fig_arch.write_html(str(arch_chart_out), include_plotlyjs="cdn")
        print(f"  → {arch_chart_out}")

        # Merge archetype_name into top100_villages.csv.
        # If archetypes were run on top100 directly, arch_df already has the
        # same index — use direct assignment.  If run on full scored CSV, join
        # on pc11_village_id.
        id_col = "pc11_village_id"
        if id_col in arch_df.columns and id_col in top100.columns:
            arch_top = arch_df[[id_col, "archetype_name"]].copy()
            arch_top[id_col] = arch_top[id_col].astype(str)
            top100[id_col]   = top100[id_col].astype(str)
            # Drop stale archetype_name column if present before merge
            top100 = top100.drop(columns=["archetype_name"], errors="ignore")
            top100 = top100.merge(arch_top, on=id_col, how="left")
        elif "archetype_name" in arch_df.columns:
            top100["archetype_name"] = arch_df["archetype_name"].values

        # Diversity guard: if the top-100 all land in a single K-means cluster,
        # OR all share the same base name (e.g. "Steady Grower 1/2/3" — which
        # happens when dist_city/highway are absent so absolute thresholds never
        # fire), override with rule_based_archetype() that uses physical signals.
        import re as _re
        n_unique   = top100["archetype_name"].nunique(dropna=True)
        base_names = set(
            _re.sub(r"\s+\d+$", "", s).strip()
            for s in top100["archetype_name"].dropna().unique()
        )
        # Degenerate if: too few names, all same base ("Steady Grower X"),
        # or any name ends with a digit suffix (uniqueness collision from assign_archetype_names)
        has_suffix = any(
            bool(_re.search(r"\s+\d+$", s))
            for s in top100["archetype_name"].dropna().unique()
        )
        degenerate = n_unique < 3 or (len(base_names) == 1) or has_suffix
        if degenerate:
            print(f"  ⚠ {n_unique} archetype(s), bases {base_names}, suffix={has_suffix} "
                  f"— applying rule-based fallback")
            top100["archetype_name"] = top100.apply(rule_based_archetype, axis=1)

        top100.to_csv(top100_path, index=False)
        print(f"  Updated {top100_path.name} with archetype_name")
        print(f"  Distribution: {top100['archetype_name'].value_counts().to_dict()}")

    print("\nStep 12 complete.")


if __name__ == "__main__":
    main()
