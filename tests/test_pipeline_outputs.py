"""
Data quality tests for the village_growth pipeline final output.

Run:  python -m pytest tests/ -v
Needs: pip install pytest pandas

Tests validate output/top_100_villages.csv against the schema produced by
04_score_rank.py → 12_shap_archetypes.py.

Row count note: the output file contains the top-N *deduplicated* villages
(N ≤ 100). Spatial deduplication (radius 5 km, same base name) collapses
multiple OSM point nodes that represent the same settlement into a single
representative entry. The current run yields 87 rows; a full 8-signal run
with a fresh OSM extract may differ slightly.
"""

import pathlib
import pytest
import pandas as pd

LOCAL_CSV = pathlib.Path(__file__).parent.parent / "output" / "top_100_villages.csv"
INDIA_LAT = (8.0, 37.5)
INDIA_LON = (68.0, 97.5)


@pytest.fixture(scope="module")
def df():
    if not LOCAL_CSV.exists():
        pytest.skip("output/top_100_villages.csv not found — run pipeline first")
    return pd.read_csv(LOCAL_CSV)


# ── Structure ──────────────────────────────────────────────────────────────────

def test_row_count(df):
    """Between 50 and 100 rows after spatial deduplication."""
    assert 50 <= len(df) <= 100, (
        f"Expected 50–100 rows after dedup, got {len(df)}. "
        "If > 100, spatial_dedup_villages() may not have run. "
        "If < 50, the OSM extract or NTL filter may be too aggressive."
    )


def test_required_columns(df):
    required = {
        "village_name", "state_name", "district_name",
        "latitude", "longitude", "composite_score", "rank",
        "ntl_2019", "ntl_2024", "ntl_growth_pct",
    }
    missing = required - set(df.columns)
    assert not missing, f"Required columns missing: {missing}"


def test_rank_column_sequential(df):
    """Rank must run from 1 to len(df) with no gaps (post-dedup renum)."""
    ranks = sorted(df["rank"].tolist())
    assert ranks == list(range(1, len(df) + 1)), (
        f"rank column is not 1–{len(df)}: {ranks[:5]}…"
    )


# ── Score integrity ────────────────────────────────────────────────────────────

def test_composite_score_no_nan(df):
    assert df["composite_score"].notna().all(), "composite_score contains NaN"


def test_composite_score_range(df):
    s = df["composite_score"]
    assert (s >= 0).all() and (s <= 100).all(), (
        f"composite_score out of [0,100]: {s.describe()}"
    )


def test_ranking_is_descending(df):
    scores = df.sort_values("rank")["composite_score"].values
    violations = [
        (i + 1, scores[i], scores[i + 1])
        for i in range(len(scores) - 1)
        if scores[i] < scores[i + 1]
    ]
    assert not violations, f"Score not descending at ranks: {violations[:5]}"


def test_score_spread(df):
    """Top-to-bottom composite score spread must exceed 1 point.

    Current 3-signal run: spread ≈ 3.8 pts (NTL + WorldCover + spatial-lag).
    Full 8-signal run with NDBI + SAR: expected ≥ 5 pts (wider per-signal
    discrimination). Threshold of 1.0 catches a collapsed/degenerate run.
    """
    spread = df["composite_score"].max() - df["composite_score"].min()
    assert spread > 1.0, (
        f"Score spread is only {spread:.4f} pts — expected > 1.0. "
        "Check that minmax_score() ran and signals are non-NaN."
    )


def test_ml_score_present_if_available(df):
    if "ml_growth_prob_score" not in df.columns:
        pytest.skip("ml_growth_prob_score not in output (pre-ML run)")
    pct_nan = df["ml_growth_prob_score"].isna().mean()
    assert pct_nan < 0.1, (
        f"ml_growth_prob_score is {pct_nan:.0%} NaN — ML step may have failed"
    )


# ── Geography ─────────────────────────────────────────────────────────────────

def test_coordinates_within_india(df):
    bad_lat = df[~df["latitude"].between(*INDIA_LAT)]
    bad_lon = df[~df["longitude"].between(*INDIA_LON)]
    assert bad_lat.empty, (
        f"Lat out of India: {bad_lat[['village_name','latitude']].to_dict('records')}"
    )
    assert bad_lon.empty, (
        f"Lon out of India: {bad_lon[['village_name','longitude']].to_dict('records')}"
    )


def test_state_diversity(df):
    n_states = df["state_name"].nunique()
    assert n_states >= 5, (
        f"Only {n_states} states represented — suspicious concentration. "
        "Expected ≥ 8 states in a healthy run."
    )


def test_no_state_dominant_majority(df):
    """No single state should exceed 50% of the shortlist.

    Current run (post-dedup): Karnataka 38%, Uttar Pradesh 16%.
    Karnataka over-representation (7× vs village-count share) is documented
    in the README limitations section — WorldCover 10 m may capture
    shade-house agriculture as built-up area in Karnataka.

    The 50% guard rail catches pathological feedback loops seen in earlier
    pipeline versions (UP at 83%, Karnataka at 95% in pre-fix runs).
    """
    counts = df["state_name"].value_counts(normalize=True)
    dominant_state = counts.index[0]
    dominant_pct   = counts.iloc[0]
    assert dominant_pct < 0.50, (
        f"{dominant_state} is {dominant_pct:.0%} of shortlist — "
        "check OSM node density bias and spatial-lag feedback loop"
    )


# ── Data quality ──────────────────────────────────────────────────────────────

def test_no_placeholder_area_names(df):
    bad = df["village_name"].dropna()
    bad = bad[bad.str.contains(r"\(area\)", na=False)]
    assert bad.empty, f"Placeholder '(area)' names still in output: {bad.tolist()}"


def test_majority_villages_named(df):
    """At least 70% of shortlist villages must have a real OSM or census name.

    Village names are sourced from OSM name= tags; unnamed nodes are resolved
    via Nominatim reverse-geocoding. Nodes that could not be resolved are shown
    as 'Village_<pc11_id>'. The current run achieves 100% named (Nominatim
    resolved all 30 unnamed nodes). Threshold 70% gives headroom for runs with
    a higher proportion of unnamed OSM extracts while still catching a fully
    unnamed output.
    """
    truly_named = df["village_name"].dropna()
    truly_named = truly_named[~truly_named.str.match(r"^Village_\d+$", na=False)]
    pct = len(truly_named) / len(df)
    assert pct >= 0.70, (
        f"Only {pct:.0%} of villages have real names (expected ≥ 70%). "
        "Run Nominatim reverse-geocoding or SHRUG PC11→name join."
    )


def test_ntl_growth_positive(df):
    if "ntl_growth_pct" not in df.columns:
        pytest.skip("ntl_growth_pct not in output")
    neg = df[df["ntl_growth_pct"] <= 0]
    assert neg.empty, (
        f"Shortlist villages with non-positive NTL growth:\n"
        f"{neg[['village_name','ntl_growth_pct']].to_string()}"
    )


def test_ntl_2024_exceeds_ntl_2019(df):
    worse = df[df["ntl_2024"] <= df["ntl_2019"]]
    assert worse.empty, (
        f"Villages with NTL 2024 ≤ NTL 2019:\n"
        f"{worse[['village_name','ntl_2019','ntl_2024']].to_string()}"
    )


# ── Multi-signal confirmation & uncertainty ───────────────────────────────────

def test_multi_signal_confirmed_present(df):
    if "multi_signal_confirmed" not in df.columns:
        pytest.fail("multi_signal_confirmed column missing — scoring step failed")
    confirmed_pct = df["multi_signal_confirmed"].mean()
    assert confirmed_pct >= 0.10, (
        f"Only {confirmed_pct:.0%} of shortlist are multi-signal confirmed. "
        "Check signal extraction steps (NTL + WorldCover minimum required)."
    )


def test_confidence_score_present(df):
    if "confidence_score" not in df.columns:
        pytest.skip("confidence_score not in output (pre-uncertainty pipeline?)")
    min_conf = df["confidence_score"].min()
    assert min_conf >= 30.0, (
        f"confidence_score = {min_conf:.1f}% — fewer than 4/14 signals active. "
        "Check signal extraction scripts."
    )


def test_signals_active_column(df):
    if "signals_active" not in df.columns:
        pytest.skip("signals_active not in output (pre-fix pipeline run?)")
    n = df["signals_active"].iloc[0]
    assert n >= 3, (
        f"Only {n} signals active — composite relies on too few independent signals"
    )


def test_ntl_growth_not_extreme_outlier(df):
    """Top NTL growth should be < 10,000% — higher values suggest a unit error."""
    if "ntl_growth_pct" not in df.columns:
        pytest.skip("ntl_growth_pct not available")
    max_growth = df["ntl_growth_pct"].max()
    assert max_growth < 10_000, (
        f"Max NTL growth {max_growth:.0f}% exceeds 10,000% — "
        "check ntl_min_absolute_delta filter and VIIRS unit consistency"
    )


# ── Deduplication sanity check ────────────────────────────────────────────────

def test_no_duplicate_settlement_clusters(df):
    """No two shortlist entries with the same base name should be within the dedup radius.

    Spatial deduplication (spatial_dedup_villages in 04_score_rank.py) merges
    OSM nodes within 5 km that share the same base name, keeping only the highest
    scorer. Multiple villages can legitimately share a name while being far apart
    (e.g., two "Bhanpur" towns in different districts 15 km apart), so this test
    only fails when same-name entries are closer than DEDUP_RADIUS_KM — confirming
    the dedup step ran and collapsed nearby duplicate nodes.
    """
    import re, math

    DEDUP_RADIUS_KM = 5.0  # matches config.yaml scoring.dedup_radius_km

    def base_name(name):
        if not isinstance(name, str):
            return ""
        return re.sub(r"\s*\(OSM\s*\d+\)\s*$", "", name).strip().lower()

    def haversine_km(lat1, lon1, lat2, lon2):
        R = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
        return R * 2 * math.asin(math.sqrt(a))

    df2 = df.copy()
    df2["base"] = df2["village_name"].apply(base_name)
    violations = []
    for name, group in df2[df2["base"] != ""].groupby("base"):
        rows = group[["village_name", "latitude", "longitude"]].values.tolist()
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                dist = haversine_km(rows[i][1], rows[i][2], rows[j][1], rows[j][2])
                if dist < DEDUP_RADIUS_KM:
                    violations.append(f"{rows[i][0]} / {rows[j][0]} ({dist:.1f} km)")
    assert not violations, (
        f"Same-name villages within dedup radius: {violations}\n"
        "spatial_dedup_villages() may not have run — check 04_score_rank.py"
    )


# ── Archetype column ──────────────────────────────────────────────────────────

def test_no_pending_archetypes(df):
    """archetype_name must not contain the 'pending' placeholder.

    The placeholder 'pending — Sentinel-2/SAR re-run required' was written in
    earlier pipeline versions when K-means collapsed to one cluster due to
    missing NDBI/SAR signals. 12_shap_archetypes.py now falls back to
    rule_based_archetype() in that case, so 'pending' should never appear.
    """
    if "archetype_name" not in df.columns:
        pytest.skip("archetype_name not in output")
    pending = df[df["archetype_name"].str.contains("pending", case=False, na=False)]
    assert pending.empty, (
        f"{len(pending)} villages still have 'pending' archetype — "
        "run 12_shap_archetypes.py to assign real archetypes"
    )


def test_archetype_names_are_known(df):
    if "archetype_name" not in df.columns:
        pytest.skip("archetype_name not in output")
    known = {
        # Full-signal K-means archetypes
        "Urban Fringe Surge",
        "Rural Electrification",
        "Construction Boom",
        "Industrial Corridor",
        "Agricultural Shift",
        "Steady Grower",
        # Sparse-signal K-means fallback names
        "NTL Breakout",
        "Active Growth",
        "Emerging Growth",
        "Latent Potential",
        "Stable Base",
        "Low Activity",
        # Rule-based archetypes (3-signal run)
        "Connectivity-Led Growth",
        "Market Corridor Growth",
        "Urban Fringe Expansion",
        "Remote Village Emergence",
    }
    unknown = set(df["archetype_name"].dropna().unique()) - known
    assert not unknown, f"Unknown archetype name(s): {unknown}"


# ── SHAP (optional) ───────────────────────────────────────────────────────────

def test_shap_columns_present(df):
    if "shap_top1" not in df.columns:
        pytest.skip("SHAP columns not in output — run 12_shap_archetypes.py")
    for col in ("shap_top1", "shap_top2", "shap_top3"):
        assert df[col].notna().mean() > 0.9, f"{col} is mostly NaN"
