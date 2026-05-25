"""
Village Economic Growth Intelligence — Streamlit Dashboard

Interactive dashboard for exploring the top 100 (and dark-track) villages
identified by the Kritter assignment pipeline.

Run locally:
    pip install streamlit plotly folium streamlit-folium pandas
    streamlit run dashboard/app.py

Deploy to Streamlit Community Cloud (free):
    Push repo to GitHub → go to share.streamlit.io → select app.py

Data is loaded from output/ directory (CSVs) or directly from S3 if configured.
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Village Economic Growth Intelligence",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded",
)

S3_BASE = ("https://kritter-village-growth-335451551223"
           ".s3.ap-south-1.amazonaws.com/")
DATA_DIR = Path(__file__).parent.parent / "output"

YEARS = [2019, 2020, 2021, 2022, 2023, 2024]
FORECAST_YRS = [2025, 2026, 2027]

TEAL  = "#0D9488"
NAVY  = "#021B2C"
LIGHT = "#F0F7FA"


# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main-header { font-size:2rem; font-weight:700; color:#021B2C; }
  .sub-header  { color:#475569; font-size:1rem; margin-top:-12px; }
  .metric-card { background:white; border-radius:10px; padding:16px;
                 box-shadow: 0 1px 4px rgba(0,0,0,0.08); text-align:center; }
  .metric-val  { font-size:2rem; font-weight:700; color:#0D9488; }
  .metric-lbl  { font-size:0.75rem; color:#94A3B8; text-transform:uppercase; }
  .village-card{ background:white; border-radius:8px; padding:14px;
                 border-left:4px solid #0D9488; margin:6px 0; }
  div[data-testid="stSelectbox"] label { font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_data():
    files = {
        "top100":    "top_100_villages.csv",
        "dark50":    "dark_top_50_villages.csv",
        "forecast":  "top_100_forecast.csv",
        "stars":     "rising_stars.csv",
        "scored":    "../processed/village_scored.csv",
        "validation":"validation_signal_correlation.csv",
        "stability": "validation_rank_stability.csv",
        "uncertainty":"village_uncertainty.csv",
    }
    dfs = {}
    for key, fname in files.items():
        path = DATA_DIR / fname
        try:
            if path.exists():
                dfs[key] = pd.read_csv(path, low_memory=False)
            else:
                # Try S3 fallback
                url = S3_BASE + Path(fname).name
                dfs[key] = pd.read_csv(url)
        except Exception:
            dfs[key] = pd.DataFrame()
    return dfs


# ── Sidebar ───────────────────────────────────────────────────────────────────

def sidebar(df_top: pd.DataFrame):
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Flag_of_India.svg/320px-Flag_of_India.svg.png",
                     width=120)
    st.sidebar.markdown("### Filters")

    states = ["All"] + sorted(df_top["state_name"].dropna().unique().tolist()) \
             if "state_name" in df_top.columns else ["All"]
    state_sel = st.sidebar.selectbox("State", states)

    score_range = st.sidebar.slider("Composite Score Range",
                                     0.0, 100.0, (0.0, 100.0), 0.5)

    track = st.sidebar.radio("Village track", ["Main (NTL ≥ 0.1)", "Dark (NTL < 0.1)", "Both"])
    confirmed_only = st.sidebar.checkbox("Multi-signal confirmed only", value=False)

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Data source:** [S3 bucket]({S3_BASE})")
    st.sidebar.markdown("**Pipeline:** VIIRS · Sentinel-2 · SAR · WorldPop · GHSL")

    return state_sel, score_range, track, confirmed_only


def apply_filters(df, state_sel, score_range, confirmed_only):
    if df.empty:
        return df
    if state_sel != "All" and "state_name" in df.columns:
        df = df[df["state_name"] == state_sel]
    if "composite_score" in df.columns:
        df = df[df["composite_score"].between(*score_range)]
    if confirmed_only and "multi_signal_confirmed" in df.columns:
        df = df[df["multi_signal_confirmed"] == True]
    return df


# ── Pages ─────────────────────────────────────────────────────────────────────

def page_overview(dfs, state_sel, score_range, confirmed_only):
    st.markdown('<div class="main-header">Village Economic Growth Intelligence</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">India · 2019–2024 · Satellite-Derived Economic Signals</div>',
                unsafe_allow_html=True)
    st.markdown("")

    top = apply_filters(dfs.get("top100", pd.DataFrame()),
                        state_sel, score_range, confirmed_only)

    # KPI row
    cols = st.columns(5)
    kpis = [
        ("255K+", "Villages Scored"),
        (str(len(top)), "In Filter"),
        (f"{top['composite_score'].max():.1f}" if not top.empty and "composite_score" in top.columns else "—",
         "Highest Score"),
        (f"{top['ntl_growth_pct'].median():.0f}%" if not top.empty and "ntl_growth_pct" in top.columns else "—",
         "Median NTL Growth"),
        (str(top["state_name"].nunique()) if "state_name" in top.columns else "—",
         "States Represented"),
    ]
    for col, (val, lbl) in zip(cols, kpis):
        col.markdown(f'<div class="metric-card"><div class="metric-val">{val}</div>'
                     f'<div class="metric-lbl">{lbl}</div></div>',
                     unsafe_allow_html=True)

    st.markdown("")

    # Map + state chart
    c1, c2 = st.columns([3, 2])

    with c1:
        st.subheader("Geographic Distribution")
        if not top.empty and "latitude" in top.columns:
            m = folium.Map(location=[20.5, 78.9], zoom_start=5, tiles="CartoDB positron")
            for _, row in top.iterrows():
                if pd.isna(row.get("latitude")):
                    continue
                score = row.get("composite_score", 50)
                r_val = int(255 * (1 - score/100))
                g_val = int(180 + 75 * score/100)
                folium.CircleMarker(
                    [row["latitude"], row["longitude"]],
                    radius=4 + (100 - row.get("rank", 50)) / 25,
                    color=f"#{r_val:02x}{g_val:02x}00",
                    fill=True, fill_opacity=0.85,
                    tooltip=f"#{int(row.get('rank',0))} {row.get('village_name','')} — {score:.1f}",
                ).add_to(m)
            st_folium(m, width=None, height=380)
        else:
            st.info("No location data available for map.")

    with c2:
        st.subheader("State Distribution")
        if not top.empty and "state_name" in top.columns:
            counts = top["state_name"].value_counts().reset_index()
            counts.columns = ["state", "count"]
            fig = px.bar(counts.head(12), x="count", y="state", orientation="h",
                         color="count", color_continuous_scale="Teal")
            fig.update_layout(height=380, margin=dict(l=0, r=0, t=0, b=0),
                               coloraxis_showscale=False,
                               yaxis=dict(categoryorder="total ascending"))
            st.plotly_chart(fig, use_container_width=True)

    # Full ranked table
    st.subheader(f"Top Villages ({len(top)} total)")
    show_cols = [c for c in ["rank", "village_name", "state_name", "district_name",
                               "composite_score", "ntl_growth_pct", "ml_growth_prob",
                               "confidence_score", "multi_signal_confirmed"]
                 if c in top.columns]
    if show_cols:
        display = top[show_cols].copy()
        if "ntl_growth_pct" in display.columns:
            display["ntl_growth_pct"] = display["ntl_growth_pct"].apply(
                lambda x: f"{x:.0f}%" if not pd.isna(x) else "—")
        if "composite_score" in display.columns:
            display["composite_score"] = display["composite_score"].round(1)
        col_cfg = {
            "rank":                  st.column_config.NumberColumn("Rank",      width="small"),
            "village_name":          st.column_config.TextColumn("Village",     width="medium"),
            "state_name":            st.column_config.TextColumn("State",       width="medium"),
            "district_name":         st.column_config.TextColumn("District",    width="medium"),
            "composite_score":       st.column_config.NumberColumn("Score",     width="small"),
            "ntl_growth_pct":        st.column_config.TextColumn("NTL Growth",  width="small"),
            "ml_growth_prob":        st.column_config.NumberColumn("ML Prob",   width="small"),
            "confidence_score":      st.column_config.NumberColumn("Confidence",width="small"),
            "multi_signal_confirmed":st.column_config.CheckboxColumn("Confirmed",width="small"),
        }
        st.dataframe(display, use_container_width=True, hide_index=True,
                     height=35 * len(display) + 38,
                     column_config={k: v for k, v in col_cfg.items() if k in display.columns})

    st.download_button("Download CSV", top.to_csv(index=False),
                       "filtered_villages.csv", "text/csv")


def page_village_explorer(dfs, state_sel, score_range, confirmed_only):
    st.header("Village Explorer")

    top = apply_filters(dfs.get("forecast", dfs.get("top100", pd.DataFrame())),
                        state_sel, score_range, confirmed_only)

    if top.empty:
        st.warning("No villages match current filters."); return

    village_names = top["village_name"].fillna("").astype(str).tolist() \
                    if "village_name" in top.columns else [str(r) for r in top["rank"]]
    sel_name = st.selectbox("Select village", village_names)

    row = top[top.get("village_name", top.index).astype(str) == sel_name].iloc[0] \
          if "village_name" in top.columns else top.iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rank",           f"#{int(row.get('rank', '?'))}")
    c2.metric("Composite Score",f"{row.get('composite_score', 0):.1f}")
    c3.metric("NTL Growth",     f"{row.get('ntl_growth_pct', 0):.0f}%"
              if not pd.isna(row.get("ntl_growth_pct")) else "—")
    c4.metric("ML Prob",        f"{row.get('ml_growth_prob', 0):.1f}%"
              if not pd.isna(row.get("ml_growth_prob")) else "—")

    st.markdown(f"**State:** {row.get('state_name','')} &nbsp; "
                f"**District:** {row.get('district_name','')} &nbsp; "
                f"**Coords:** {row.get('latitude',''):.4f}°N, {row.get('longitude',''):.4f}°E"
                if not pd.isna(row.get("latitude")) else "")

    # NTL time series + forecast
    st.subheader("NTL Time Series & Forecast")
    hist_y = [row.get(f"ntl_{y}", np.nan) for y in YEARS]
    fc_y   = [row.get(f"ntl_forecast_{y}", np.nan) for y in FORECAST_YRS]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=YEARS, y=hist_y, mode="lines+markers",
                              name="Historical", line=dict(color=TEAL, width=2),
                              marker=dict(size=7)))
    if any(not pd.isna(v) for v in fc_y):
        fig.add_trace(go.Scatter(x=FORECAST_YRS, y=fc_y, mode="lines+markers",
                                  name="Forecast (ensemble)", line=dict(color="#F59E0B",
                                  dash="dash", width=2), marker=dict(size=7, symbol="diamond")))
        fig.add_vrect(x0=2024.5, x1=2027.5, fillcolor="#FEF3C7",
                      opacity=0.3, line_width=0, annotation_text="Forecast")
    fig.update_layout(height=300, xaxis_title="Year",
                       yaxis_title="NTL (nW/cm²/sr)",
                       paper_bgcolor=LIGHT, plot_bgcolor="#E8F4F8")
    st.plotly_chart(fig, use_container_width=True)

    # Signal breakdown
    st.subheader("Signal Breakdown")
    sig_cols = {
        "NTL Growth %":    "ntl_growth_pct",
        "NTL Trend Slope": "ntl_trend_slope",
        "NDBI Growth":     "ndbi_growth",
        "GHSL Change":     "ghsl_change",
        "SAR VV Delta":    "s1_vv_delta",
        "Tower Growth":    "tower_growth",
        "Pop Growth":      "pop_growth_rate",
        "WorldCover Δ":    "builtup_change",
    }
    sig_data = {lbl: row.get(col, np.nan) for lbl, col in sig_cols.items()
                if not pd.isna(row.get(col, np.nan))}
    if sig_data:
        sig_df = pd.DataFrame({"Signal": list(sig_data.keys()),
                                "Value": list(sig_data.values())})
        st.dataframe(sig_df, use_container_width=True, hide_index=True)
    else:
        st.info("Signal data not available for this village.")


def page_analytics(dfs):
    st.header("Analytics & Validation")

    tabs = st.tabs(["Score Distribution", "Signal Correlation",
                    "Rank Stability", "Rising Stars", "Dark-track Villages"])

    top100 = dfs.get("top100", pd.DataFrame())
    all_scored = dfs.get("scored", pd.DataFrame())

    with tabs[0]:
        if "composite_score" in top100.columns:
            fig = px.histogram(top100, x="composite_score", nbins=20,
                               title="Top-100 Score Distribution",
                               color_discrete_sequence=[TEAL])
            fig.update_layout(paper_bgcolor=LIGHT, plot_bgcolor="#E8F4F8")
            st.plotly_chart(fig, use_container_width=True)

            if "confidence_score" in top100.columns:
                fig2 = px.scatter(top100, x="confidence_score", y="composite_score",
                                  hover_name=top100.get("village_name"),
                                  title="Score vs Data Confidence",
                                  color_discrete_sequence=[TEAL])
                st.plotly_chart(fig2, use_container_width=True)

    with tabs[1]:
        corr = dfs.get("validation", pd.DataFrame())
        if not corr.empty:
            try:
                cm = corr.set_index(corr.columns[0]).astype(float)
                labels = [c.replace("_", " ") for c in cm.columns]
                fig = px.imshow(cm.values, x=labels, y=labels,
                                color_continuous_scale="RdBu", zmin=-1, zmax=1,
                                text_auto=".2f",
                                title="Signal Spearman Correlation Matrix")
                fig.update_layout(height=450)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not render correlation matrix: {e}")
        else:
            st.info("Run 07_validate.py to generate correlation data.")

    with tabs[2]:
        stab = dfs.get("stability", pd.DataFrame())
        if not stab.empty and "bootstrap_inclusion_pct" in stab.columns:
            fig = px.histogram(stab, x="bootstrap_inclusion_pct", nbins=20,
                               title="Rank Stability — Bootstrap Inclusion % (top 100 villages)",
                               labels={"bootstrap_inclusion_pct": "Bootstrap inclusion %"},
                               color_discrete_sequence=[TEAL])
            st.plotly_chart(fig, use_container_width=True)
            n_stable = (stab["bootstrap_inclusion_pct"] >= 80).sum()
            st.metric("Highly stable villages (≥80% inclusion)", f"{n_stable}/100")
        else:
            st.info("Run 07_validate.py to generate stability data.")

    with tabs[3]:
        stars = dfs.get("stars", pd.DataFrame())
        if not stars.empty:
            show = [c for c in ["rank","village_name","state_name",
                                  "ntl_trend_slope","composite_score","likely_top100_2027"]
                    if c in stars.columns]
            st.dataframe(stars[show].head(50), use_container_width=True, hide_index=True)
            if "ntl_trend_slope" in stars.columns:
                fig = px.bar(stars.head(20).sort_values("ntl_trend_slope"),
                             x="ntl_trend_slope", y=stars.head(20).get("village_name",
                             stars.head(20)["rank"].astype(str)),
                             orientation="h", title="Rising Stars — NTL Trend Slope",
                             color_discrete_sequence=[TEAL])
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run 08_forecast.py to generate rising stars data.")

    with tabs[4]:
        dark = dfs.get("dark50", pd.DataFrame())
        if not dark.empty:
            st.markdown("Villages with near-zero nighttime light in 2019 — scored on "
                        "NDBI, GHSL built-up change, and population growth.")
            show = [c for c in ["rank","village_name","state_name","composite_score",
                                  "ndbi_growth","ghsl_change","pop_growth_rate"]
                    if c in dark.columns]
            st.dataframe(dark[show].head(25), use_container_width=True, hide_index=True)
        else:
            st.info("No dark-track data found.")


def page_methodology():
    st.header("Methodology")
    st.markdown(f"""
### Data Sources

| Source | Signal | Resolution | Purpose |
|--------|--------|-----------|---------|
| NASA VIIRS VNP46A4 | Nighttime light radiance | 500 m | Primary economic activity proxy |
| Sentinel-2 L2A (Element84) | NDBI, NDVI annual composites | ~100 m | Built-up & vegetation change |
| Sentinel-1 RTC (Element84) | SAR VV backscatter | ~100 m | Cloud-independent built-up |
| ESA WorldCover | Built-up fraction | 10 m | 2020→2021 change |
| GHSL R2023A (EU JRC) | Built-up surface | 100 m | Validated 2015→2020 change |
| WorldPop | Population density | 100 m | Demographic growth signal |
| OpenStreetMap | Telecom tower locations | — | Mobile connectivity proxy |

### Scoring Model

**Main track** (villages with NTL ≥ 0.1 nW/cm²/sr):
1. Feature engineering: BFAST change-point detection, spatial lag, signal agreement
2. Self-supervised ML label: top-10% on ≥2 primary signals simultaneously
3. GradientBoostingClassifier → ML growth probability (0–100)
4. Composite: 40% ML + 20% NTL growth + 15% NDBI + 15% GHSL + 10% SAR

**Dark track** (villages with NTL < 0.1):
Scored separately on NDBI (40%) + GHSL (30%) + tower growth (20%) + population (10%)

### Validation
- Bootstrap rank stability (200 random weight draws)
- Signal correlation analysis (Spearman ρ)
- PMGSY road data cross-reference

### Links
- [Top 100 CSV]({S3_BASE}top_100_villages.csv)
- [Interactive Map]({S3_BASE}map.html)
- [GitHub Repository](https://github.com/ishunteam-png/kritter-village-growth)
""")


# ── Main app ──────────────────────────────────────────────────────────────────

def main():
    dfs = load_data()

    top_df = dfs.get("top100", pd.DataFrame())
    state_sel, score_range, track, confirmed_only = sidebar(top_df)

    if track == "Dark (NTL < 0.1)":
        dfs["top100"] = dfs.get("dark50", pd.DataFrame())
    elif track == "Both":
        main_t = dfs.get("top100", pd.DataFrame())
        dark_t = dfs.get("dark50", pd.DataFrame())
        dfs["top100"] = pd.concat([main_t, dark_t], ignore_index=True) if not dark_t.empty else main_t

    page = st.sidebar.radio("Page", ["Overview", "Village Explorer", "Analytics", "Methodology"])

    if page == "Overview":
        page_overview(dfs, state_sel, score_range, confirmed_only)
    elif page == "Village Explorer":
        page_village_explorer(dfs, state_sel, score_range, confirmed_only)
    elif page == "Analytics":
        page_analytics(dfs)
    elif page == "Methodology":
        page_methodology()


if __name__ == "__main__":
    main()
