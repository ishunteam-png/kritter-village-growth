"""
Step 5: Generate all visualisations for the top 100 villages.

Outputs (all in /data/satellite/kritter/output/):
  map.html                   – interactive Folium map (main + dark-track villages)
  chart_top20.html           – bar chart of top 20 by composite score
  chart_state_dist.html      – state distribution bar chart
  chart_scatter.html         – NTL growth vs NDBI growth scatter
  chart_score_breakdown.html – stacked bar of score components (top 20)
  chart_signal_coverage.html – signal availability heatmap across top 100
  chart_dark_track.html      – dark-village top 50 (NTL-excluded villages)
"""

import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
from pathlib import Path

TOP100_FILE  = Path("/data/satellite/kritter/output/top_100_villages.csv")
DARK50_FILE  = Path("/data/satellite/kritter/output/dark_top_50_villages.csv")
OUTPUT_DIR   = Path("/data/satellite/kritter/output")
INDIA_CENTER = [20.5937, 78.9629]


def load():
    if not TOP100_FILE.exists():
        raise FileNotFoundError(f"{TOP100_FILE} not found. Run 04_score_rank.py first.")
    top = pd.read_csv(TOP100_FILE)
    dark = pd.read_csv(DARK50_FILE) if DARK50_FILE.exists() else pd.DataFrame()
    return top, dark


# ── Map ───────────────────────────────────────────────────────────────────────

def build_map(df: pd.DataFrame, dark: pd.DataFrame) -> folium.Map:
    m = folium.Map(location=INDIA_CENTER, zoom_start=5, tiles="CartoDB positron")

    lat_col = next((c for c in ["latitude", "lat"]  if c in df.columns), None)
    lon_col = next((c for c in ["longitude", "lon"] if c in df.columns), None)

    if lat_col and lon_col:
        cluster = MarkerCluster(name="Top 100 Villages (main)").add_to(m)
        smin, smax = df["composite_score"].min(), df["composite_score"].max()

        for _, row in df.iterrows():
            if pd.isna(row.get(lat_col)):
                continue
            t = (row["composite_score"] - smin) / (smax - smin + 1e-9)
            colour = f"#{int(255*(1-t)):02x}{int(180+75*t):02x}00"
            vname = row.get("village_name", f"#{int(row['rank'])}")
            ndbi_line = (f"NDBI growth: <b>{row.get('ndbi_growth', 0):.4f}</b><br>"
                         if "ndbi_growth" in df.columns else "")
            popup = (
                f"<b>#{int(row['rank'])} {vname}</b><br>"
                f"{row.get('district_name','')}, {row.get('state_name','')}<br>"
                f"<hr style='margin:3px 0'>"
                f"Composite Score: <b>{row['composite_score']:.1f}</b><br>"
                f"NTL Growth: <b>{row.get('ntl_growth_pct', 0):.1f}%</b><br>"
                + ndbi_line +
                f"Built-up Change: <b>{row.get('builtup_change', 0):.4f}</b>"
            )
            folium.CircleMarker(
                location=[row[lat_col], row[lon_col]],
                radius=5 + (100 - row["rank"]) / 20,
                color=colour, fill=True, fill_opacity=0.85,
                popup=folium.Popup(popup, max_width=280),
                tooltip=f"#{int(row['rank'])} {vname}",
            ).add_to(cluster)

    # Dark-track villages in a separate layer
    if not dark.empty and lat_col and lon_col:
        dark_layer = folium.FeatureGroup(name="Dark-track Top 50").add_to(m)
        for _, row in dark.iterrows():
            if pd.isna(row.get(lat_col)):
                continue
            vname = row.get("village_name", f"#{int(row['rank'])}")
            folium.CircleMarker(
                location=[row[lat_col], row[lon_col]],
                radius=4,
                color="#6366F1", fill=True, fill_opacity=0.7,
                popup=folium.Popup(
                    f"<b>Dark-track #{int(row['rank'])} {vname}</b><br>"
                    f"{row.get('state_name','')}", max_width=240),
                tooltip=f"Dark #{int(row['rank'])} {vname}",
            ).add_to(dark_layer)

    folium.LayerControl().add_to(m)
    return m


# ── Charts ────────────────────────────────────────────────────────────────────

def chart_top20(df):
    top20 = df.head(20).copy()
    top20["label"] = (top20.get("village_name", top20["rank"]).astype(str)
                      + " (" + top20.get("state_name", "").astype(str) + ")")
    fig = px.bar(
        top20.sort_values("composite_score"),
        x="composite_score", y="label", orientation="h",
        color="composite_score", color_continuous_scale="Teal",
        title="Top 20 Villages — Composite Economic Growth Score",
        labels={"composite_score": "Score", "label": "Village"},
    )
    fig.update_layout(coloraxis_showscale=False, height=620,
                      paper_bgcolor="#F0F7FA", plot_bgcolor="#E8F4F8")
    return fig


def chart_state_dist(df):
    if "state_name" not in df.columns:
        return go.Figure()
    counts = df["state_name"].value_counts().reset_index()
    counts.columns = ["state", "count"]
    fig = px.bar(counts, x="state", y="count", color="count",
                 color_continuous_scale="Teal",
                 title="State Distribution of Top 100 Villages")
    fig.update_layout(xaxis_tickangle=-40, coloraxis_showscale=False,
                      paper_bgcolor="#F0F7FA", plot_bgcolor="#E8F4F8")
    return fig


def chart_scatter(df):
    x = "ntl_growth_pct"
    # Prefer NDBI growth if available, else builtup_change
    if "ndbi_growth" in df.columns and df["ndbi_growth"].notna().sum() > 20:
        y, ylabel = "ndbi_growth", "NDBI Growth 2019→2024"
    elif "builtup_change" in df.columns:
        y, ylabel = "builtup_change", "Built-up Change 2020→2021"
    else:
        y, ylabel = "ntl_trend_slope", "NTL Trend Slope"

    if x not in df.columns:
        return go.Figure()

    hover = "village_name" if "village_name" in df.columns else "rank"
    fig = px.scatter(
        df, x=x, y=y,
        size="composite_score", color="composite_score",
        color_continuous_scale="Teal", hover_name=hover,
        title=f"NTL Growth vs {ylabel} — Top 100 Villages",
        labels={x: "NTL Growth 2019→2024 (%)", y: ylabel},
    )
    fig.update_traces(marker=dict(sizemin=5, opacity=0.85))
    fig.update_layout(paper_bgcolor="#F0F7FA", plot_bgcolor="#E8F4F8")
    return fig


def chart_score_breakdown(df):
    top20 = df.head(20).sort_values("composite_score")
    labels = top20.get("village_name", top20["rank"]).astype(str)

    # Map old or new column names
    components = [
        ("ntl_growth_log",        "ntl_growth_log_score",   "NTL Growth",   "#2196F3"),
        ("ntl_trend_slope",       "ntl_trend_slope_score",  "NTL Trend",    "#9C27B0"),
        ("ntl_2024",              "ntl_2024_score",          "NTL Level",    "#FF9800"),
        ("ndbi_growth",           "ndbi_growth_score",       "NDBI Growth",  "#0D9488"),
        ("ndbi_trend_slope",      "ndbi_trend_slope_score",  "NDBI Trend",   "#14B8A6"),
        ("ndvi_trend_slope",      "ndvi_trend_slope_score",  "NDVI Trend",   "#22C55E"),
        ("builtup_change",        "builtup_change_score",    "Built-up",     "#4CAF50"),
        ("pop_growth_rate",       "pop_growth_score",        "Population",   "#F59E0B"),
    ]

    fig = go.Figure()
    for col_a, col_b, name, colour in components:
        col = col_b if col_b in top20.columns else (col_a if col_a in top20.columns else None)
        if col:
            fig.add_trace(go.Bar(
                name=name, y=labels, x=top20[col],
                orientation="h", marker_color=colour,
            ))

    fig.update_layout(
        barmode="stack",
        title="Score Component Breakdown — Top 20 Villages",
        xaxis_title="Component Score (rank-percentile 0–100)",
        height=640, legend=dict(orientation="h", y=1.05),
        paper_bgcolor="#F0F7FA", plot_bgcolor="#E8F4F8",
    )
    return fig


def chart_signal_coverage(df):
    """Heatmap showing which signals are available for the top 100 villages."""
    signal_cols = [c for c in [
        "ntl_2019", "ntl_2024", "ndbi_2019", "ndbi_2024",
        "ndvi_2019", "builtup_change", "pop_growth_rate",
    ] if c in df.columns]

    if not signal_cols:
        return go.Figure()

    presence = df[signal_cols].notna().astype(int).values
    labels_y = df.get("village_name", df["rank"]).astype(str).values[:50]
    presence  = presence[:50]

    fig = go.Figure(go.Heatmap(
        z=presence,
        x=[c.replace("_", " ") for c in signal_cols],
        y=labels_y,
        colorscale=[[0, "#E2E8F0"], [1, "#0D9488"]],
        showscale=False,
        xgap=2, ygap=1,
    ))
    fig.update_layout(
        title="Signal Coverage — Top 50 Villages",
        height=800, margin=dict(l=180),
        paper_bgcolor="#F0F7FA",
    )
    return fig


def chart_dark_track(dark):
    if dark.empty:
        return go.Figure()
    top20d = dark.head(20).copy()
    top20d["label"] = (top20d.get("village_name", top20d["rank"]).astype(str)
                       + " (" + top20d.get("state_name", "").astype(str) + ")")
    fig = px.bar(
        top20d.sort_values("composite_score"),
        x="composite_score", y="label", orientation="h",
        color="composite_score", color_continuous_scale="Purples",
        title="Top 20 Dark-Track Villages (NTL < 0.1 — scored on NDBI + Population)",
        labels={"composite_score": "Score", "label": "Village"},
    )
    fig.update_layout(coloraxis_showscale=False, height=620,
                      paper_bgcolor="#F0F7FA", plot_bgcolor="#E8F4F8")
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df, dark = load()
    print(f"Loaded {len(df)} main-track villages, {len(dark)} dark-track villages")

    print("Building map...")
    m = build_map(df, dark)
    m.save(str(OUTPUT_DIR / "map.html"))
    print(f"  → map.html")

    charts = [
        ("chart_top20.html",            chart_top20(df)),
        ("chart_state_dist.html",       chart_state_dist(df)),
        ("chart_scatter.html",          chart_scatter(df)),
        ("chart_score_breakdown.html",  chart_score_breakdown(df)),
        ("chart_signal_coverage.html",  chart_signal_coverage(df)),
        ("chart_dark_track.html",       chart_dark_track(dark)),
    ]
    for name, fig in charts:
        if fig.data:
            fig.write_html(str(OUTPUT_DIR / name), include_plotlyjs="cdn")
            print(f"  → {name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
