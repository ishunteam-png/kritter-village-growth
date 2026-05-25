"""
Step 10: Year-by-year NTL animation for top villages.

Outputs:
  output/animation_top20_ntl.html   — Plotly animated choropleth (open in browser)
  output/animation_top20_ntl.gif    — GIF (requires kaleido + pillow)
"""

import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

warnings.filterwarnings("ignore")

TOP100_FILE = Path("/data/satellite/kritter/output/top_100_villages.csv")
OUTPUT_DIR  = Path("/data/satellite/kritter/output")
YEARS       = [2019, 2020, 2021, 2022, 2023, 2024]

TEAL  = "#0D9488"
NAVY  = "#021B2C"
LIGHT = "#F0F7FA"


def build_long_df(df: pd.DataFrame) -> pd.DataFrame:
    """Melt top-100 into long format: one row per village per year."""
    rows = []
    for _, row in df.iterrows():
        for yr in YEARS:
            ntl = row.get(f"ntl_{yr}", np.nan)
            if pd.isna(ntl):
                continue
            rows.append({
                "year":         yr,
                "village_name": row.get("village_name", "Unknown"),
                "state_name":   row.get("state_name", ""),
                "district_name":row.get("district_name", ""),
                "latitude":     row.get("latitude", np.nan),
                "longitude":    row.get("longitude", np.nan),
                "rank":         int(row.get("rank", 0)),
                "ntl":          float(ntl),
                "composite_score": row.get("composite_score", np.nan),
            })
    return pd.DataFrame(rows)


def animated_bubble_map(long_df: pd.DataFrame) -> go.Figure:
    """Animated bubble map: bubble size = NTL value, color = rank."""
    long_df = long_df.dropna(subset=["latitude", "longitude", "ntl"])

    # Normalize NTL for bubble size (0–40 pixel range)
    ntl_max = long_df["ntl"].max() or 1
    long_df["bubble_size"] = (long_df["ntl"] / ntl_max * 35 + 5).clip(5, 40)

    long_df["hover"] = (
        long_df["village_name"] + "<br>"
        + "Rank: #" + long_df["rank"].astype(str) + "<br>"
        + "State: " + long_df["state_name"] + "<br>"
        + "NTL: " + long_df["ntl"].round(3).astype(str) + " nW/cm²/sr"
    )

    fig = px.scatter_geo(
        long_df,
        lat="latitude", lon="longitude",
        size="bubble_size",
        color="rank",
        color_continuous_scale=px.colors.sequential.Teal_r,
        hover_name="village_name",
        hover_data={"hover": True, "bubble_size": False, "rank": False,
                    "latitude": False, "longitude": False},
        animation_frame="year",
        title="Top 100 Village NTL Growth Animation (2019–2024)",
        scope="asia",
    )
    fig.update_geos(
        fitbounds="locations",
        visible=True,
        showcountries=True, countrycolor="#CBD5E1",
        showcoastlines=True, coastlinecolor="#CBD5E1",
        showland=True, landcolor="#F8FAFC",
        showocean=True, oceancolor="#E0F2FE",
        bgcolor=LIGHT,
    )
    fig.update_layout(
        paper_bgcolor=NAVY,
        font_color="white",
        title_font_size=16,
        coloraxis_colorbar=dict(title="Rank", tickfont_color="white", title_font_color="white"),
        geo=dict(bgcolor=LIGHT),
        sliders=[{
            "currentvalue": {"prefix": "Year: ", "font": {"color": "white", "size": 14}},
            "font": {"color": "white"},
        }],
        updatemenus=[{
            "buttons": [
                {"args": [None, {"frame": {"duration": 900, "redraw": True},
                                 "fromcurrent": True}],
                 "label": "▶ Play", "method": "animate"},
                {"args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}],
                 "label": "⏸ Pause", "method": "animate"},
            ],
            "bgcolor": TEAL, "bordercolor": "white",
            "font": {"color": "white", "size": 12},
            "direction": "left", "showactive": False, "type": "buttons",
            "x": 0.05, "y": -0.05,
        }],
        margin=dict(l=10, r=10, t=50, b=60),
    )
    return fig


def ntl_growth_bar_race(long_df: pd.DataFrame) -> go.Figure:
    """Bar chart race: top-10 villages by NTL each year."""
    frames = []
    for yr in YEARS:
        yr_df = long_df[long_df["year"] == yr].nlargest(10, "ntl")
        frames.append(go.Frame(
            name=str(yr),
            data=[go.Bar(
                x=yr_df["ntl"].values,
                y=yr_df["village_name"].values,
                orientation="h",
                marker_color=TEAL,
                text=yr_df["ntl"].round(3).astype(str) + " nW",
                textposition="outside",
                hovertemplate="%{y}<br>NTL: %{x:.3f}<extra></extra>",
            )],
            layout=go.Layout(title_text=f"Top 10 Villages by NTL — {yr}"),
        ))

    # Initial frame
    init_df = long_df[long_df["year"] == YEARS[0]].nlargest(10, "ntl")
    fig = go.Figure(
        data=[go.Bar(
            x=init_df["ntl"].values,
            y=init_df["village_name"].values,
            orientation="h",
            marker_color=TEAL,
        )],
        frames=frames,
        layout=go.Layout(
            title=f"Top 10 Villages by NTL — {YEARS[0]}",
            paper_bgcolor=NAVY,
            plot_bgcolor="#0A2942",
            font_color="white",
            xaxis=dict(title="NTL (nW/cm²/sr)", gridcolor="#1E4060"),
            yaxis=dict(autorange="reversed"),
            height=420,
            margin=dict(l=160, r=80, t=50, b=50),
            updatemenus=[{
                "buttons": [
                    {"args": [None, {"frame": {"duration": 1000, "redraw": True},
                                     "fromcurrent": True}],
                     "label": "▶ Play", "method": "animate"},
                    {"args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}],
                     "label": "⏸ Pause", "method": "animate"},
                ],
                "bgcolor": TEAL, "font": {"color": "white"},
                "direction": "left", "showactive": False, "type": "buttons",
                "x": 0.0, "y": -0.12,
            }],
            sliders=[{
                "steps": [{"args": [[f.name], {"frame": {"duration": 0}, "mode": "immediate"}],
                           "label": f.name, "method": "animate"} for f in frames],
                "currentvalue": {"prefix": "Year: ", "font": {"color": "white"}},
                "font": {"color": "white"},
            }],
        )
    )
    return fig


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not TOP100_FILE.exists():
        print(f"Missing {TOP100_FILE} — run 04_score_rank.py first")
        return

    df = pd.read_csv(TOP100_FILE)
    print(f"Loaded {len(df)} villages")

    long_df = build_long_df(df)
    print(f"Long format: {len(long_df)} rows ({len(long_df['year'].unique())} years)")

    # Animated bubble map
    print("Building animated bubble map...")
    fig_map = animated_bubble_map(long_df)
    map_path = OUTPUT_DIR / "animation_top100_map.html"
    fig_map.write_html(str(map_path), include_plotlyjs="cdn")
    print(f"  → {map_path}")

    # Bar chart race (top 10)
    print("Building bar chart race...")
    fig_bar = ntl_growth_bar_race(long_df)
    bar_path = OUTPUT_DIR / "animation_top10_bar_race.html"
    fig_bar.write_html(str(bar_path), include_plotlyjs="cdn")
    print(f"  → {bar_path}")

    # Combined HTML (both in one page)
    combined = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Village NTL Animation — Kritter</title>
<style>
  body {{ background: {NAVY}; font-family: Calibri, Arial; color: white; padding: 20px; }}
  h1 {{ color: #4ECDC4; }}
  .subtitle {{ color: #94A3B8; font-size: 14px; margin-bottom: 20px; }}
  .chart-section {{ margin-bottom: 30px; }}
</style>
</head>
<body>
<h1>Village Economic Growth — NTL Animation</h1>
<p class="subtitle">NASA VIIRS Night-Time Light | Top 100 Villages | India 2019–2024</p>
<div class="chart-section">
  <h2 style="color:#0D9488;font-size:15px">Animated Map (bubble size = NTL brightness)</h2>
  <iframe src="animation_top100_map.html" width="100%" height="520" frameborder="0"></iframe>
</div>
<div class="chart-section">
  <h2 style="color:#0D9488;font-size:15px">Bar Chart Race — Top 10 by NTL each year</h2>
  <iframe src="animation_top10_bar_race.html" width="100%" height="480" frameborder="0"></iframe>
</div>
<p style="color:#475569;font-size:11px;margin-top:30px">
  Kritter Software Technologies Candidate Assignment · Data: NASA VIIRS VNP46A4 v2
</p>
</body></html>"""

    combo_path = OUTPUT_DIR / "animation_combined.html"
    combo_path.write_text(combined, encoding="utf-8")
    print(f"\nCombined animation page → {combo_path}")
    print(f"Open: file://{combo_path.absolute()}")


if __name__ == "__main__":
    main()
