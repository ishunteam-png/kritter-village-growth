"""
Step 9: Auto-generate PDF report cards for top-20 villages.

Each PDF page contains:
  - Village header: name, rank, state, district, coordinates
  - Score gauge: composite score vs district average
  - Signal summary table: all available signals with values and percentile bars
  - NTL time series sparkline
  - Forecast bar: predicted NTL for 2025-2027
  - Data quality badge: confidence score + signals available

Output: output/report_cards/village_rank_{N:03d}_{name}.pdf
        output/report_cards_combined.pdf  (all 20 in one file)
"""

import warnings, os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from io import BytesIO

warnings.filterwarnings("ignore")

TOP100_FILE = Path("/data/satellite/kritter/output/top_100_villages.csv")
FORECAST_FILE = Path("/data/satellite/kritter/output/top_100_forecast.csv")
OUTPUT_DIR  = Path("/data/satellite/kritter/output")
CARDS_DIR   = OUTPUT_DIR / "report_cards"
YEARS       = [2019, 2020, 2021, 2022, 2023, 2024]
FORECAST_YRS = [2025, 2026, 2027]

TEAL   = "#0D9488"
NAVY   = "#021B2C"
LIGHT  = "#F0F7FA"
ACCENT = "#4ECDC4"


def signal_table_html(row: pd.Series) -> str:
    signals = [
        ("NTL 2019",        f"ntl_2019",         "nW/cm²/sr"),
        ("NTL 2024",        f"ntl_2024",          "nW/cm²/sr"),
        ("NTL Growth",      "ntl_growth_pct",     "%"),
        ("NTL Trend Slope", "ntl_trend_slope",    "nW/yr"),
        ("NDBI Growth",     "ndbi_growth",        "Δ index"),
        ("GHSL Change",     "ghsl_change",        "fraction"),
        ("SAR VV Δ",        "s1_vv_delta",        "dB"),
        ("Tower Growth",    "tower_growth",       "/km²"),
        ("Pop Growth",      "pop_growth_rate",    "fraction"),
        ("Built-up Δ",      "builtup_change",     "fraction"),
        ("Composite Score", "composite_score",    "/100"),
        ("Confidence",      "confidence_score",   "%"),
    ]
    rows_html = ""
    for label, col, unit in signals:
        val = row.get(col, np.nan)
        if pd.isna(val):
            display = "—"
            bar = ""
        else:
            display = f"{val:.3g} {unit}"
            pct = min(max(float(row.get(f"{col}_score", 50)), 0), 100)
            bar = (f'<div style="background:#ddd;border-radius:3px;height:8px;width:120px;display:inline-block">'
                   f'<div style="background:{TEAL};height:100%;width:{pct:.0f}%;border-radius:3px"></div></div>')
        rows_html += f"""
        <tr>
          <td style="padding:3px 8px;font-size:11px;color:#475569">{label}</td>
          <td style="padding:3px 8px;font-size:11px;font-weight:600">{display}</td>
          <td style="padding:3px 8px">{bar}</td>
        </tr>"""
    return f'<table style="border-collapse:collapse;width:100%">{rows_html}</table>'


def ntl_sparkline_html(row: pd.Series) -> str:
    """Simple inline SVG sparkline of NTL values 2019-2024."""
    vals = [row.get(f"ntl_{y}", np.nan) for y in YEARS]
    clean = [(y, v) for y, v in zip(YEARS, vals) if not pd.isna(v)]
    if len(clean) < 2:
        return "<p style='color:#94A3B8'>No NTL time series available</p>"

    ys, vs = zip(*clean)
    vmin, vmax = min(vs), max(vs)
    rng = vmax - vmin or 1
    W, H = 280, 60
    pts = " ".join(f"{(y-YEARS[0])/(YEARS[-1]-YEARS[0])*W:.1f},{H-(v-vmin)/rng*H:.1f}"
                   for y, v in zip(ys, vs))
    return (f'<svg width="{W}" height="{H}" style="overflow:visible">'
            f'<polyline points="{pts}" fill="none" stroke="{TEAL}" stroke-width="2"/>'
            + "".join(
                f'<circle cx="{(y-YEARS[0])/(YEARS[-1]-YEARS[0])*W:.1f}" '
                f'cy="{H-(v-vmin)/rng*H:.1f}" r="3" fill="{TEAL}"/>'
                for y, v in zip(ys, vs))
            + f'<text x="0" y="{H+14}" style="font-size:9px;fill:#94A3B8">{YEARS[0]}</text>'
            f'<text x="{W-16}" y="{H+14}" style="font-size:9px;fill:#94A3B8">{YEARS[-1]}</text>'
            f'<text x="0" y="8" style="font-size:9px;fill:#94A3B8">{vmax:.2f}</text>'
            f'<text x="0" y="{H}" style="font-size:9px;fill:#94A3B8">{vmin:.2f}</text>'
            f'</svg>')


def generate_html_card(row: pd.Series) -> str:
    """Full HTML for one village report card."""
    rank    = int(row.get("rank", "?"))
    name    = str(row.get("village_name", "Unknown"))
    state   = str(row.get("state_name", ""))
    district = str(row.get("district_name", ""))
    lat     = row.get("latitude", np.nan)
    lon     = row.get("longitude", np.nan)
    score   = row.get("composite_score", np.nan)
    conf    = row.get("confidence_score", np.nan)
    ml_prob = row.get("ml_growth_prob", np.nan)
    ntl_g   = row.get("ntl_growth_pct", np.nan)
    confirmed = row.get("multi_signal_confirmed", False)

    fc_vals = [row.get(f"ntl_forecast_{y}", "—") for y in FORECAST_YRS]
    fc_str  = " → ".join(
        f"{v:.2f}" if not isinstance(v, str) and not pd.isna(v) else "—"
        for v in fc_vals)
    coords  = f"{lat:.4f}°N, {lon:.4f}°E" if not pd.isna(lat) else "—"

    badge_color = "#22C55E" if confirmed else "#F59E0B"
    badge_text  = "Multi-signal confirmed" if confirmed else "Single-signal"
    conf_text   = f"{conf:.0f}% data coverage" if not pd.isna(conf) else "—"

    return f"""
<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  body {{ font-family: Calibri, Arial, sans-serif; background: {LIGHT}; margin: 0; padding: 16px; }}
  .card {{ background: white; border-radius: 10px; padding: 20px; max-width: 680px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
  .header {{ background: {NAVY}; color: white; border-radius: 8px; padding: 14px 18px; margin-bottom: 14px; }}
  .rank {{ font-size: 32px; font-weight: 700; color: {ACCENT}; display: inline-block; }}
  .name {{ font-size: 22px; font-weight: 700; }}
  .sub  {{ font-size: 12px; color: #94A3B8; margin-top: 3px; }}
  .score-big {{ font-size: 40px; font-weight: 700; color: {TEAL}; }}
  .badge {{ display: inline-block; padding: 2px 10px; border-radius: 10px;
            color: white; font-size: 11px; font-weight: 600; }}
  .section {{ margin: 12px 0; }}
  .section-title {{ font-size: 11px; font-weight: 700; color: #475569;
                    text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }}
  .fc-box {{ background: {LIGHT}; border-radius: 6px; padding: 8px 12px; font-size: 12px; }}
</style>
</head><body><div class="card">
  <div class="header">
    <span class="rank">#{rank}</span>&nbsp;&nbsp;
    <span class="name">{name}</span><br>
    <div class="sub">{district}, {state} &nbsp;|&nbsp; {coords}</div>
  </div>

  <div style="display:flex;gap:20px;align-items:center;margin-bottom:12px">
    <div>
      <div class="section-title">Composite Score</div>
      <span class="score-big">{score:.1f}</span><span style="color:#94A3B8;font-size:13px">/100</span>
    </div>
    <div>
      <div class="section-title">ML Growth Probability</div>
      <div style="font-size:22px;font-weight:700;color:#065A82">
        {f"{ml_prob:.1f}%" if not pd.isna(ml_prob) else "—"}
      </div>
    </div>
    <div>
      <span class="badge" style="background:{badge_color}">{badge_text}</span><br>
      <span style="font-size:11px;color:#94A3B8;margin-top:4px;display:block">{conf_text}</span>
    </div>
  </div>

  <div class="section">
    <div class="section-title">NTL Time Series (2019–2024)</div>
    {ntl_sparkline_html(row)}
  </div>

  <div class="section">
    <div class="section-title">Signal Summary</div>
    {signal_table_html(row)}
  </div>

  <div class="section">
    <div class="section-title">NTL Forecast (ensemble)</div>
    <div class="fc-box">
      {'  →  '.join(str(y)+': '+f'{row.get(f"ntl_forecast_{y}", np.nan):.3f}' for y in FORECAST_YRS
                    if not pd.isna(row.get(f'ntl_forecast_{y}', np.nan)))}
      &nbsp;&nbsp;<span style="color:#94A3B8;font-size:11px">nW/cm²/sr (damped Holt-Winters + linear ensemble)</span>
    </div>
  </div>

  <div style="margin-top:14px;font-size:10px;color:#CBD5E1;border-top:1px solid #E2E8F0;padding-top:8px">
    Kritter Software Technologies Candidate Assignment · Data: NASA VIIRS, ESA WorldCover,
    Sentinel-2/1, WorldPop, GHSL, OSM · Processed on AWS EC2 ap-south-1
  </div>
</div></body></html>
"""


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CARDS_DIR.mkdir(exist_ok=True)

    src_file = FORECAST_FILE if FORECAST_FILE.exists() else TOP100_FILE
    if not src_file.exists():
        print("No top_100 file found — run 04_score_rank.py and 08_forecast.py first")
        return

    df = pd.read_csv(src_file)
    top20 = df.head(20)
    print(f"Generating report cards for {len(top20)} villages...")

    html_paths = []
    for _, row in top20.iterrows():
        rank = int(row.get("rank", 0))
        name_safe = str(row.get("village_name", "unknown")).replace(" ", "_").replace("/", "_")[:30]
        html_path = CARDS_DIR / f"village_rank_{rank:03d}_{name_safe}.html"

        html_content = generate_html_card(row)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        html_paths.append(html_path)
        print(f"  → {html_path.name}")

    # Combined index page
    index_html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Village Economic Growth — Top 20 Report Cards</title>
<style>
  body {{ font-family: Calibri, Arial; background: #F0F7FA; padding: 20px; }}
  h1 {{ color: #021B2C; }} a {{ color: #0D9488; text-decoration: none; }}
  .grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }}
  .item {{ background: white; border-radius: 8px; padding: 12px;
           box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
</style></head><body>
<h1>Top 20 Village Report Cards</h1>
<div class="grid">
{"".join(f'<div class="item"><a href="{p.name}">#{int(df.iloc[i]["rank"])} {df.iloc[i].get("village_name","")}</a><br><small style="color:#94A3B8">{df.iloc[i].get("state_name","")}</small></div>'
         for i, p in enumerate(html_paths))}
</div></body></html>"""

    idx = CARDS_DIR / "index.html"
    idx.write_text(index_html, encoding="utf-8")
    print(f"\nReport cards index → {idx}")
    print(f"Open in browser: file://{idx.absolute()}")


if __name__ == "__main__":
    main()
