"""
Step 8: Growth forecasting — project NTL for 2025–2027, identify rising stars.

Methods applied per village:
  1. Linear extrapolation   — extends ntl_trend_slope forward
  2. Holt-Winters (additive) — handles trend + dampening
  3. Ensemble average

Rising stars: villages currently ranked 100–1000 but with the steepest
trend slopes — likely to enter top 100 within 2–3 years.

Output:
  output/top_100_forecast.csv    — 2025-2027 NTL forecasts for top 100
  output/rising_stars.csv        — top 50 rising stars (currently outside top 100)
  output/chart_forecast.html     — interactive forecast chart
"""

import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

warnings.filterwarnings("ignore")

INPUT_SCORED = Path("/data/satellite/kritter/processed/village_scored.csv")
TOP100_FILE  = Path("/data/satellite/kritter/output/top_100_villages.csv")
OUTPUT_DIR   = Path("/data/satellite/kritter/output")
YEARS        = [2019, 2020, 2021, 2022, 2023, 2024]
FORECAST_YRS = [2025, 2026, 2027]


# ── Forecasting models ────────────────────────────────────────────────────────

def linear_forecast(ntl_series: np.ndarray, years_ahead: int) -> np.ndarray:
    """Extrapolate linear trend from the 6-year series."""
    x = np.array(YEARS)
    valid = ~np.isnan(ntl_series)
    if valid.sum() < 2:
        return np.full(years_ahead, np.nan)
    from scipy.stats import linregress
    m, b, *_ = linregress(x[valid], ntl_series[valid])
    future_x = np.array([YEARS[-1] + i for i in range(1, years_ahead + 1)])
    forecast = m * future_x + b
    return np.maximum(forecast, 0)


def holt_winters_forecast(ntl_series: np.ndarray, years_ahead: int,
                           alpha: float = 0.5, beta: float = 0.2,
                           phi: float = 0.9) -> np.ndarray:
    """
    Damped Holt-Winters additive trend — prevents explosive growth projection.
    phi < 1 dampens the trend over time.
    """
    y = ntl_series.copy()
    valid_mask = ~np.isnan(y)
    if valid_mask.sum() < 3:
        return np.full(years_ahead, np.nan)

    # Fill NaN with linear interpolation
    idx = np.arange(len(y))
    y[~valid_mask] = np.interp(idx[~valid_mask], idx[valid_mask], y[valid_mask])

    L = y[0]
    T = y[1] - y[0]
    forecasts = []
    for t in range(1, len(y)):
        L_new = alpha * y[t] + (1 - alpha) * (L + phi * T)
        T_new = beta * (L_new - L) + (1 - beta) * phi * T
        L, T = L_new, T_new

    phi_cum = 1.0
    for h in range(1, years_ahead + 1):
        phi_cum *= phi
        forecasts.append(max(L + phi_cum * T, 0))
    return np.array(forecasts)


def forecast_village(row: pd.Series) -> dict:
    ntl_vals = np.array([row.get(f"ntl_{y}", np.nan) for y in YEARS])
    lin  = linear_forecast(ntl_vals, len(FORECAST_YRS))
    hw   = holt_winters_forecast(ntl_vals, len(FORECAST_YRS))
    ens  = np.where(np.isnan(lin) | np.isnan(hw), np.nanmean([lin, hw], axis=0),
                    (lin + hw) / 2)
    result = {}
    for i, yr in enumerate(FORECAST_YRS):
        result[f"ntl_linear_{yr}"]  = round(float(lin[i]),  4) if not np.isnan(lin[i]) else np.nan
        result[f"ntl_hw_{yr}"]      = round(float(hw[i]),   4) if not np.isnan(hw[i])  else np.nan
        result[f"ntl_forecast_{yr}"] = round(float(ens[i]), 4) if not np.isnan(ens[i]) else np.nan
    return result


# ── Rising stars ──────────────────────────────────────────────────────────────

def find_rising_stars(df: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    """
    Villages ranked 100–2000 with steepest NTL trend slope.
    Projected to enter top 100 within 3 years if trend continues.
    """
    if "rank" not in df.columns:
        return pd.DataFrame()

    stars = df[(df["rank"] >= 100) & (df["rank"] <= 2000)].copy()
    if "ntl_trend_slope" in stars.columns:
        stars = stars.sort_values("ntl_trend_slope", ascending=False)

    # Project: will they enter top-100 composite territory?
    if "composite_score" in df.columns and len(df) > 100:
        top100_threshold = df["composite_score"].nlargest(100).min()
        if "ntl_trend_slope" in stars.columns:
            projected_score_bump = stars["ntl_trend_slope"].rank(pct=True) * 20
            stars["projected_score_2027"] = (stars.get("composite_score", 0)
                                              + projected_score_bump)
            stars["likely_top100_2027"] = stars["projected_score_2027"] > top100_threshold

    return stars.head(top_n)


# ── Charts ────────────────────────────────────────────────────────────────────

def chart_forecast(df_top100: pd.DataFrame, stars: pd.DataFrame) -> go.Figure:
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Top 10 — NTL Forecast 2025–2027",
                                        "Rising Stars (current rank 100–2000)"])

    colors = px.colors.qualitative.Safe
    top10 = df_top100.head(10)

    for i, (_, row) in enumerate(top10.iterrows()):
        hist_x = YEARS
        hist_y = [row.get(f"ntl_{y}", np.nan) for y in YEARS]
        fc_x   = FORECAST_YRS
        fc_y   = [row.get(f"ntl_forecast_{y}", np.nan) for y in FORECAST_YRS]
        name   = str(row.get("village_name", f"#{int(row.get('rank',i+1))}"))[:20]
        col    = colors[i % len(colors)]

        fig.add_trace(go.Scatter(x=hist_x + fc_x, y=hist_y + fc_y,
                                  mode="lines+markers", name=name,
                                  line=dict(color=col, dash="solid"),
                                  marker=dict(symbol=["circle"]*6 + ["diamond"]*3)),
                      row=1, col=1)

    if not stars.empty and "ntl_trend_slope" in stars.columns:
        top_stars = stars.head(20).sort_values("ntl_trend_slope")
        lbl = (top_stars.get("village_name", top_stars.get("rank", top_stars.index))
               .astype(str))
        fig.add_trace(go.Bar(x=top_stars["ntl_trend_slope"], y=lbl,
                              orientation="h", marker_color="#0D9488",
                              name="NTL trend slope"),
                      row=1, col=2)

    fig.update_layout(height=520, title="Village Growth Forecast — 2025–2027",
                      paper_bgcolor="#F0F7FA", plot_bgcolor="#E8F4F8",
                      showlegend=False)
    fig.update_xaxes(title_text="NTL (nW/cm²/sr)", row=1, col=1)
    fig.update_xaxes(title_text="NTL Trend Slope (nW/yr)", row=1, col=2)
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_SCORED.exists():
        print("village_scored.csv not found — run 04_score_rank.py first"); return

    print("Loading scored villages...")
    df_all  = pd.read_csv(INPUT_SCORED, low_memory=False)
    df_top  = pd.read_csv(TOP100_FILE) if TOP100_FILE.exists() else df_all.head(100)

    # Forecast for top 100
    print(f"Forecasting {len(df_top)} top villages...")
    fc_rows = []
    for _, row in df_top.iterrows():
        fc = forecast_village(row)
        fc_rows.append(fc)
    fc_df = pd.DataFrame(fc_rows)
    df_top = pd.concat([df_top.reset_index(drop=True), fc_df], axis=1)

    df_top.to_csv(OUTPUT_DIR / "top_100_forecast.csv", index=False)
    print(f"  → top_100_forecast.csv")

    # Rising stars from full scored set
    stars = find_rising_stars(df_all)
    if not stars.empty:
        stars.head(50).to_csv(OUTPUT_DIR / "rising_stars.csv", index=False)
        print(f"  → rising_stars.csv  ({len(stars)} rising stars)")

    # Chart
    fig = chart_forecast(df_top, stars)
    fig.write_html(str(OUTPUT_DIR / "chart_forecast.html"), include_plotlyjs="cdn")
    print(f"  → chart_forecast.html")

    print("\nForecast complete.")


if __name__ == "__main__":
    main()
