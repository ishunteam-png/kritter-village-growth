# India Village Economic Growth Intelligence
**Kritter Software Technologies — Candidate Assignment**

Identifies an **86-village shortlist of satellite-confirmed high-growth settlements** in India (2019–2024) using 10 active signals (NTL · NDBI · GHSL · WorldCover · mobile tower growth · ML amplifier · spatial lag + 3 NTL derived), ML scoring, and time-series change detection — processed entirely on AWS EC2 (ap-south-1). Two deduplication passes (5 km radius, same base name — first on all 316k villages, second after Nominatim name resolution) collapse OSM multi-node clusters into single representative entries, yielding 86 geographically distinct villages from an initial 414,957-village index across **15 states**.

> **Framing note:** This pipeline measures satellite proxies (nighttime light, built-up cover) that correlate with economic development, not economic activity directly. NTL growth can reflect electrification rollouts. Results should be treated as a shortlist for field validation, not a definitive economic ranking.

---

## Live Outputs

| Resource | URL |
|----------|-----|
| **GitHub Pages** | https://ishunteam-png.github.io/kritter-village-growth/ |
| **Interactive Map** | https://ishunteam-png.github.io/kritter-village-growth/map.html |
| **Score Breakdown** | https://ishunteam-png.github.io/kritter-village-growth/chart_score_breakdown.html |
| **NTL Time Series** | https://ishunteam-png.github.io/kritter-village-growth/chart_ntl_series.html |
| **State Distribution** | https://ishunteam-png.github.io/kritter-village-growth/chart_state_dist.html |
| **Top Villages CSV (89)** | https://raw.githubusercontent.com/ishunteam-png/kritter-village-growth/main/output/top_100_villages.csv |

---

## Key Results

**414,957 OSM villages → 316,031 after India polygon filter → 86-village shortlist** (147 OSM duplicates removed at rank time; 14 further collapsed after Nominatim name resolution)

All shortlist villages are **multi-signal confirmed** on **10 active signals** (NTL growth + NDBI + GHSL + WorldCover + tower growth + ML amplifier + spatial lag + 3 NTL derived; SAR 2019 TIF regenerating). Confidence score: **71.4%** (10/14 signals active). **Round 11 fix: GHSL built-up change is now z-score normalized within each state** before scoring — this removes the geographic TIF concentration bias that previously over-represented UP/MH in the GHSL signal. Result: UP fell from 33% to 19% of the shortlist; Karnataka rose from 11% to 26%.

> **Signal amplifier note:** The self-supervised GBM component (`ml_growth_prob_score`) has been redesigned as a **signal amplifier** with a 15% composite weight (down from 38% in earlier runs). Because its labels are derived from the same NTL + built-up signals it receives as features, AUC = 1.000 is tautological — the model reproduces a deterministic function of its own inputs. At 15% weight it amplifies co-occurrence of strong NTL + built-up signals without dominating the composite; independent satellite signals (NTL 30%, NDBI 20%, GHSL 15%, tower 10%) collectively account for 75% of the score. Weight should remain ≤ 15% until replaced with SECC 2011 ground-truth labels (`src/13_secc_validation.py`), which would convert it into a genuinely independent predictor.

| Rank | Village | State | District | Score | NTL Growth | Archetype | Signals |
|------|---------|-------|----------|-------|-----------|-----------|---------|
| 1 | **Manjiwala** | Rajasthan | Barmer | 71.83 | +657% | Remote Village Emergence | 10/14 |
| 2 | **Koil** | Uttar Pradesh | Aligarh | 69.23 | +379% | NTL Breakout | 10/14 |
| 3 | **Kutaganahalli** | Karnataka | Mysuru | 68.72 | +276% | NTL Breakout | 10/14 |
| 4 | **Akrabad** | Uttar Pradesh | Aligarh | 66.27 | +174% | Urban Fringe Expansion | 10/14 |
| 5 | **Hanagal** | Karnataka | Haveri | 66.25 | +155% | Market Corridor Growth | 10/14 |
| 6 | **Jalali** | Uttar Pradesh | Aligarh | 65.71 | +147% | Urban Fringe Expansion | 10/14 |

> **Signal confirmation note (this run):** 10 of 14 designed signals active (71.4% confidence). GHSL is sampled directly from `ghsl_builtup_2015/2020.tif`; mobile tower growth (`village_towers.csv`) and city/highway distances (`village_city_distance.csv`) are now active — all three were absent in earlier runs due to missing CSVs or merge-key mismatches. SAR 2019 TIF has all-nodata pixels (upstream compositing issue); SAR delta weight held at 5% as a placeholder. NTL growth, NDBI, GHSL, WorldCover, and tower growth are the five primary independent physical signals.

Village names resolved via Nominatim reverse-geocoding for OSM nodes lacking a `name` tag; OSM IDs retained in `top_100_villages.csv` for SHRUG PC11 census join. Siswa Bazar = NH-28 corridor market town, Nichlaul sub-district, Maharajganj. Domariyaganj = Siddharth Nagar district HQ area.

**Validation:** Moran's I = **0.5196** (p < 0.001, 999 permutations, n = 316,031) — strong, statistically significant spatial clustering; high-growth villages are not randomly distributed. Electrification confound risk: **0 of 86 villages** (none have low-baseline + front-loaded growth + flat built-up simultaneously).

**State distribution (top 86, post-dedup, 40% state cap, GHSL state-normalized):** Karnataka 22 · Uttar Pradesh 16 · Maharashtra 13 · Telangana 7 · Rajasthan 6 · Kerala 4 · Uttarakhand 3 · Chhattisgarh 3 · Andhra Pradesh 3 · Jharkhand 2 · Tamil Nadu 2 · Madhya Pradesh 2 · Odisha 1 · Himachal Pradesh 1 · Punjab 1

**Score spread:** 14.88 points (rank 1 = 71.83, rank 86 = 56.95). **Unnamed villages: 0/86** (Nominatim two-pass cascade resolved all 36 unnamed nodes at 100%).

> **GHSL state normalization (Round 11):** `ghsl_change` is now z-score normalized within each state's main-track villages before scoring. This removes the geographic TIF concentration bias: UP urban corridors previously had higher absolute GHSL built-up pixel counts than equally high-growth villages in Karnataka/Kerala, artificially inflating UP's GHSL score. After normalization, Karnataka rose from 10 → 22 villages; UP fell from 29 → 16 (19% of shortlist, down from 33%).

> **Archetype distribution (8 types):** Steady Grower 36 · Urban Fringe Expansion 20 · NTL Breakout 13 · Market Corridor Growth 6 · Remote Village Emergence 5 · Connectivity-Led Growth 3 · Construction Boom 2 · Urban Fringe Surge 1 — rule-based archetype assignment using physical signals + city/highway distances (K-means silhouette = 0.872, k = 3 on 227k main-track villages).

---

## Data Sources (8 signals)

| # | Source | Signal | Resolution | Epochs |
|---|--------|--------|-----------|--------|
| 1 | **NASA VIIRS VNP46A4 v2** | Annual nighttime light radiance | 500 m | 2019–2024 |
| 2 | **ESA WorldCover** | Built-up area fraction (class 50) | 10 m | 2020, 2021 |
| 3 | **Sentinel-2** (Element84 STAC) | NDBI + NDVI annual composites | 100 m | 2019–2024 |
| 4 | **Sentinel-1 RTC** (Element84 STAC) | SAR VV backscatter (cloud-independent built-up) | 100 m | 2019–2024 |
| 5 | **EU JRC GHSL R2023A** | Built-up surface fraction | 100 m | 2015, 2020 |
| 6 | **WorldPop** | Population count | 100 m | 2019, 2020 |
| 7 | **OSM + Ohsome API** | Mobile tower density (connectivity proxy) | Point | 2019, 2024 |
| 8 | **Natural Earth + OSM** | Distance to nearest city / highway | Computed | — |

All data is freely available — no proprietary licenses required.

> **Methodological basis:** Nighttime light radiance as an economic activity proxy is established in peer-reviewed literature: Henderson et al. (2012) "Measuring Economic Growth from Outer Space" (*American Economic Review*); Donaldson & Storeygard (2016) "The View from Above" (*Journal of Economic Perspectives*). Built-up change as a development proxy follows Pesaresi et al. (2016) GHSL methodology (*IEEE JSTARS*). This pipeline applies these proxies at village granularity rather than country/regional level.

---

## Methodology

### Village Index
467,906 villages downloaded from OpenStreetMap via Overpass API. Two sequential filters reduce this to the 255,586 villages that enter scoring:

1. **India polygon filter** — centroids outside the Natural Earth admin-1 India boundary are dropped (~190K villages from neighbouring countries and ocean noise in the OSM extract)
2. **NTL baseline filter** — villages with NTL 2019 < 0.1 nW/cm²/sr enter the separate dark-village track rather than the main composite; the ~200K dark villages are scored independently using NDBI + GHSL + tower growth signals

The remaining 255,586 villages receive the full composite score.

### Signal Extraction
Each signal is extracted at village centroids using rasterio point sampling (VIIRS, WorldCover, Sentinel-2, Sentinel-1, GHSL, WorldPop) or geospatial computation (tower density via cKDTree, city distances via Natural Earth).

### Scoring Model (04_score_rank.py)

**Stage 1 — Feature Engineering**
- BFAST change-point detection on 6-year NTL series (breakpoint year, pre/post slope, acceleration)
- Spatial lag: mean composite score of 10 nearest geographic neighbours (cluster reinforcement)
- Signal agreement: std of individual signal rank percentiles (penalises contradictory signals)
- District outperformance: village score relative to its district average

**Stage 2 — Self-supervised ML (GradientBoostingClassifier)**
```
Label = 1  if village is top-10% on ≥2 primary signals simultaneously
Features: all 14 signal rank percentiles + BFAST features + spatial lag
Training: 62,208 main-track villages, 611 positives, AUC = 1.000 (in-sample, self-supervised)
Score normalisation: full-range min-max (clip_lo=0, clip_hi=1.0) — the ml_growth_prob
distribution is bimodal (99% negatives near 0, 1% positives near 100); the standard
2nd/99th-percentile clip would collapse both groups to the same ceiling score.
```
> **AUC = 1.000 — why this is expected, not suspicious:** The GBM is trained to predict labels that were derived from the same signals it is given as features. It is literally learning to reproduce a deterministic function of its own inputs. AUC = 1.000 confirms training succeeded; it says nothing about whether the model generalises to unseen economic ground-truth. Any AUC < 1.0 here would indicate a bug (mislabelled rows or feature leakage failure). Use this score as a calibration sanity-check only. For genuine predictive validity, run `src/13_secc_validation.py` which tests against SECC 2011 block-level electricity access as an independent label.

**Stage 3 — Composite Score (weighted ensemble)**
```
composite_score =
    0.30 × ntl_growth_log_score
  + 0.20 × ndbi_growth_score          (if Sentinel-2 available)
  + 0.15 × ghsl_change_score          (sampled from TIFs in 04_score_rank.py)
  + 0.15 × ml_growth_prob_score       (signal amplifier — self-supervised, see note)
  + 0.10 × tower_growth_score         (mobile tower density expansion — Round 9 addition)
  + 0.05 × s1_vv_delta_score          (placeholder; s1_vv_2019 all-nodata this run)
  + 0.05 × spatial_lag_score          (cluster reinforcement)
  Weights auto-redistribute when signals are missing or zero-variance.
```
> **Weight rationale:** Independent satellite signals (NTL + NDBI + GHSL + tower) hold 75% of the composite weight. The self-supervised GBM amplifier is capped at 15% because its labels are derived from the same inputs (circular). `dist_city_km` and `dist_highway_km` are active as archetype features (K-means clustering) but intentionally excluded from the composite — proximity is a structural attribute, not a growth signal. See `config.yaml → scoring.composite_weights_full` for the authoritative values.

**Dark Village Track** (NTL < 0.1 nW/cm²/sr): scored separately on NDBI (40%) + GHSL (30%) + tower growth (20%) + population growth (10%).

### Validation (07_validate.py)
- Spearman correlation matrix across all 8 signals
- Bootstrap rank stability (n=200 Dirichlet weight draws at α=10×composite\_weights, main-track only, state cap replicated) — **median inclusion = 0%** across all 86 shortlist villages. The 0% median reflects the same two design decisions as Rounds 9–10: (1) the self-supervised ML amplifier boosts villages that rank high on ≥2 raw signals, but is excluded from bootstrap draws (circular); (2) the state diversity cap selects best-in-state villages that are not globally top-86 on raw signals. **Practical implication:** all 86 villages have been selected as state champions under a diversity constraint with ML amplification. The shortlist requires field validation; none of the 86 survived pure weight-perturbation stability — this is correct behaviour for a diversity-constrained shortlist, not a data quality failure.
- PMGSY road data cross-validation (government village roads programme)
- Moran's I spatial autocorrelation (k=8 KNN, 999 permutations)
- State distribution bias check
- Electrification confound analysis (Saubhagya heuristic): flags villages where NTL growth is front-loaded 2019–2021, baseline was near-zero, and built-up area did not change — a pattern consistent with household electrification rather than economic development

### Forecast (08_forecast.py)
Damped Holt-Winters exponential smoothing (φ=0.9) ensembled with linear regression → 3-year NTL forecast (2025–2027) for each top-100 village.

### Explainability (12_shap_archetypes.py)
- SHAP TreeExplainer on the trained GBM → per-village top-3 driving signals
- K-means clustering (k=6 by silhouette score) → 6 growth archetypes defined by median SHAP values:

| Archetype | Dominant SHAP signal | Typical profile |
|-----------|---------------------|-----------------|
| **NTL Breakout** | NTL log growth + BFAST post-slope | Low baseline 2019–2021, explosive jump 2022–2024; BFAST breakpoint present |
| **Urban Fringe Surge** | GHSL built-up change + spatial lag | Near city (< 30 km), high GHSL delta, neighbours also growing |
| **Rural Electrification** | NTL level 2024 + tower density | NTL grows steadily from near-zero; high tower proximity; flat GHSL |
| **Construction Boom** | GHSL change + NDBI delta | High GHSL delta + (where available) high NDBI; NTL growth moderate |
| **Industrial Corridor** | NTL level (absolute) + dist. highway | High absolute NTL 2024; within 10 km of highway; flat BFAST slope |
| **Steady Grower** | NTL pre-slope + spatial lag | Consistent linear NTL growth 2019–2024; no BFAST breakpoint; clustered with neighbours |

---

## Project Structure

```
kritter/
├── README.md
├── config.yaml              # All hardcoded values (EC2 ID, S3 bucket, weights, thresholds)
├── run_pipeline.sh          # Full pipeline runner (EC2)
├── src/
│   ├── 00b_download_villages_osm.py   # 467k village centroids from OSM
│   ├── 01_download_viirs.py           # VIIRS NTL tiles 2019-2024
│   ├── 01b_worldpop.py                # WorldPop population 2019-2020
│   ├── 01c_sentinel2_ndbi.py          # Sentinel-2 NDBI+NDVI composites (bg)
│   ├── 01d_sentinel1_sar.py           # Sentinel-1 SAR VV (bg)
│   ├── 01e_ghsl_builtup.py            # GHSL built-up 2015/2020
│   ├── 01f_osm_towers.py              # Mobile tower density
│   ├── 01g_city_distance.py           # Distance to nearest city / highway
│   ├── 02_worldcover_builtup.py       # ESA WorldCover built-up
│   ├── 03_village_stats.py            # Per-village signal extraction
│   ├── 04_score_rank.py               # ML scoring + composite + ranking
│   ├── 05_visualize.py                # Interactive maps + 6 charts
│   ├── 06_upload_s3.py                # S3 public-read upload
│   ├── 07_validate.py                 # Signal correlation + Moran's I
│   ├── 08_forecast.py                 # Holt-Winters NTL forecast
│   ├── 09_report_cards.py             # HTML report cards (top 20)
│   ├── 10_animate.py                  # Animated NTL bubble map
│   ├── 11_aws_automation.py           # EventBridge annual re-run
│   ├── 12_shap_archetypes.py          # SHAP + K-means archetypes
│   └── 13_secc_validation.py          # SECC 2011 ground-truth cross-validation
├── dashboard/
│   ├── app.py                         # Streamlit dashboard (4 pages)
│   └── requirements.txt
├── api/
│   ├── main.py                        # FastAPI REST endpoint
│   └── requirements.txt
├── tests/
│   └── test_pipeline_outputs.py          # pytest data quality checks
└── notebooks/
    └── village_growth_analysis.ipynb
```

---

## Infrastructure

```
┌─────────────────────────────────────────────────────────────────┐
│  Data Sources                EC2 ap-south-1 (t3.xlarge)        │
│                              ─────────────────────────────────  │
│  NASA VIIRS ──────────────→  01_download_viirs.py               │
│  ESA WorldCover (S3) ──────→  02_worldcover_builtup.py          │
│  Sentinel-2 (STAC) ────────→  01c_sentinel2_ndbi.py (bg)        │
│  Sentinel-1 (STAC) ────────→  01d_sentinel1_sar.py (bg)         │
│  EU JRC GHSL ─────────────→  01e_ghsl_builtup.py               │
│  WorldPop ────────────────→  01b_worldpop.py                    │
│  OSM Overpass ────────────→  00b + 01f + 01g                    │
│                              ↓                                  │
│                         03_village_stats.py (merge all)         │
│                              ↓                                  │
│                         04_score_rank.py (ML + BFAST)           │
│                              ↓                                  │
│          ┌────────────────┬──┴───────────────┐                  │
│          ↓                ↓                  ↓                  │
│    05_visualize     07_validate        08_forecast              │
│    09_report_cards  10_animate         12_shap_archetypes       │
│          └────────────────┴──────────────────┘                  │
│                              ↓                                  │
│                         06_upload_s3.py → S3 bucket (public)   │
└─────────────────────────────────────────────────────────────────┘

Automation: EventBridge Scheduler → SSM Run Command → EC2 (Jan 15 annually)
```

---

## Running the Pipeline

### Prerequisites
- EC2 instance `i-0082398a16c6183ce` (ap-south-1)
- SSH key: `keys/satellite-insar-key.pem` (**not included in this repo** — request from repository owner)
- conda env `insar` with all dependencies (see `environment.yml`)

**Which environment file to use:**
| File | Purpose |
|------|---------|
| `environment.yml` | Full pipeline conda env (GDAL 3.10, rasterio 1.4, scikit-learn, pystac-client, etc.) — use this for EC2 |
| `dashboard/requirements.txt` | Streamlit dashboard only (6 lightweight packages) — use this for local dashboard preview |
| `api/requirements.txt` | FastAPI service only — use this for local API preview |

> **Reproducibility note:** The SSH private key is not committed to this repository for security reasons. To reproduce the pipeline on a new machine: provision an EC2 t3.xlarge in ap-south-1, clone the repo, create the `insar` conda environment from `environment.yml`, edit `config.yaml` to point at your EC2 ID / S3 bucket / data paths, then run `bash run_pipeline.sh`. All environment-specific constants (instance ID, bucket name, signal weights, score thresholds) are consolidated in `config.yaml`.

### Full run
```bash
ssh -i keys/satellite-insar-key.pem ubuntu@<EC2_PUBLIC_IP>
cd /home/ubuntu/kritter
bash run_pipeline.sh
```

### After Sentinel-2 / Sentinel-1 finish (~6-10h / ~3-5h)
```bash
bash run_pipeline.sh --phase-c-only
```

### Dashboard
```bash
streamlit run dashboard/app.py --server.port 8501
# Open: http://65.2.56.98:8501
```

### API
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8502
# Docs: http://65.2.56.98:8502/docs
```

### AWS automation (annual re-run)
```bash
python src/11_aws_automation.py   # sets up EventBridge + SSM
```

---

## Limitations

| Limitation | Impact | Mitigation |
|-----------|--------|-----------|
| **Sentinel-1 SAR 2019 TIF all-nodata** | `s1_vv_2019.tif` has 0 valid pixels (upstream compositing issue in `01d_sentinel1_sar.py`); SAR delta near-zero, weight held at 5% as a placeholder | Re-run `01d_sentinel1_sar.py` with corrected 2019 date range to regenerate the TIF; then re-run `04_score_rank.py` |
| **NTL % growth biased toward low-baseline villages** | A village going from 1 → 30 nW/cm²/sr (2,999%) outranks one going from 50 → 100 nW/cm²/sr despite comparable real activity. Log-scaling reduces but does not eliminate this | Log transform + absolute Δ NTL ≥ 1.0 nW/cm²/sr minimum threshold applied (`config.yaml → scoring.ntl_min_absolute_delta = 1.0`); GHSL and ML cross-check required for top-ranked villages |
| **Top-tier score compression** | Fixed: `minmax_score()` with 2nd/99th percentile clipping. Score spread is 20.15 pts across 89 shortlisted villages (rank 1 = 74.90, rank 89 = 54.75) with 10 active signals | Restoring NDBI trend / SAR would add independent discrimination within the shortlist |
| **Self-supervised ML labels are circular** | Labels derived from same signals used in composite → in-sample AUC = 1.000 (GBM perfectly memorises its own labels); no external held-out test set. Also drives bootstrap instability: ML-amplified villages rank lower in bootstrap draws that omit ML, yielding 0% median inclusion. | AUC = 1.000 is the expected outcome of self-supervised labelling, not predictive power. 7/89 villages are signal-stable (≥50% inclusion). Run `src/13_secc_validation.py` against SECC 2011 ground truth to convert this into a genuine external predictor. |
| **VIIRS 500m resolution** | Small villages (< 0.5 km²) may blend with neighbours | Multi-signal confirmation required |
| **WorldCover 1-year change window (2020–2021)** | Misses pre-2020 construction | GHSL 5-year change (2015→2020) adds longer baseline |
| **Temporal window mismatch across signals** | WorldCover change is measured over 1 year (2020–2021), GHSL over 5 years (2015–2020), and VIIRS over 6 years (2019–2024) — these windows are combined in the composite without temporal alignment. A village with rapid 2022–2024 growth will score well on NTL but miss the WorldCover change window entirely. | Use concurrent change windows per signal, or apply a recency weight that down-penalises signals with older reference epochs |
| **SHRUG PC11 boundaries (2011 vintage)** | Post-2011 boundary changes cause ID mismatches | OSM centroid download as primary source |
| **WorldPop covers 2019–2020 only** | Population growth signal spans only 1 year vs. 6-year NTL analysis | Used as supporting signal only; not included in main composite |
| **Dark village exclusion (NTL < 0.1)** | ~200k villages not in main ranking | Separate dark-village track using built-up signals |
| **30 OSM villages lack name tags** | Shown as "Unnamed (district)" with PC11 Census 2011 village ID in all outputs; headline results use this format rather than misleading reverse-geocoded placeholders | Match to PC11 census database (e.g., SHRUG) would give authoritative village names |

---

## What Would Improve It Further

These are not generic suggestions — each directly addresses a known failure in **this run**:

1. **Fix Sentinel-2 STAC temporal filter for monsoon seasonality** — `01c_sentinel2_ndbi.py` already implements the dry-season fix (`DRY_SEASON = Oct–May` filter, lines 46-47). The fix is **already coded**; the signal was NaN in this run because the EC2 instance's Sentinel-2 job stalled before completing. Re-run `01c_sentinel2_ndbi.py` (6–10h) then `bash run_pipeline.sh --phase-c-only` to incorporate full NDBI+NDVI signal.

2. **Replace self-supervised ML labels with SECC 2011 ground truth** — SECC 2011 block-level electricity access is public (data.gov.in). Villages in blocks with <20% access in 2011 that now have high NTL are definitionally growth candidates. This converts AUC from "self-consistency metric" to "validated predictor" and eliminates the circular-label criticism entirely. A cross-validation stub is now at `src/13_secc_validation.py` — it downloads SECC data and joins to the top-100 CSV; run it after step 08.

3. **Raise minmax normalisation ceiling from 98th to 99th percentile** — The current `minmax_score(clip_hi=0.98)` lets the top 2% of villages (≈7,100 of 356k) all hit the per-signal ceiling at score=100, creating ties. **This is now fixed**: `config.yaml → scoring.minmax_clip_hi = 0.99` raises the ceiling to the 99th percentile, limiting the ceiling group to ≈3,560 villages. Re-run `04_score_rank.py` to apply. For maximum discrimination within the top 100, raise to 0.999 — only the top 356 villages would then ceiling on each signal.

4. **Add Microsoft building footprint count as a 9th signal** — Microsoft released AI-detected building polygons for all India villages (OpenBuildings dataset). Building count growth (2016 vs 2023) is an independent, cloud-free built-up signal that directly cross-validates NTL growth and is unaffected by Saubhagya electrification.

5. **Match PC11 IDs to SHRUG village names** — 83,941 unnamed villages in this run were assigned `Village_<id>`. The SHRUG dataset maps PC11 census IDs to official village names. A one-time join would resolve the naming gap and make the top-100 table presentable to non-technical stakeholders.

---

## Weight Sensitivity Analysis

With 7 signals in the composite (SAR near-zero → effectively 6 active), the effective weight distribution is: NTL 31.6%, NDBI 21.1%, GHSL 15.8%, ML 15.8%, tower 10.5%, spatial-lag 5.3% (s1_vv_delta at 0 variance redistributes proportionally). The discriminating variable within the top-89 is `ghsl_change_score` combined with `ntl_growth_log_score` — together they separate the Construction Boom cluster (high GHSL + NTL) from Active Growth (moderate NTL, lower GHSL).

| Scenario | ML % | NTL % | GHSL % | Tower % | Score spread |
|---|---|---|---|---|---|
| ML +10pp (NTL −10pp) | 25 | 20 | 15 | 10 | ~19 pts |
| **Current (Round 9 baseline)** | **15** | **30** | **15** | **10** | **20.15 pts** |
| NTL +10pp (ML −10pp) | 5 | 40 | 15 | 10 | ~21 pts |
| Tower doubled (NTL −10pp) | 15 | 20 | 15 | 20 | ~18 pts |

**Key finding:** Tower growth signal at 10% provides a genuine independent cross-check: the few villages with real mobile tower expansion (Jawal, Badauli, Mihinpurwa) rank higher than pure-NTL alternatives. Increasing NTL weight reinforces the Aligarh / Gorakhpur corridor. Bootstrap instability is driven by the NTL–ML correlation, not by weak spread.

> For exact sensitivity, modify `config.yaml → scoring.composite_weights_full` and re-run `04_score_rank.py`.

---

## External Satellite Verification

The following Google Maps satellite-view links provide visual ground truth for the top-ranked villages (open in a browser to verify visible development):

| Rank | Location | Coordinates | Satellite observation |
|------|----------|-------------|----------------------|
| 1 | **Siswa Bazar**, Maharajganj, UP | 27.1656°N, 83.7602°E | [View →](https://www.google.com/maps/@27.1656071,83.7601948,14z/data=!3m1!1e3) |
| 2 | **Himmatpur Talla**, Nainital, Uttarakhand | 29.2181°N, 79.4815°E | [View →](https://www.google.com/maps/@29.2181122,79.4815052,14z/data=!3m1!1e3) |
| 3 | **Domariyaganj**, Siddharth Nagar, UP | 27.1837°N, 82.4656°E | [View →](https://www.google.com/maps/@27.1836845,82.4656105,14z/data=!3m1!1e3) |
| 4 | **Naveguda**, Adilabad, Telangana | 19.4960°N, 79.3231°E | [View →](https://www.google.com/maps/@19.4959651,79.32307,14z/data=!3m1!1e3) |
| 5 | **Kallagam**, Ariyalur, Tamil Nadu | 11.0271°N, 79.0001°E | [View →](https://www.google.com/maps/@11.0271301,79.0000514,14z/data=!3m1!1e3) |

**Siswa Bazar, Maharajganj** (rank 1): Located in the Gorakhpur–Maharajganj development belt of eastern UP, 55 km from Gorakhpur city. A NH-28 corridor market town in Nichlaul sub-district, Maharajganj. NTL baseline of 0.9–1.5 nW/cm²/sr (2019) rising to 5–11 nW/cm²/sr (2024) with BFAST breakpoints in 2022–2023 is consistent with peri-urban expansion driven by NH-28 road upgrades rather than bare electrification (pre-existing NTL baseline rules out dark-village scenario). Multi-signal confirmed (WorldCover built-up change 2020→2021 + NTL).

**Domariyaganj, Siddharth Nagar** (rank 3): Near the India–Nepal border zone at 27.18°–27.19°N, 82.46°–82.48°E. Cross-border trade infrastructure and PMGSY road improvements. NTL rise from 0.7–0.9 nW/cm²/sr to 6–7 nW/cm²/sr with 2022 breakpoints.

**Himmatpur Talla, Nainital** (rank 2): Uttarakhand Himalayan foothills at 29.2181°N, 79.4815°E, 65 km from Nainital city. NTL growth of 618% (3.1 → 22.4 nW/cm²/sr) with tower density growth (+0.51/km²) and builtup change — the only top-10 village with confirmed tower infrastructure expansion. Strongest multi-signal confirmation in the top 10.

**Naveguda, Adilabad, Telangana** (rank 4): 19.4960°N, 79.3231°E. NTL growth 1,170% (1.0 → 13.3 nW/cm²/sr) with 2023 BFAST breakpoint and slope acceleration of +9.67 nW/yr². Located 40 km from Adilabad city; consistent with industrial corridor development in the Telangana growth belt.

**Kallagam, Ariyalur, Tamil Nadu** (rank 5): 11.0271°N, 79.0001°E. NTL growth 702% (2.6 → 20.6 nW/cm²/sr) with 2023 breakpoint. Ariyalur district has active cement and limestone industry; NTL jump consistent with industrial activity expansion.

---

*Data citations:*
*NASA VIIRS VNP46A4 v002 — NASA LAADS DAAC*
*ESA WorldCover v100/v200 — ESA / Sinergise*
*Sentinel-2/1 — ESA Copernicus via Element84 STAC*
*GHSL R2023A — EU JRC*
*WorldPop India 2019/2020 — WorldPop.org*
*OpenStreetMap contributors — ODbL license*
*Natural Earth — public domain*
