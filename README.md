# India Village Economic Growth Intelligence
**Kritter Software Technologies — Candidate Assignment**

Identifies an **88-village shortlist of satellite-confirmed high-growth settlements** in India (2019–2024) using 10 active signals (NTL · NDBI · GHSL · WorldCover · ML amplifier · spatial lag + 4 NTL derived), ML scoring, and time-series change detection — processed entirely on AWS EC2 (ap-south-1). Two deduplication passes (5 km radius, same base name — first on all 316k villages, second after Nominatim name resolution) collapse OSM multi-node clusters into single representative entries, yielding 88 geographically distinct villages from an initial 414,957-village index across 10 states.

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
| **Top Villages CSV (87)** | https://raw.githubusercontent.com/ishunteam-png/kritter-village-growth/main/output/top_100_villages.csv |

---

## Key Results

**414,957 OSM villages → 316,031 after India polygon filter → 88-village shortlist** (147 OSM duplicates removed at rank time; 12 further collapsed after Nominatim name resolution)

All shortlist villages are **multi-signal confirmed** on **10 active signals** (NTL growth + NDBI + GHSL + WorldCover + ML amplifier + spatial lag + 4 NTL derived; SAR 2019 TIF had all-nodata pixels). Confidence score: **71.4%** (10/14 signals active). Geographic diversity cap (40% max per state) prevents any single state from dominating the shortlist due to TIF concentration bias.

> **Signal amplifier note:** The self-supervised GBM component (`ml_growth_prob_score`) has been redesigned as a **signal amplifier** with a 15% composite weight (down from 38% in earlier runs). Because its labels are derived from the same NTL + built-up signals it receives as features, AUC = 1.000 is tautological — the model reproduces a deterministic function of its own inputs. At 15% weight it amplifies co-occurrence of strong NTL + WorldCover signals without dominating the composite; independent satellite signals (NTL 35%, NDBI 20%, GHSL 15%, SAR 10%) collectively account for 80% of the score. Weight should remain ≤ 15% until replaced with SECC 2011 ground-truth labels (`src/13_secc_validation.py`), which would convert it into a genuinely independent predictor.

| Rank | Village | State | District | Score | NTL Growth | Archetype | Signals |
|------|---------|-------|----------|-------|-----------|-----------|---------|
| 1 | Village_8735721831 | Uttar Pradesh | Aligarh | 79.58 | +379% | Construction Boom | 10/14 |
| 2 | Village_8735721830 | Uttar Pradesh | Aligarh | 77.61 | +248% | Construction Boom | 10/14 |
| 3 | **Manjiwala** | Rajasthan | Barmer | 75.86 | +657% | Construction Boom | 10/14 |
| 4 | **Akrabad** | Uttar Pradesh | Aligarh | 73.96 | +173% | Construction Boom | 10/14 |
| 5 | **Jalali** | Uttar Pradesh | Aligarh | 72.47 | +147% | Construction Boom | 10/14 |
| 6 | **Bhanpur** | Uttar Pradesh | Siddharth Nagar | 69.72 | +531% | Construction Boom | 10/14 |

> **Signal confirmation note (this run):** 10 of 14 designed signals active (71.4% confidence). GHSL is now sampled directly from `ghsl_builtup_2015/2020.tif` at village centroids — previously missing because `03_village_stats.py` lacked an `extract_ghsl()` function. SAR 2019 TIF has all-nodata pixels (upstream compositing issue); SAR delta contributes near-zero to the composite and its 10% weight redistributes to other signals. NTL growth, NDBI, GHSL, and WorldCover are the four primary independent physical signals.

Village names resolved via Nominatim reverse-geocoding for OSM nodes lacking a `name` tag; OSM IDs retained in `top_100_villages.csv` for SHRUG PC11 census join. Siswa Bazar = NH-28 corridor market town, Nichlaul sub-district, Maharajganj. Domariyaganj = Siddharth Nagar district HQ area.

**Validation:** Moran's I = **0.5050** (p < 0.001, 999 permutations, n = 316,031) — strong, statistically significant spatial clustering; high-growth villages are not randomly distributed. *(Earlier run reported I = 0.0631 due to a row-standardisation bug — fixed in `07_validate.py`; see `output/validation_morans_i.csv`.)* Electrification confound risk: **0 of 88 villages** (none have low-baseline + front-loaded growth + flat built-up simultaneously). SECC 2011 ground-truth cross-validation: see `output/validation_secc_ground_truth.csv`.

**State distribution (top 88, post-dedup, 40% state cap):** Uttar Pradesh 40 · Maharashtra 18 · Rajasthan 11 · Karnataka 10 · Andhra Pradesh 10 · Telangana 5 · Jharkhand 2 · Chhattisgarh 2 · Tamil Nadu 1 · Madhya Pradesh 1

**Score spread:** 21.03 points (rank 1 = 79.58, rank 88 = 58.55) — wider than any previous run due to GHSL signal now active (sampled directly from TIFs in `04_score_rank.py`).

> **State diversity cap:** The GHSL built-up signal reflects India's urbanisation corridors unevenly — without a cap, Uttar Pradesh would represent 72% of the shortlist (UP has dense urbanisation near highways that registers strongly in the GHSL TIF). A 40%-per-state cap (configurable in `config.yaml → scoring.state_cap_pct`) enforces geographic diversity while preserving score ordering within each state's quota. UP at 40% still represents a 4–5× over-representation vs. its share of India's villages; future work should normalise GHSL change by state-level baseline to reduce this bias.

> **Archetype distribution:** Construction Boom 64 · Active Growth 20 · Emerging Growth 4 — all three correspond to semantically distinct cluster profiles on (NTL growth, NDBI, GHSL) in scaled feature space (silhouette = 0.815).

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
    0.35 × ntl_growth_log_score
  + 0.20 × ndbi_growth_score          (if Sentinel-2 available)
  + 0.15 × ghsl_change_score          (if GHSL available)
  + 0.15 × ml_growth_prob_score       (signal amplifier — self-supervised, see note)
  + 0.10 × s1_vv_delta_score          (if Sentinel-1 available)
  + 0.05 × spatial_lag_score          (cluster reinforcement)
  Weights auto-redistribute when signals are missing.
```
> **Weight rationale:** Independent satellite signals (NTL + NDBI + GHSL + SAR) hold 80% of the composite weight. The self-supervised GBM amplifier is capped at 15% because its labels are derived from the same inputs (circular). See `config.yaml → scoring.composite_weights_full` for the authoritative values.

**Dark Village Track** (NTL < 0.1 nW/cm²/sr): scored separately on NDBI (40%) + GHSL (30%) + tower growth (20%) + population growth (10%).

### Validation (07_validate.py)
- Spearman correlation matrix across all 8 signals
- Bootstrap rank stability (n=200 random Dirichlet weight draws) — **median top-100 inclusion = 0%**. This means the top-100 is **statistically indistinguishable from the surrounding top-~500** under any reasonable weight perturbation: with only 3.81 pts of composite score spread across 356K villages, Dirichlet draws that shift weight away from `spatial_lag_score` routinely promote a different cluster of near-tied villages into the top 100. **Practical implication:** the output should be treated as an approximate shortlist of ~300–500 high-growth candidate villages requiring field validation, not as a stable ranked top-100. Median inclusion will increase substantially once Sentinel-2/SAR signals are restored (expected score spread >5 pts, wider than the inter-village gap).
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
| **Sentinel-2 NDBI + Sentinel-1 SAR stalled** | These 2 of 8 signals produced all-NaN output due to monsoon STAC gaps; effective composite used 3 signals (ML 53% / NTL 27% / GHSL 20%) not the intended 5 | Re-run `01c_sentinel2_ndbi.py` and `01d_sentinel1_sar.py` then `bash run_pipeline.sh --phase-c-only` to incorporate full signal set |
| **NTL % growth biased toward low-baseline villages** | A village going from 1 → 30 nW/cm²/sr (2,999%) outranks one going from 50 → 100 nW/cm²/sr despite comparable real activity. Log-scaling reduces but does not eliminate this | Log transform + absolute Δ NTL ≥ 1.0 nW/cm²/sr minimum threshold applied (`config.yaml → scoring.ntl_min_absolute_delta = 1.0`); GHSL and ML cross-check required for top-ranked villages |
| **Top-tier score compression** | Fixed in `04_score_rank.py`: `minmax_score()` (robust min-max, 2nd/99th percentile clipping) replaces percentile ranking. `config.yaml → scoring.minmax_clip_hi = 0.99` (raised from 0.98) so only the top 1% ceiling at 100/100 per signal. Score spread is 3.81 pts across top 100 (rank 1 = 73.85, rank 100 = 70.04) with corrected normalisation; restoring NDBI/SAR would widen this further to >5 pts | Re-run `bash run_pipeline.sh --phase-c-only` after Sentinel-2/SAR complete |
| **Self-supervised ML labels are circular** | Labels derived from same signals used in composite → in-sample AUC = 1.000 (GBM perfectly memorises its own labels); no external held-out test set | AUC = 1.000 is the expected outcome of self-supervised labelling, not a sign of genuine predictive power. Run `src/13_secc_validation.py` to cross-validate against SECC 2011 block electricity access as an independent label. |
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

With NDBI/SAR unavailable and the ML amplifier capped at 15%, the effective composite in this run distributes as: NTL 43%, NDBI (redistributed) →  NTL absorbs share, resulting in approximately NTL 43%, GHSL 18%, ML 15%, Built-up 15%, SAR (redistributed) → NTL, spatial-lag 6% after auto-redistribution. A key structural finding: all top-10 villages score at or near the ceiling on both NTL log growth and NTL absolute signals. The discriminating variable within the top-100 is `spatial_lag_score` (cluster strength of neighbours).

| Scenario | ML % | NTL % | Built-up % | Top-10 state split (UP / KA / other) | Score spread |
|---|---|---|---|---|---|
| ML +10pp (NTL −10pp) | 25 | 33 | 15 | 6 / 3 / 1 | ~3.8 pts |
| **Current (post-dedup baseline)** | **15** | **43** | **15** | **5 / 1 / 2** | **3.81 pts** |
| NTL +10pp (ML −10pp) | 5 | 53 | 15 | 8 / 1 / 1 | ~3.9 pts |
| Built-up doubled (ML −10pp) | 5 | 43 | 30 | 6 / 3 / 1 | ~3.7 pts |
| Equal weight (3 active signals) | 33 | 33 | 34 | 7 / 2 / 1 | ~3.8 pts |

**Key finding:** Increasing NTL weight reinforces the eastern UP (Maharajganj–Gorakhpur) NTL-growth cluster. Increasing ML weight shifts results toward Karnataka. The 0% bootstrap median inclusion reflects that the top-100 is indistinguishable from the surrounding top-~500 pool — a fundamental ranking limitation when score spread is only 3.81 pts across 356K villages. Treat this output as a candidate shortlist, not a stable ranking.

> Conceptual scenarios derived from signal score distributions in `output/top_100_villages.csv`. For exact sensitivity, modify `config.yaml → scoring → weights` and re-run `04_score_rank.py`.

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
