"use strict";
const pptxgen = require("pptxgenjs");

let pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.title   = "India Village Economic Growth Intelligence";
pres.author  = "Candidate";

// ── Palette ──────────────────────────────────────────────────────────────────
const DARK_BG   = "021B2C";
const MID_BLUE  = "065A82";
const TEAL      = "1C7293";
const ACCENT    = "0D9488";
const LIGHT_BG  = "F0F7FA";
const WHITE     = "FFFFFF";
const TEXT_DARK = "1E293B";
const TEXT_MID  = "475569";
const MINT      = "4ECDC4";
const CARD_MUTED = "CBD5E1";

// ── Slide 1 · Title ───────────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: DARK_BG };

  // Top accent strip
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 10, h: 0.07, fill: { color: ACCENT }, line: { type: "none" }
  });

  // Decorative circles (satellite motif)
  s.addShape(pres.shapes.OVAL, {
    x: 6.9, y: 0.6, w: 4.0, h: 4.0,
    fill: { color: MID_BLUE, transparency: 78 }, line: { type: "none" }
  });
  s.addShape(pres.shapes.OVAL, {
    x: 7.6, y: 1.1, w: 2.8, h: 2.8,
    fill: { color: TEAL, transparency: 65 }, line: { type: "none" }
  });
  s.addShape(pres.shapes.OVAL, {
    x: 8.1, y: 1.55, w: 1.8, h: 1.8,
    fill: { color: ACCENT, transparency: 50 }, line: { type: "none" }
  });

  // Title
  s.addText("India Village Economic Growth", {
    x: 0.55, y: 1.0, w: 7.0, h: 0.85,
    fontSize: 36, bold: true, color: WHITE,
    fontFace: "Calibri", align: "left", margin: 0
  });
  s.addText("Intelligence", {
    x: 0.55, y: 1.82, w: 7.0, h: 0.85,
    fontSize: 36, bold: true, color: MINT,
    fontFace: "Calibri", align: "left", margin: 0
  });

  // Subtitle
  s.addText("Satellite-Derived Signals for Village-Level Economic Activity  ·  2019–2024", {
    x: 0.55, y: 2.82, w: 8.0, h: 0.42,
    fontSize: 14, color: "94A3B8", fontFace: "Calibri", align: "left", margin: 0
  });

  // Thin rule
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.55, y: 3.38, w: 1.8, h: 0.05,
    fill: { color: ACCENT }, line: { type: "none" }
  });

  // Stats row
  const stats = [
    { v: "255K", l: "Villages\nScored" },
    { v: "6 yrs", l: "VIIRS NTL\n2019–2024" },
    { v: "100",   l: "Top Villages\nShortlisted" }
  ];
  stats.forEach((st, i) => {
    const x = 0.55 + i * 3.0;
    s.addText(st.v, {
      x, y: 3.6, w: 2.7, h: 0.65,
      fontSize: 30, bold: true, color: MINT,
      fontFace: "Calibri", align: "left", margin: 0
    });
    s.addText(st.l, {
      x, y: 4.22, w: 2.7, h: 0.55,
      fontSize: 11.5, color: "94A3B8", fontFace: "Calibri", align: "left", margin: 0
    });
  });

  s.addText("467K OSM centroids downloaded  ·  255K passed India polygon + NTL baseline filter  →  scored and shortlisted", {
    x: 0.55, y: 4.88, w: 8.5, h: 0.22,
    fontSize: 8.5, color: "475569", italic: true, fontFace: "Calibri", align: "left", margin: 0
  });

  // Footer
  s.addText("Kritter Software Technologies — Candidate Assignment", {
    x: 0.55, y: 5.18, w: 6.5, h: 0.28,
    fontSize: 9.5, color: "475569", fontFace: "Calibri", align: "left", margin: 0
  });
  s.addText("May 2026", {
    x: 7.8, y: 5.18, w: 1.7, h: 0.28,
    fontSize: 9.5, color: "475569", fontFace: "Calibri", align: "right", margin: 0
  });
}

// ── Slide 2 · Problem & Approach ─────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: LIGHT_BG };

  s.addText("Problem & Approach", {
    x: 0.5, y: 0.32, w: 9, h: 0.58,
    fontSize: 28, bold: true, color: TEXT_DARK, fontFace: "Calibri", align: "left", margin: 0
  });

  // Helper: card
  const card = (x, y, w, h, topColor) => {
    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w, h, fill: { color: WHITE },
      line: { color: "E2E8F0", width: 0.5 },
      shadow: { type: "outer", color: "000000", blur: 8, offset: 2, angle: 135, opacity: 0.08 }
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w, h: 0.06, fill: { color: topColor }, line: { type: "none" }
    });
  };

  // Left card: The Challenge
  card(0.5, 1.05, 4.3, 4.0, MID_BLUE);
  s.addText("The Challenge", {
    x: 0.7, y: 1.22, w: 3.9, h: 0.42,
    fontSize: 15, bold: true, color: MID_BLUE, fontFace: "Calibri", align: "left", margin: 0
  });
  [
    "India has ~640k villages — traditional surveys are slow, expensive, and years out of date",
    "No real-time economic data exists at village scale",
    "Census boundary data is from 2011 (PC11); split/merged villages cause mismatches"
  ].forEach((t, i) => {
    s.addShape(pres.shapes.OVAL, {
      x: 0.72, y: 1.88 + i * 0.98, w: 0.18, h: 0.18,
      fill: { color: ACCENT }, line: { type: "none" }
    });
    s.addText(t, {
      x: 1.04, y: 1.82 + i * 0.98, w: 3.55, h: 0.6,
      fontSize: 12, color: TEXT_DARK, fontFace: "Calibri", align: "left", margin: 0
    });
  });

  // Right card: Our Solution
  card(5.2, 1.05, 4.3, 4.0, ACCENT);
  s.addText("Satellite Proxy Signals", {
    x: 5.4, y: 1.22, w: 3.9, h: 0.42,
    fontSize: 15, bold: true, color: ACCENT, fontFace: "Calibri", align: "left", margin: 0
  });
  [
    { sig: "Nighttime Lights (VIIRS)",    desc: "Annual composites 2019–2024 at 500m. Proxy for electrification, commerce & settlement density." },
    { sig: "Built-up Change (WorldCover)", desc: "ESA 10m land cover — detects new infrastructure 2020→2021." },
    { sig: "6-Year Trend",                desc: "Linear regression slope separates sustained growth from single-year spikes." }
  ].forEach((item, i) => {
    s.addText(item.sig, {
      x: 5.4, y: 1.82 + i * 0.98, w: 3.9, h: 0.3,
      fontSize: 13, bold: true, color: MID_BLUE, fontFace: "Calibri", align: "left", margin: 0
    });
    s.addText(item.desc, {
      x: 5.4, y: 2.1 + i * 0.98, w: 3.9, h: 0.52,
      fontSize: 11, color: TEXT_MID, fontFace: "Calibri", align: "left", margin: 0
    });
  });

  s.addText("3 principal signals shown above — full 8-signal architecture (incl. Sentinel-2, Sentinel-1, GHSL, WorldPop) on Slide 3", {
    x: 0.5, y: 5.06, w: 9.0, h: 0.20,
    fontSize: 9, color: "64748B", italic: true, fontFace: "Calibri", align: "center", margin: 0
  });
  s.addText("All compute runs on AWS EC2 (ap-south-1). No proprietary tools or licensed datasets required.", {
    x: 0.5, y: 5.24, w: 9.0, h: 0.22,
    fontSize: 9.5, color: TEXT_MID, italic: true, fontFace: "Calibri", align: "center", margin: 0
  });
}

// ── Slide 3 · Data & Pipeline ─────────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: LIGHT_BG };

  s.addText("Data Sources & Pipeline  —  8-Signal Framework (3 active this run)", {
    x: 0.5, y: 0.32, w: 9.2, h: 0.58,
    fontSize: 24, bold: true, color: TEXT_DARK, fontFace: "Calibri", align: "left", margin: 0
  });

  // 8 source cards in 2 rows of 4
  // Row 1: 3 active signals + OSM infrastructure
  // Row 2: 2 stalled Sentinel signals + 2 supporting datasets
  const sourceRows = [
    [
      { title: "NASA VIIRS\nVNP46A4 v2",    detail: "Annual NTL radiance\n500m · 2019–2024\n16 India tiles",  color: MID_BLUE, badge: "✓ ACTIVE",    bc: "16A34A" },
      { title: "ESA WorldCover\n2020 & 2021", detail: "Built-up fraction\n10m COGs · v100/v200",              color: TEAL,     badge: "✓ ACTIVE",    bc: "16A34A" },
      { title: "EU JRC GHSL\nR2023A",         detail: "Built-up surface\n100m · 2015 & 2020",                color: ACCENT,   badge: "✓ ACTIVE",    bc: "16A34A" },
      { title: "OSM + Natural\nEarth",         detail: "467K centroids\ntower density · city dist.",          color: "0891B2", badge: "✓ ACTIVE",    bc: "16A34A" }
    ],
    [
      { title: "Sentinel-2\n(Element84 STAC)", detail: "NDBI + NDVI 100m\n2019–2024 composites",             color: "64748B", badge: "⚠ STALLED",   bc: "D97706" },
      { title: "Sentinel-1 RTC\n(Element84)",   detail: "SAR VV backscatter\n100m · cloud-free",             color: "64748B", badge: "⚠ STALLED",   bc: "D97706" },
      { title: "WorldPop India\n2019–2020",      detail: "Population count\n100m · auxiliary only",          color: "94A3B8", badge: "◑ SUPPORTING", bc: "475569" },
      { title: "OSM Overpass\n+ Ohsome API",     detail: "Mobile tower density\n2019 & 2024 snapshots",      color: "94A3B8", badge: "◑ SUPPORTING", bc: "475569" }
    ]
  ];

  const CW = 2.10;
  const rowY  = [1.03, 2.50];
  const rowH  = [1.25, 1.02];

  sourceRows.forEach((row, ri) => {
    row.forEach((src, ci) => {
      const x = 0.38 + ci * 2.37;
      const y = rowY[ri];
      const h = rowH[ri];
      s.addShape(pres.shapes.RECTANGLE, {
        x, y, w: CW, h,
        fill: { color: src.color }, line: { type: "none" },
        shadow: { type: "outer", color: "000000", blur: 8, offset: 2, angle: 135, opacity: ri === 0 ? 0.16 : 0.08 }
      });
      // Status badge
      s.addShape(pres.shapes.RECTANGLE, {
        x: x + 0.08, y: y + 0.07, w: 0.95, h: 0.16,
        fill: { color: src.bc, transparency: 20 }, line: { type: "none" }
      });
      s.addText(src.badge, {
        x: x + 0.08, y: y + 0.07, w: 0.95, h: 0.16,
        fontSize: 7, bold: true, color: WHITE, fontFace: "Calibri", align: "center", valign: "middle", margin: 0
      });
      // Title
      s.addText(src.title, {
        x: x + 0.1, y: y + 0.26, w: CW - 0.2, h: ri === 0 ? 0.52 : 0.42,
        fontSize: ri === 0 ? 11.5 : 10.5, bold: true, color: WHITE, fontFace: "Calibri", align: "left", margin: 0
      });
      // Detail
      s.addText(src.detail, {
        x: x + 0.1, y: y + (ri === 0 ? 0.80 : 0.68), w: CW - 0.2, h: 0.34,
        fontSize: 9.5, color: ri === 0 ? "C7DFF0" : "CBD5E1", fontFace: "Calibri", align: "left", margin: 0
      });
    });
  });

  // Output row
  s.addText("Outputs", {
    x: 0.5, y: 3.65, w: 1.8, h: 0.34,
    fontSize: 13, bold: true, color: TEXT_DARK, fontFace: "Calibri", margin: 0
  });
  [
    { name: "village_all_stats.csv", desc: "467K rows · NTL 2019–2024 + per-village signals" },
    { name: "village_scored.csv",    desc: "255K India villages · composite score + rank" },
    { name: "top_100_villages.csv\n+ map.html",  desc: "Final shortlist · public S3 bucket (ap-south-1)" }
  ].forEach((o, i) => {
    const x = 0.5 + i * 3.18;
    s.addShape(pres.shapes.RECTANGLE, {
      x, y: 4.04, w: 2.98, h: 1.32,
      fill: { color: WHITE }, line: { color: "E2E8F0", width: 0.5 },
      shadow: { type: "outer", color: "000000", blur: 5, offset: 1, angle: 135, opacity: 0.07 }
    });
    s.addText(o.name, {
      x: x + 0.15, y: 4.15, w: 2.68, h: 0.38,
      fontSize: 10, bold: true, color: MID_BLUE, fontFace: "Consolas", align: "left", margin: 0
    });
    s.addText(o.desc, {
      x: x + 0.15, y: 4.53, w: 2.68, h: 0.68,
      fontSize: 10.5, color: TEXT_MID, fontFace: "Calibri", align: "left", margin: 0
    });
  });
}

// ── Slide 4 · ML Model ────────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: LIGHT_BG };

  s.addText("ML Scoring Model — 3-Stage Pipeline", {
    x: 0.5, y: 0.32, w: 9, h: 0.58,
    fontSize: 28, bold: true, color: TEXT_DARK, fontFace: "Calibri", align: "left", margin: 0
  });

  // Stage boxes
  const stages = [
    {
      num: "01", title: "Feature Engineering",
      color: MID_BLUE,
      bullets: [
        "BFAST-style piecewise regression on 6-year NTL series → breakpoint year, pre/post slope, acceleration",
        "Spatial lag: mean composite score of 10 nearest geographic neighbours (cluster reinforcement)",
        "Signal agreement: std of rank percentiles — penalises contradictory signals"
      ]
    },
    {
      num: "02", title: "Self-supervised Labelling",
      color: TEAL,
      bullets: [
        "Label = 1  if village is top-10% on ≥ 2 primary signals simultaneously",
        "~28K positive labels from 255K villages (~11% prevalence)",
        "⚠ Labels derived from same features as final composite — AUC measures self-consistency, not external ground truth"
      ]
    },
    {
      num: "03", title: "GradientBoostingClassifier",
      color: ACCENT,
      bullets: [
        "200 estimators · max_depth=4 · learning_rate=0.05 · subsample=0.8",
        "Inputs: 14 signal rank percentiles + BFAST features + spatial lag",
        "Output: ml_growth_prob (0–100) · AUC = 1.000 (in-sample, self-supervised) — GBM memorises its own self-supervised labels; this is the mathematically expected result, not a sign of external validity"
      ]
    }
  ];

  stages.forEach((st, i) => {
    const x = 0.35 + i * 3.18;
    // Card
    s.addShape(pres.shapes.RECTANGLE, {
      x, y: 1.05, w: 2.98, h: 4.15,
      fill: { color: WHITE }, line: { color: "E2E8F0", width: 0.5 },
      shadow: { type: "outer", color: "000000", blur: 8, offset: 2, angle: 135, opacity: 0.08 }
    });
    // Top colour strip
    s.addShape(pres.shapes.RECTANGLE, {
      x, y: 1.05, w: 2.98, h: 0.06,
      fill: { color: st.color }, line: { type: "none" }
    });
    // Stage number badge
    s.addShape(pres.shapes.OVAL, {
      x: x + 0.12, y: 1.2, w: 0.42, h: 0.42,
      fill: { color: st.color }, line: { type: "none" }
    });
    s.addText(st.num, {
      x: x + 0.12, y: 1.2, w: 0.42, h: 0.42,
      fontSize: 11, bold: true, color: WHITE,
      fontFace: "Calibri", align: "center", valign: "middle", margin: 0
    });
    // Title
    s.addText(st.title, {
      x: x + 0.62, y: 1.22, w: 2.25, h: 0.38,
      fontSize: 12.5, bold: true, color: st.color,
      fontFace: "Calibri", align: "left", margin: 0
    });
    // Divider
    s.addShape(pres.shapes.RECTANGLE, {
      x: x + 0.15, y: 1.72, w: 2.68, h: 0.03,
      fill: { color: "E2E8F0" }, line: { type: "none" }
    });
    // Bullets
    st.bullets.forEach((b, j) => {
      s.addShape(pres.shapes.OVAL, {
        x: x + 0.18, y: 1.88 + j * 0.98, w: 0.14, h: 0.14,
        fill: { color: st.color }, line: { type: "none" }
      });
      s.addText(b, {
        x: x + 0.38, y: 1.83 + j * 0.98, w: 2.48, h: 0.82,
        fontSize: 10, color: TEXT_DARK, fontFace: "Calibri", align: "left", margin: 0
      });
    });
    // Arrow between stages
    if (i < 2) {
      s.addShape(pres.shapes.RECTANGLE, {
        x: x + 3.0, y: 2.9, w: 0.15, h: 0.04,
        fill: { color: TEXT_MID }, line: { type: "none" }
      });
    }
  });

  // AUC note at bottom
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.35, y: 5.25, w: 9.3, h: 0.32,
    fill: { color: "EFF6FF" }, line: { color: "BFDBFE", width: 0.5 }
  });
  s.addText(
    "Model saved to ml_model.pkl · SHAP TreeExplainer applied post-hoc for per-village signal attribution · " +
    "K-means (k=6 by silhouette) clusters all scored villages into 6 growth archetypes",
    {
      x: 0.5, y: 5.28, w: 9.1, h: 0.26,
      fontSize: 9, color: TEXT_MID, fontFace: "Calibri", align: "left", margin: 0
    }
  );
}

// ── Slide 5 · Composite Score ─────────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: LIGHT_BG };

  s.addText("Composite Score — Actual Weights Used", {
    x: 0.5, y: 0.32, w: 9, h: 0.58,
    fontSize: 28, bold: true, color: TEXT_DARK, fontFace: "Calibri", align: "left", margin: 0
  });

  // Honest weights: Sentinel-2 NDBI and Sentinel-1 SAR stalled (NaN for all villages)
  // redistribute() dropped them → effective weights are ml:53% ntl:27% ghsl:20%
  const signals = [
    { pct: "53%", name: "ML Growth Probability",   detail: "GBM output (Stage 3)  ·  dominant signal  ·  captures multi-signal interaction",  color: ACCENT,   bar: 5.3 },
    { pct: "27%", name: "NTL Growth (log-scaled)", detail: "log(1 + ntl_growth_pct) 2019→2024  ·  log dampens extreme % from low baselines", color: MID_BLUE, bar: 2.7 },
    { pct: "20%", name: "GHSL Built-up Change",    detail: "EU JRC 2015→2020 built-up surface fraction  ·  cloud-independent, validated",     color: TEAL,     bar: 2.0 }
  ];

  // Note banner
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 1.02, w: 9.1, h: 0.55,
    fill: { color: "FEF3C7" }, line: { color: "FCD34D", width: 0.5 }
  });
  s.addText(
    "⚠  Sentinel-2 NDBI and Sentinel-1 SAR signals stalled during this pipeline run (monsoon STAC gaps)." +
    "  Weights auto-redistributed from 5-signal spec (40%+20%+15%+15%+10%) to 3-signal above." +
    "  Re-running --phase-c-only after S2/S1 complete will restore full 5-signal composite.",
    {
      x: 0.65, y: 1.06, w: 8.8, h: 0.46,
      fontSize: 9, color: "92400E", fontFace: "Calibri", align: "left", margin: 0
    }
  );

  signals.forEach((sig, i) => {
    const y = 1.78 + i * 1.0;
    s.addText(sig.pct, {
      x: 0.5, y, w: 0.78, h: 0.7,
      fontSize: 22, bold: true, color: sig.color,
      fontFace: "Calibri", align: "right", valign: "middle", margin: 0
    });
    s.addText(sig.name, {
      x: 1.42, y, w: 5.3, h: 0.32,
      fontSize: 13.5, bold: true, color: TEXT_DARK, fontFace: "Calibri", align: "left", margin: 0
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: 1.42, y: y + 0.38, w: 5.5, h: 0.26,
      fill: { color: "DDE6EE" }, line: { type: "none" }
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: 1.42, y: y + 0.38, w: sig.bar, h: 0.26,
      fill: { color: sig.color }, line: { type: "none" }
    });
    s.addText(sig.detail, {
      x: 7.1, y: y + 0.05, w: 2.6, h: 0.62,
      fontSize: 9.5, color: TEXT_MID, fontFace: "Calibri", align: "left", margin: 0
    });
  });

  // Pre-filter info box
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 5.06, w: 9.2, h: 0.38,
    fill: { color: "EFF6FF" }, line: { color: "BFDBFE", width: 0.5 }
  });
  s.addText(
    "Pre-filters: NTL 2019 ≥ 0.1 nW/cm²/sr + India polygon filter + Δ NTL ≥ 1.0 nW/cm²/sr absolute minimum  ·  Normalisation: min-max (2nd/99th pct clip) → 0–100  ·  " +
    "Score spread: top-100 range 73.85–70.04 (3.81 pts); rank-50 = 71.65. Spread was < 0.01 pt before minmax fix. Restoring Sentinel-2/SAR will widen it further. Treat top-100 as a statistical shortlist.",
    {
      x: 0.65, y: 5.1, w: 9.0, h: 0.3,
      fontSize: 9, color: TEXT_MID, fontFace: "Calibri", align: "left", margin: 0
    }
  );
}

// ── Slide 6 · Results ─────────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: LIGHT_BG };

  s.addText("Results — Top 100 Shortlist  (3-signal composite: NTL 27% · GHSL 20% · ML amplifier 53%)", {
    x: 0.5, y: 0.32, w: 9, h: 0.58,
    fontSize: 21, bold: true, color: TEXT_DARK, fontFace: "Calibri", align: "left", margin: 0
  });

  // Top-10 table
  const hdr = [
    { text: "#",             options: { bold: true, color: WHITE, fill: { color: MID_BLUE }, align: "center" } },
    { text: "Village",       options: { bold: true, color: WHITE, fill: { color: MID_BLUE } } },
    { text: "State / Dist.", options: { bold: true, color: WHITE, fill: { color: MID_BLUE } } },
    { text: "NTL Growth",    options: { bold: true, color: WHITE, fill: { color: MID_BLUE }, align: "right" } }
  ];
  const rows = [
    ["1",  "nr. Siswa Bazar †",       "Maharajganj, UP",      "+628%"],
    ["2",  "nr. Siswa Bazar †",       "Maharajganj, UP",      "+551%"],
    ["3",  "Himmatpur Talla",         "Nainital, Uttarakhand", "+618%"],
    ["4",  "nr. Domariyaganj †",      "Siddharth Nagar, UP",  "+831%"],
    ["5",  "nr. Siswa Bazar †",       "Maharajganj, UP",      "+476%"],
    ["6",  "nr. Domariyaganj †",      "Siddharth Nagar, UP",  "+541%"],
    ["7",  "nr. Siswa Bazar †",       "Maharajganj, UP",      "+682%"],
    ["8",  "nr. Siswa Bazar †",       "Maharajganj, UP",      "+511%"],
    ["9",  "Naveguda",                "Adilabad, Telangana",  "+1,170%"],
    ["10", "Kallagam",                "Ariyalur, Tamil Nadu", "+702%"]
  ];
  const tableData = [
    hdr,
    ...rows.map((r, i) => {
      const bg = i % 2 === 0 ? "F1F5F9" : WHITE;
      return [
        { text: r[0], options: { fill: { color: bg }, align: "center", color: TEXT_MID, fontSize: 10 } },
        { text: r[1], options: { fill: { color: bg }, color: TEXT_DARK, fontSize: 10 } },
        { text: r[2], options: { fill: { color: bg }, color: TEXT_MID,  fontSize: 10 } },
        { text: r[3], options: { fill: { color: bg }, bold: true, align: "right", color: MID_BLUE, fontSize: 10 } }
      ];
    })
  ];
  s.addTable(tableData, {
    x: 0.5, y: 1.02, w: 5.45, h: 4.3,
    colW: [0.38, 1.62, 2.0, 1.15],
    fontFace: "Calibri",
    border: { pt: 0.5, color: "E2E8F0" }
  });

  // State bar chart
  s.addChart(pres.charts.BAR, [{
    name: "Villages in Top 100",
    labels: ["Karnataka", "Uttar Pradesh", "Maharashtra", "Tamil Nadu", "Chhattisgarh", "Andhra Pradesh", "Telangana", "Uttarakhand", "Others"],
    values: [33, 25, 7, 7, 6, 5, 5, 3, 9]
  }], {
    x: 6.15, y: 0.98, w: 3.55, h: 4.35,
    barDir: "bar",
    chartColors: [MID_BLUE, TEAL, ACCENT, "0891B2", MID_BLUE, TEAL, ACCENT, "0891B2", MID_BLUE, TEAL],
    chartArea: { fill: { color: LIGHT_BG } },
    catAxisLabelColor: TEXT_MID,
    valAxisLabelColor: TEXT_MID,
    valGridLine: { color: "E2E8F0", size: 0.5 },
    catGridLine: { style: "none" },
    showValue: true,
    dataLabelColor: TEXT_DARK,
    dataLabelFontSize: 9,
    showLegend: false,
    showTitle: true,
    title: "Top 100 by State",
    titleFontSize: 12,
    titleColor: TEXT_DARK
  });

  // Footnote
  s.addText(
    "† OSM hamlet/village with no name tag — Nominatim reverse-geocoded to nearest named place (Siswa Bazar or Domariyaganj). 30 of 100 are unnamed OSM nodes.  " +
    "Score spread: 73.85–70.04 (3.81 pts). Bootstrap median inclusion = 0% (3 of 8 signals active) — treat as a statistical shortlist for field validation.",
    {
      x: 0.5, y: 5.35, w: 5.5, h: 0.28,
      fontSize: 7.5, color: TEXT_MID, italic: true, fontFace: "Calibri", align: "left", margin: 0
    }
  );
}

// ── Slide 7 · Case Study — Himmatpur Talla ───────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: LIGHT_BG };

  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 10, h: 0.07, fill: { color: ACCENT }, line: { type: "none" }
  });
  s.addText("Case Study — Himmatpur Talla, Uttarakhand  (Rank #3 · strongest multi-signal confirmation in top 10)", {
    x: 0.5, y: 0.28, w: 9.1, h: 0.55,
    fontSize: 20, bold: true, color: TEXT_DARK, fontFace: "Calibri", align: "left", margin: 0
  });

  // Left: profile card
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 0.97, w: 3.1, h: 4.62,
    fill: { color: WHITE }, line: { color: "E2E8F0", width: 0.5 },
    shadow: { type: "outer", color: "000000", blur: 6, offset: 1, angle: 135, opacity: 0.07 }
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 0.97, w: 3.1, h: 0.06, fill: { color: ACCENT }, line: { type: "none" }
  });
  s.addText("Village Profile  (Rank #3)", {
    x: 0.66, y: 1.07, w: 2.8, h: 0.30,
    fontSize: 11.5, bold: true, color: ACCENT, fontFace: "Calibri", align: "left", margin: 0
  });

  [
    ["District",     "Nainital, Uttarakhand"],
    ["Coordinates",  "29.2181°N, 79.4815°E"],
    ["Dist. to city","64.7 km  (Himalayan foothills)"],
    ["Archetype",    "NTL Breakout"],
    ["BFAST year",   "2022 (structural change)"],
    ["Pre-slope",    "0.23 nW/cm²/sr/yr (flat)"],
    ["Post-slope",   "3.97 nW/cm²/sr/yr  (17× acceleration)"],
    ["Tower growth", "+0.51 towers/km²  (only top-10 with confirmed connectivity expansion)"]
  ].forEach(([k, v], i) => {
    s.addText(k + ":", {
      x: 0.66, y: 1.46 + i * 0.33, w: 1.06, h: 0.28,
      fontSize: 9, bold: true, color: TEXT_MID, fontFace: "Calibri", align: "left", margin: 0
    });
    s.addText(v, {
      x: 1.75, y: 1.46 + i * 0.33, w: 1.73, h: 0.28,
      fontSize: 9, color: TEXT_DARK, fontFace: "Calibri", align: "left", margin: 0
    });
  });

  // SHAP section inside card
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.55, y: 4.18, w: 3.0, h: 0.03, fill: { color: "E2E8F0" }, line: { type: "none" }
  });
  s.addText("Top SHAP Feature Drivers", {
    x: 0.66, y: 4.24, w: 2.8, h: 0.24,
    fontSize: 9.5, bold: true, color: TEXT_MID, fontFace: "Calibri", align: "left", margin: 0
  });
  [
    { lbl: "Built-up Change (GHSL)", pct: 0.82 },
    { lbl: "NTL Log Growth",         pct: 0.78 },
    { lbl: "NTL Trend Slope",         pct: 0.71 }
  ].forEach((d, i) => {
    const y = 4.52 + i * 0.28;
    s.addText(d.lbl, {
      x: 0.66, y, w: 1.72, h: 0.24,
      fontSize: 8.5, color: TEXT_DARK, fontFace: "Calibri", margin: 0
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: 2.4, y: y + 0.07, w: 0.95, h: 0.12,
      fill: { color: "DDE6EE" }, line: { type: "none" }
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: 2.4, y: y + 0.07, w: 0.95 * d.pct, h: 0.12,
      fill: { color: ACCENT }, line: { type: "none" }
    });
  });

  // Right: 6-year NTL line chart — Himmatpur Talla actual values from pipeline output
  s.addChart(pres.charts.LINE, [
    {
      name: "NTL Radiance (nW/cm²/sr)",
      labels: ["2019", "2020", "2021", "2022", "2023", "2024"],
      values: [3.12, 2.51, 3.59, 14.49, 21.81, 22.42]
    }
  ], {
    x: 3.82, y: 0.94, w: 5.9, h: 3.88,
    chartColors: [ACCENT],
    lineDataSymbol: "dot",
    lineDataSymbolSize: 7,
    chartArea: { fill: { color: WHITE } },
    plotArea: { fill: { color: "F8FAFC" } },
    catAxisLabelColor: TEXT_MID,
    valAxisLabelColor: TEXT_MID,
    valGridLine: { color: "E2E8F0", size: 0.5 },
    catGridLine: { style: "none" },
    showTitle: true,
    title: "Nighttime Light Radiance (nW/cm²/sr)  ·  2019–2024",
    titleFontSize: 11,
    titleColor: TEXT_DARK,
    showLegend: false,
    showValue: true,
    dataLabelFontSize: 9,
    dataLabelColor: TEXT_MID
  });

  // Annotation box
  s.addShape(pres.shapes.RECTANGLE, {
    x: 7.55, y: 2.8, w: 2.0, h: 0.82,
    fill: { color: "FEF9C3" }, line: { color: "FCD34D", width: 0.5 }
  });
  s.addText("BFAST breakpoint\n2022 — pre-slope 0.23\npost-slope 3.97 nW/yr", {
    x: 7.62, y: 2.84, w: 1.86, h: 0.72,
    fontSize: 8.5, color: "92400E", fontFace: "Calibri", italic: true, align: "left", margin: 0
  });

  // Bottom interpretation bar
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 5.02, w: 9.2, h: 0.55,
    fill: { color: "EFF6FF" }, line: { color: "BFDBFE", width: 0.5 }
  });
  s.addText(
    "Why rank #3?  Himmatpur Talla is the only top-10 village with confirmed tower infrastructure expansion (+0.51 towers/km²), " +
    "ruling out pure electrification as the growth driver. NTL grew +618% (3.1 → 22.4 nW/cm²/sr). " +
    "BFAST structural break in 2022; post-break acceleration 17× pre-break rate. " +
    "Selected as case study because it is named, geographically isolated (64.7 km from nearest city), and has 4/4 available signals confirming → confidence_score = 100, inter_signal_agreement = 60.0.",
    {
      x: 0.66, y: 5.07, w: 8.9, h: 0.46,
      fontSize: 8.5, color: TEXT_MID, fontFace: "Calibri", align: "left", margin: 0
    }
  );
}

// ── Slide 8 · Conclusion ──────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: DARK_BG };

  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 10, h: 0.07, fill: { color: ACCENT }, line: { type: "none" }
  });

  s.addText("Key Findings & Next Steps", {
    x: 0.55, y: 0.42, w: 8.5, h: 0.62,
    fontSize: 28, bold: true, color: WHITE, fontFace: "Calibri", align: "left", margin: 0
  });

  // 3 finding cards
  const findings = [
    {
      title: "Growth is Geographically Concentrated",
      body: "Top 100 span 8+ states. Karnataka (33) and UP (25) dominate — consistent with Bengaluru peri-urban expansion and UP highway corridor districts. Karnataka is ~4× over-represented vs. its India village share; see README for confound analysis."
    },
    {
      title: "Spatial Signal Confirmed  ·  Ranking is a Shortlist",
      body: "Moran's I = 0.0631 (p < 0.001, 999 permutations, k=8 KNN, n=356K) — weak but significant clustering; scores are not spatially random. Bootstrap stability (n=200 Dirichlet draws): 14 villages ≥80% stable; 71 below 50%. Treat as a field-validation shortlist, not a strict ordered ranking."
    },
    {
      title: "Pipeline is Scalable & Repeatable",
      body: "255K scored villages on a single EC2 t3.xlarge in < 12 hrs. Annual automated re-run via EventBridge. Completing Sentinel-2/SAR will raise active signals from 3 to 5, widen score spread from 3.8 to >5 pts, and stabilise bootstrap rankings."
    }
  ];
  findings.forEach((f, i) => {
    const x = 0.5 + i * 3.18;
    s.addShape(pres.shapes.RECTANGLE, {
      x, y: 1.32, w: 2.98, h: 2.55,
      fill: { color: MID_BLUE, transparency: 55 },
      line: { color: TEAL, width: 0.5 }
    });
    s.addText(f.title, {
      x: x + 0.15, y: 1.45, w: 2.68, h: 0.55,
      fontSize: 12, bold: true, color: MINT, fontFace: "Calibri", align: "left", margin: 0
    });
    s.addText(f.body, {
      x: x + 0.15, y: 2.0, w: 2.68, h: 1.72,
      fontSize: 10.5, color: CARD_MUTED, fontFace: "Calibri", align: "left", margin: 0
    });
  });

  // Next steps (left)
  s.addText("What I Would Do Next", {
    x: 0.55, y: 4.05, w: 4.5, h: 0.35,
    fontSize: 12.5, bold: true, color: "94A3B8", fontFace: "Calibri", align: "left", margin: 0
  });
  s.addText([
    { text: "Complete Sentinel-2/SAR stalled signals — restores full 5-signal composite & resolves circular label issue", options: { bullet: true, breakLine: true } },
    { text: "Replace self-supervised labels with SHRUG/SECC consumption quintiles as external ground truth",              options: { bullet: true, breakLine: true } },
    { text: "Run PMGSY road correlation — validation framework built in 07_validate.py; requires EC2 re-run with PMGSY shapefiles", options: { bullet: true, breakLine: true } },
    { text: "Panel data fixed-effects model across all 6 years — replaces endpoint comparison",                           options: { bullet: true } }
  ], {
    x: 0.55, y: 4.45, w: 4.9, h: 1.0,
    fontSize: 10, color: "94A3B8", fontFace: "Calibri", margin: 0
  });

  // S3 link box (right)
  s.addShape(pres.shapes.RECTANGLE, {
    x: 5.55, y: 3.95, w: 4.1, h: 1.55,
    fill: { color: "041420" }, line: { color: ACCENT, width: 0.75 }
  });
  s.addText("Live Dashboard & Results", {
    x: 5.75, y: 4.06, w: 3.7, h: 0.35,
    fontSize: 11, bold: true, color: ACCENT, fontFace: "Calibri", align: "left", margin: 0
  });
  s.addText("ishunteam-png.github.io/kritter-village-growth", {
    x: 5.75, y: 4.4, w: 3.7, h: 0.27,
    fontSize: 8.5, color: MINT, fontFace: "Consolas", align: "left", margin: 0
  });
  s.addText("Top-100 table · 6 Plotly charts · interactive map", {
    x: 5.75, y: 4.65, w: 3.7, h: 0.27,
    fontSize: 9.5, color: CARD_MUTED, fontFace: "Calibri", align: "left", margin: 0
  });
  s.addText("S3: top_100_villages.csv · map.html · archetypes · forecast", {
    x: 5.75, y: 4.9, w: 3.7, h: 0.27,
    fontSize: 9, color: "475569", fontFace: "Calibri", align: "left", margin: 0
  });
}

// ── Write ─────────────────────────────────────────────────────────────────────
pres.writeFile({ fileName: "kritter_village_growth.pptx" })
  .then(() => console.log("Saved: kritter_village_growth.pptx"))
  .catch(err => { console.error(err); process.exit(1); });
