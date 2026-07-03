# 🎆 Fireworks & Air Quality — Causal Inference Pipeline

> **Can we measure the exact PM2.5 spike caused by backyard fireworks on July 4th?**
> Yes — using real EPA monitoring data and two rigorous causal inference methods.

This project builds a complete, reproducible tutorial measuring the **causal effect
of July 4th fireworks on PM2.5 air quality** in Richmond, Virginia.
It showcases two complementary estimators that solve different statistical
vulnerabilities inherent to environmental data.

---

## 🧪 The Core Problem: Atmospheric Confounding

Weather metrics like wind speed and relative humidity **don't behave linearly**.
Dead-calm air traps fireworks smoke exponentially. High humidity locks fine
particulates near the ground. A naïve regression gives a biased answer.

This project demonstrates two strategies to eliminate that bias:

| Method | Library | Strengths |
|--------|---------|-----------|
| **Synthetic DiD** | `diff-diff` | Temporal panel; builds weather-adjusted synthetic twin |
| **DoubleML IRM** | `doubleml` | ML scrubs non-linear weather confounders; policy-grade ATE |

---

## 🗺️ Study Design

**Focal city (treated)**: Richmond, VA  
**Donor pool (controls)**: Roanoke VA · Virginia Beach VA · Raleigh NC · Baltimore MD  
**Data**: EPA AQS PM2.5 (param 88101) + NOAA weather via Meteostat  
**Years**: 2025 (primary) · 2024 · 2023 (placebo checks)  
**Study window**: June 29 – July 8 (hourly)

Richmond sits in a geographic pocket prone to **thermal inversions and atmospheric
stagnation** on hot summer evenings — the worst-case environment for fireworks
smoke to pool. This makes it the ideal focal city.

---

## 📓 The Notebook

Everything lives in a single monolith:
**`notebooks/fireworks_causal_inference.ipynb`** (83 cells)

| Chapter | Title | Key Output |
|---------|-------|-----------|
| **0** | Environment & Credentials | Station map, API ping |
| **1** | Data Acquisition | Downloads 2023 / 2024 / 2025 EPA + NOAA data → Parquet |
| **2** | Exploratory Data Analysis | Parallel trends check, wind/humidity confounding |
| **3** | **Synthetic DiD — Three Acts** | ATT + divergence + decay plots |
| **4** | DoubleML IRM | ATE + naïve OLS bias comparison |
| **5** | Placebo & Robustness Checks | Multi-year, false-date, permutation test |

> Run cells top-to-bottom. Data is cached after the first run — no repeated API calls.

### The Three Acts (Chapter 3)

- **Act I** — *Pre-Trend Alignment*: The SDID algorithm assigns weights to
  Roanoke and Virginia Beach to construct a "Synthetic Richmond" that perfectly
  tracks the real city's baseline PM2.5 through the week before the holiday.

- **Act II** — *Pyrotechnic Divergence*: At 9:00 PM July 4, the real Richmond
  line spikes while the synthetic counterfactual stays flat. The gap *is* the
  causal effect.

- **Act III** — *Dispersion Phase*: Track the decay curve — how many hours
  before PM2.5 returns to baseline? This is the "environmental hangover."

---

## 🚀 Quick Start

### Prerequisites

1. **Python 3.11+** and **[uv](https://docs.astral.sh/uv/)** installed
2. A **free EPA AQS API key** — register at https://aqs.epa.gov/data/api/signup

### Setup

```bash
# 1. Clone and enter the project
git clone <repo-url>
cd "fireworks and-air-quality"

# 2. Copy the env template and fill in your credentials
cp .env.example .env
# edit .env with your AQS_EMAIL and AQS_KEY

# 3. Install all dependencies with uv
uv sync

# 4. Generate the notebooks
uv run python scripts/build_notebooks.py

# 5. Launch Jupyter
uv run jupyter notebook notebooks/
```

### Run

```bash
uv run jupyter notebook notebooks/fireworks_causal_inference.ipynb
```

> Execute cells top-to-bottom. Chapter banners with the 🎆 gradient header
> divide the six logical sections — use the Table of Contents cell to jump around.

---

## 📁 Project Structure

```
fireworks and-air-quality/
├── pyproject.toml              # uv project + all dependencies
├── .env.example                # Credentials template
├── .gitignore                  # Excludes data/, .env, .venv/
├── README.md                   # This file
│
├── src/                        # Data pipeline modules
│   ├── aqs_client.py           # EPA AQS API + bulk ZIP fallback
│   ├── weather_client.py       # NOAA weather via Meteostat v2
│   └── panel_builder.py        # Merge → Parquet + treatment columns
│
├── utils/                      # Shared utilities
│   ├── plotting.py             # Dark-theme visualization helpers
│   └── stats_helpers.py        # AQI, decay half-life, numpy SDID fallback
│
├── scripts/
│   └── build_notebooks.py      # Generates the monolith notebook
│
├── notebooks/
│   └── fireworks_causal_inference.ipynb   # ← THE notebook (83 cells)
│
└── data/
    ├── raw/                    # .gitignore'd — EPA ZIPs, weather cache
    └── processed/              # .gitignore'd — panel_YYYY.parquet files
```

---

## 🔬 Methods

### Synthetic Difference-in-Differences (Arkhangelsky et al., 2021)
- Implemented via the `diff-diff` library (numpy fallback available)
- Solves the **parallel trends failure** by reweighting control units
- Unit weights: linear combination of donor cities that minimizes pre-period distance to Richmond
- Inference: bootstrap confidence intervals

### DoubleML Interactive Regression Model (Chernozhukov et al., 2018)
- Implemented via the `doubleml` library with LightGBM nuisance models
- Achieves **Neyman Orthogonality** — estimation error in nuisance models doesn't
  bias the ATE at first order
- 5-fold cross-fitting × 3 repetitions for variance-stable inference
- Compares against naïve OLS to quantify atmospheric confounding bias

---

## 📊 Data Sources

| Dataset | Source | Access |
|---------|--------|--------|
| Hourly PM2.5 (FRM/FEM) | EPA AQS — parameter 88101 | API (free key) or bulk ZIP |
| Hourly weather | NOAA ISD via Meteostat | `meteostat` Python library |
| City metadata | USCB 2024 census estimates | Hardcoded in `src/panel_builder.py` |

**Note on 2025 data availability**: EPA regulatory data undergoes quality
assurance review with a typical 3–6 month lag. Summer 2025 data should be
available by late 2025. If the API returns no data for July 2025, the notebooks
automatically retry and fall back to the bulk pre-generated file download.

---

## 📦 Key Dependencies

```toml
doubleml    >= 0.8    # Double ML causal inference
diff-diff   >= 0.3    # Synthetic DiD + TWFE + Callaway-Sant'Anna
lightgbm    >= 4.3    # ML nuisance learners
meteostat   >= 1.6    # NOAA weather
pandas      >= 2.2    # Data wrangling
pyarrow     >= 16.0   # Parquet I/O
plotly      >= 5.22   # Interactive station map
```

---

## 🤝 Contributing

PRs welcome! Especially:
- Additional donor cities from the Mid-Atlantic region
- Alternative ML learners for the DoubleML nuisance (XGBoost, CatBoost)
- Interactive Plotly version of the Act II divergence plot

---

## 📄 License

MIT License — see LICENSE file.

---

*Last updated: July 3, 2026 — built with ❤️ and 🎆 for the eve of Independence Day*
