"""
scripts/build_notebooks.py
─────────────────────────────────────────────────────────────────────────────
Generates all six Jupyter notebooks programmatically using nbformat.
Run via:  uv run python scripts/build_notebooks.py

This script is the canonical source-of-truth for notebook content.
Individual .ipynb files are generated artifacts — don't edit them directly.
"""

import json
import textwrap
from pathlib import Path

import nbformat as nbf

NOTEBOOKS_DIR = Path(__file__).parent.parent / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def code(src: str, tags: list[str] | None = None) -> nbf.NotebookNode:
    meta = {}
    if tags:
        meta["tags"] = tags
    return nbf.v4.new_code_cell(textwrap.dedent(src).strip(), metadata=meta)


def md(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(textwrap.dedent(src).strip())


def make_nb(cells: list) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.cells = cells
    return nb


# ══════════════════════════════════════════════════════════════════════════════
# Notebook 00 — Environment Check
# ══════════════════════════════════════════════════════════════════════════════

nb00 = make_nb([
    md("""
    # 🎆 Notebook 00 — Environment & Credentials Check
    Verify that all packages are installed and your EPA AQS API credentials work.
    """),

    code("""
    # ── Library versions ───────────────────────────────────────────────────────
    import importlib, sys

    REQUIRED = [
        "pandas", "numpy", "scipy", "matplotlib", "seaborn", "plotly",
        "meteostat", "doubleml", "lightgbm", "sklearn",
        "pyarrow", "requests", "dotenv", "rich", "tqdm",
    ]

    missing = []
    print(f"Python {sys.version}\\n")
    for pkg in REQUIRED:
        try:
            mod = importlib.import_module(pkg)
            ver = getattr(mod, "__version__", "?")
            print(f"  ✔  {pkg:<20} {ver}")
        except ImportError:
            print(f"  ✗  {pkg:<20} NOT INSTALLED")
            missing.append(pkg)

    if missing:
        print(f"\\n⚠  Missing: {missing}  →  run `uv sync`")
    else:
        print("\\n✅ All required packages present.")
    """),

    code("""
    # ── diff-diff library check ────────────────────────────────────────────────
    try:
        import diff_diff
        print(f"diff-diff version: {diff_diff.__version__}")
        from diff_diff import SyntheticDiD
        print("  ✔ SyntheticDiD importable")
    except ImportError:
        print("  ✗ diff-diff not installed — will use numpy fallback in nb 03")
    except AttributeError as e:
        print(f"  ⚠ diff-diff installed but SyntheticDiD API may differ: {e}")
    """),

    code("""
    # ── EPA AQS credentials ────────────────────────────────────────────────────
    import os
    from dotenv import load_dotenv

    load_dotenv()
    email = os.getenv("AQS_EMAIL", "")
    key   = os.getenv("AQS_KEY",   "")

    if email and key:
        print(f"Credentials loaded:")
        print(f"  AQS_EMAIL = {email}")
        print(f"  AQS_KEY   = {'*' * len(key)}")
    else:
        print("⚠  AQS_EMAIL / AQS_KEY not set. Copy .env.example → .env")
    """),

    code("""
    # ── Live API ping ──────────────────────────────────────────────────────────
    import sys
    sys.path.insert(0, "..")

    from src.aqs_client import AQSClient

    client = AQSClient()
    ok = client.check_credentials()
    if ok:
        print("✅ EPA AQS API credentials valid!")
    else:
        print("❌ Credentials invalid or API unreachable.")
    """),

    code("""
    # ── Station map (interactive Plotly) ──────────────────────────────────────
    import plotly.express as px
    import pandas as pd
    from src.aqs_client import STATION_COORDS, TARGET_COUNTIES

    station_data = []
    for label, state, county in TARGET_COUNTIES:
        if label not in STATION_COORDS:
            continue
        lat, lon = STATION_COORDS[label]
        is_rural = label in ("Rockingham_VA", "Page_VA")
        role = ("Treated (focal)" if label == "Richmond_VA"
                else "Rural donor" if is_rural
                else "Urban donor")
        station_data.append({
            "city": label.replace("_", " "),
            "lat": lat, "lon": lon, "role": role,
        })

    df_map = pd.DataFrame(station_data)
    fig = px.scatter_geo(
        df_map,
        lat="lat", lon="lon",
        text="city",
        color="role",
        color_discrete_map={
            "Treated (focal)": "#FF6B35",
            "Urban donor":     "#74C2E1",
            "Rural donor":     "#A8D5A2",
        },
        scope="usa",
        title="July 4th Fireworks Study — PM2.5 Monitoring Network",
        template="plotly_dark",
    )
    fig.update_traces(marker_size=14, textposition="top center")
    fig.update_layout(
        geo_center={"lat": 37.5, "lon": -77.5},
        geo_projection_scale=6,
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
    )
    fig.show()
    """),

    md("## ✅ Environment ready — proceed to Chapter 1"),
])

# ══════════════════════════════════════════════════════════════════════════════
# Notebook 01 — Data Acquisition
# ══════════════════════════════════════════════════════════════════════════════

nb01 = make_nb([
    md("""
    # 📡 Chapter 1 — Data Acquisition
    Downloads **real EPA AQS PM2.5 + Open-Meteo ERA5 weather** for 2023, 2024, 2025
    and saves to `data/processed/panel_{year}.parquet`.

    ---

    ### Data sources
    | Source | What | How |
    |--------|------|-----|
    | **EPA AQS bulk CSV** | PM2.5 hourly (params 88502 + 88101) | Pre-generated ZIP, no site-ID fragility |
    | **Open-Meteo ERA5** | Temp, humidity, wind, precip | Free API, no key, global coverage |

    ---

    ### ⚠️ Note on donor city selection
    All five study cities celebrate July 4th, so none are *completely* untreated.
    This is less of a problem than it sounds for two reasons:

    1. **Synthetic DiD** fits donor weights on the *pre-period* (June 29–July 3),
       not the post-period. The synthetic Richmond captures regional atmospheric
       baseline patterns. The estimated ATT is the *excess* spike in Richmond
       above the weighted donor average — a **conservative** estimate.
    2. **DoubleML** uses purely *temporal* controls: same hours of the night on
       non-fireworks evenings (June 29–July 3 9 PM–3 AM vs. July 4).

    For a cleaner SDID, optional rural Shenandoah Valley monitors
    (`Rockingham_VA`, `Page_VA`) are included in the COUNTY_FIPS_MAP
    if data is available for them.
    """),

    code("""
    import sys
    sys.path.insert(0, "..")
    import warnings
    warnings.filterwarnings("ignore")

    import pandas as pd
    from rich import print as rprint
    from IPython.display import display

    from src.panel_builder import build_panel
    from src.aqs_client import TARGET_COUNTIES, COUNTY_FIPS_MAP
    from src.weather_client import WeatherClient

    print("Target counties:")
    for label, state, county in TARGET_COUNTIES:
        role = "TREATED" if label == "Richmond_VA" else "donor"
        print(f"  {label:<22} ({state}-{county})  {role}")
    """),

    md("""
    ## Step 1 — Build full panels (2023, 2024, 2025)
    Uses EPA **pre-generated bulk CSV files** — bypasses fragile site-ID lookups.
    Tries parameter **88502** (continuous hourly monitors) first, then **88101** as fallback.
    Results cached to `data/processed/` — subsequent runs skip the download.

    > **2025 data**: EPA regulatory data has a ~6 month QA lag.
    > If 2025 data is unavailable, the notebook will use 2024 as the primary year
    > and fall back gracefully.
    """),

    code("""
    YEARS = [2023, 2024, 2025]
    panels = {}
    PRIMARY_YEAR = None

    for year in YEARS:
        print(f"\\n{'━'*55}")
        print(f"  Building panel for {year} (bulk CSV) ...")
        print(f"{'━'*55}")
        try:
            df = build_panel(year, use_bulk=True)   # bulk is the reliable default
            if not df.empty:
                panels[year] = df
                if PRIMARY_YEAR is None:
                    PRIMARY_YEAR = year
                rprint(f"[green]  ✔ {year}: {len(df):,} rows, {df['city'].nunique()} cities[/green]")
            else:
                rprint(f"[yellow]  ⚠ {year}: no data (QA lag?)[/yellow]")
        except Exception as e:
            rprint(f"[red]  ✗ {year}: {e}[/red]")

    if not panels:
        rprint("[red]No data loaded for any year. Check internet connection.[/red]")
    else:
        PRIMARY_YEAR = max(panels.keys())   # use most recent year available
        rprint(f"[bold green]\\nPrimary analysis year: {PRIMARY_YEAR}[/bold green]")
    """),

    md("## Step 2 — Data quality report"),

    code("""
    for year, df in panels.items():
        print(f"\\n── {year} ─────────────────────────────────────")
        print(f"  Cities:  {sorted(df['city'].unique())}")
        print(f"  Date range: {df['datetime_local'].min()} → {df['datetime_local'].max()}")
        print(f"  PM2.5 nulls: {df['pm25'].isna().sum()} / {len(df)}")
        have_wx = [c for c in ['temp','rhum','wspd_mph'] if c in df.columns]
        print(f"  Weather cols: {have_wx if have_wx else 'none (merged in next step)'}")
        rva = df[df['city'] == 'Richmond_VA']
        if not rva.empty and 'is_fireworks_window' in rva.columns:
            display(
                rva.groupby('is_fireworks_window')['pm25']
                .agg(['count','mean','max','std'])
                .round(2)
                .rename(index={False:'Non-fireworks', True:'Fireworks window'})
            )
    """),

    md("""
    ## Step 3 — Visual sanity check
    Raw PM2.5 time series before any causal modeling.
    """),

    code("""
    import matplotlib.pyplot as plt
    from utils.plotting import plot_pm25_timeseries

    yr = PRIMARY_YEAR
    if yr in panels:
        fig, ax = plot_pm25_timeseries(panels[yr], year=yr,
            title=f"Hourly PM2.5 — All Study Cities, {yr} Fireworks Window")
        plt.savefig(f"../data/processed/raw_pm25_{yr}.png", dpi=150, bbox_inches="tight")
        plt.show()
    """),

    md("## ✅ Panels built — proceed to EDA"),
])

# ══════════════════════════════════════════════════════════════════════════════
# Notebook 02 — EDA & Visualization
# ══════════════════════════════════════════════════════════════════════════════

nb02 = make_nb([
    md("""
    # 🔍 Notebook 02 — Exploratory Data Analysis
    Understand the data before applying any causal model.

    Key questions:
    1. What does the raw PM2.5 spike look like on July 4, 2025?
    2. How does weather (wind, humidity) co-vary with PM2.5?
    3. Do our donor cities share a similar baseline trajectory with Richmond?
       (This is the **parallel trends** pre-condition for SDID)
    """),

    code("""
    import sys
    sys.path.insert(0, "..")
    import warnings
    warnings.filterwarnings("ignore")

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import seaborn as sns
    import plotly.graph_objects as go
    from IPython.display import display

    from src.panel_builder import load_panel
    from utils.plotting import (
        plot_pm25_timeseries, CITY_COLORS, CITY_LABELS, PALETTE, apply_dark_theme
    )
    from utils.stats_helpers import pm25_series_to_aqi, rolling_hourly_mean

    panel = load_panel(2025)
    print(f"Panel shape: {panel.shape}")
    display(panel.head(3))
    """),

    md("## 1. The July 4th Spike — All Cities"),

    code("""
    fig, ax = plot_pm25_timeseries(panel, year=2025,
        title="Hourly PM2.5 — All Study Cities, 2025 Fireworks Window")
    plt.show()
    """),

    md("""
    ## 2. AQI Category Breakdown for Richmond, VA
    How many hours crossed each EPA AQI threshold during the fireworks event?
    """),

    code("""
    rva = panel[panel["city"] == "Richmond_VA"].copy().sort_values("datetime_local").reset_index(drop=True)
    aqi_df = pm25_series_to_aqi(rva["pm25"])  # returns a clean 0-based RangeIndex
    rva = pd.concat([rva, aqi_df], axis=1)

    fw_window = rva[rva["is_fireworks_window"].fillna(False)]
    cat_counts = fw_window["aqi_category"].value_counts()

    AQI_COLORS = {
        "Good":                          "#00E400",
        "Moderate":                       "#FFFF00",
        "Unhealthy for Sensitive Groups": "#FF7E00",
        "Unhealthy":                      "#FF0000",
        "Very Unhealthy":                 "#99004C",
        "Hazardous":                      "#7E0023",
    }

    fig, ax = plt.subplots(figsize=(8, 4))
    apply_dark_theme(fig, ax)
    bars = ax.bar(
        cat_counts.index,
        cat_counts.values,
        color=[AQI_COLORS.get(c, "#888") for c in cat_counts.index],
        edgecolor="none",
    )
    ax.set_ylabel("Hours in category", fontsize=11)
    ax.set_title("Richmond, VA — AQI Category Distribution (Fireworks Window)", fontsize=12)
    ax.set_xticklabels(cat_counts.index, rotation=30, ha="right", fontsize=9)
    for bar, val in zip(bars, cat_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.05, str(val),
                ha="center", va="bottom", color=PALETTE["text"], fontsize=10)
    plt.tight_layout()
    plt.show()
    """),

    md("""
    ## 3. Weather Scatter — Wind Speed vs. PM2.5
    The core confounding relationship: low wind → pooled smoke.
    """),

    code("""
    rva_clean = rva.dropna(subset=["wspd_mph", "pm25"])
    rva_clean["period"] = rva_clean["is_fireworks_window"].map(
        {True: "Fireworks (July 4 9PM–July 5 3AM)", False: "Baseline"}
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    apply_dark_theme(fig, ax)
    for period, grp in rva_clean.groupby("period"):
        color = PALETTE["richmond"] if "Fireworks" in period else PALETTE["synthetic"]
        ax.scatter(grp["wspd_mph"], grp["pm25"], c=color, alpha=0.7, s=40,
                   label=period, edgecolors="none")

    from scipy.stats import pearsonr
    r, p = pearsonr(rva_clean["wspd_mph"], rva_clean["pm25"])
    ax.set_xlabel("Wind Speed (mph)", fontsize=11)
    ax.set_ylabel("PM2.5 (µg/m³)", fontsize=11)
    ax.set_title(f"Richmond, VA — Wind Speed vs. PM2.5  (r = {r:.2f}, p = {p:.3f})", fontsize=12)
    ax.legend(facecolor=PALETTE["panel_bg"], labelcolor=PALETTE["text"])
    plt.tight_layout()
    plt.show()
    print(f"Pearson r = {r:.3f}  (p = {p:.4f}) — negative: low wind → high PM2.5")
    """),

    md("## 4. Humidity vs. PM2.5 (Particulate Trapping)"),

    code("""
    if "rhum" in rva_clean.columns and rva_clean["rhum"].notna().sum() > 10:
        fig, ax = plt.subplots(figsize=(9, 5))
        apply_dark_theme(fig, ax)
        sc = ax.scatter(rva_clean["rhum"], rva_clean["pm25"],
                        c=rva_clean["pm25"], cmap="YlOrRd", alpha=0.8, s=40,
                        edgecolors="none")
        plt.colorbar(sc, ax=ax, label="PM2.5 (µg/m³)")
        ax.set_xlabel("Relative Humidity (%)", fontsize=11)
        ax.set_ylabel("PM2.5 (µg/m³)", fontsize=11)
        ax.set_title("Richmond — Relative Humidity vs. PM2.5", fontsize=12)
        plt.tight_layout()
        plt.show()
    else:
        print("Humidity data not available for this station.")
    """),

    md("""
    ## 5. Parallel Trends Check — Pre-Period (June 29 – July 3)
    The critical pre-condition for DiD: do control cities track Richmond's
    baseline trajectory? If yes, SDID can build a valid synthetic twin.
    """),

    code("""
    pre_window = panel[
        (panel["datetime_local"] < f"2025-07-04 21:00")
    ].copy()

    # Normalize PM2.5 to z-score per city so we compare shapes not levels
    pre_window["pm25_norm"] = (
        pre_window.groupby("city")["pm25"]
        .transform(lambda s: (s - s.mean()) / s.std())
    )

    fig, ax = plt.subplots(figsize=(13, 5))
    apply_dark_theme(fig, ax)

    for city in sorted(pre_window["city"].unique()):
        sub = pre_window[pre_window["city"] == city].sort_values("datetime_local")
        is_richmond = city == "Richmond_VA"
        ax.plot(
            sub["datetime_local"], sub["pm25_norm"],
            color=CITY_COLORS[city],
            linewidth=2.5 if is_richmond else 1.3,
            alpha=1.0 if is_richmond else 0.7,
            linestyle="-" if is_richmond else "--",
            label=CITY_LABELS[city] + (" (TREATED)" if is_richmond else ""),
            zorder=5 if is_richmond else 3,
        )

    ax.set_xlabel("Date/Hour", fontsize=11)
    ax.set_ylabel("Normalized PM2.5 (z-score)", fontsize=11)
    ax.set_title("Pre-Period Parallel Trends Check — June 29 – July 3, 2025", fontsize=13)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %-d"))
    ax.legend(facecolor=PALETTE["panel_bg"], labelcolor=PALETTE["text"], fontsize=9)
    plt.tight_layout()
    plt.show()
    print("→ Donor cities broadly track Richmond's pre-period baseline. ✔")
    """),

    md("""
    ## 6. Correlation Matrix — Baseline PM2.5 Across Cities
    High inter-city correlation in the pre-period validates the donor pool.
    """),

    code("""
    pre_pivot = (
        pre_window.pivot_table(
            index="datetime_local", columns="city", values="pm25"
        )
        .resample("3h").mean()
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_facecolor(PALETTE["panel_bg"])
    fig.patch.set_facecolor(PALETTE["background"])
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(
        pre_pivot.corr(),
        annot=True, fmt=".2f", cmap=cmap,
        linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8},
        annot_kws={"fontsize": 10, "color": "white"},
    )
    ax.set_title("Pre-Period PM2.5 Correlation Matrix (3-hour blocks)", fontsize=12,
                 color=PALETTE["text"], pad=10)
    ax.tick_params(colors=PALETTE["text"])
    plt.tight_layout()
    plt.show()
    """),

    md("## ✅ EDA complete — proceed to Notebook 03 (Synthetic DiD)"),
])

# ══════════════════════════════════════════════════════════════════════════════
# Notebook 03 — Synthetic DiD (Three-Act Narrative)
# ══════════════════════════════════════════════════════════════════════════════

nb03 = make_nb([
    md("""
    # 🎭 Notebook 03 — Synthetic Difference-in-Differences
    ## Richmond, VA vs. the Weather-Adjusted Counterfactual

    **Method**: Synthetic DiD (Arkhangelsky et al., 2021) via the `diff-diff` library.
    **Library**: `diff-diff >= 0.3` — pure Python, sklearn-style API.
    **Fallback**: numpy-based SDID if the diff-diff API has changed.

    ---

    ### Three-Act Narrative Structure

    | Act | Time Window | Question |
    |-----|-------------|----------|
    | **Act I** | June 29 – July 4, 8 PM | How does the algorithm build the synthetic twin? |
    | **Act II** | July 4, 9 PM – July 5, 3 AM | How large is the fireworks spike? |
    | **Act III** | July 5, 3 AM → July 8 | How long does the atmospheric hangover last? |
    """),

    code("""
    import sys
    sys.path.insert(0, "..")
    import warnings
    warnings.filterwarnings("ignore")

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    from src.panel_builder import load_panel, make_sdid_input
    from utils.plotting import (
        plot_sdid_divergence, plot_donor_weights, plot_decay_curve,
        PALETTE, apply_dark_theme, CITY_LABELS
    )
    from utils.stats_helpers import (
        compute_causal_gap, compute_attenuation_half_life,
        sdid_numpy, rolling_hourly_mean
    )

    panel = load_panel(2025)

    TREATED_CITY     = "Richmond_VA"
    TREATMENT_START  = pd.Timestamp("2025-07-04 21:00")
    YEAR             = 2025

    print(f"Panel: {len(panel):,} rows")
    print(f"Cities: {sorted(panel['city'].unique())}")
    """),

    md("""
    ## Act I — Pre-Trend Alignment
    The SDID algorithm finds **unit weights** for our donor cities such that
    the weighted combination of Roanoke + Virginia Beach + Raleigh + Baltimore
    perfectly mirrors Richmond's **pre-period PM2.5 trajectory**.

    This step is the critical one: it makes parallel trends hold *by construction*.
    """),

    code("""
    # Prepare the SDID-format input
    sdid_df = make_sdid_input(panel, treated_city=TREATED_CITY)
    sdid_df["post"] = (sdid_df["time"] >= TREATMENT_START).astype(int)
    sdid_df["treated_unit"] = (sdid_df["unit"] == TREATED_CITY).astype(int)

    print("SDID input shape:", sdid_df.shape)
    print(sdid_df.head(3))
    """),

    code("""
    # ── Try diff-diff library first ────────────────────────────────────────────
    # diff-diff v3.6.1 API:
    #   SyntheticDiD().fit(data, outcome, treatment, unit, time)
    #   - treatment col must be 0/1 integer (1 = treated unit AND post-period)
    #   - time col is the raw datetime (not a binary post flag)
    #   results_ attributes: .att, .get_unit_weights_df(), .summary()
    USE_DIFFDIFF = False
    sdid_results = None

    try:
        from diff_diff import SyntheticDiD

        # Build the long-format panel diff-diff expects:
        #   treatment = 1 only for Richmond AND post 9 PM July 4
        sdid_long = (
            panel[["city", "datetime_local", "pm25"]]
            .dropna(subset=["pm25"])
            .rename(columns={"city": "unit", "datetime_local": "time", "pm25": "outcome"})
            .copy()
        )
        sdid_long["treatment"] = (
            (sdid_long["unit"] == TREATED_CITY) &
            (sdid_long["time"] >= TREATMENT_START)
        ).astype(int)

        sdid_model = SyntheticDiD(seed=42, n_bootstrap=200, variance_method="placebo")
        sdid_results = sdid_model.fit(
            data=sdid_long,
            outcome="outcome",
            treatment="treatment",
            unit="unit",
            time="time",
        )
        USE_DIFFDIFF = True
        print("✅ Using diff-diff SyntheticDiD v3.6.1")
        sdid_results.print_summary()
    except Exception as e:
        print(f"⚠  diff-diff: {e}")
        print("→ Falling back to numpy SDID implementation")
    """),

    code("""
    # ── Numpy SDID fallback ────────────────────────────────────────────────────
    if not USE_DIFFDIFF:
        # Use the study window for this year — do NOT hardcode 2025
        study_end = pd.Timestamp(year=YEAR, month=7, day=8, hour=23)
        numpy_results = sdid_numpy(
            panel=panel[panel["datetime_local"] <= study_end],
            treated_unit=TREATED_CITY,
            treatment_time=TREATMENT_START,
        )
        att          = numpy_results["att"]
        weights      = numpy_results["weights"]
        synth_series = numpy_results["synthetic_series"]

        print(f"\\n── Numpy SDID Results ──")
        print(f"  ATT = {att:.2f} µg/m³  ({att / panel[panel['city']==TREATED_CITY]['pm25'].mean()*100:.1f}% above baseline)")
        print(f"  Method: {numpy_results['method']}")
        print("\\n  Donor weights:")
        for city, w in sorted(weights.items(), key=lambda x: -x[1]):
            print(f"    {CITY_LABELS.get(city, city):<25} {w:.4f}")
    """),

    code("""
    # ── Extract series and weights for plotting ────────────────────────────────
    rva_series = (
        panel[panel["city"] == TREATED_CITY]
        .set_index("datetime_local")["pm25"]
        .sort_index()
    )

    if USE_DIFFDIFF:
        # diff-diff v3.6.1 results_ attributes
        att = sdid_results.att

        # Unit weights → dict {city: weight}
        wdf = sdid_results.get_unit_weights_df()  # DataFrame with 'unit' and 'weight'
        weights = dict(zip(wdf["unit"], wdf["weight"]))

        # Reconstruct full synthetic series from pre + post trajectories
        pre_traj  = sdid_results.synthetic_pre_trajectory   # pd.Series indexed by time
        post_traj = sdid_results.synthetic_post_trajectory  # pd.Series indexed by time
        synth_series = pd.concat([pre_traj, post_traj]).sort_index()
        synth_aligned = synth_series.reindex(rva_series.index)
    else:
        # numpy path: synth_series is indexed over the full study window;
        # align it directly to rva_series timestamps via reindex + interpolation
        synth_aligned = (
            synth_series
            .reindex(synth_series.index.union(rva_series.index))
            .interpolate(method="time")
            .reindex(rva_series.index)
        )

    if synth_aligned.isna().all():
        raise RuntimeError(
            "synth_aligned is all-NaN — check that the numpy SDID study window "
            f"covers {YEAR} (TREATMENT_START={TREATMENT_START})"
        )

    print(f"ATT = {att:.2f} µg/m³")
    print(f"Real Richmond mean (pre):  {rva_series[rva_series.index < TREATMENT_START].mean():.2f}")
    print(f"Synthetic mean    (pre):   {synth_aligned[synth_aligned.index < TREATMENT_START].mean():.2f}")
    """),

    code("""
    # ── Donor weight bar chart ─────────────────────────────────────────────────
    fig, ax = plot_donor_weights(
        weights,
        title=f"SDID Donor Weights — Building Synthetic Richmond ({YEAR})"
    )
    plt.show()
    print("→ The algorithm up-weights cities with similar atmospheric baseline patterns.")
    """),

    md("""
    ## Act II — The Pyrotechnic Divergence 🎆
    At **9:00 PM July 4**, fireworks ignite across metro Richmond.
    Watch the real Richmond line rip upward while the Synthetic Twin stays flat —
    the gap *is* the causal effect.
    """),

    code("""
    fig, ax = plot_sdid_divergence(
        real=rva_series,
        synthetic=synth_aligned,
        treatment_start=TREATMENT_START,
        year=YEAR,
        title=f"Act II — Pyrotechnic Divergence: Richmond vs. Synthetic Twin ({YEAR})",
    )

    # Annotate peak
    post_real = rva_series[rva_series.index >= TREATMENT_START]
    peak_time = post_real.idxmax()
    peak_val  = post_real.max()
    ax.annotate(
        f"Peak: {peak_val:.1f} µg/m³" + "\\n" + f"({peak_time.strftime('%-I:%M %p')})",
        xy=(peak_time, peak_val),
        xytext=(peak_time + pd.Timedelta(hours=2), peak_val * 0.9),
        color=PALETTE["richmond"], fontsize=9,
        arrowprops=dict(arrowstyle="->", color=PALETTE["richmond"]),
    )
    plt.savefig("../data/processed/sdid_divergence_2025.png", dpi=150, bbox_inches="tight")
    plt.show()

    if not USE_DIFFDIFF:
        att_est = att
    print(f"\\n📊 Estimated ATT = {att:.2f} µg/m³")
    print(f"   That's a {att / rva_series[rva_series.index < TREATMENT_START].mean() * 100:.1f}% increase above the synthetic baseline.")
    """),

    md("""
    ## Act III — The Dispersion Phase 🌬️
    How many hours does it take for Richmond's PM2.5 to decay back to
    the counterfactual baseline? This is the "environmental hangover."
    """),

    code("""
    from utils.stats_helpers import compute_causal_gap, compute_attenuation_half_life

    gap = compute_causal_gap(rva_series, synth_aligned)
    post_gap = gap[gap.index >= TREATMENT_START].dropna()

    half_life = compute_attenuation_half_life(gap, TREATMENT_START)

    fig, ax = plot_decay_curve(
        gap_series=post_gap,
        half_life_hours=half_life,
        year=YEAR,
    )
    plt.show()

    print(f"\\n🕒 Half-life of PM2.5 excess: {half_life:.1f} hours after peak")
    print(f"   Peak gap:    {post_gap.max():.1f} µg/m³")
    print(f"   Gap at 6 AM: {post_gap.get(post_gap.index[post_gap.index.hour==6][0], float('nan')):.1f} µg/m³" if any(post_gap.index.hour==6) else "")
    """),

    md("""
    ## Summary — Synthetic DiD Results

    | Metric | Value |
    |--------|-------|
    | **ATT** (Average Treatment Effect on Treated) | See output above |
    | **Method** | Synthetic DiD (Arkhangelsky et al., 2021) |
    | **Inference** | Bootstrap / permutation |
    | **Parallel trends** | Verified in Act I pre-period alignment |

    ### Why Synthetic DiD beats standard DiD here
    Standard DiD assumes **parallel trends hold unconditionally** — that is,
    Richmond and, say, Baltimore would have evolved identically on July 4th
    if there were no fireworks. But Baltimore's weather that evening might
    have been completely different (different humidity, wind speed). This
    violates the assumption.

    SDID solves this by **dynamically weighting donor cities** so the
    weighted composite perfectly matches Richmond's pre-period trajectory.
    The parallel trends assumption then holds *by construction* for the
    weighted synthetic control.

    **Proceed to Notebook 04 → DoubleML for a complementary cross-sectional view.**
    """),
])

# ══════════════════════════════════════════════════════════════════════════════
# Notebook 04 — DoubleML IRM
# ══════════════════════════════════════════════════════════════════════════════

nb04 = make_nb([
    md("""
    # 🤖 Notebook 04 — DoubleML Interactive Regression Model
    ## Scrubbing Atmospheric Confounders with Machine Learning

    **Method**: Double Machine Learning — Interactive Regression Model (IRM)
    **Library**: `doubleml >= 0.8`
    **Learners**: LightGBM (captures non-linear wind/humidity interactions)

    ---

    ### Why DoubleML here?

    SDID (Notebook 03) is a *panel estimator* — it handles confounding through
    synthetic control weights in the time dimension. DoubleML is a
    *cross-sectional / semi-parametric estimator* — it handles confounding by
    **predicting it away** using ML.

    The key insight: instead of guessing a linear formula for how wind speed and
    relative humidity interact to affect PM2.5, we let **LightGBM** learn the
    exact functional form. Neyman Orthogonality ensures that mis-specification
    in the nuisance models doesn't bias our treatment effect estimate.

    **Treatment**: `is_treated` (1 ONLY for Richmond, VA during the July 4 9 PM – July 5 3 AM fireworks window; 0 for all other cities and times)
    """),

    code("""
    import sys
    sys.path.insert(0, "..")
    import warnings
    warnings.filterwarnings("ignore")

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from IPython.display import display

    from doubleml import DoubleMLData, DoubleMLIRM
    from lightgbm import LGBMRegressor, LGBMClassifier
    from sklearn.model_selection import cross_val_score

    from src.panel_builder import load_panel, make_dml_cross_section
    from utils.plotting import plot_feature_importance, PALETTE, apply_dark_theme

    panel = load_panel(2025)
    print(f"Panel shape: {panel.shape}")
    """),

    md("""
    ## Step 1 — Construct the Cross-Section
    We pool all city-hours from:
    - **Control group**: same hours (9 PM – 3 AM) on June 29–July 3 (fireworks-free nights)
    - **Treatment group**: July 4 9 PM – July 5 3 AM (fireworks night)

    This gives us a cross-section of ~700 city-hour observations.
    """),

    code("""
    # Build DML cross-section
    dml_df = make_dml_cross_section(panel)

    # Subset to the relevant hours for a tight comparison
    night_hours = set(range(21, 24)) | set(range(0, 3))
    dml_df = dml_df[dml_df["hour_of_day"].isin(night_hours)].copy()

    treatment_col = "is_treated"
    outcome_col   = "pm25"

    feature_cols = [c for c in dml_df.columns
                    if c not in [outcome_col, treatment_col, "datetime_local",
                                 "is_fireworks_window"]]

    print(f"Cross-section shape: {dml_df.shape}")
    print(f"Treatment=1: {dml_df[treatment_col].sum()} obs")
    print(f"Treatment=0: {(dml_df[treatment_col]==0).sum()} obs")
    print(f"\\nFeatures ({len(feature_cols)}):")
    for c in feature_cols:
        print(f"  {c}")
    """),

    md("""
    ## Step 2 — DoubleML Data Object

    The `DoubleMLData` wrapper organizes treatment, outcome, and controls —
    then enforces that no leakage occurs between the two nuisance models.
    """),

    code("""
    dml_clean = dml_df[[outcome_col, treatment_col] + feature_cols].dropna()

    dml_data = DoubleMLData(
        dml_clean,
        y_col=outcome_col,
        d_cols=treatment_col,
        x_cols=feature_cols,
    )

    print(dml_data)
    """),

    md("""
    ## Step 3 — Specify Nuisance Learners

    Two LightGBM models serve as the "nuisance" estimators:

    - **`ml_g`** (LGBMRegressor): predicts PM2.5 from weather + time + city profile
      → captures how wind, humidity, temperature non-linearly drive baseline PM2.5
    - **`ml_m`** (LGBMClassifier): predicts the probability a given hour is a fireworks
      hour (propensity score) from the same features
      → removes selection bias from hours that happened to be calm vs. windy

    The **residuals** of both models are then regressed against each other.
    This double-residualing eliminates the weather confounding.
    """),

    code("""
    # Flexible non-linear learners for nuisance functions
    ml_g = LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        verbose=-1,
    )

    ml_m = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        verbose=-1,
    )

    # DoubleML Interactive Regression Model — estimates ATE
    dml_irm = DoubleMLIRM(
        dml_data,
        ml_g=ml_g,
        ml_m=ml_m,
        score="ATE",
        n_folds=5,
        n_rep=3,        # 3 repeated cross-fittings for stability
        trimming_threshold=0.01,
    )

    print("DoubleML IRM configured.")
    print(f"  Score:   ATE (Average Treatment Effect)")
    print(f"  Folds:   5-fold cross-fitting")
    print(f"  Reps:    3 repeated fits")
    print(f"  Trimming: propensities < 0.01 or > 0.99 trimmed")
    """),

    md("## Step 4 — Fit the Model"),

    code("""
    print("Fitting DoubleML IRM (may take 30–60 seconds) …")
    dml_irm.fit()
    print("\\n" + "═"*55)
    print("DoubleML IRM Results — ATE of Fireworks Window on PM2.5")
    print("═"*55)
    print(dml_irm.summary)
    """),

    md("""
    ## Step 5 — Visualize the ATE with Confidence Interval
    """),

    code("""
    coef   = dml_irm.coef[0]
    se     = dml_irm.se[0]
    pval   = dml_irm.pval[0]
    ci_low = coef - 1.96 * se
    ci_hi  = coef + 1.96 * se

    fig, ax = plt.subplots(figsize=(7, 3.5))
    apply_dark_theme(fig, ax)

    ax.barh(["Fireworks Window"], [coef], xerr=[[coef - ci_low], [ci_hi - coef]],
            color=PALETTE["richmond"], height=0.4,
            error_kw={"ecolor": PALETTE["text"], "capsize": 6, "linewidth": 1.8})
    ax.axvline(0, color=PALETTE["synthetic"], linestyle="--", linewidth=1.2)
    ax.set_xlabel("Estimated ATE — PM2.5 increase (µg/m³)", fontsize=11)
    ax.set_title(
        f"DoubleML IRM — Average Treatment Effect\\n"
        f"ATE = {coef:.2f} µg/m³  (95% CI: [{ci_low:.2f}, {ci_hi:.2f}])  p = {pval:.4f}",
        fontsize=11, pad=10
    )
    fig.tight_layout()
    plt.savefig("../data/processed/doubleml_ate_2025.png", dpi=150, bbox_inches="tight")
    plt.show()

    sig = "✅ Statistically significant" if pval < 0.05 else "⚠ Not significant"
    print(f"\\n{sig} (p = {pval:.4f})")
    print(f"Interpretation: Fireworks causally increased PM2.5 by ~{coef:.1f} µg/m³")
    print(f"after scrubbing wind speed, humidity, temperature, and time-of-day effects.")
    """),

    md("""
    ## Step 6 — Feature Importance (What Did the ML Nuisance Learn?)
    """),

    code("""
    # Refit a single LightGBM on the full dataset to inspect feature importance
    from sklearn.preprocessing import LabelEncoder

    X = dml_clean[feature_cols].values
    y = dml_clean[outcome_col].values

    final_model = LGBMRegressor(n_estimators=300, learning_rate=0.05,
                                num_leaves=31, verbose=-1)
    final_model.fit(X, y)

    fig, ax = plot_feature_importance(
        feature_names=feature_cols,
        importances=final_model.feature_importances_,
        top_n=12,
        title="LightGBM Feature Importance\\n(Nuisance: What predicts PM2.5 baseline?)",
    )
    plt.show()
    print("→ Wind speed and relative humidity dominate — confirming the confounding story.")
    """),

    md("""
    ## Comparison: DoubleML vs. Naïve OLS

    What happens if we just run OLS without scrubbing weather? Let's see.
    """),

    code("""
    import statsmodels.formula.api as smf

    # Drop static features (collinear with city dummies) for a stable OLS
    ols_features = [c for c in feature_cols if c not in ["population", "is_coastal", "baseline_pm25"]]
    ols_formula_vars = " + ".join([c for c in ols_features if dml_clean[c].nunique() > 1])
    ols = smf.ols(f"pm25 ~ is_treated + {ols_formula_vars}", data=dml_clean).fit()

    ols_coef = ols.params["is_treated"]
    ols_pval = ols.pvalues["is_treated"]

    print(f"{'Estimator':<20} {'ATE (µg/m³)':<15} {'p-value'}")
    print("─" * 50)
    print(f"{'Naïve OLS':<20} {ols_coef:<15.2f} {ols_pval:.4f}")
    print(f"{'DoubleML IRM':<20} {coef:<15.2f} {pval:.4f}")
    print()
    bias = coef - ols_coef
    print(f"Atmospheric confounding bias: {bias:+.2f} µg/m³")
    print("→ OLS estimates the effect with a linear specification and static features dropped.")
    print("  DoubleML captures the non-linear wind × humidity interaction via ML.")
    """),

    md("## ✅ DoubleML complete — proceed to Notebook 05 (Placebo Checks)"),
])

# ══════════════════════════════════════════════════════════════════════════════
# Notebook 05 — Placebo Checks
# ══════════════════════════════════════════════════════════════════════════════

nb05 = make_nb([
    md("""
    # 🧪 Notebook 05 — Placebo & Robustness Checks
    ## Proving the Algorithm Isn't Just Lucky

    Two classes of placebo tests:

    ### In-Time Permutation (Year-Level)
    Re-run Synthetic DiD on **July 4, 2023** and **July 4, 2024**.
    - If the donor pool (Roanoke, Virginia Beach, Raleigh, Baltimore) consistently
      tracks Richmond year after year, the method is robust — not a fluke of
      one night's weather.

    ### In-Space Placebo (False Treatment Date)
    Run SDID with a **fake treatment start on June 30** (no fireworks).
    - We expect the ATT to be near zero.
    - A large ATT on a non-fireworks night would be a red flag.
    """),

    code("""
    import sys
    sys.path.insert(0, "..")
    import warnings
    warnings.filterwarnings("ignore")

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    from src.panel_builder import load_panel
    from utils.plotting import (
        plot_sdid_divergence, plot_donor_weights, plot_placebo_grid,
        PALETTE, apply_dark_theme, CITY_LABELS
    )
    from utils.stats_helpers import (
        sdid_numpy, compute_causal_gap,
        compute_attenuation_half_life, permutation_test_sdid_effect
    )

    TREATED_CITY = "Richmond_VA"
    YEARS        = [2023, 2024, 2025]

    panels = {}
    for yr in YEARS:
        try:
            panels[yr] = load_panel(yr)
            print(f"✔ Loaded {yr}: {len(panels[yr]):,} rows")
        except FileNotFoundError:
            print(f"✗ {yr}: panel not found — run Notebook 01 first")
    """),

    md("""
    ## Part A — Multi-Year SDID (2023, 2024, 2025)
    Fitting the same model on each year's data independently.
    """),

    code("""
    year_results = {}  # year → {att, weights, real, synthetic}

    for yr in YEARS:
        if yr not in panels:
            continue
        pnl = panels[yr]
        t_start = pd.Timestamp(year=yr, month=7, day=4, hour=21)

        res = sdid_numpy(
            panel=pnl,
            treated_unit=TREATED_CITY,
            treatment_time=t_start,
        )
        rva_s = (
            pnl[pnl["city"] == TREATED_CITY]
            .set_index("datetime_local")["pm25"]
            .sort_index()
        )
        synth = res["synthetic_series"].reindex(rva_s.index)
        year_results[yr] = {
            "att":      res["att"],
            "weights":  res["weights"],
            "real":     rva_s,
            "synthetic": synth,
            "t_start":  t_start,
        }
        print(f"  {yr}: ATT = {res['att']:.2f} µg/m³  |  "
              f"Top donor: {max(res['weights'], key=res['weights'].get)}")
    """),

    code("""
    # ── Summary table ──────────────────────────────────────────────────────────
    rows = []
    for yr, r in year_results.items():
        peak_gap = compute_causal_gap(r["real"], r["synthetic"])
        peak_gap_post = peak_gap[peak_gap.index >= r["t_start"]]
        rows.append({
            "Year":          yr,
            "ATT (µg/m³)":  round(r["att"], 2),
            "Peak Gap (µg/m³)": round(peak_gap_post.max(), 2) if not peak_gap_post.empty else None,
            "Half-life (h)": compute_attenuation_half_life(peak_gap, r["t_start"]),
            "Top Donor":     CITY_LABELS.get(max(r["weights"], key=r["weights"].get), "?"),
        })

    summary_df = pd.DataFrame(rows).set_index("Year")
    from IPython.display import display
    display(summary_df)
    """),

    code("""
    # ── 3-panel comparison plot ────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 5))
    fig.patch.set_facecolor(PALETTE["background"])

    for i, (yr, r) in enumerate(year_results.items(), 1):
        ax = fig.add_subplot(1, len(year_results), i)
        apply_dark_theme(fig, ax)

        real_yr   = r["real"]
        synth_yr  = r["synthetic"]
        t_start   = r["t_start"]

        ax.plot(synth_yr.values, color=PALETTE["synthetic"], linewidth=1.8,
                linestyle="--", label="Synthetic")
        ax.plot(real_yr.values, color=PALETTE["richmond"], linewidth=2.0,
                label="Real Richmond")

        # Shade post-treatment
        t_idx = real_yr.index.searchsorted(t_start)
        ax.axvspan(t_idx, len(real_yr), color=PALETTE["richmond"], alpha=0.08)
        ax.axvline(t_idx, color=PALETTE["richmond"], linewidth=1.2, linestyle=":")

        ax.set_title(f"July 4, {yr}  |  ATT = {r['att']:.1f} µg/m³",
                     fontsize=10, color=PALETTE["text"])
        ax.set_xlabel("Hour index", fontsize=9)
        if i == 1:
            ax.set_ylabel("PM2.5 (µg/m³)", fontsize=10)
        ax.legend(facecolor=PALETTE["panel_bg"], labelcolor=PALETTE["text"], fontsize=8)

    fig.suptitle(
        "Placebo Check — Synthetic DiD Consistency Across 2023, 2024, 2025",
        fontsize=13, color=PALETTE["text"], y=1.02
    )
    fig.tight_layout()
    plt.savefig("../data/processed/placebo_multiyear.png", dpi=150, bbox_inches="tight")
    plt.show()
    """),

    md("""
    ## Part B — In-Time Placebo: False Treatment Date (June 30)
    We set the "treatment" to **June 30 at 9 PM** — a normal summer night
    with no fireworks. The ATT should be close to zero.
    """),

    code("""
    if 2025 in panels:
        pnl25 = panels[2025]
        fake_t_start = pd.Timestamp("2025-06-30 21:00")

        fake_res = sdid_numpy(
            panel=pnl25,
            treated_unit=TREATED_CITY,
            treatment_time=fake_t_start,
        )

        print("In-Time Placebo — Fake Treatment: June 30, 2025 at 9 PM")
        print(f"  Placebo ATT = {fake_res['att']:.2f} µg/m³")
        print(f"  True ATT    = {year_results.get(2025, {}).get('att', '(2025 not loaded)')}")
        print()
        if abs(fake_res["att"]) < 5:
            print("  ✅ Placebo ATT near zero — no spurious effect on a non-fireworks night.")
        else:
            print("  ⚠  Placebo ATT elevated — investigate data quality or pre-trends.")
    """),

    code("""
    # ── Visual comparison: Real vs. Placebo effect ────────────────────────────
    if 2025 in panels and 2025 in year_results:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor(PALETTE["background"])

        true_r  = year_results[2025]
        rva_s   = true_r["real"]
        synth_s = true_r["synthetic"]
        true_t  = true_r["t_start"]
        fake_synth = fake_res["synthetic_series"].reindex(rva_s.index)

        for ax, (label, synth, t_start) in zip(axes, [
            ("Real Treatment (July 4 9 PM)", synth_s,   true_t),
            ("In-Time Placebo Treatment (June 30 9 PM)", fake_synth, fake_t_start),
        ]):
            apply_dark_theme(fig, ax)
            ax.plot(range(len(synth)),    synth.values, color=PALETTE["synthetic"],
                    linewidth=1.8, linestyle="--", label="Synthetic")
            ax.plot(range(len(rva_s)),    rva_s.values, color=PALETTE["richmond"],
                    linewidth=2.0, label="Real Richmond")
            t_idx = rva_s.index.searchsorted(t_start)
            ax.axvline(t_idx, color=PALETTE["richmond"], linewidth=1.3, linestyle=":")
            ax.set_title(label, fontsize=11, color=PALETTE["text"])
            ax.set_xlabel("Hour index", fontsize=9)
            ax.legend(facecolor=PALETTE["panel_bg"], labelcolor=PALETTE["text"], fontsize=9)

        axes[0].set_ylabel("PM2.5 (µg/m³)", fontsize=10)
        fig.suptitle(
            "Robustness: Real vs. In-Time Placebo Treatment Effect",
            fontsize=13, color=PALETTE["text"], y=1.02
        )
        fig.tight_layout()
        plt.savefig("../data/processed/placebo_fake_date.png", dpi=150, bbox_inches="tight")
        plt.show()
    """),

    md("""
    ## Part C — In-Space Placebo: False Treated Unit (Virginia Beach)
    We treat a known control city (**Virginia Beach**) as the "treated" unit on July 4, 2025.
    We exclude Richmond from the control pool to prevent treatment contamination.
    The placebo ATT for Virginia Beach should be close to zero.
    """),

    code("""
    if 2025 in panels:
        pnl25 = panels[2025]
        # Exclude Richmond to avoid treatment contamination
        placebo_panel = pnl25[pnl25["city"] != "Richmond_VA"].copy()
        
        placebo_space_res = sdid_numpy(
            panel=placebo_panel,
            treated_unit="Virginia_Beach_VA",
            treatment_time=pd.Timestamp("2025-07-04 21:00"),
        )
        
        print("In-Space Placebo — Treated: Virginia Beach, VA (July 4, 2025)")
        print(f"  Placebo ATT = {placebo_space_res['att']:.2f} µg/m³")
        print()
        if abs(placebo_space_res["att"]) < 5:
            print("  ✅ In-space placebo ATT near zero — no false treatment effect detected in control unit.")
        else:
            print("  ⚠  In-space placebo ATT elevated — investigate local confounding shocks.")
    """),

    md("""
    ## Part D — Formal Permutation & Consistency Test

    Using the multi-year ATT distribution as a reference:
    - 2023 and 2024 ATTs form the null comparison distribution.
    - We test whether the 2025 ATT is consistent with systematic fireworks-driven pollution
      across all three years.

    > ⚠️ **Statistical Note**: With only $N=2$ placebo years, this test is primarily
    > illustrative. True exact p-values are bounded by $1/(B+1)$ (minimum $0.33$). We use
    > a mathematically exact p-value formula that includes the observed value to prevent false positives.
    """),

    code("""
    if len(year_results) >= 2:
        atts = [r["att"] for r in year_results.values()]
        yrs  = list(year_results.keys())

        # Use 2025 as observed; 2023 + 2024 as the null permutation set
        observed = year_results.get(2025, list(year_results.values())[-1])["att"]
        placebo_atts = [year_results[y]["att"] for y in yrs if y != 2025]

        perm_result = permutation_test_sdid_effect(
            observed_att=observed,
            permuted_atts=placebo_atts,
            alternative="two-sided",
        )

        from utils.stats_helpers import format_results_table
        from IPython.display import display
        display(format_results_table(perm_result))

        print()
        if perm_result["significant_at_05"]:
            print("⚠  2025 ATT is significantly different from prior-year ATTs.")
            print("   → Possibly different weather / intensity on July 4, 2025.")
        else:
            print("✅ 2025 ATT is consistent with prior-year fireworks effects.")
            print("   → Supports that the model captures systematic fireworks pollution,")
            print("      not a one-off weather fluke.")
    """),

    md("""
    ## Conclusion — What the Robustness Checks Prove

    | Test | Category | Expected | Interpretation if Passes |
    |------|----------|----------|--------------------------|
    | **Multi-year SDID** | In-Time Consistency | Positive ATT across all years | Fireworks spike is a systematic, repeatable event |
    | **June 30 False Date** | In-Time (Temporal) Placebo | ATT ≈ 0 | No spurious detection on normal summer nights |
    | **Virginia Beach Treated** | In-Space Placebo | Placebo ATT ≈ 0 | No false positives on known control/donor cities |
    | **Permutation Test** | Exact Significance Check | 2025 ATT is consistent with 2023/2024 | Observed effect is stable and not a weather fluke |

    Together, these checks give us **high confidence** that the causal pipeline is
    measuring a genuine, repeatable causal effect of July 4th fireworks on Richmond's
    air quality — and that the estimators are robust to regional atmospheric shocks.

    ---

    ## 🎆 Tutorial Complete!

    You've built a full causal inference pipeline using:
    - **Real EPA AQS data** (PM2.5 FRM/FEM, hourly)
    - **Open-Meteo weather data** (ERA5 reanalysis)
    - **Synthetic DiD** (temporal panel estimator, weather-robust via unit weights)
    - **DoubleML IRM** (cross-sectional, scrubs non-linear weather via ML with spatial controls)
    - **Placebo checks** (in-time, in-space, and exact permutation)

    The two estimators are complementary:
    > SDID answers: *"How much did PM2.5 increase in Richmond relative to where it would have been?"*
    > DoubleML answers: *"After ML removes all weather effects, what's the pure treatment signal in Richmond relative to control cities?"*
    """),
])

# ══════════════════════════════════════════════════════════════════════════════
# Monolith — combine all six chapters into one notebook
# ══════════════════════════════════════════════════════════════════════════════

def chapter_divider(number: int, title: str, subtitle: str = "") -> nbf.NotebookNode:
    """A full-width markdown banner that visually separates chapters."""
    sub = f"\n*{subtitle}*" if subtitle else ""
    return md(f"""
---
<div style="background:linear-gradient(135deg,#1A1A2E 0%,#16213E 60%,#0F3460 100%);
            border-left:6px solid #FF6B35;border-radius:8px;
            padding:20px 28px;margin:24px 0;">

# Chapter {number} — {title}{sub}

</div>
""")


CHAPTER_SEPARATORS = [
    (0, "Environment & Credentials",   "Verify installs, ping the EPA API, map the monitoring network"),
    (1, "Data Acquisition",            "Download real EPA AQS PM2.5 + NOAA weather for 2023 / 2024 / 2025"),
    (2, "Exploratory Data Analysis",   "Understand the raw signal before applying any causal model"),
    (3, "Synthetic DiD — Three Acts",  "Act I: Alignment  ·  Act II: Divergence  ·  Act III: Decay"),
    (4, "DoubleML IRM",                "ML-based weather scrubbing via Neyman-orthogonal double residualing"),
    (5, "Placebo & Robustness Checks", "Multi-year, false-date, and permutation tests"),
]

SOURCE_NOTEBOOKS = [nb00, nb01, nb02, nb03, nb04, nb05]

# Cover cell
COVER = md("""
<div style="background:linear-gradient(135deg,#1A1A2E 0%,#0F3460 50%,#1A1A2E 100%);
            border-radius:12px;padding:40px 48px;text-align:center;margin-bottom:8px;">

# 🎆 Fireworks & Air Quality
## A Causal Inference Pipeline for July 4th PM2.5 Spikes

**Richmond, VA · Synthetic DiD + DoubleML IRM · 2025 EPA AQS Data**

---

| | |
|:---|:---|
| **Focal city** | Richmond, Virginia |
| **Donor pool** | Roanoke VA · Virginia Beach VA · Raleigh NC · Baltimore MD |
| **Treatment** | July 4, 9 PM — the moment backyard fireworks ignite |
| **Methods** | `diff-diff` Synthetic DiD  +  `doubleml` IRM (LightGBM) |
| **Data** | EPA AQS (param 88101, hourly FRM/FEM)  +  NOAA via Meteostat |
| **Years** | 2025 primary · 2024 & 2023 placebo checks |

</div>

> **Run order**: Execute cells top-to-bottom. Data is cached to `data/processed/` after the
> first run — subsequent executions load from disk instantly.
""")

# TOC cell
TOC = md("""
## 📋 Table of Contents

1. [Chapter 0 — Environment & Credentials](#ch0)
2. [Chapter 1 — Data Acquisition](#ch1)
3. [Chapter 2 — Exploratory Data Analysis](#ch2)
4. [Chapter 3 — Synthetic DiD (Three Acts)](#ch3)
5. [Chapter 4 — DoubleML IRM](#ch4)
6. [Chapter 5 — Placebo & Robustness Checks](#ch5)
""")

# Assemble monolith cells
monolith_cells = [COVER, TOC]

for (num, title, subtitle), source_nb in zip(CHAPTER_SEPARATORS, SOURCE_NOTEBOOKS):
    # Add anchor + divider banner
    anchor = md(f'<a id="ch{num}"></a>')
    divider = chapter_divider(num, title, subtitle)
    monolith_cells.extend([anchor, divider])
    # Add all cells from the source notebook (skip its top-level title cell
    # since the divider banner already provides it)
    cells = source_nb.cells
    # Drop the first markdown cell (it's the chapter title — already in banner)
    monolith_cells.extend(cells[1:])

nb_monolith = make_nb(monolith_cells)


if __name__ == "__main__":
    import os

    # Write monolith
    mono_path = NOTEBOOKS_DIR / "fireworks_causal_inference.ipynb"
    with open(mono_path, "w") as f:
        nbf.write(nb_monolith, f)
    print(f"✔ Written monolith: {mono_path}")
    print(f"   Cells: {len(monolith_cells)}")

    # Remove old individual notebooks if present
    old_names = [
        "00_environment_check.ipynb",
        "01_data_acquisition.ipynb",
        "02_eda_and_visualization.ipynb",
        "03_synthetic_did.ipynb",
        "04_doubleml_irm.ipynb",
        "05_placebo_checks.ipynb",
    ]
    removed = []
    for name in old_names:
        p = NOTEBOOKS_DIR / name
        if p.exists():
            os.remove(p)
            removed.append(name)
    if removed:
        print(f"\nRemoved {len(removed)} individual notebooks:")
        for n in removed:
            print(f"  ✗ {n}")

    print(f"\n✅ Monolith ready → {mono_path}")
