"""
src/panel_builder.py
─────────────────────────────────────────────────────────────────────────────
Merges EPA AQS PM2.5 data with NOAA weather data into analysis-ready
panel DataFrames saved as Parquet files.

Output files
────────────
data/processed/panel_{year}.parquet        — hourly panel, all cities
data/processed/panel_july4_{year}.parquet  — June 29–July 8 window

Key derived columns
───────────────────
  hour_of_day         0–23
  day_of_week         0=Mon … 6=Sun
  is_july4            bool, True on July 4th
  is_post_9pm         bool, hour >= 21
  is_fireworks_window bool, July 4 AND hour in [21, 22, 23, 0, 1, 2]
  is_treated          bool = is_fireworks_window (primary treatment indicator)
  minutes_since_9pm   continuous version (0 at 9 PM July 4, NaN outside)
  baseline_pm25       city-level median PM2.5 for June 29 – July 3
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.aqs_client import AQSClient, download_bulk_pm25, TARGET_COUNTIES
from src.weather_client import WeatherClient

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"


# ── City metadata (population 2024 census estimate) ───────────────────────────

CITY_META: dict[str, dict] = {
    # Treated city
    "Richmond_VA":        {"population": 228_783, "state": "VA", "is_coastal": False, "is_rural": False},
    # Confirmed donors (have EPA 88101 hourly monitors)
    "Charlottesville_VA": {"population":  46_597, "state": "VA", "is_coastal": False, "is_rural": False},
    "Virginia_Beach_VA":  {"population": 460_297, "state": "VA", "is_coastal": True,  "is_rural": False},
    "Raleigh_NC":         {"population": 479_576, "state": "NC", "is_coastal": False, "is_rural": False},
    "Baltimore_MD":       {"population": 568_271, "state": "MD", "is_coastal": True,  "is_rural": False},
    # Rural donors (optional — may not have data every year)
    "Rockingham_VA":      {"population":  80_000, "state": "VA", "is_coastal": False, "is_rural": True},
    "Page_VA":            {"population":  23_000, "state": "VA", "is_coastal": False, "is_rural": True},
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _add_treatment_cols(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Add causal inference treatment / time columns."""
    df = df.copy()

    # Drop rows where datetime parsing failed (NaT) — they'd produce NaN booleans
    df = df.dropna(subset=["datetime_local"]).reset_index(drop=True)

    dt = df["datetime_local"]
    df["hour_of_day"]  = dt.dt.hour
    df["day_of_week"]  = dt.dt.dayofweek
    df["date"]         = dt.dt.date

    july4 = pd.Timestamp(year=year, month=7, day=4)
    df["is_july4"]    = (dt.dt.date == july4.date()).fillna(False)
    df["is_post_9pm"] = (dt.dt.hour >= 21).fillna(False)

    # Fireworks window: July 4 9 PM through July 5 2 AM (incl.)
    df["is_fireworks_window"] = (
        (
            (dt.dt.date == july4.date()) & (dt.dt.hour >= 21)
        ) | (
            (dt.dt.date == (july4 + pd.Timedelta(days=1)).date()) & (dt.dt.hour < 3)
        )
    ).fillna(False)
    df["is_treated"] = df["is_fireworks_window"].astype(int)

    # Continuous: minutes since 9 PM July 4 (for decay analysis)
    nine_pm = july4.replace(hour=21)
    df["minutes_since_9pm"] = (dt - nine_pm).dt.total_seconds().div(60).where(
        df["is_fireworks_window"], other=np.nan
    )

    # DiD post-period flag: July 4 9pm onward vs. same hours June 29–July 3
    df["is_post"] = (df["is_july4"] & df["is_post_9pm"]).fillna(False)

    return df


def _add_baseline_pm25(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Add per-city baseline PM2.5 (median over June 29 – July 3)."""
    baseline_mask = (
        (df["datetime_local"].dt.month == 6) & (df["datetime_local"].dt.day >= 29)
    ) | (
        (df["datetime_local"].dt.month == 7) & (df["datetime_local"].dt.day <= 3)
    )
    baseline_stats = (
        df[baseline_mask]
        .groupby("city")["pm25"]
        .agg(baseline_pm25="median", baseline_pm25_std="std")
        .reset_index()
    )
    return df.merge(baseline_stats, on="city", how="left")


def _add_city_meta(df: pd.DataFrame) -> pd.DataFrame:
    """Join in population and coastal flag."""
    meta_df = pd.DataFrame(CITY_META).T.reset_index().rename(columns={"index": "city"})
    meta_df["population"] = meta_df["population"].astype(int)
    return df.merge(meta_df, on="city", how="left")


def _add_wind_vectors(df: pd.DataFrame) -> pd.DataFrame:
    """Decompose wind speed (wspd_mph) and direction (wdir) into U and V components."""
    df = df.copy()
    if "wspd_mph" in df.columns and "wdir" in df.columns:
        # Convert wind direction from degrees to radians.
        # Meteorological convention: 0 degrees is North, increasing clockwise.
        # Wind vector points in the direction of flow.
        rad = np.radians(df["wdir"])
        df["wind_u"] = -df["wspd_mph"] * np.sin(rad)
        df["wind_v"] = -df["wspd_mph"] * np.cos(rad)
    else:
        df["wind_u"] = 0.0
        df["wind_v"] = 0.0
    return df


# ── Main build functions ───────────────────────────────────────────────────────

def build_panel(
    year: int,
    use_bulk: bool = True,   # bulk CSV is the reliable default for 2023–2025
    include_rural: bool = True, # include rural controls (Rockingham) by default
    aqs_email: Optional[str] = None,
    aqs_key:   Optional[str] = None,
    force:     bool = False,
) -> pd.DataFrame:
    """
    Download (or load cached) and merge PM2.5 + weather for `year`.

    Parameters
    ----------
    year          : Calendar year (2023, 2024, 2025 …)
    use_bulk      : If True, use EPA pre-generated bulk CSV instead of API
    include_rural : If True, include rural background counties (Rockingham_VA)
    force         : Re-download even if parquet cache exists

    Returns
    -------
    Hourly panel DataFrame saved to data/processed/panel_{year}.parquet
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / f"panel_{year}.parquet"

    if out_path.exists() and not force:
        print(f"✔ Loaded cached panel: {out_path}")
        return pd.read_parquet(out_path)

    # ── Step 1: AQS PM2.5 ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Building panel for {year}")
    print(f"{'='*60}")

    bdate = f"{year}0629"
    edate = f"{year}0708"

    if use_bulk:
        print("Using EPA bulk pre-generated file (default — most reliable) …")
        pm_df = download_bulk_pm25(year, include_rural=include_rural)
        # Filter to study window
        pm_df = pm_df[
            (pm_df["datetime_local"] >= f"{year}-06-29")
            & (pm_df["datetime_local"] <= f"{year}-07-08 23:59")
        ]
    else:
        print("Fetching via EPA AQS county-level API …")
        client = AQSClient(email=aqs_email, key=aqs_key)
        pm_df = client.get_all_counties_hourly(bdate, edate, include_rural=include_rural)
        # API fallback: if no data, try bulk
        if pm_df.empty:
            print("  API returned no data — falling back to bulk download")
            pm_df = download_bulk_pm25(year, include_rural=include_rural)
            pm_df = pm_df[
                (pm_df["datetime_local"] >= f"{year}-06-29")
                & (pm_df["datetime_local"] <= f"{year}-07-08 23:59")
            ]

    if pm_df.empty:
        raise RuntimeError(
            f"No PM2.5 data for {year}. "
            "Note: 2025 EPA data may not yet be certified (QA lag ~6 months). "
            "Try year=2024 or year=2023."
        )

    pm_df["datetime_local"] = pd.to_datetime(pm_df["datetime_local"])

    # ── Step 2: Weather ────────────────────────────────────────────────────────
    print("\nFetching weather data via Meteostat …")
    wx_client = WeatherClient()
    wx_df = wx_client.get_weather_window(year, pre_days=5, post_days=4)

    if wx_df.empty:
        print("⚠ Weather data unavailable; proceeding without it.")
    else:
        wx_df["datetime_local"] = pd.to_datetime(wx_df["datetime_local"])
        # Round to nearest hour for join
        wx_df["datetime_local"] = wx_df["datetime_local"].dt.round("h")

    # ── Step 3: Merge ──────────────────────────────────────────────────────────
    if not wx_df.empty:
        panel = pm_df.merge(wx_df, on=["city", "datetime_local"], how="left")
    else:
        panel = pm_df.copy()

    # ── Step 4: Derived columns ────────────────────────────────────────────────
    panel = _add_treatment_cols(panel, year)
    panel = _add_baseline_pm25(panel, year)
    panel = _add_city_meta(panel)
    panel = _add_wind_vectors(panel)

    # Sort
    panel = panel.sort_values(["city", "datetime_local"]).reset_index(drop=True)

    panel.to_parquet(out_path, index=False)
    print(f"\n✔ Saved panel → {out_path}  ({len(panel):,} rows × {panel.shape[1]} cols)")
    return panel


def load_panel(year: int) -> pd.DataFrame:
    """Load a cached panel, raising if not yet built."""
    path = PROCESSED_DIR / f"panel_{year}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Panel for {year} not found. Run build_panel({year}) first."
        )
    return pd.read_parquet(path)


def make_sdid_input(
    panel: pd.DataFrame,
    treated_city: str = "Richmond_VA",
) -> pd.DataFrame:
    """
    Reshape the hourly panel into the city × hour format expected by
    diff-diff's SyntheticDiD.

    Returns a DataFrame with columns:
        unit (city), time (datetime_local), outcome (pm25),
        treated (1 if treated_city AND is_post else 0)
    """
    df = panel[["city", "datetime_local", "pm25", "is_post", "is_july4"]].copy()
    df = df.dropna(subset=["pm25"])
    df = df.rename(columns={"city": "unit", "datetime_local": "time", "pm25": "outcome"})
    df["treated"] = (df["unit"] == treated_city) & df["is_post"]
    return df


def make_dml_cross_section(
    panel: pd.DataFrame,
    treated_city: str = "Richmond_VA",
    feature_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Create a city-hour cross-section for DoubleML.

    Treatment = is_fireworks_window only for the treated city (spatial controls).
    Features  = weather + time-of-day + city metadata.
    """
    default_features = [
        "temp", "rhum", "wspd_mph", "wdir", "prcp",
        "hour_of_day", "day_of_week",
        "population", "is_coastal",
        "baseline_pm25",
    ]
    feat_cols = feature_cols or default_features

    df = panel[["city", "datetime_local", "pm25", "is_fireworks_window"] + feat_cols].copy()
    df = df.dropna(subset=["pm25"])
    
    # Correct causal framing: treatment is 1 ONLY for the focal treated city during fireworks window
    df["is_treated"] = ((df["city"] == treated_city) & df["is_fireworks_window"]).astype(int)

    # One-hot encode categorical city (for DoubleML cross-fitting)
    city_dummies = pd.get_dummies(df["city"], prefix="city", drop_first=True)
    df = pd.concat([df.drop(columns=["city"]), city_dummies], axis=1)

    return df.dropna(subset=feat_cols).reset_index(drop=True)
