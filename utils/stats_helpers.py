"""
utils/stats_helpers.py
─────────────────────────────────────────────────────────────────────────────
Statistical and domain-specific helper functions for the fireworks tutorial.

Covers:
  - EPA AQI calculation from PM2.5 24-hour averages
  - Rolling hourly smoothing
  - Causal gap / decay analysis (Act III)
  - Placebo permutation test utilities
  - Simple SDID fallback (numpy) when diff-diff API has changed
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


# ── 1. AQI conversion (PM2.5 → AQI) ──────────────────────────────────────────

# EPA PM2.5 breakpoints (µg/m³) → AQI breakpoints (unitless)
# Source: EPA Technical Assistance Document for PM2.5 NAAQS AQI (2024)
_AQI_BREAKPOINTS = [
    # (C_lo, C_hi, AQI_lo, AQI_hi, category)
    (0.0,    9.0,    0,   50,  "Good"),
    (9.1,   35.4,   51,  100,  "Moderate"),
    (35.5,  55.4,  101,  150,  "Unhealthy for Sensitive Groups"),
    (55.5, 125.4,  151,  200,  "Unhealthy"),
    (125.5, 225.4, 201,  300,  "Very Unhealthy"),
    (225.5, 325.4, 301,  500,  "Hazardous"),
]


def pm25_to_aqi(pm25: float) -> tuple[float, str]:
    """
    Convert a single PM2.5 concentration (µg/m³) to an AQI value + category.

    Uses EPA truncation (truncate PM2.5 to 1 decimal before lookup).
    Returns (aqi, category_string).
    """
    if np.isnan(pm25) or pm25 < 0:
        return np.nan, "Unknown"
    pm25 = float(int(pm25 * 10)) / 10  # truncate to 1 decimal
    for c_lo, c_hi, aqi_lo, aqi_hi, category in _AQI_BREAKPOINTS:
        if c_lo <= pm25 <= c_hi:
            aqi = ((aqi_hi - aqi_lo) / (c_hi - c_lo)) * (pm25 - c_lo) + aqi_lo
            return round(aqi), category
    # Beyond hazardous
    return 500.0, "Hazardous"


def pm25_series_to_aqi(series: pd.Series) -> pd.DataFrame:
    """Vectorised AQI conversion; returns DataFrame with aqi and aqi_category columns.

    Always returns a zero-based RangeIndex so it can be safely pd.concat'd
    with any DataFrame that has been reset_index(drop=True).
    """
    results = [pm25_to_aqi(v) for v in series]
    aqi_vals, cats = zip(*results) if results else ([], [])
    return pd.DataFrame({"aqi": list(aqi_vals), "aqi_category": list(cats)})


# ── 2. Rolling averages ───────────────────────────────────────────────────────

def rolling_hourly_mean(
    df: pd.DataFrame,
    value_col: str = "pm25",
    window: int = 3,
    group_col: str = "city",
) -> pd.DataFrame:
    """
    Compute a rolling mean over `window` hours, grouped by city.
    Returns a copy of df with a new column f"{value_col}_roll{window}h".
    """
    df = df.copy().sort_values([group_col, "datetime_local"])
    df[f"{value_col}_roll{window}h"] = (
        df.groupby(group_col)[value_col]
        .transform(lambda s: s.rolling(window, min_periods=1, center=True).mean())
    )
    return df


# ── 3. Causal gap / decay analysis ───────────────────────────────────────────

def compute_causal_gap(
    real: pd.Series,
    synthetic: pd.Series,
) -> pd.Series:
    """
    Point-wise difference: real − synthetic.
    Both Series must share the same index (datetime).
    """
    return (real - synthetic.reindex(real.index)).fillna(0.0)


def compute_attenuation_half_life(
    gap: pd.Series,
    treatment_start: pd.Timestamp,
) -> float:
    """
    Estimate the 'environmental hangover' half-life.

    Finds the first hour after treatment start where the gap falls below
    half of its peak value (post-peak). Returns hours from peak.
    """
    post_gap = gap[gap.index >= treatment_start].dropna()
    if post_gap.empty or post_gap.max() <= 0:
        return float("nan")

    peak_val  = post_gap.max()
    half_val  = peak_val / 2.0
    below_half = post_gap[post_gap.index > post_gap.idxmax()][post_gap <= half_val]

    if below_half.empty:
        return float("nan")

    peak_time  = post_gap.idxmax()
    decay_time = below_half.index[0]
    hours = (decay_time - peak_time).total_seconds() / 3600
    return round(hours, 2)


# ── 4. Placebo permutation test ───────────────────────────────────────────────

def permutation_test_sdid_effect(
    observed_att: float,
    permuted_atts: list[float],
    alternative: str = "greater",
) -> dict:
    """
    Permutation test: does the observed ATT significantly exceed
    the distribution of placebo ATTs?

    Uses the standard exact p-value formula: (sum(placebo >= observed) + 1) / (B + 1)
    to avoid false positives with small permutation sizes.

    Parameters
    ----------
    observed_att  : ATT from the real July 4 analysis
    permuted_atts : ATTs from placebo runs (fake treatment dates / years)
    alternative   : "greater" | "two-sided"

    Returns dict with p_value, mean_placebo, std_placebo, z_score.
    """
    arr = np.array(permuted_atts)
    mean_p = arr.mean() if len(arr) > 0 else np.nan
    std_p  = arr.std(ddof=1) if len(arr) > 1 else np.nan
    z = (observed_att - mean_p) / std_p if std_p and not np.isnan(std_p) else np.nan

    B = len(arr)
    if B == 0:
        p_val = 1.0
    elif alternative == "greater":
        p_val = (np.sum(arr >= observed_att) + 1) / (B + 1)
    else:
        p_val = (np.sum(np.abs(arr - mean_p) >= np.abs(observed_att - mean_p)) + 1) / (B + 1)

    return {
        "observed_att":  observed_att,
        "mean_placebo":  round(mean_p, 4) if not np.isnan(mean_p) else None,
        "std_placebo":   round(std_p, 4) if not np.isnan(std_p) else None,
        "z_score":       round(z, 3)    if not np.isnan(z)    else None,
        "p_value":       round(p_val, 4),
        "n_permutations": B,
        "significant_at_05": p_val < 0.05,
    }



# ── 5. Numpy SDID fallback ────────────────────────────────────────────────────

def sdid_numpy(
    panel: pd.DataFrame,
    treated_unit: str,
    treatment_time: pd.Timestamp,
    outcome_col: str = "pm25",
    unit_col: str = "city",
    time_col: str = "datetime_local",
) -> dict:
    """
    Pure-numpy Synthetic DiD estimator — fallback when diff-diff API is
    unavailable or its signature has changed.

    Implements the Arkhangelsky et al. (2021) SDID algorithm:
      1. Solve for unit weights ω so that pre-period control average
         matches treated unit's pre-period trajectory.
      2. Solve for time weights λ (uniform by default in simple version).
      3. Compute ATT as weighted post-period diff-in-diff.

    This is a simplified version (no regularization λ). For the full
    estimator with inference, use diff_diff.SyntheticDiD.

    Returns dict with keys: att, synthetic_series, weights.
    """
    df = panel[[unit_col, time_col, outcome_col]].dropna()
    pivot = df.pivot_table(index=time_col, columns=unit_col, values=outcome_col, aggfunc="mean")
    pivot = pivot.sort_index()

    # ── Fill gaps (EPA hourly data has sporadic missing readings) ─────────────
    # Forward-fill up to 2 hours, then back-fill, then drop rows that still
    # have NaN so nnls never receives an ill-conditioned matrix.
    n_raw = len(pivot)
    pivot = pivot.ffill(limit=2).bfill(limit=2)
    pivot = pivot.dropna()
    if len(pivot) < n_raw:
        print(f"  sdid_numpy: dropped {n_raw - len(pivot)} rows with unfillable gaps")

    control_cols = [c for c in pivot.columns if c != treated_unit]
    if not control_cols:
        raise ValueError("Need at least one control unit.")
    if treated_unit not in pivot.columns:
        raise ValueError(f"Treated unit '{treated_unit}' not found in pivot columns: {list(pivot.columns)}")

    pre  = pivot[pivot.index <  treatment_time]
    post = pivot[pivot.index >= treatment_time]

    if pre.empty:
        raise ValueError(f"No pre-period rows found before {treatment_time}. Check treatment_time.")

    # Step 1: solve for unit weights via non-negative least squares
    from scipy.optimize import nnls
    A = pre[control_cols].values        # (T_pre × N_control)
    b = pre[treated_unit].values        # (T_pre,)

    # Sanity check — should be impossible after dropna above
    if not (np.isfinite(A).all() and np.isfinite(b).all()):
        raise ValueError("NaN/Inf in pre-period matrix after gap-fill. Check input data.")

    raw_weights, _ = nnls(A, b)
    weights = raw_weights / (raw_weights.sum() + 1e-12)

    # Step 2: synthetic series
    synthetic = pivot[control_cols].values @ weights
    synthetic_series = pd.Series(synthetic, index=pivot.index, name="synthetic")

    # Step 3: ATT
    treated_pre_mean  = pre[treated_unit].mean()
    treated_post_mean = post[treated_unit].mean()
    synth_pre_mean    = (pre[control_cols].values @ weights).mean()
    synth_post_mean   = (post[control_cols].values @ weights).mean()

    att = (treated_post_mean - treated_pre_mean) - (synth_post_mean - synth_pre_mean)

    return {
        "att": round(att, 4),
        "synthetic_series": synthetic_series,
        "weights": dict(zip(control_cols, weights)),
        "method": "sdid_numpy_fallback",
    }


# ── 6. Summary table formatter ────────────────────────────────────────────────

def format_results_table(results: dict) -> pd.DataFrame:
    """
    Pretty-print a results dict as a Pandas DataFrame row for display
    in a Jupyter notebook.
    """
    return pd.DataFrame([results])
