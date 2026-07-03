"""
utils/plotting.py
─────────────────────────────────────────────────────────────────────────────
Reusable, publication-quality plot helpers for the fireworks tutorial.

All functions return (fig, ax) unless otherwise noted.
Color palette is smoke-and-fire themed: charcoal backgrounds, ember oranges,
sky blues, and muted greens for controls.
"""

from __future__ import annotations

from typing import Optional, Union

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ── Global style ───────────────────────────────────────────────────────────────

PALETTE = {
    "richmond":    "#FF6B35",   # ember orange (treated city)
    "synthetic":   "#74C2E1",   # sky blue    (synthetic twin)
    "causal_fill": "#FF6B3530", # translucent orange fill
    "roanoke":     "#8BC34A",
    "va_beach":    "#29B6F6",
    "raleigh":     "#CE93D8",
    "baltimore":   "#FFCA28",
    "background":  "#1A1A2E",
    "panel_bg":    "#16213E",
    "grid":        "#2D3561",
    "text":        "#E0E0E0",
    "accent":      "#FF6B35",
}

AQI_BANDS = [
    (0,   12,   "#00E400", "Good"),
    (12,  35.4, "#FFFF00", "Moderate"),
    (35.4,55.4, "#FF7E00", "Unhealthy for Sensitive"),
    (55.4,150.4,"#FF0000", "Unhealthy"),
    (150.4,250.4,"#99004C","Very Unhealthy"),
]

CITY_COLORS: dict[str, str] = {
    "Richmond_VA":       PALETTE["richmond"],
    "Roanoke_VA":        PALETTE["roanoke"],
    "Virginia_Beach_VA": PALETTE["va_beach"],
    "Raleigh_NC":        PALETTE["raleigh"],
    "Baltimore_MD":      PALETTE["baltimore"],
}

CITY_LABELS: dict[str, str] = {
    "Richmond_VA":       "Richmond, VA",
    "Roanoke_VA":        "Roanoke, VA",
    "Virginia_Beach_VA": "Virginia Beach, VA",
    "Raleigh_NC":        "Raleigh, NC",
    "Baltimore_MD":      "Baltimore, MD",
}


def apply_dark_theme(fig: plt.Figure, ax: plt.Axes) -> None:
    fig.patch.set_facecolor(PALETTE["background"])
    ax.set_facecolor(PALETTE["panel_bg"])
    ax.tick_params(colors=PALETTE["text"], labelsize=10)
    ax.xaxis.label.set_color(PALETTE["text"])
    ax.yaxis.label.set_color(PALETTE["text"])
    ax.title.set_color(PALETTE["text"])
    for spine in ax.spines.values():
        spine.set_color(PALETTE["grid"])
    ax.grid(color=PALETTE["grid"], linewidth=0.6, linestyle="--", alpha=0.7)


# ── 1. Raw PM2.5 time series ──────────────────────────────────────────────────

def plot_pm25_timeseries(
    panel: pd.DataFrame,
    cities: Optional[list[str]] = None,
    year: int = 2025,
    title: Optional[str] = None,
    figsize: tuple = (14, 5),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot hourly PM2.5 for one or more cities over the fireworks study window.
    Shades the fireworks window (July 4 9 PM – July 5 3 AM) in amber.
    """
    cities = cities or list(CITY_LABELS.keys())
    fig, ax = plt.subplots(figsize=figsize)
    apply_dark_theme(fig, ax)

    for city in cities:
        sub = panel[panel["city"] == city].sort_values("datetime_local")
        ax.plot(
            sub["datetime_local"], sub["pm25"],
            color=CITY_COLORS.get(city, "white"),
            linewidth=1.8,
            alpha=0.9,
            label=CITY_LABELS.get(city, city),
        )

    # Shade fireworks window
    fw_start = pd.Timestamp(year=year, month=7, day=4, hour=21)
    fw_end   = pd.Timestamp(year=year, month=7, day=5, hour=3)
    ax.axvspan(fw_start, fw_end, color="#FF6B35", alpha=0.12, label="Fireworks window")
    ax.axvline(fw_start, color="#FF6B35", linewidth=1.2, linestyle="--", alpha=0.7)

    ax.set_xlabel("Date / Hour (local time)", fontsize=11)
    ax.set_ylabel("PM2.5 (µg/m³)", fontsize=11)
    ax.set_title(title or f"Hourly PM2.5 — July 4th Study Window ({year})", fontsize=13, pad=14)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %-d\n%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    ax.legend(facecolor=PALETTE["panel_bg"], labelcolor=PALETTE["text"], fontsize=9)
    fig.tight_layout()
    return fig, ax


# ── 2. Synthetic DiD divergence plot (Act II) ─────────────────────────────────

def plot_sdid_divergence(
    real: pd.Series,
    synthetic: pd.Series,
    treatment_start: pd.Timestamp,
    treatment_end: Optional[pd.Timestamp] = None,
    ci_lower: Optional[pd.Series] = None,
    ci_upper: Optional[pd.Series] = None,
    year: int = 2025,
    figsize: tuple = (14, 6),
    title: Optional[str] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    The money plot. Overlays real Richmond vs. Synthetic Twin,
    fills the causal gap, and annotates AQI bands.

    Parameters
    ----------
    real, synthetic : pd.Series indexed by datetime
    treatment_start : pd.Timestamp — moment fireworks begin (9 PM July 4)
    ci_lower/upper  : optional confidence interval bands for synthetic
    """
    fig, ax = plt.subplots(figsize=figsize)
    apply_dark_theme(fig, ax)

    # AQI horizontal band overlays
    for lo, hi, color, label in AQI_BANDS:
        ax.axhspan(lo, hi, alpha=0.06, color=color)

    # Synthetic twin (dashed)
    ax.plot(
        synthetic.index, synthetic.values,
        color=PALETTE["synthetic"], linewidth=2.2, linestyle="--",
        label="Synthetic Richmond (counterfactual)", zorder=4
    )

    # CI shading
    if ci_lower is not None and ci_upper is not None:
        ax.fill_between(
            synthetic.index, ci_lower.values, ci_upper.values,
            color=PALETTE["synthetic"], alpha=0.15, label="95% CI (bootstrap)"
        )

    # Real Richmond (solid)
    ax.plot(
        real.index, real.values,
        color=PALETTE["richmond"], linewidth=2.5,
        label="Real Richmond, VA", zorder=5
    )

    # Causal gap fill (post-treatment only)
    post_mask = real.index >= treatment_start
    if post_mask.any():
        ax.fill_between(
            real.index[post_mask],
            synthetic.reindex(real.index[post_mask]).values,
            real.values[post_mask],
            color=PALETTE["richmond"],
            alpha=0.20,
            label="Causal effect (ATT)",
            zorder=3,
        )

    # Treatment line
    ax.axvline(treatment_start, color="#FF6B35", linewidth=1.5, linestyle=":")
    ax.annotate(
        "🎆 Fireworks begin\n(9:00 PM)",
        xy=(treatment_start, ax.get_ylim()[1] * 0.85),
        xytext=(treatment_start + pd.Timedelta(hours=1.5), ax.get_ylim()[1] * 0.85),
        color="#FF6B35", fontsize=9,
        arrowprops=dict(arrowstyle="->", color="#FF6B35", lw=1.2),
    )

    ax.set_xlabel("Date / Hour (local time)", fontsize=11)
    ax.set_ylabel("PM2.5 (µg/m³)", fontsize=11)
    ax.set_title(
        title or f"Synthetic DiD — Richmond, VA vs. Counterfactual ({year})",
        fontsize=13, pad=14
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %-d\n%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    ax.legend(facecolor=PALETTE["panel_bg"], labelcolor=PALETTE["text"], fontsize=9)
    fig.tight_layout()
    return fig, ax


# ── 3. Donor weights bar chart ────────────────────────────────────────────────

def plot_donor_weights(
    weights: dict[str, float],
    figsize: tuple = (7, 4),
    title: str = "Synthetic Control Donor Weights",
) -> tuple[plt.Figure, plt.Axes]:
    """Bar chart of SDID donor weights assigned to each control city."""
    labels = [CITY_LABELS.get(k, k) for k in weights]
    values = list(weights.values())
    colors = [CITY_COLORS.get(k, "#aaa") for k in weights]

    fig, ax = plt.subplots(figsize=figsize)
    apply_dark_theme(fig, ax)
    bars = ax.barh(labels, values, color=colors, edgecolor="none", height=0.55)
    for bar, val in zip(bars, values):
        ax.text(
            val + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", color=PALETTE["text"], fontsize=10
        )
    ax.set_xlim(0, max(values) * 1.2)
    ax.set_xlabel("Weight", fontsize=11)
    ax.set_title(title, fontsize=12, pad=10)
    fig.tight_layout()
    return fig, ax


# ── 4. Placebo comparison grid ────────────────────────────────────────────────

def plot_placebo_grid(
    results: dict[int, tuple[pd.Series, pd.Series]],
    treatment_hours: dict[int, pd.Timestamp],
    figsize: tuple = (15, 4.5),
) -> tuple[plt.Figure, list[plt.Axes]]:
    """
    Multi-panel plot comparing real vs. synthetic for multiple years
    (e.g. 2023, 2024, 2025) side-by-side for placebo validation.

    Parameters
    ----------
    results  : {year: (real_series, synthetic_series)}
    treatment_hours : {year: pd.Timestamp of fireworks start}
    """
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=figsize, sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (year, (real, synthetic)) in zip(axes, results.items()):
        apply_dark_theme(fig, ax)
        ax.plot(synthetic.values, color=PALETTE["synthetic"], linewidth=1.8,
                linestyle="--", label="Synthetic")
        ax.plot(real.values, color=PALETTE["richmond"], linewidth=2.0,
                label="Real Richmond")
        # Mark treatment hour
        t_idx = real.index.get_indexer([treatment_hours[year]], method="nearest")[0]
        ax.axvline(t_idx, color="#FF6B35", linewidth=1.2, linestyle=":")
        ax.set_title(f"July 4, {year}", fontsize=11, color=PALETTE["text"])
        ax.set_xlabel("Hour (relative)", fontsize=9)

    axes[0].set_ylabel("PM2.5 (µg/m³)", fontsize=10)
    axes[0].legend(facecolor=PALETTE["panel_bg"], labelcolor=PALETTE["text"], fontsize=8)
    fig.suptitle(
        "Placebo Check — Synthetic DiD Consistency Across Years",
        fontsize=13, color=PALETTE["text"], y=1.02
    )
    fig.tight_layout()
    return fig, axes


# ── 5. Decay / dispersion curve (Act III) ────────────────────────────────────

def plot_decay_curve(
    gap_series: pd.Series,
    half_life_hours: Optional[float] = None,
    year: int = 2025,
    figsize: tuple = (10, 4.5),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot the causal gap (real − synthetic) from fireworks start,
    decaying back to zero. Annotates the half-life if provided.
    """
    fig, ax = plt.subplots(figsize=figsize)
    apply_dark_theme(fig, ax)

    ax.fill_between(range(len(gap_series)), 0, gap_series.values,
                    color=PALETTE["richmond"], alpha=0.35)
    ax.plot(gap_series.values, color=PALETTE["richmond"], linewidth=2.2,
            label="PM2.5 causal gap (ATT)")
    ax.axhline(0, color=PALETTE["text"], linewidth=0.8, alpha=0.5)

    if half_life_hours is not None:
        ax.axvline(half_life_hours, color=PALETTE["synthetic"],
                   linewidth=1.4, linestyle="--",
                   label=f"Half-life ≈ {half_life_hours:.1f} h")
        ax.annotate(
            f"t½ = {half_life_hours:.1f} h",
            xy=(half_life_hours, gap_series.max() / 2),
            xytext=(half_life_hours + 1, gap_series.max() * 0.65),
            color=PALETTE["synthetic"], fontsize=9,
            arrowprops=dict(arrowstyle="->", color=PALETTE["synthetic"]),
        )

    ax.set_xlabel("Hours after 9 PM (July 4)", fontsize=11)
    ax.set_ylabel("PM2.5 excess (µg/m³)", fontsize=11)
    ax.set_title(f"Act III — Environmental Hangover Decay Curve ({year})", fontsize=13, pad=12)
    ax.legend(facecolor=PALETTE["panel_bg"], labelcolor=PALETTE["text"], fontsize=9)
    fig.tight_layout()
    return fig, ax


# ── 6. Feature importance (LightGBM) ─────────────────────────────────────────

def plot_feature_importance(
    feature_names: list[str],
    importances: np.ndarray,
    top_n: int = 15,
    figsize: tuple = (8, 5),
    title: str = "LightGBM Feature Importance (DoubleML Nuisance)",
) -> tuple[plt.Figure, plt.Axes]:
    """Horizontal bar chart of LightGBM feature importances."""
    idx = np.argsort(importances)[-top_n:]
    names  = [feature_names[i] for i in idx]
    values = importances[idx]
    colors = [PALETTE["richmond"] if v == values.max() else PALETTE["synthetic"]
              for v in values]

    fig, ax = plt.subplots(figsize=figsize)
    apply_dark_theme(fig, ax)
    ax.barh(names, values, color=colors, edgecolor="none")
    ax.set_xlabel("Importance", fontsize=11)
    ax.set_title(title, fontsize=12, pad=10)
    fig.tight_layout()
    return fig, ax
