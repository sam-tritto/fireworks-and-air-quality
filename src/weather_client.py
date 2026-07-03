"""
src/weather_client.py
─────────────────────────────────────────────────────────────────────────────
Hourly weather data via the Open-Meteo Historical Archive API.

Open-Meteo advantages over Meteostat:
  ✔ Completely free, no API key
  ✔ ERA5 reanalysis — global coverage, never missing
  ✔ Consistent hourly grid — no station-dropout gaps
  ✔ Returns: temp (°C), rhum (%), wind speed (mph), wind direction (°), prcp (mm)

API docs: https://open-meteo.com/en/docs/historical-weather-api
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

# ── City coordinates (lat, lon) ────────────────────────────────────────────────

CITY_COORDS: dict[str, tuple[float, float]] = {
    "Richmond_VA":       (37.5338, -77.4349),
    "Roanoke_VA":        (37.2788, -79.9581),
    "Virginia_Beach_VA": (36.7681, -76.0507),
    "Raleigh_NC":        (35.7796, -78.6382),
    "Baltimore_MD":      (39.2904, -76.6122),
}

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

HOURLY_VARS = [
    "temperature_2m",          # °C
    "relative_humidity_2m",    # %
    "wind_speed_10m",          # returned in mph (we set wind_speed_unit=mph)
    "wind_direction_10m",      # °
    "precipitation",           # mm
    "surface_pressure",        # hPa
]

RENAME_MAP = {
    "temperature_2m":       "temp",
    "relative_humidity_2m": "rhum",
    "wind_speed_10m":       "wspd_mph",
    "wind_direction_10m":   "wdir",
    "precipitation":        "prcp",
    "surface_pressure":     "pres",
}


def _fetch_open_meteo(
    lat: float,
    lon: float,
    start: datetime,
    end: datetime,
    timezone: str = "America/New_York",
) -> pd.DataFrame:
    """
    Call the Open-Meteo ERA5 archive for one location.
    Returns a DataFrame indexed by 'datetime_local'.
    """
    params = {
        "latitude":        lat,
        "longitude":       lon,
        "start_date":      start.strftime("%Y-%m-%d"),
        "end_date":        end.strftime("%Y-%m-%d"),
        "hourly":          ",".join(HOURLY_VARS),
        "wind_speed_unit": "mph",        # get wind already in mph
        "timezone":        timezone,
    }
    resp = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    hourly = data.get("hourly", {})
    if not hourly or "time" not in hourly:
        return pd.DataFrame()

    df = pd.DataFrame(hourly)
    df = df.rename(columns={"time": "datetime_local"})
    df["datetime_local"] = pd.to_datetime(df["datetime_local"])
    df = df.rename(columns={k: v for k, v in RENAME_MAP.items() if k in df.columns})
    return df


class WeatherClient:
    """Fetch and cache hourly weather for study cities using Open-Meteo ERA5."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or (
            Path(__file__).parent.parent / "data" / "raw" / "weather"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_city_weather(
        self,
        city: str,
        start: datetime,
        end: datetime,
        timezone: str = "America/New_York",
        force: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch ERA5 hourly weather for a single city.
        Cached to parquet; subsequent calls load from disk.
        """
        cache_file = (
            self.cache_dir
            / f"weather_{city}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.parquet"
        )
        if cache_file.exists() and not force:
            return pd.read_parquet(cache_file)

        if city not in CITY_COORDS:
            raise ValueError(f"Unknown city '{city}'. Valid: {list(CITY_COORDS)}")

        lat, lon = CITY_COORDS[city]
        df = _fetch_open_meteo(lat, lon, start, end, timezone=timezone)

        if df.empty:
            print(f"  ⚠ No weather data for {city}")
            return df

        df["city"] = city
        keep = ["city", "datetime_local", "temp", "rhum", "wspd_mph", "wdir", "prcp", "pres"]
        df = df[[c for c in keep if c in df.columns]].copy()
        df.to_parquet(cache_file, index=False)
        return df

    def get_all_cities_weather(
        self,
        start: datetime,
        end: datetime,
        force: bool = False,
    ) -> pd.DataFrame:
        frames = []
        for city in CITY_COORDS:
            print(f"  → {city}")
            try:
                df = self.get_city_weather(city, start, end, force=force)
                if not df.empty:
                    frames.append(df)
            except Exception as exc:
                print(f"    ✗ {city}: {exc}")
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def get_weather_window(
        self,
        year: int,
        pre_days: int = 5,
        post_days: int = 4,
        force: bool = False,
    ) -> pd.DataFrame:
        """Fetch June 29 – July 8 for a given year (default window)."""
        july4 = datetime(year, 7, 4)
        start = july4 - timedelta(days=pre_days)
        end   = july4 + timedelta(days=post_days, hours=23)
        print(f"Weather window: {start.date()} → {end.date()}")
        return self.get_all_cities_weather(start, end, force=force)
