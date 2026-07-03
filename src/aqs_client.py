"""
src/aqs_client.py
─────────────────────────────────────────────────────────────────────────────
EPA Air Quality System (AQS) Data Mart API + bulk CSV client.

PM2.5 parameter notes
─────────────────────
  88101  FRM/FEM (federal reference/equivalent, often 24h integrated)
  88502  Acceptable (PM2.5 non-FRM, typically continuous hourly monitors)

For *hourly* data we query both parameters and take 88502 preferentially,
since most continuous monitors report under 88502.

Key fix vs. v1: removed the `duration` filter — it excluded many continuous
monitors that report under different duration codes.

Station discovery
─────────────────
The v1 hardcoded site numbers were unreliable. We now use county-level
queries (`sampleData/byCounty`) and take the best-coverage site per county.

Target counties (state FIPS, county FIPS):
  Richmond City, VA      51-760   ← treated city
  Charlottesville, VA    51-540   ← donor
  Virginia Beach City    51-810   ← donor (coastal)
  Wake County, NC        37-183   ← donor (Raleigh)
  Baltimore City, MD     24-510   ← donor (urban north)

⚠ Methodological note on donor selection
──────────────────────────────────────────
Nearby cities also celebrate July 4th, so they are NOT "untreated" in the
strict DiD sense. However, for Synthetic DiD this is less critical:
  • Donor weights are fit on the PRE-period (June 29–July 3), not the post.
  • The synthetic Richmond captures regional atmospheric baseline patterns.
  • The estimated ATT is "excess urban fireworks effect above regional
    background" — a conservative estimate of the true fireworks impact.
  • For DoubleML we use temporal controls (same hours, non-July-4 nights).

For a cleaner control pool, add rural background monitors:
  Shenandoah Valley, VA  51-165 or 51-171  (rural, NPS Class I area)
  These are added as optional RURAL_DONORS below.

API docs: https://aqs.epa.gov/aqsweb/documents/data_api.html
"""

from __future__ import annotations

import io
import os
import time
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

BASE_URL  = "https://aqs.epa.gov/data/api"
PARAMS_PM25 = ["88502", "88101"]   # 88502 first (continuous hourly), 88101 fallback

RAW_DATA_DIR       = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DATA_DIR = Path(__file__).parent.parent / "data" / "processed"

# ── Target counties ────────────────────────────────────────────────────────────
# (label, state_code, county_code)
TARGET_COUNTIES: list[tuple[str, str, str]] = [
    ("Richmond_VA",       "51", "760"),   # treated (Richmond City)
    ("Charlottesville_VA","51", "540"),   # donor (small city, lower fireworks density)
    ("Virginia_Beach_VA", "51", "810"),   # donor (coastal)
    ("Raleigh_NC",        "37", "183"),   # donor (Wake County)
    ("Baltimore_MD",      "24", "510"),   # donor (Baltimore City)
]

# Optional rural background donors — lower fireworks density
RURAL_DONORS: list[tuple[str, str, str]] = [
    ("Rockingham_VA",     "51", "165"),   # Shenandoah Valley (rural)
    ("Page_VA",           "51", "171"),   # Shenandoah NP area (rural)
]

# City coordinates (for weather / metadata joins)
STATION_COORDS: dict[str, tuple[float, float]] = {
    "Richmond_VA":        (37.5338, -77.4349),
    "Charlottesville_VA": (38.0293, -78.4767),
    "Virginia_Beach_VA":  (36.7681, -76.0507),
    "Raleigh_NC":         (35.7796, -78.6382),
    "Baltimore_MD":       (39.2904, -76.6122),
    "Rockingham_VA":      (38.4760, -78.8690),
    "Page_VA":            (38.6820, -78.3080),
}


# ── API Client ─────────────────────────────────────────────────────────────────

class AQSClient:
    """EPA AQS Data Mart REST API client (county-level queries)."""

    def __init__(
        self,
        email: Optional[str] = None,
        key: Optional[str] = None,
        sleep_between_calls: float = 1.5,
    ):
        self.email = email or os.environ["AQS_EMAIL"]
        self.key   = key   or os.environ["AQS_KEY"]
        self.sleep = sleep_between_calls
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "fireworks-air-quality/0.2"})

    def _get(self, endpoint: str, params: dict) -> list[dict]:
        params = dict(params)
        params.update({"email": self.email, "key": self.key})
        resp = self.session.get(f"{BASE_URL}/{endpoint}", params=params, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        header = payload.get("Header", [{}])[0]
        if header.get("status", "") == "Failed":
            raise RuntimeError(f"AQS API: {header.get('error', 'unknown error')}")
        return payload.get("Data", [])

    def check_credentials(self) -> bool:
        try:
            self._get("list/states", {})
            return True
        except Exception as exc:
            print(f"Credential check failed: {exc}")
            return False

    def get_hourly_pm25_county(
        self,
        state: str,
        county: str,
        bdate: str,
        edate: str,
        label: str = "",
    ) -> pd.DataFrame:
        """
        Fetch hourly PM2.5 for an entire county (all sites, both 88502 + 88101).
        This avoids fragile site-number hardcoding.

        Tries 88502 (continuous) first, then 88101 (FRM) as fallback.
        Does NOT pass a duration filter so all sample types are included.
        """
        frames = []
        for param in PARAMS_PM25:
            try:
                data = self._get(
                    "sampleData/byCounty",
                    {
                        "param":   param,
                        "bdate":   bdate,
                        "edate":   edate,
                        "state":   state,
                        "county":  county,
                    },
                )
                if data:
                    df = pd.DataFrame(data)
                    df["param_code"] = param
                    frames.append(df)
                    break   # got data — no need to try fallback param
            except Exception as e:
                print(f"    Param {param} error: {e}")
            time.sleep(self.sleep)

        if not frames:
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True)

        df["datetime_local"] = pd.to_datetime(
            df["date_local"] + " " + df["time_local"], errors="coerce"
        )
        df["pm25"] = pd.to_numeric(df["sample_measurement"], errors="coerce")
        df["city"] = label
        df["aqs_site_id"] = (
            df["state_code"].astype(str) + "-" +
            df["county_code"].astype(str) + "-" +
            df["site_num"].astype(str)
        )

        # Aggregate multiple sites → county hourly mean
        agg = (
            df.dropna(subset=["pm25"])
            .groupby(["city", "datetime_local"])["pm25"]
            .mean()
            .reset_index()
        )
        return agg

    def get_all_counties_hourly(
        self,
        bdate: str,
        edate: str,
        include_rural: bool = False,
    ) -> pd.DataFrame:
        """Fetch PM2.5 for all target counties."""
        counties = list(TARGET_COUNTIES)
        if include_rural:
            counties += RURAL_DONORS

        frames = []
        for label, state, county in tqdm(counties, desc="Fetching counties"):
            print(f"  → {label} ({state}-{county})")
            df = self.get_hourly_pm25_county(state, county, bdate, edate, label)
            if df.empty:
                print(f"    ⚠ No data for {label}")
            else:
                print(f"    ✔ {len(df)} hourly records")
                frames.append(df)

        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ── Bulk CSV download (pre-generated, no site-ID dependency) ───────────────────
# These files are updated twice yearly by EPA and cover all certified data.
# STRONGLY preferred over the API for 2023, 2024, 2025 historical analysis.

BULK_URL_88502 = "https://aqs.epa.gov/aqsweb/airdata/hourly_88502_{year}.zip"
BULK_URL_88101 = "https://aqs.epa.gov/aqsweb/airdata/hourly_88101_{year}.zip"

# (state_fips, county_fips) → label  — county-level matching, no site needed
COUNTY_FIPS_MAP: dict[tuple[int, int], str] = {
    (51, 760): "Richmond_VA",
    (51, 540): "Charlottesville_VA",
    (51, 810): "Virginia_Beach_VA",
    (37, 183): "Raleigh_NC",
    (24, 510): "Baltimore_MD",
    # Rural donors (optional)
    (51, 165): "Rockingham_VA",
    (51, 171): "Page_VA",
}


def _download_zip(url: str, out_path: Path) -> None:
    """Stream-download a ZIP with a progress bar."""
    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with (
        open(out_path, "wb") as fout,
        tqdm(total=total, unit="B", unit_scale=True, desc=out_path.name) as bar,
    ):
        for chunk in resp.iter_content(chunk_size=65536):
            fout.write(chunk)
            bar.update(len(chunk))


def download_bulk_pm25(
    year: int,
    param: str = "88502",
    include_rural: bool = False,
    force: bool = False,
) -> pd.DataFrame:
    """
    Download EPA pre-generated hourly PM2.5 bulk CSV for `year`.

    Tries parameter 88502 (continuous) first, then 88101 if 88502 has no
    data for target counties.

    No API key required. Files cached to data/raw/.
    Returns a tidy DataFrame filtered to target counties.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    target_map = dict(COUNTY_FIPS_MAP) if include_rural else {
        k: v for k, v in COUNTY_FIPS_MAP.items()
        if v not in ("Rockingham_VA", "Page_VA")
    }

    for try_param in [param, "88101" if param == "88502" else "88502"]:
        url      = (BULK_URL_88502 if try_param == "88502" else BULK_URL_88101).format(year=year)
        zip_path = RAW_DATA_DIR / f"hourly_{try_param}_{year}.zip"
        csv_name = f"hourly_{try_param}_{year}.csv"

        if not zip_path.exists() or force:
            print(f"Downloading {url} …")
            try:
                _download_zip(url, zip_path)
            except requests.HTTPError as e:
                print(f"  ✗ {try_param} bulk not available for {year}: {e}")
                continue

        print(f"Parsing bulk ZIP for param {try_param} year {year} …")
        try:
            with zipfile.ZipFile(zip_path) as zf:
                # EPA sometimes nests the CSV in a subdirectory (88101 zips do this)
                csv_entries = [n for n in zf.namelist() if n.endswith(".csv")]
                if not csv_entries:
                    raise ValueError(f"No .csv found in {zip_path.name}: {zf.namelist()}")
                csv_entry = csv_entries[0]
                with zf.open(csv_entry) as f:
                    # 2023+ EPA format uses underscores; older files used spaces.
                    # Read without usecols first to detect format, then rename.
                    df = pd.read_csv(f, low_memory=False)

                    # Normalise column names: replace spaces → underscores
                    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

                    required = ["State_Code", "County_Code", "Date_Local",
                                "Time_Local", "Sample_Measurement"]
                    missing = [c for c in required if c not in df.columns]
                    if missing:
                        raise ValueError(f"Missing columns {missing}; found: {list(df.columns)[:10]}")

                    df = df[["State_Code", "County_Code", "Date_Local",
                              "Time_Local", "Sample_Measurement"]].copy()
                    df["State_Code"]  = pd.to_numeric(df["State_Code"],  errors="coerce")
                    df["County_Code"] = pd.to_numeric(df["County_Code"], errors="coerce")
                    df = df.dropna(subset=["State_Code", "County_Code"])
                    df["State_Code"]  = df["State_Code"].astype(int)
                    df["County_Code"] = df["County_Code"].astype(int)
        except Exception as e:
            print(f"  ✗ Parse error: {e}")
            continue

        df["_county_key"] = list(zip(df["State_Code"], df["County_Code"]))
        df = df[df["_county_key"].isin(target_map)].copy()

        if df.empty:
            print(f"  ⚠ No target-county data in {try_param} {year}")
            continue

        df["city"] = df["_county_key"].map(target_map)
        df["datetime_local"] = pd.to_datetime(
            df["Date_Local"] + " " + df["Time_Local"], errors="coerce"
        )
        df["pm25"] = pd.to_numeric(df["Sample_Measurement"], errors="coerce")

        # Average across sites within same county-hour
        result = (
            df.dropna(subset=["pm25"])
            .groupby(["city", "datetime_local"])["pm25"]
            .mean()
            .reset_index()
        )
        print(f"  ✔ {len(result):,} county-hour records (param {try_param})")
        return result

    return pd.DataFrame()
