"""CPCB ground-station data via OpenAQ: hourly PM2.5, PM10, NO2, SO2, CO, O3 -> daily.

1. Stations: OpenAQ v3 /locations (country IN, reference monitors), filtered to
   providers matching ingest.openaq.provider_regex (CPCB). Needs an API key in
   $OPENAQ_API_KEY (free, https://explore.openaq.org/register).
   -> data/raw/cpcb/stations.csv
2. Hourly values:
   --source s3   (default) OpenAQ public archive s3://openaq-data-archive, one
                 csv.gz per location per day, anonymous access, cached under
                 data/raw/cpcb/s3/ so re-runs only fetch new days.
   --source api  v3 /sensors/{id}/hours, rate-limited (~55 req/min). Fine for a
                 season, too slow for all 7 years.
3. Cleaning: units -> ug/m3 (CO -> mg/m3; ppm/ppb converted at 25 C, 1 atm),
   negatives and implausible values dropped, sub-hourly records averaged into
   IST hour-beginning bins.
4. Daily (IST calendar day), needing >= min_hours_per_day (16) valid hours:
     PM2.5, PM10, NO2, SO2: 24 h mean
     O3, CO: max of the 8 h running means (each needs >= 6 of 8 hours)
   -> data/raw/cpcb/daily_obs.parquet
      columns: location_id, date, pm25, pm10, no2, so2, co, o3, <param>_n (hours)

    python -m src.ingest.cpcb --start 2023-10-01 --end 2023-11-30
"""

from __future__ import annotations

import argparse
import gzip

import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.config import add_date_args, date_range, load_settings, raw_dir, resolve_dates, setup_logging

log = logging.getLogger("ingest.cpcb")

# molecular weights (g/mol) for ppb/ppm -> mass concentration at 25 C, 1 atm (molar volume 24.45 L)
MW = {"no2": 46.0055, "so2": 64.066, "o3": 47.997, "co": 28.010}
MOLAR_VOLUME = 24.45
ROLLING_8H = ("o3", "co")


def cfg() -> dict:
    return load_settings()["ingest"]["openaq"]


# =========================================================================== OpenAQ v3 API
class RateLimiter:
    def __init__(self, per_minute: int):
        self.interval = 60.0 / per_minute
        self.lock = threading.Lock()
        self.next_t = 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            if now < self.next_t:
                time.sleep(self.next_t - now)
            self.next_t = max(now, self.next_t) + self.interval


class OpenAQ:
    def __init__(self):
        c = cfg()
        key = os.environ.get(c["api_key_env"])
        if not key:
            raise SystemExit(f"set ${c['api_key_env']} (OpenAQ v3 API key)")
        self.base = c["base_url"].rstrip("/")
        self.s = requests.Session()
        self.s.headers.update({"X-API-Key": key, "Accept": "application/json"})
        self.limiter = RateLimiter(c["requests_per_minute"])

    def get(self, path: str, params: dict | None = None, retries: int = 5) -> dict:
        for attempt in range(retries):
            self.limiter.wait()
            try:
                r = self.s.get(f"{self.base}{path}", params=params, timeout=60)
            except requests.RequestException as e:
                wait = 2 ** attempt * 5
                log.warning("OpenAQ %s: %s; retry in %ds", path, e, wait)
                time.sleep(wait)
                continue
            if r.status_code == 429 or r.status_code >= 500:
                wait = int(r.headers.get("x-ratelimit-reset") or r.headers.get("retry-after") or 2 ** attempt * 5)
                log.warning("OpenAQ %s: HTTP %d; retry in %ds", path, r.status_code, wait)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError(f"OpenAQ {path}: failed after {retries} attempts")

    def paginate(self, path: str, params: dict, limit: int = 1000):
        page = 1
        while True:
            res = self.get(path, {**params, "limit": limit, "page": page}).get("results", [])
            yield from res
            if len(res) < limit:
                return
            page += 1


# =========================================================================== stations
def fetch_stations(api: OpenAQ) -> pd.DataFrame:
    c = cfg()
    params_wanted = set(c["parameters"])
    provider_re = re.compile(c["provider_regex"], re.I)
    rows, skipped = [], 0
    for loc in api.paginate("/locations", {"iso": c["country_iso"], "monitor": "true"}):
        provider = (loc.get("provider") or {}).get("name") or ""
        if not provider_re.search(provider):
            skipped += 1
            continue
        row = {
            "location_id": loc["id"],
            "name": loc.get("name"),
            "locality": loc.get("locality"),
            "lat": (loc.get("coordinates") or {}).get("latitude"),
            "lon": (loc.get("coordinates") or {}).get("longitude"),
            "provider": provider,
            "owner": (loc.get("owner") or {}).get("name"),
            "datetime_first": ((loc.get("datetimeFirst") or {}).get("utc")),
            "datetime_last": ((loc.get("datetimeLast") or {}).get("utc")),
        }
        for s in loc.get("sensors", []):
            p = (s.get("parameter") or {}).get("name")
            if p in params_wanted and f"sensor_{p}" not in row:
                row[f"sensor_{p}"] = s["id"]
                row[f"units_{p}"] = s["parameter"].get("units")
        if any(f"sensor_{p}" in row for p in params_wanted):
            rows.append(row)
    df = pd.DataFrame(rows).sort_values("location_id").reset_index(drop=True)
    log.info("stations: %d %s locations kept (%d non-matching providers skipped)",
             len(df), c["provider_regex"], skipped)
    return df


def stations_active(st: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    first = pd.to_datetime(st["datetime_first"], utc=True, errors="coerce").dt.date
    last = pd.to_datetime(st["datetime_last"], utc=True, errors="coerce").dt.date
    ok = (first.isna() | (first <= end)) & (last.isna() | (last >= start))
    return st[ok]


# =========================================================================== hourly: S3 archive
_s3_local = threading.local()


def _s3():
    if not hasattr(_s3_local, "client"):
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config

        _s3_local.client = boto3.client(
            "s3", region_name="us-east-1", config=Config(signature_version=UNSIGNED, retries={"max_attempts": 8})
        )
    return _s3_local.client


def _s3_keys(location_id: int, start: date, end: date) -> list[str]:
    c = cfg()
    keys = []
    for year in range(start.year, end.year + 1):
        prefix = f"{c['s3_prefix']}/locationid={location_id}/year={year}/"
        for page in _s3().get_paginator("list_objects_v2").paginate(Bucket=c["s3_bucket"], Prefix=prefix):
            for obj in page.get("Contents", []):
                m = re.search(r"-(\d{8})\.csv\.gz$", obj["Key"])
                if m and start <= date(int(m[1][:4]), int(m[1][4:6]), int(m[1][6:])) <= end:
                    keys.append(obj["Key"])
    return keys


def _s3_download(key: str, cache: Path) -> Path:
    out = cache / key.split("/", 2)[-1]  # drop "records/csv.gz/"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".part")
        _s3().download_file(cfg()["s3_bucket"], key, str(tmp))
        tmp.replace(out)
    return out


def archive_time(dt: pd.Series) -> pd.Series:
    """Archive `datetime` is the END of the averaging period (CPCB: 15/30-min or hourly).
    Shift by 1 s so floor("h") in clean_hourly maps e.g. 00:30 and 01:00 -> hour 00:00."""
    return pd.to_datetime(dt, utc=True, format="ISO8601") - pd.Timedelta(seconds=1)


def hourly_s3(location_id: int, start: date, end: date, pool: ThreadPoolExecutor) -> pd.DataFrame:
    """Raw hourly rows for one location. Pads one day each side for the IST shift."""
    cache = raw_dir("cpcb", "s3")
    keys = _s3_keys(location_id, start - timedelta(days=1), end + timedelta(days=1))
    files = list(pool.map(lambda k: _s3_download(k, cache), keys))
    frames = []
    for f in files:
        try:
            with gzip.open(f, "rt") as fh:
                frames.append(pd.read_csv(fh, usecols=["datetime", "parameter", "units", "value"]))
        except (OSError, EOFError, pd.errors.ParserError, ValueError) as e:
            log.warning("bad cached file %s (%s); deleting", f, e)
            f.unlink(missing_ok=True)
    if not frames:
        return pd.DataFrame(columns=["time", "parameter", "units", "value"])
    df = pd.concat(frames, ignore_index=True)
    df["time"] = archive_time(df.pop("datetime"))
    return df


# =========================================================================== hourly: v3 API
def hourly_api(api: OpenAQ, station: pd.Series, start: date, end: date) -> pd.DataFrame:
    cache = raw_dir("cpcb", "api")
    frames = []
    for p in cfg()["parameters"]:
        sid = station.get(f"sensor_{p}")
        if pd.isna(sid):
            continue
        sid = int(sid)
        out = cache / f"sensor_{sid}_{start:%Y%m%d}_{end:%Y%m%d}.parquet"
        if out.exists():
            frames.append(pd.read_parquet(out))
            continue
        params = {
            "datetime_from": f"{start - timedelta(days=1)}T00:00:00Z",
            "datetime_to": f"{end + timedelta(days=1)}T23:59:59Z",
        }
        rows = [
            {
                "time": r["period"]["datetimeFrom"]["utc"],
                "parameter": r["parameter"]["name"],
                "units": r["parameter"]["units"],
                "value": r["value"],
            }
            for r in api.paginate(f"/sensors/{sid}/hours", params)
        ]
        df = pd.DataFrame(rows, columns=["time", "parameter", "units", "value"])
        df["time"] = pd.to_datetime(df["time"], utc=True)  # hour-beginning already
        df.to_parquet(out, index=False)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["time", "parameter", "units", "value"])
    return pd.concat(frames, ignore_index=True)


# =========================================================================== cleaning + daily
def _unit_key(u: str) -> str:
    u = str(u).lower().replace("µ", "u").replace("μ", "u").replace("³", "3").replace(" ", "")
    return {"ug/m3": "ug/m3", "mg/m3": "mg/m3", "ppm": "ppm", "ppb": "ppb"}.get(u, u)


def to_cpcb_units(df: pd.DataFrame) -> pd.DataFrame:
    """Convert `value` to CPCB units: ug/m3, CO in mg/m3. Drops unknown units."""
    df = df.copy()
    unit = df["units"].map(_unit_key)
    mw = df["parameter"].map(MW)
    is_co = df["parameter"] == "co"
    val = df["value"].astype("f8")

    # ug/m3 equivalent first
    ug = pd.Series(np.nan, index=df.index)
    ug[unit == "ug/m3"] = val[unit == "ug/m3"]
    ug[unit == "mg/m3"] = val[unit == "mg/m3"] * 1000
    ppb = unit == "ppb"
    ppm = unit == "ppm"
    ug[ppb] = val[ppb] * mw[ppb] / MOLAR_VOLUME
    ug[ppm] = val[ppm] * 1000 * mw[ppm] / MOLAR_VOLUME
    df["value"] = np.where(is_co, ug / 1000, ug)

    bad = df["value"].isna() & val.notna()
    if bad.any():
        log.debug("dropping %d rows with unconvertible units %s", bad.sum(),
                  sorted(df.loc[bad, "units"].astype(str).unique()))
    return df.drop(columns="units")


def clean_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """-> one value per (parameter, IST hour-beginning), CPCB units, plausible range."""
    c = cfg()
    df = df[df["parameter"].isin(c["parameters"])]
    df = to_cpcb_units(df)
    limits = df["parameter"].map(c["max_plausible"])
    df = df[(df["value"] >= 0) & (df["value"] <= limits)]
    t = pd.to_datetime(df["time"], utc=True).dt.tz_convert(c["timezone"]).dt.floor("h").dt.tz_localize(None)
    df = df.assign(time=t)
    # sub-hourly records, several sensors or duplicates within the same hour -> mean
    return df.groupby(["parameter", "time"], as_index=False)["value"].mean()


def daily_from_hourly(hourly: pd.DataFrame, min_hours: int = 16, rolling_min: int = 6) -> pd.DataFrame:
    """hourly: columns parameter, time (local, hour-beginning, tz-naive), value.

    Returns a wide frame indexed by date with <param> and <param>_n columns.
    """
    out = {}
    for p, g in hourly.groupby("parameter"):
        s = g.set_index("time")["value"].sort_index()
        s = s[~s.index.duplicated()]
        day = s.index.normalize()
        n = s.groupby(day).count()
        if p in ROLLING_8H:
            # 8 h running mean labelled by the window's last hour; window = [t-7h, t]
            roll = s.asfreq("h").rolling(8, min_periods=rolling_min).mean().dropna()
            val = roll.groupby(roll.index.normalize()).max()
        else:
            val = s.groupby(day).mean()
        val = val.reindex(n.index)
        val[n < min_hours] = np.nan
        out[p] = val
        out[f"{p}_n"] = n
    if not out:
        return pd.DataFrame()
    df = pd.DataFrame(out)
    df.index = df.index.date
    df.index.name = "date"
    return df


# =========================================================================== driver
def load_or_fetch_stations(refresh: bool) -> pd.DataFrame:
    path = raw_dir("cpcb") / "stations.csv"
    if path.exists() and not refresh:
        log.info("using cached %s (--refresh-stations to re-query)", path)
        return pd.read_csv(path)
    st = fetch_stations(OpenAQ())
    st.to_csv(path, index=False)
    log.info("wrote %s", path)
    return st


def run(start: date, end: date, source: str, workers: int, refresh_stations: bool, stations_only: bool) -> None:
    c = cfg()
    st = load_or_fetch_stations(refresh_stations)
    if stations_only:
        return
    st = stations_active(st, start, end)
    log.info("%d stations active in %s..%s; hourly source: %s", len(st), start, end, source)
    api = OpenAQ() if source == "api" else None

    params = c["parameters"]
    days = pd.Index(list(date_range(start, end)), name="date")
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (_, row) in enumerate(st.iterrows(), 1):
            loc = int(row["location_id"])
            try:
                raw = hourly_s3(loc, start, end, pool) if source == "s3" else hourly_api(api, row, start, end)
            except Exception as e:
                log.warning("[%d/%d] location %d: fetch failed (%s); skipping", i, len(st), loc, e)
                continue
            if raw.empty:
                log.info("[%d/%d] location %d (%s): no data", i, len(st), loc, row.get("name"))
                continue
            daily = daily_from_hourly(clean_hourly(raw), c["min_hours_per_day"], c["rolling_8h_min_hours"])
            daily = daily[(daily.index >= start) & (daily.index <= end)]
            if daily.empty:
                continue
            daily = daily.reindex(columns=[x for p in params for x in (p, f"{p}_n")])
            daily = daily.dropna(how="all", subset=params)
            daily.insert(0, "location_id", loc)
            results.append(daily.reset_index())
            log.info("[%d/%d] location %d (%s): %d valid days", i, len(st), loc, row.get("name"), len(daily))

    if not results:
        log.warning("no daily observations produced")
        return
    df = pd.concat(results, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    for p in params:
        df[f"{p}_n"] = df[f"{p}_n"].fillna(0).astype("int16")
    out = raw_dir("cpcb") / "daily_obs.parquet"
    if out.exists():
        # merge with existing file: new rows win for overlapping (location, date)
        old = pd.read_parquet(out)
        df = (pd.concat([old, df], ignore_index=True)
              .drop_duplicates(["location_id", "date"], keep="last"))
    df = df.sort_values(["location_id", "date"]).reset_index(drop=True)
    df.to_parquet(out, index=False)
    log.info("wrote %s (%d station-days, %d stations, %d days requested)",
             out, len(df), df["location_id"].nunique(), len(days))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_date_args(p)
    p.add_argument("--source", choices=["s3", "api"], default="s3")
    p.add_argument("--workers", type=int, default=16, help="parallel S3 downloads")
    p.add_argument("--refresh-stations", action="store_true")
    p.add_argument("--stations-only", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    setup_logging(args.verbose)
    start, end = resolve_dates(args)
    run(start, end, args.source, args.workers, args.refresh_stations, args.stations_only)


if __name__ == "__main__":
    main()
