"""Active-fire ingest: daily fire counts and FRP per 0.1 deg cell.

Two sources, written to the same monthly NetCDF:

* fire_count_modis  GEE `FIRMS` collection (MODIS, ~1 km raster; bands T21 /
                    confidence, no FRP). Fire pixels with confidence >=
                    ingest.firms.min_confidence, summed per 0.1 deg cell.
* fire_count_viirs, frp_sum_viirs
                    NASA FIRMS VIIRS 375 m archive CSVs (they include FRP, MW).
                    Download from https://firms.modaps.eosdis.nasa.gov/download/
                    (country = India or the bbox, CSV) into ingest.firms.archive_dir,
                    or pass --map-key / FIRMS_MAP_KEY to fetch them via the FIRMS
                    area API.

Days are UTC (acq_date). A cell is 0 on days with data but no fires and NaN on
days the source does not cover.

Output: data/raw/firms/firms_YYYY-MM.nc  (time, lat, lon)

    python -m src.ingest.gee_firms --start 2023-10-01 --end 2023-11-30
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests
import xarray as xr

from src.config import (
    add_date_args, chunk_tag, date_range, load_settings, month_chunks, project_path, raw_dir,
    resolve_dates, setup_logging,
)
from src.grid.regrid import bin_points_to_grid, india_grid
from src.ingest import _gee

log = logging.getLogger("ingest.firms")

FIRMS_API = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"


def cfg() -> dict:
    return load_settings()["ingest"]["firms"]


# --------------------------------------------------------------------------- GEE (MODIS) counts
def modis_count_image(day: date):
    import ee

    c = cfg()
    t0, t1 = _gee.day_window(day)
    coll = ee.ImageCollection(c["gee_collection"]).filterDate(t0, t1)

    def build(col):
        native = ee.Image(col.first()).projection()
        fire = col.select("confidence").max().gte(c["min_confidence"]).unmask(0)
        return (
            fire.setDefaultProjection(native)
            .reduceResolution(ee.Reducer.sum(), bestEffort=True, maxPixels=1024)
            .round()  # edge pixels are fractionally weighted
        )

    return _gee.day_or_empty(coll, build)


# --------------------------------------------------------------------------- VIIRS archive CSVs
def fetch_viirs_api(start: date, end: date, map_key: str) -> None:
    """Download VIIRS fire CSVs for [start, end] via the FIRMS area API (5-day chunks)."""
    c = cfg()
    g = india_grid()
    out_dir = project_path(c["archive_dir"]) / "api"
    out_dir.mkdir(parents=True, exist_ok=True)
    area = ",".join(str(v) for v in g.bounds)  # west,south,east,north
    for source in c["api_sources"]:
        d = start
        while d <= end:
            n = min(5, (end - d).days + 1)  # API day range is capped
            out = out_dir / f"{source}_{d:%Y%m%d}_{n}d.csv"
            if not out.exists():
                url = f"{FIRMS_API}/{map_key}/{source}/{area}/{n}/{d.isoformat()}"
                r = requests.get(url, timeout=120)
                r.raise_for_status()
                if r.text.lstrip().lower().startswith(("invalid", "error")):
                    raise RuntimeError(f"FIRMS API: {r.text[:200]}")
                out.write_text(r.text)
                log.info("FIRMS API: %s %s +%dd -> %d rows", source, d, n, r.text.count("\n") - 1)
                time.sleep(1)
            d += timedelta(days=n)


def load_viirs(start: date, end: date) -> tuple[pd.DataFrame, set[date]]:
    """All VIIRS detections in [start, end] + the set of days the CSVs cover."""
    c = cfg()
    files = sorted(project_path(c["archive_dir"]).rglob("*.csv"))
    if not files:
        log.warning("no VIIRS CSVs in %s; FRP will be NaN", c["archive_dir"])
        return pd.DataFrame(), set()

    cols = ["latitude", "longitude", "acq_date", "frp", "confidence", "satellite", "type"]
    frames, covered = [], set()
    for f in files:
        df = pd.read_csv(f, usecols=lambda col: col in cols, dtype={"confidence": str, "satellite": str})
        if df.empty:
            continue
        df["acq_date"] = pd.to_datetime(df["acq_date"]).dt.date
        # A file covers every day between its first and last detection. Fires occur
        # somewhere in India every day, so internal gaps are real zeros.
        covered.update(date_range(df["acq_date"].min(), df["acq_date"].max()))
        frames.append(df[(df["acq_date"] >= start) & (df["acq_date"] <= end)])
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)

    drop_conf = {s.lower() for s in c.get("viirs_drop_confidence", [])}
    df = df[~df["confidence"].astype(str).str.lower().isin(drop_conf)]
    if "type" in df and c.get("viirs_keep_types") is not None:
        df = df[df["type"].isin(c["viirs_keep_types"])]
    if c.get("viirs_satellites"):
        n0 = len(df)
        df = df[df["satellite"].astype(str).isin([str(s) for s in c["viirs_satellites"]])]
        if n0 and df.empty:
            log.warning("VIIRS: satellite filter %s removed all %d rows; check ingest.firms.viirs_satellites",
                        c["viirs_satellites"], n0)
    df = df.drop_duplicates(["latitude", "longitude", "acq_date", "frp", "satellite"])
    log.info("VIIRS: %d detections in %s..%s from %d file(s)", len(df), start, end, len(files))
    return df, covered


def viirs_grids(df: pd.DataFrame, covered: set[date], days: list[date]) -> tuple[xr.DataArray, xr.DataArray]:
    g = india_grid()
    count = np.full((len(days), g.nlat, g.nlon), np.nan, dtype="f4")
    frp = count.copy()
    by_day = dict(tuple(df.groupby("acq_date"))) if not df.empty else {}
    for i, d in enumerate(days):
        if d not in covered:
            continue
        sub = by_day.get(d)
        if sub is None:
            count[i] = 0
            frp[i] = 0
            continue
        count[i] = bin_points_to_grid(sub["latitude"], sub["longitude"], how="count", grid=g)
        frp[i] = bin_points_to_grid(sub["latitude"], sub["longitude"], sub["frp"], how="sum", grid=g)
    coords = {"time": pd.to_datetime(days), **g.coords()}
    dims = ("time", "lat", "lon")
    return (
        xr.DataArray(count, dims=dims, coords=coords, name="fire_count_viirs"),
        xr.DataArray(frp, dims=dims, coords=coords, name="frp_sum_viirs"),
    )


# --------------------------------------------------------------------------- driver
def run(start: date, end: date, skip_gee: bool, skip_csv: bool, map_key: str | None, overwrite: bool) -> None:
    c = cfg()
    if not skip_gee:
        _gee.init_ee()
    if not skip_csv and map_key:
        fetch_viirs_api(start, end, map_key)
    viirs, covered = (pd.DataFrame(), set()) if skip_csv else load_viirs(start, end)

    for m0, m1 in month_chunks(start, end):
        label = f"firms {m0:%Y-%m}"
        out = raw_dir("firms") / f"firms_{chunk_tag(m0, m1)}.nc"
        if out.exists() and not overwrite:
            log.info("%s: exists, skipping", label)
            continue
        days = list(date_range(m0, m1))
        ds = xr.Dataset()

        if not skip_gee:
            log.info("%s: fetching MODIS fire counts from GEE", label)
            da = _gee.fetch_days(modis_count_image, days, "fire_count_modis")
            _gee.log_coverage(da, f"{label} modis")
            da.attrs.update(units="1", source=c["gee_collection"],
                            description=f"MODIS fire pixels (confidence>={c['min_confidence']}) per cell")
            ds["fire_count_modis"] = da

        if not skip_csv:
            cnt, frp = viirs_grids(viirs, covered, days)
            _gee.log_coverage(cnt, f"{label} viirs")
            cnt.attrs.update(units="1", source="NASA FIRMS VIIRS 375m archive",
                             description="VIIRS fire detections per cell")
            frp.attrs.update(units="MW", source="NASA FIRMS VIIRS 375m archive",
                             description="sum of fire radiative power of VIIRS detections per cell")
            ds["fire_count_viirs"] = cnt
            ds["frp_sum_viirs"] = frp

        if not ds.data_vars:
            raise SystemExit("nothing to do: both --skip-gee and --skip-csv given")
        _gee.write_nc(ds, out)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_date_args(p)
    p.add_argument("--skip-gee", action="store_true", help="skip MODIS counts from GEE")
    p.add_argument("--skip-csv", action="store_true", help="skip VIIRS archive counts/FRP")
    p.add_argument("--map-key", default=os.environ.get("FIRMS_MAP_KEY"),
                   help="FIRMS MAP_KEY to download VIIRS CSVs via the API (env FIRMS_MAP_KEY)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    setup_logging(args.verbose)
    start, end = resolve_dates(args)
    run(start, end, args.skip_gee, args.skip_csv, args.map_key, args.overwrite)


if __name__ == "__main__":
    main()
