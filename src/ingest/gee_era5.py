"""ERA5 / ERA5-Land meteorology ingest (T2m, Td2m, u/v10, surface pressure, precip, BLH).

* ERA5-Land DAILY_AGGR (GEE): one image per day; bilinear-resampled onto the
  common 0.1 deg grid. ERA5-Land is land-only, so ocean cells are NaN.
* Boundary layer height (not in ERA5-Land):
    --blh-source auto  probe `ingest.era5.blh_gee_collection` in GEE for the BLH
                       band covering the requested dates; else fall back to CDS
    --blh-source gee   GEE hourly collection, daily (UTC) mean
    --blh-source cds   Copernicus CDS (cdsapi; needs ~/.cdsapirc), hourly -> daily
                       (UTC) mean, bilinear-interpolated onto the grid
    --blh-source none  skip BLH

Units are as delivered: K, Pa, m s-1, m (precip daily sum), m (BLH).

Output: data/raw/era5/<var>/era5_<var>_YYYY-MM.nc  (time, lat, lon)

    python -m src.ingest.gee_era5 --start 2023-10-01 --end 2023-11-30
"""

from __future__ import annotations

import argparse
import logging
import tempfile
from datetime import date
from pathlib import Path

import pandas as pd
import xarray as xr

from src.config import (
    add_date_args, chunk_tag, date_range, load_settings, month_chunks, raw_dir, resolve_dates, setup_logging,
)
from src.grid.regrid import india_grid, to_grid
from src.ingest import _gee

log = logging.getLogger("ingest.era5")

UNITS = {"t2m": "K", "d2m": "K", "u10": "m s-1", "v10": "m s-1", "sp": "Pa", "tp": "m", "blh": "m"}


def cfg() -> dict:
    return load_settings()["ingest"]["era5"]


# --------------------------------------------------------------------------- ERA5-Land
def land_daily_image(band: str, day: date):
    import ee

    t0, t1 = _gee.day_window(day)
    coll = ee.ImageCollection(cfg()["land_collection"]).filterDate(t0, t1).select(band)
    return _gee.day_or_empty(coll, lambda c: ee.Image(c.first()).resample("bilinear"))


# --------------------------------------------------------------------------- BLH via GEE
def gee_blh_available(start: date, end: date) -> bool:
    """True if the configured GEE hourly collection has the BLH band for start and end."""
    import ee

    c = cfg()
    coll = ee.ImageCollection(c["blh_gee_collection"])
    try:
        info = ee.Dictionary({
            "first_bands": ee.Algorithms.If(
                coll.filterDate(*_gee.day_window(start)).size().gt(0),
                ee.Image(coll.filterDate(*_gee.day_window(start)).first()).bandNames(),
                ee.List([]),
            ),
            "n_end": coll.filterDate(*_gee.day_window(end)).size(),
        }).getInfo()
    except Exception as e:
        log.info("BLH probe: %s not usable (%s)", c["blh_gee_collection"], e)
        return False
    ok = c["blh_band"] in info["first_bands"] and info["n_end"] > 0
    log.info("BLH probe: %s %s band %r for %s..%s", c["blh_gee_collection"],
             "has" if ok else "lacks", c["blh_band"], start, end)
    return ok


def gee_blh_daily_image(day: date):
    import ee

    c = cfg()
    t0, t1 = _gee.day_window(day)
    coll = ee.ImageCollection(c["blh_gee_collection"]).filterDate(t0, t1).select(c["blh_band"])
    return _gee.day_or_empty(
        coll,
        lambda col: col.mean().setDefaultProjection(ee.Image(col.first()).projection()).resample("bilinear"),
    )


# --------------------------------------------------------------------------- BLH via CDS
def cds_blh_month(m0: date, m1: date) -> xr.DataArray:
    """Download hourly ERA5 BLH for [m0, m1] from CDS -> daily mean on the common grid."""
    import cdsapi

    c = cfg()
    g = india_grid()
    pad = 0.5  # pad so bilinear interpolation has neighbours at the bbox edge
    days = list(date_range(m0, m1))
    request = {
        "product_type": ["reanalysis"],
        "variable": [c["blh_cds_variable"]],
        "year": [f"{m0.year}"],
        "month": [f"{m0.month:02d}"],
        "day": [f"{d.day:02d}" for d in days],
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": [g.lat_max + pad, g.lon_min - pad, g.lat_min - pad, g.lon_max + pad],  # N, W, S, E
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "blh.nc"
        log.info("CDS: requesting BLH %s..%s (queued requests can take minutes)", m0, m1)
        cdsapi.Client().retrieve(c["blh_cds_dataset"], request, str(target))
        with xr.open_dataset(target) as ds:
            ds = ds.load()

    var = "blh" if "blh" in ds else next(iter(ds.data_vars))
    da = ds[var]
    tdim = "valid_time" if "valid_time" in da.dims else "time"
    da = da.rename({tdim: "time"}).resample(time="1D").mean()
    da = to_grid(da, method="linear").reindex(time=pd.to_datetime(days))
    return da.astype("f4").transpose("time", "lat", "lon")


# --------------------------------------------------------------------------- driver
def _write(da: xr.DataArray, var: str, m0: date, m1: date, fmt: str, attrs: dict) -> None:
    out_dir = raw_dir("era5", var)
    da = da.rename(var)
    da.attrs.update(units=UNITS.get(var, ""), **attrs)
    if fmt == "tif":
        _gee.write_daily_tifs(da, out_dir, f"era5_{var}")
    else:
        _gee.write_nc(da.to_dataset(), out_dir / f"era5_{var}_{chunk_tag(m0, m1)}.nc")


def _exists(var: str, m0: date, m1: date) -> bool:
    return (raw_dir("era5", var) / f"era5_{var}_{chunk_tag(m0, m1)}.nc").exists()


def run(start: date, end: date, variables: list[str], blh_source: str, fmt: str, overwrite: bool) -> None:
    _gee.init_ee()
    c = cfg()
    bands: dict[str, str] = c["bands"]
    unknown = set(variables) - set(bands) - {"blh"}
    if unknown:
        raise SystemExit(f"unknown variables {sorted(unknown)}; choose from {sorted(bands) + ['blh']}")

    # ERA5-Land variables
    for var in [v for v in variables if v != "blh"]:
        band = bands[var]
        for m0, m1 in month_chunks(start, end):
            label = f"era5 {var} {m0:%Y-%m}"
            if fmt == "nc" and _exists(var, m0, m1) and not overwrite:
                log.info("%s: exists, skipping", label)
                continue
            days = list(date_range(m0, m1))
            log.info("%s: fetching %d days", label, len(days))
            da = _gee.fetch_days(lambda d: land_daily_image(band, d), days, var)
            _gee.log_coverage(da, label)
            _write(da, var, m0, m1, fmt, {"source": c["land_collection"], "band": band,
                                          "resampling": "bilinear"})

    # Boundary layer height
    if "blh" not in variables or blh_source == "none":
        return
    if blh_source == "auto":
        blh_source = "gee" if gee_blh_available(start, end) else "cds"
    log.info("BLH source: %s", blh_source)

    for m0, m1 in month_chunks(start, end):
        label = f"era5 blh {m0:%Y-%m}"
        if fmt == "nc" and _exists("blh", m0, m1) and not overwrite:
            log.info("%s: exists, skipping", label)
            continue
        days = list(date_range(m0, m1))
        if blh_source == "gee":
            da = _gee.fetch_days(gee_blh_daily_image, days, "blh")
            attrs = {"source": c["blh_gee_collection"], "band": c["blh_band"]}
        else:
            da = cds_blh_month(m0, m1)
            attrs = {"source": f"CDS {c['blh_cds_dataset']}", "band": c["blh_cds_variable"]}
        _gee.log_coverage(da, label)
        _write(da, "blh", m0, m1, fmt, {**attrs, "composite": "daily mean of hourly values (UTC day)"})


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_date_args(p)
    p.add_argument("--vars", nargs="+", default=None,
                   help="subset of variables (default: all in settings.yaml + blh)")
    p.add_argument("--blh-source", choices=["auto", "gee", "cds", "none"], default="auto")
    p.add_argument("--format", choices=["nc", "tif"], default="nc")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    setup_logging(args.verbose)
    start, end = resolve_dates(args)
    variables = args.vars or [*cfg()["bands"], "blh"]
    run(start, end, variables, args.blh_source, args.format, args.overwrite)


if __name__ == "__main__":
    main()
