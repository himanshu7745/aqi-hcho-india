"""Sentinel-5P TROPOMI ingest via Google Earth Engine (NO2, SO2, CO, O3, HCHO).

Daily mean composite of the OFFL L3 granules whose acquisition falls in the UTC
day (India overpass ~08 UTC, so one overpass = one UTC day). The ~1.1 km L3
pixels are area-averaged onto the 0.1 deg common grid with reduceResolution.
Units are unchanged (mol/m^2). Days with no granules are written as all-NaN.

Output: data/raw/s5p/<product>/s5p_<product>_YYYY-MM.nc   (time, lat, lon)
        or one GeoTIFF per day with --format tif.

    python -m src.ingest.gee_s5p --start 2023-10-01 --end 2023-11-30
    python -m src.ingest.gee_s5p --start 2023-10-01 --end 2023-11-30 --products hcho no2
"""

from __future__ import annotations

import argparse
import logging
from datetime import date

from src.config import (
    add_date_args, chunk_tag, date_range, load_settings, month_chunks, raw_dir, resolve_dates, setup_logging,
)
from src.ingest import _gee

log = logging.getLogger("ingest.s5p")


def products() -> dict[str, tuple[str, str]]:
    return {k: tuple(v) for k, v in load_settings()["ingest"]["s5p"]["products"].items()}


def daily_image(collection: str, band: str, day: date):
    import ee

    t0, t1 = _gee.day_window(day)
    coll = (
        ee.ImageCollection(collection)
        .filterDate(t0, t1)
        .filterBounds(_gee.ee_region())
        .select(band)
    )

    def build(c):
        native = ee.Image(c.first()).projection()
        return (
            c.mean()
            .setDefaultProjection(native)
            .reduceResolution(ee.Reducer.mean(), bestEffort=True, maxPixels=1024)
        )

    return _gee.day_or_empty(coll, build)


def run(start: date, end: date, names: list[str], fmt: str = "nc", overwrite: bool = False) -> None:
    _gee.init_ee()
    prods = products()
    unknown = set(names) - set(prods)
    if unknown:
        raise SystemExit(f"unknown products {sorted(unknown)}; choose from {sorted(prods)}")

    for name in names:
        collection, band = prods[name]
        for m0, m1 in month_chunks(start, end):
            label = f"s5p {name} {m0:%Y-%m}"
            out_dir = raw_dir("s5p", name)
            nc_path = out_dir / f"s5p_{name}_{chunk_tag(m0, m1)}.nc"
            if fmt == "nc" and nc_path.exists() and not overwrite:
                log.info("%s: exists, skipping (%s)", label, nc_path.name)
                continue

            days = list(date_range(m0, m1))
            log.info("%s: fetching %d days (%s)", label, len(days), collection)
            da = _gee.fetch_days(lambda d: daily_image(collection, band, d), days, name)
            _gee.log_coverage(da, label)
            da.attrs.update(units="mol m-2", source=collection, band=band,
                            composite="daily mean of OFFL L3 granules (UTC day), area-mean to grid")

            if fmt == "tif":
                _gee.write_daily_tifs(da, out_dir, f"s5p_{name}")
            else:
                _gee.write_nc(da.to_dataset(), nc_path)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_date_args(p)
    p.add_argument("--products", nargs="+", default=None,
                   help="subset of products (default: all in settings.yaml)")
    p.add_argument("--format", choices=["nc", "tif"], default="nc")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    setup_logging(args.verbose)
    start, end = resolve_dates(args)
    run(start, end, args.products or list(products()), args.format, args.overwrite)


if __name__ == "__main__":
    main()
