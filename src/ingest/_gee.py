"""Shared Earth Engine helpers for the ingest modules.

Every product is pulled onto the exact common grid (EPSG:4326, 0.1 deg, origin
at lon_min/lat_max from settings.yaml) with `ee.data.computePixels`, one band per
day, one request per product-month (~12 MB, under the 48 MB request limit).
Requests that fail (timeouts, memory) are split in half and retried.
"""

from __future__ import annotations

import logging
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import xarray as xr

from src.config import load_settings
from src.grid.regrid import Grid, india_grid

log = logging.getLogger(__name__)

NODATA = -9999.0
_EE_READY = False


def init_ee() -> None:
    """Initialise Earth Engine with the project id from settings.yaml."""
    global _EE_READY
    if _EE_READY:
        return
    import ee

    project = load_settings()["project"]["gee_project_id"]
    if not project or project.startswith("your-"):
        raise SystemExit("Set project.gee_project_id in config/settings.yaml")
    try:
        ee.Initialize(project=project)
    except Exception:  # not authenticated yet
        log.info("Earth Engine not authenticated; running ee.Authenticate()")
        ee.Authenticate()
        ee.Initialize(project=project)
    _EE_READY = True
    log.info("Earth Engine initialised (project=%s)", project)


def ee_region():
    import ee

    g = india_grid()
    return ee.Geometry.Rectangle(list(g.bounds), proj="EPSG:4326", geodesic=False)


def ee_grid_projection():
    import ee

    g = india_grid()
    return ee.Projection(g.crs, g.crs_transform)


def day_window(d: date):
    """[d 00:00 UTC, d+1 00:00 UTC) as EE dates."""
    import ee

    return ee.Date(d.isoformat()), ee.Date((d + timedelta(days=1)).isoformat())


def masked_image():
    """A single fully-masked float band (stand-in for a day with no data)."""
    import ee

    return ee.Image.constant(0).toFloat().updateMask(ee.Image.constant(0))


def day_or_empty(coll, build: Callable):
    """`build(coll)` if the (already date-filtered) collection is non-empty,
    else a fully-masked image. Lets missing days become NaN instead of erroring."""
    import ee

    return ee.Image(ee.Algorithms.If(coll.size().gt(0), build(coll), masked_image()))


def band_name(d: date) -> str:
    return "d" + d.strftime("%Y%m%d")


def _compute_pixels(image, grid: Grid) -> np.ndarray:
    import ee

    t = grid.crs_transform
    return ee.data.computePixels(
        {
            "expression": image,
            "fileFormat": "NUMPY_NDARRAY",
            "grid": {
                "dimensions": {"width": grid.nlon, "height": grid.nlat},
                "affineTransform": {
                    "scaleX": t[0], "shearX": t[1], "translateX": t[2],
                    "shearY": t[3], "scaleY": t[4], "translateY": t[5],
                },
                "crsCode": grid.crs,
            },
        }
    )


def fetch_days(
    day_image: Callable[[date], object],
    days: list[date],
    name: str,
    retries: int = 3,
) -> xr.DataArray:
    """Fetch one daily image per day onto the common grid -> DataArray(time, lat, lon).

    `day_image(d)` must return a single-band ee.Image (masked where no data).
    """
    import ee

    grid = india_grid()
    stack = ee.Image.cat([ee.Image(day_image(d)).rename(band_name(d)) for d in days])
    stack = stack.toFloat().unmask(NODATA, False)

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            arr = _compute_pixels(stack, grid)
            break
        except ee.EEException as e:
            last_err = e
            msg = str(e).lower()
            splittable = len(days) > 1 and any(
                k in msg for k in ("timed out", "memory", "too large", "too many", "limit")
            )
            if splittable:
                mid = len(days) // 2
                log.warning("%s: request too heavy (%s); splitting %d days", name, e, len(days))
                return xr.concat(
                    [fetch_days(day_image, days[:mid], name), fetch_days(day_image, days[mid:], name)],
                    dim="time",
                )
            wait = 2 ** attempt * 5
            log.warning("%s: EE error (%s); retry %d/%d in %ds", name, e, attempt + 1, retries, wait)
            time.sleep(wait)
        except Exception as e:  # network hiccups
            last_err = e
            wait = 2 ** attempt * 5
            log.warning("%s: %s; retry %d/%d in %ds", name, e, attempt + 1, retries, wait)
            time.sleep(wait)
    else:
        raise RuntimeError(f"{name}: failed after {retries} attempts") from last_err

    data = np.stack([arr[band_name(d)].astype("f4") for d in days])
    data[data <= NODATA + 1] = np.nan
    return xr.DataArray(
        data,
        dims=("time", "lat", "lon"),
        coords={"time": pd.to_datetime(days), **grid.coords()},
        name=name,
    )


def log_coverage(da: xr.DataArray, label: str) -> list[date]:
    """Log valid-day count; return the days that are entirely NaN."""
    valid = da.notnull().any(dim=("lat", "lon")).values
    missing = [pd.Timestamp(t).date() for t, ok in zip(da.time.values, valid) if not ok]
    log.info("%s: %d/%d days with data", label, int(valid.sum()), len(valid))
    if missing:
        log.warning("%s: no data on %s", label, ", ".join(str(d) for d in missing))
    return missing


def grid_attrs() -> dict:
    g = india_grid()
    return {
        "crs": g.crs,
        "grid_resolution_deg": g.res,
        "bbox_lon_lat": list(g.bounds),
        "crs_transform": g.crs_transform,
    }


def write_nc(ds: xr.Dataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.attrs.update(grid_attrs())
    ds["lat"].attrs.update(units="degrees_north", standard_name="latitude")
    ds["lon"].attrs.update(units="degrees_east", standard_name="longitude")
    enc = {v: {"zlib": True, "complevel": 4, "dtype": "float32"} for v in ds.data_vars}
    tmp = path.with_suffix(".nc.part")
    ds.to_netcdf(tmp, encoding=enc)
    tmp.replace(path)
    log.info("wrote %s", path)


def write_daily_tifs(da: xr.DataArray, out_dir: Path, prefix: str) -> None:
    """One single-band GeoTIFF per day (same grid as the NetCDF output)."""
    import importlib

    importlib.import_module("rioxarray")  # registers the .rio accessor

    out_dir.mkdir(parents=True, exist_ok=True)
    g = india_grid()
    for t in da.time.values:
        day = pd.Timestamp(t)
        out = out_dir / f"{prefix}_{day:%Y%m%d}.tif"
        r = (
            da.sel(time=t)
            .rename({"lon": "x", "lat": "y"})
            .rio.write_crs(g.crs)
            .rio.write_nodata(np.nan)
        )
        r.rio.to_raster(out, driver="COG", compress="DEFLATE")
    log.info("wrote %d GeoTIFFs to %s", da.sizes["time"], out_dir)
