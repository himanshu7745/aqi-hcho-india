"""INSAT-3D / 3DR AOD ingest from MOSDAC (3DIMG_L2G_AOD HDF5, downloaded manually).

Step 1, confirm variable names (prints the HDF5 tree of one file):

    python -m src.ingest.mosdac_insat inspect [path/to/file.h5]

Then set ingest.insat.{aod_var, lat_var, lon_var, qc_var, qc_good} in
config/settings.yaml.

Step 2, daily product:

    python -m src.ingest.mosdac_insat process [--start 2023-10-01 --end 2023-11-30]

For each scene (~30-min) in data/raw/insat/: apply scale/offset, mask fill
values, values outside ingest.insat.valid_range, and (if qc_var is set) pixels
whose QC is not in qc_good. Every valid pixel is binned into the 0.1 deg grid.
The daily AOD is sum/count over all of that UTC day's scenes, i.e. weighted by
observations. n_obs (valid native pixels) and n_scenes (scenes contributing to
each cell) are saved alongside.

Output: data/raw/insat/daily/insat_aod_YYYYMMDD.nc  (lat, lon)
"""

from __future__ import annotations

import argparse
import logging
import re
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

import numpy as np
import xarray as xr

from src.config import load_settings, parse_date, project_path, setup_logging
from src.grid.regrid import bin_points_to_grid, india_grid
from src.ingest import _gee

log = logging.getLogger("ingest.insat")

# e.g. 3DIMG_01NOV2023_0815_L2G_AOD.h5 / 3RIMG_01NOV2023_0815_L2G_AOD_V01R00.h5
FNAME_RE = re.compile(r"_(\d{2}[A-Z]{3}\d{4})_(\d{4})_")


def cfg() -> dict:
    return load_settings()["ingest"]["insat"]


def scene_time(path: Path) -> datetime | None:
    m = FNAME_RE.search(path.name.upper())
    if not m:
        return None
    return datetime.strptime(m.group(1) + m.group(2), "%d%b%Y%H%M")


# --------------------------------------------------------------------------- inspect
def _fmt_attrs(attrs) -> str:
    parts = []
    for k, v in attrs.items():
        if isinstance(v, bytes):
            v = v.decode(errors="replace")
        elif isinstance(v, np.ndarray) and v.size == 1:
            v = v.item()
        s = repr(v)
        parts.append(f"{k}={s[:80] + '...' if len(s) > 80 else s}")
    return ", ".join(parts)


def inspect(path: Path) -> None:
    import h5py

    print(f"FILE {path}")
    with h5py.File(path, "r") as f:
        if f.attrs:
            print(f"  [root attrs] {_fmt_attrs(f.attrs)}")

        def visit(name, obj):
            depth = name.count("/")
            pad = "  " * (depth + 1)
            if isinstance(obj, h5py.Group):
                print(f"{pad}{name}/")
            else:
                print(f"{pad}{name}  shape={obj.shape} dtype={obj.dtype}")
                if obj.attrs:
                    print(f"{pad}    attrs: {_fmt_attrs(obj.attrs)}")
                if obj.size and obj.dtype.kind in "fiu":
                    sl = obj[...] if obj.size <= 5_000_000 else obj[tuple(slice(0, None, 4) for _ in obj.shape)]
                    print(f"{pad}    min={np.nanmin(sl)!r} max={np.nanmax(sl)!r}")

        f.visititems(visit)


# --------------------------------------------------------------------------- read one scene
def _read_var(f, name: str) -> np.ndarray:
    """Read a dataset, apply _FillValue -> NaN and scale_factor/add_offset."""
    ds = f[name]
    raw = ds[...]
    data = raw.astype("f8")
    attrs = {k.lower(): v for k, v in ds.attrs.items()}
    for key in ("_fillvalue", "fill_value", "fillvalue", "missing_value"):
        if key in attrs:
            fill = np.atleast_1d(attrs[key]).astype("f8")
            data[np.isin(raw.astype("f8"), fill)] = np.nan
    scale = float(np.atleast_1d(attrs.get("scale_factor", 1.0))[0])
    offset = float(np.atleast_1d(attrs.get("add_offset", 0.0))[0])
    data = data * scale + offset
    return np.squeeze(data)


def read_scene(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (lat, lon, aod) flattened, valid pixels only."""
    import h5py

    c = cfg()
    with h5py.File(path, "r") as f:
        for key in ("aod_var", "lat_var", "lon_var"):
            if c[key] not in f:
                raise KeyError(f"{path.name}: dataset {c[key]!r} ({key}) not found; "
                               f"run `python -m src.ingest.mosdac_insat inspect` and fix settings.yaml")
        aod = _read_var(f, c["aod_var"])
        lat = _read_var(f, c["lat_var"])
        lon = _read_var(f, c["lon_var"])
        qc = np.squeeze(f[c["qc_var"]][...]) if c.get("qc_var") else None

    # L2G may ship 1-D lat/lon axes for a regular grid.
    if lat.ndim == 1 and lon.ndim == 1 and aod.ndim == 2:
        if aod.shape == (lat.size, lon.size):
            lon, lat = np.meshgrid(lon, lat)
        elif aod.shape == (lon.size, lat.size):
            lat, lon = np.meshgrid(lat, lon)
    if not (lat.shape == lon.shape == aod.shape):
        raise ValueError(f"{path.name}: shape mismatch aod{aod.shape} lat{lat.shape} lon{lon.shape}")

    lo, hi = c["valid_range"]
    ok = np.isfinite(aod) & np.isfinite(lat) & np.isfinite(lon) & (aod >= lo) & (aod <= hi)
    ok &= (np.abs(lat) <= 90) & (np.abs(lon) <= 180)
    if qc is not None:
        ok &= np.isin(qc, c["qc_good"])
    return lat[ok], lon[ok], aod[ok]


# --------------------------------------------------------------------------- daily product
def find_scenes(start: date | None, end: date | None) -> dict[date, list[tuple[datetime, Path]]]:
    c = cfg()
    raw = project_path(c["raw_dir"])
    by_day: dict[date, list[tuple[datetime, Path]]] = defaultdict(list)
    for p in sorted(raw.rglob(c["file_glob"])):
        if "daily" in p.parts:
            continue
        t = scene_time(p)
        if t is None:
            log.warning("cannot parse date/time from %s; skipping", p.name)
            continue
        if (start and t.date() < start) or (end and t.date() > end):
            continue
        by_day[t.date()].append((t, p))
    return dict(sorted(by_day.items()))


def process_day(day: date, scenes: list[tuple[datetime, Path]], out: Path) -> None:
    g = india_grid()
    total = np.zeros((g.nlat, g.nlon))
    count = np.zeros((g.nlat, g.nlon))
    n_scenes = np.zeros((g.nlat, g.nlon))
    used = []
    for t, p in sorted(scenes):
        try:
            lat, lon, aod = read_scene(p)
        except KeyError:
            raise
        except Exception as e:  # corrupt / truncated download
            log.warning("%s: unreadable (%s); skipping", p.name, e)
            continue
        if aod.size == 0:
            log.debug("%s: no valid pixels", p.name)
            continue
        cnt = bin_points_to_grid(lat, lon, how="count", grid=g)
        total += bin_points_to_grid(lat, lon, aod, how="sum", grid=g)
        count += cnt
        n_scenes += cnt > 0
        used.append(t.strftime("%H%M"))

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, total / count, np.nan)
    ds = xr.Dataset(
        {
            "aod": (("lat", "lon"), mean.astype("f4"), {"units": "1", "long_name": "aerosol optical depth (550 nm)"}),
            "n_obs": (("lat", "lon"), count.astype("f4"), {"long_name": "valid native pixels averaged"}),
            "n_scenes": (("lat", "lon"), n_scenes.astype("f4"), {"long_name": "scenes contributing"}),
        },
        coords=g.coords(),
        attrs={
            "date_utc": day.isoformat(),
            "source": "MOSDAC INSAT-3D/3DR L2G AOD",
            "scenes_used_utc": ",".join(used),
            "n_files": len(scenes),
        },
    )
    _gee.write_nc(ds, out)
    log.info("%s: %d/%d scenes used, %.1f%% cells with AOD", day, len(used), len(scenes),
             100 * np.isfinite(mean).mean())


def process(start: date | None, end: date | None, overwrite: bool) -> None:
    scenes = find_scenes(start, end)
    if not scenes:
        log.warning("no files matching %r in %s", cfg()["file_glob"], cfg()["raw_dir"])
        return
    out_dir = project_path(cfg()["raw_dir"]) / "daily"
    log.info("%d day(s), %d scene(s)", len(scenes), sum(map(len, scenes.values())))
    for day, sc in scenes.items():
        out = out_dir / f"insat_aod_{day:%Y%m%d}.nc"
        if out.exists() and not overwrite:
            log.info("%s: exists, skipping", day)
            continue
        process_day(day, sc, out)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("inspect", help="print the HDF5 tree of one file")
    pi.add_argument("file", nargs="?", help="default: first matching file in ingest.insat.raw_dir")
    pp = sub.add_parser("process", help="build daily 0.1 deg AOD NetCDFs")
    pp.add_argument("--start")
    pp.add_argument("--end")
    pp.add_argument("--overwrite", action="store_true")
    for sp in (pi, pp):
        sp.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    setup_logging(args.verbose)

    if args.cmd == "inspect":
        if args.file:
            path = Path(args.file)
        else:
            raw = project_path(cfg()["raw_dir"])
            files = sorted(f for f in raw.rglob("*.h5") if "daily" not in f.parts)
            if not files:
                raise SystemExit(f"no .h5 files in {raw}; download a 3DIMG_*_L2G_AOD file from MOSDAC first")
            path = files[0]
        inspect(path)
    else:
        process(
            parse_date(args.start) if args.start else None,
            parse_date(args.end) if args.end else None,
            args.overwrite,
        )


if __name__ == "__main__":
    main()
