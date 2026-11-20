"""Common 0.1 deg India grid (EPSG:4326) and helpers to put data onto it.

Grid geometry comes from `grid` in config/settings.yaml. Cells are north-up:
`lat` runs from lat_max - res/2 down to lat_min + res/2, `lon` from lon_min + res/2
up to lon_max - res/2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from src.config import load_settings


@dataclass(frozen=True)
class Grid:
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    res: float
    crs: str

    @property
    def nlon(self) -> int:
        return int(round((self.lon_max - self.lon_min) / self.res))

    @property
    def nlat(self) -> int:
        return int(round((self.lat_max - self.lat_min) / self.res))

    @property
    def lon(self) -> np.ndarray:
        return np.round(self.lon_min + self.res * (np.arange(self.nlon) + 0.5), 6)

    @property
    def lat(self) -> np.ndarray:
        return np.round(self.lat_max - self.res * (np.arange(self.nlat) + 0.5), 6)

    @property
    def crs_transform(self) -> list[float]:
        """Affine [xScale, xShear, x0, yShear, yScale, y0] (GDAL/EE order)."""
        return [self.res, 0.0, self.lon_min, 0.0, -self.res, self.lat_max]

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (self.lon_min, self.lat_min, self.lon_max, self.lat_max)

    def coords(self) -> dict[str, np.ndarray]:
        return {"lat": self.lat, "lon": self.lon}

    def empty(self, fill: float = np.nan, dtype=np.float32) -> np.ndarray:
        return np.full((self.nlat, self.nlon), fill, dtype=dtype)


def india_grid() -> Grid:
    g = load_settings()["grid"]
    b = g["bbox"]
    return Grid(
        lon_min=float(b["lon_min"]),
        lon_max=float(b["lon_max"]),
        lat_min=float(b["lat_min"]),
        lat_max=float(b["lat_max"]),
        res=float(g["resolution_deg"]),
        crs=g["crs"],
    )


def bin_points_to_grid(
    lat: np.ndarray,
    lon: np.ndarray,
    values: np.ndarray | None = None,
    how: str = "mean",
    grid: Grid | None = None,
) -> np.ndarray:
    """Aggregate scattered points into grid cells.

    how: "count" (values ignored), "sum", or "mean". Points outside the grid
    or with non-finite value/coords are dropped. Empty cells are 0 for
    count/sum and NaN for mean. Returns (nlat, nlon) float64, north-up.
    """
    grid = grid or india_grid()
    lat = np.asarray(lat, dtype="f8").ravel()
    lon = np.asarray(lon, dtype="f8").ravel()
    ok = np.isfinite(lat) & np.isfinite(lon)
    if values is not None:
        values = np.asarray(values, dtype="f8").ravel()
        ok &= np.isfinite(values)
    lat, lon = lat[ok], lon[ok]

    # Ascending edges for histogram2d; flip lat afterwards to north-up.
    lat_edges = grid.lat_min + grid.res * np.arange(grid.nlat + 1)
    lon_edges = grid.lon_min + grid.res * np.arange(grid.nlon + 1)
    bins = (lat_edges, lon_edges)

    count, _, _ = np.histogram2d(lat, lon, bins=bins)
    if how == "count":
        return count[::-1]
    if values is None:
        raise ValueError(f"how={how!r} requires values")
    total, _, _ = np.histogram2d(lat, lon, bins=bins, weights=values[ok])
    if how == "sum":
        return total[::-1]
    if how == "mean":
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(count > 0, total / count, np.nan)[::-1]
    raise ValueError(f"unknown how={how!r}")


def to_grid(da: xr.DataArray, method: str = "linear") -> xr.DataArray:
    """Interpolate a regular lat/lon DataArray onto the common grid."""
    grid = india_grid()
    rename = {k: v for k, v in {"latitude": "lat", "longitude": "lon"}.items() if k in da.dims}
    da = da.rename(rename).sortby("lat").sortby("lon")
    return da.interp(lat=grid.lat, lon=grid.lon, method=method)
