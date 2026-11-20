"""Unit tests for ingest helpers (no network)."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.config import chunk_tag, month_chunks
from src.grid.regrid import bin_points_to_grid, india_grid
from src.ingest.cpcb import archive_time, clean_hourly, daily_from_hourly, to_cpcb_units


# ----------------------------------------------------------------- grid
def test_india_grid_geometry():
    g = india_grid()
    assert (g.nlat, g.nlon) == (320, 300)
    assert g.lat[0] == pytest.approx(37.95) and g.lat[-1] == pytest.approx(6.05)
    assert g.lon[0] == pytest.approx(68.05) and g.lon[-1] == pytest.approx(97.95)
    assert g.crs_transform == [0.1, 0.0, 68.0, 0.0, -0.1, 38.0]


def test_bin_points_to_grid():
    g = india_grid()
    lat = np.array([28.61, 28.62, 28.65, 10.0, 50.0])   # last one outside bbox
    lon = np.array([77.21, 77.22, 77.25, 80.0, 77.0])
    val = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
    cnt = bin_points_to_grid(lat, lon, how="count")
    mean = bin_points_to_grid(lat, lon, val, how="mean")
    total = bin_points_to_grid(lat, lon, val, how="sum")
    i = int(np.argmin(np.abs(g.lat - 28.65)))
    j = int(np.argmin(np.abs(g.lon - 77.25)))
    assert cnt[i, j] == 3 and mean[i, j] == pytest.approx(3.0) and total[i, j] == 9
    assert cnt.sum() == 4  # out-of-bbox point dropped
    assert np.isnan(mean).sum() == g.nlat * g.nlon - 2


def test_month_chunks_and_tags():
    ch = list(month_chunks(date(2023, 10, 15), date(2023, 12, 31)))
    assert ch[0] == (date(2023, 10, 15), date(2023, 10, 31))
    assert [chunk_tag(*c) for c in ch] == ["20231015_20231031", "2023-11", "2023-12"]


# ----------------------------------------------------------------- CPCB daily rules
def _hours(day: str, values) -> pd.Series:
    idx = pd.date_range(day, periods=len(values), freq="h")
    return pd.Series(values, index=idx, dtype="f8")


def _frame(param: str, s: pd.Series) -> pd.DataFrame:
    s = s.dropna()
    return pd.DataFrame({"parameter": param, "time": s.index, "value": s.values})


def test_24h_mean_requires_16_hours():
    full = _hours("2023-11-01", np.arange(24))           # 24 h -> mean 11.5
    sparse = _hours("2023-11-02", [10.0] * 15)           # 15 h -> invalid
    ok16 = _hours("2023-11-03", [20.0] * 16)             # 16 h -> valid
    h = pd.concat([_frame("pm25", s) for s in (full, sparse, ok16)])
    d = daily_from_hourly(h, min_hours=16)
    assert d.loc[date(2023, 11, 1), "pm25"] == pytest.approx(11.5)
    assert np.isnan(d.loc[date(2023, 11, 2), "pm25"])
    assert d.loc[date(2023, 11, 2), "pm25_n"] == 15
    assert d.loc[date(2023, 11, 3), "pm25"] == pytest.approx(20.0)


def test_o3_is_max_8h_running_mean():
    vals = np.full(24, 10.0)
    vals[12:20] = 100.0                                   # one 8 h block at 100
    d = daily_from_hourly(_frame("o3", _hours("2023-11-01", vals)), min_hours=16)
    assert d.loc[date(2023, 11, 1), "o3"] == pytest.approx(100.0)

    vals = np.full(24, 10.0)
    vals[12] = 170.0                                      # single spike: 8h mean = 30
    d = daily_from_hourly(_frame("o3", _hours("2023-11-01", vals)), min_hours=16)
    assert d.loc[date(2023, 11, 1), "o3"] == pytest.approx(30.0)


def test_8h_window_needs_6_of_8_hours():
    vals = np.full(24, np.nan)
    vals[0:5] = 50.0                                      # only 5 h anywhere in a window
    vals[8:24] = 10.0
    s = _hours("2023-11-01", vals)
    d = daily_from_hourly(_frame("co", s), min_hours=16, rolling_min=6)
    # windows over 00-04 have only 5 values -> rejected, so the 50s never reach the max
    assert d.loc[date(2023, 11, 1), "co"] == pytest.approx(
        pd.Series(vals, index=s.index).rolling(8, min_periods=6).mean().max()
    )
    assert d.loc[date(2023, 11, 1), "co"] < 50


def test_unit_conversion():
    df = pd.DataFrame({
        "parameter": ["no2", "co", "co", "o3", "pm25"],
        "units": ["ppb", "ppm", "µg/m³", "ppb", "µg/m³"],
        "value": [10.0, 1.0, 1500.0, 24.45, 55.0],
    })
    v = to_cpcb_units(df)["value"].to_numpy()
    assert v[0] == pytest.approx(10 * 46.0055 / 24.45)
    assert v[1] == pytest.approx(28.01 / 24.45)           # mg/m3
    assert v[2] == pytest.approx(1.5)                      # ug/m3 -> mg/m3
    assert v[3] == pytest.approx(47.997)
    assert v[4] == 55.0


def test_clean_hourly_ist_and_filters():
    df = pd.DataFrame({
        "time": pd.to_datetime(["2023-10-31T18:30:00Z", "2023-10-31T18:30:00Z", "2023-10-31T19:30:00Z",
                                "2023-10-31T20:30:00Z"], utc=True),
        "parameter": ["pm25", "pm25", "pm25", "pm25"],
        "units": ["µg/m³"] * 4,
        "value": [100.0, 120.0, -5.0, 99999.0],
    })
    out = clean_hourly(df)
    assert len(out) == 1                                   # negative + implausible dropped, dupes averaged
    assert out["time"].iloc[0] == pd.Timestamp("2023-11-01 00:00")  # 18:30Z -> 00:00 IST
    assert out["value"].iloc[0] == pytest.approx(110.0)


def test_subhourly_period_end_records_bin_into_hour():
    # archive rows: 30-min values stamped at period END, IST offset
    raw = pd.DataFrame({
        "datetime": ["2023-11-01T00:30:00+05:30", "2023-11-01T01:00:00+05:30", "2023-11-01T01:30:00+05:30"],
        "parameter": ["pm25"] * 3, "units": ["µg/m³"] * 3, "value": [200.0, 210.0, 300.0],
    })
    raw["time"] = archive_time(raw.pop("datetime"))
    out = clean_hourly(raw).set_index("time")["value"]
    assert out[pd.Timestamp("2023-11-01 00:00")] == pytest.approx(205.0)
    assert out[pd.Timestamp("2023-11-01 01:00")] == pytest.approx(300.0)
