"""Project settings loader. `config/settings.yaml` is the single source of truth."""

from __future__ import annotations

import argparse
import logging
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

import yaml

ROOT = Path(__file__).resolve().parents[1]
SETTINGS_PATH = ROOT / "config" / "settings.yaml"


@lru_cache(maxsize=1)
def load_settings(path: str | Path = SETTINGS_PATH) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def project_path(rel: str | Path) -> Path:
    """Resolve a settings path (relative to repo root) to an absolute Path."""
    p = Path(rel)
    return p if p.is_absolute() else ROOT / p


def raw_dir(*parts: str) -> Path:
    """data/raw/<parts...>, created on demand."""
    d = project_path(load_settings()["paths"]["raw"]).joinpath(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d


def parse_date(s: str | date) -> date:
    return s if isinstance(s, date) else date.fromisoformat(str(s))


def date_range(start: date, end: date) -> Iterator[date]:
    """Inclusive daily range."""
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def month_chunks(start: date, end: date) -> Iterator[tuple[date, date]]:
    """Split [start, end] (inclusive) into calendar-month pieces."""
    cur = start
    while cur <= end:
        nxt = (cur.replace(day=1) + timedelta(days=32)).replace(day=1)
        yield cur, min(end, nxt - timedelta(days=1))
        cur = nxt


def add_date_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--start", help="YYYY-MM-DD (default: dates.start in settings.yaml)")
    p.add_argument("--end", help="YYYY-MM-DD inclusive (default: dates.end in settings.yaml)")


def resolve_dates(args: argparse.Namespace) -> tuple[date, date]:
    cfg = load_settings()["dates"]
    start = parse_date(args.start or cfg["start"])
    end = parse_date(args.end or cfg["end"])
    if end < start:
        raise SystemExit(f"--end {end} is before --start {start}")
    return start, end


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def chunk_tag(m0: date, m1: date) -> str:
    """'YYYY-MM' for a full calendar month, else 'YYYYMMDD_YYYYMMDD'."""
    import calendar

    full = m0.day == 1 and m1.day == calendar.monthrange(m1.year, m1.month)[1] and m0.month == m1.month
    return f"{m0:%Y-%m}" if full else f"{m0:%Y%m%d}_{m1:%Y%m%d}"
