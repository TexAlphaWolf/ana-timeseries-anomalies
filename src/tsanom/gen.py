"""Synthetic series with anomalies injected at known positions.

A series is a level, a slow trend, a daily and a weekly cycle and AR(1) noise. The first `train_fraction` of it is clean;
the anomalies are placed in the rest, never closer than `MIN_GAP` points to each other, and every one is recorded as an
event (kind, first point, end point). Sizes of the anomalies are in units of the noise standard deviation, so the same
difficulty holds for any level.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

Floats = NDArray[np.float64]

KINDS = ("spike", "level_shift", "variance_burst", "dropout")
MIN_GAP = 400  # points between two events, so that one cannot hide or mimic another
LEVEL = 100.0
NOISE_SD = 2.0
PHI = 0.5  # AR(1) coefficient of the noise
START = datetime(2026, 1, 1)


class SeriesError(ValueError):
    """The request cannot be met (too short a series, a rate that does not fit, a bad file)."""


@dataclass(frozen=True)
class Event:
    kind: str
    start: int
    end: int  # exclusive

    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class Series:
    values: Floats
    events: tuple[Event, ...]
    train_n: int
    period: int
    seed: int

    def labels(self) -> NDArray[np.bool_]:
        out = np.zeros(len(self.values), dtype=bool)
        for e in self.events:
            out[e.start : e.end] = True
        return out


def base(n: int, period: int, rng: np.random.Generator) -> Floats:
    t = np.arange(n, dtype=np.float64)
    daily = 10.0 * np.sin(2 * np.pi * t / period) + 4.0 * np.sin(4 * np.pi * t / period + 0.7)
    weekly = 5.0 * np.sin(2 * np.pi * t / (7 * period) + 1.1)
    trend = 0.5 * t / max(n, 1)  # half a unit over the whole series
    eps = rng.normal(0.0, NOISE_SD * np.sqrt(1 - PHI**2), n)
    noise = np.empty(n)
    noise[0] = rng.normal(0.0, NOISE_SD)
    for i in range(1, n):
        noise[i] = PHI * noise[i - 1] + eps[i]
    return LEVEL + daily + weekly + trend + noise


def place(count: int, lo: int, hi: int, lengths: list[int], rng: np.random.Generator) -> list[tuple[int, int]] | None:
    """Starts for `count` events in [lo, hi), each at least MIN_GAP from the others; None if it does not fit."""
    taken: list[tuple[int, int]] = []
    for length in lengths:
        for _ in range(500):
            s = int(rng.integers(lo, hi - length))
            if all(s >= b + MIN_GAP or s + length + MIN_GAP <= a for a, b in taken):
                taken.append((s, s + length))
                break
        else:
            return None
    return taken


def generate(n: int, seed: int, period: int = 144, train_fraction: float = 0.25, rate: float = 4.0) -> Series:
    """`rate` events of each kind per 10,000 points of the test part (at least one of each kind)."""
    if period < 8:
        raise SeriesError("period must be at least 8 points")
    if not 0.1 <= train_fraction <= 0.6:
        raise SeriesError("train_fraction must be between 0.1 and 0.6")
    train_n = int(n * train_fraction)
    if train_n < 3 * period:
        raise SeriesError(f"need at least {3 * period} clean points (three periods); got {train_n}: use more points")
    if rate <= 0:
        raise SeriesError("rate must be positive")
    rng = np.random.default_rng(seed)
    x = base(n, period, rng)
    test_n = n - train_n
    per_kind = max(1, round(rate * test_n / 10_000))
    lengths: list[tuple[str, int]] = []
    for kind in KINDS:
        for _ in range(per_kind):
            span = {
                "spike": 1,
                "level_shift": int(rng.integers(20, 61)),
                "variance_burst": int(rng.integers(20, 61)),
                "dropout": int(rng.integers(5, 21)),
            }[kind]
            lengths.append((kind, span))
    order = rng.permutation(len(lengths))
    lengths = [lengths[i] for i in order]
    spots = place(len(lengths), train_n + period, n - period, [s for _, s in lengths], rng)
    if spots is None:
        raise SeriesError(
            f"{len(lengths)} events do not fit in {test_n} test points; lower the rate or use more points"
        )
    events: list[Event] = []
    for (kind, _), (a, b) in zip(lengths, spots, strict=True):
        sign = 1.0 if rng.random() < 0.5 else -1.0
        if kind == "spike":
            x[a] += sign * rng.uniform(5.0, 10.0) * NOISE_SD
        elif kind == "level_shift":
            x[a:b] += sign * rng.uniform(4.0, 8.0) * NOISE_SD
        elif kind == "variance_burst":
            x[a:b] += rng.normal(0.0, rng.uniform(3.0, 5.0) * NOISE_SD, b - a)
        else:  # dropout: the sensor reports almost nothing
            x[a:b] = rng.uniform(0.0, 2.0, b - a)
        events.append(Event(kind, a, b))
    events.sort(key=lambda e: e.start)
    return Series(x, tuple(events), train_n, period, seed)


def timestamps(n: int, period: int) -> list[str]:
    step = timedelta(seconds=86400 // period)
    return [(START + i * step).isoformat() for i in range(n)]


def write(series: Series, out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    labels = series.labels()
    ts = timestamps(len(series.values), series.period)
    path = out / "series.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["ts", "value", "anomaly"])
        for t, v, lab in zip(ts, series.values, labels, strict=True):
            w.writerow([t, f"{v:.4f}", int(lab)])
    meta = out / "events.json"
    meta.write_text(
        json.dumps(
            {
                "points": len(series.values),
                "period": series.period,
                "train_points": series.train_n,
                "seed": series.seed,
                "events": [asdict(e) for e in series.events],
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    return [path, meta]


def read_values(path: Path) -> Floats:
    """Values from a CSV with a `value` column (or a single column of numbers)."""
    try:
        with path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
    except OSError as exc:
        raise SeriesError(f"cannot read {path}: {exc}") from exc
    if not rows:
        raise SeriesError(f"{path} is empty")
    head = [c.strip().lower() for c in rows[0]]
    col = head.index("value") if "value" in head else None
    body = rows[1:] if (col is not None or not _is_number(rows[0][0])) else rows
    if col is None:
        if len(rows[0]) != 1:
            raise SeriesError(f"{path}: no 'value' column in the header {rows[0]}")
        col = 0
    try:
        vals = np.array([float(r[col]) for r in body], dtype=np.float64)
    except (ValueError, IndexError) as exc:
        raise SeriesError(f"{path}: a value is missing or is not a number ({exc})") from exc
    if not np.all(np.isfinite(vals)):
        raise SeriesError(f"{path}: contains nan or infinite values")
    return vals


def _is_number(s: str) -> bool:
    try:
        float(s)
    except ValueError:
        return False
    return True


def read_events(path: Path) -> tuple[list[Event], dict[str, Any]]:
    meta = json.loads(path.read_text(encoding="utf-8"))
    return [Event(**e) for e in meta["events"]], meta
