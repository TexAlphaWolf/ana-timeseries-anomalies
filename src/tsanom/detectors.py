"""Four detectors. Each one is fitted on the clean start of a series and then scores every point of it (higher is more
anomalous), using only points up to the one being scored, as a live monitor would.

* rolling_median: distance from the median of the last `period` points, in robust standard deviations. Knows nothing
  of seasonality.
* seasonal_profile: the median shape of one period, learnt on the clean start, is subtracted; what is left is compared
  with its own recent median.
* seasonal_diff: difference from the same moment one period earlier. Needs no training for the shape, but an anomaly
  echoes one period later.
* isolation_forest: scikit-learn's forest on four features of the seasonal residual.

The threshold is the same rule for all: the largest score seen on the clean start.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import IsolationForest

Floats = NDArray[np.float64]
MAD_TO_SD = 1.4826
CHUNK = 20_000


class DetectorError(ValueError):
    """A detector cannot run with these inputs."""


def rolling_median(x: Floats, window: int) -> Floats:
    """Median of the `window` points before each one (not including it); nan for the first `window` points."""
    n = len(x)
    out = np.full(n, np.nan)
    if n <= window:
        return out
    view = np.lib.stride_tricks.sliding_window_view(x, window)  # row i is x[i : i + window]
    for lo in range(0, n - window, CHUNK):
        hi = min(lo + CHUNK, n - window)
        out[window + lo : window + hi] = np.median(view[lo:hi], axis=1)
    return out


def rolling_mean_std(x: Floats, window: int) -> tuple[Floats, Floats]:
    """Mean and standard deviation of the `window` points ending at each one (partial at the start)."""
    c1 = np.concatenate(([0.0], np.cumsum(x)))
    c2 = np.concatenate(([0.0], np.cumsum(x * x)))
    idx = np.arange(1, len(x) + 1)
    lo = np.maximum(idx - window, 0)
    k = (idx - lo).astype(np.float64)
    mean = (c1[idx] - c1[lo]) / k
    var = np.maximum((c2[idx] - c2[lo]) / k - mean**2, 0.0)
    return mean, np.sqrt(var)


def robust_sd(v: Floats) -> float:
    v = v[np.isfinite(v)]
    if len(v) == 0:
        raise DetectorError("no clean points to learn from")
    s = float(MAD_TO_SD * np.median(np.abs(v - np.median(v))))
    if s <= 0.0:
        raise DetectorError("the clean start is constant: nothing to compare against")
    return s


def profile_of(train: Floats, period: int) -> Floats:
    if len(train) < 3 * period:
        raise DetectorError(f"need at least {3 * period} clean points (three periods); got {len(train)}")
    return np.array([np.median(train[k::period]) for k in range(period)])


def residual(x: Floats, profile: Floats) -> Floats:
    return x - profile[np.arange(len(x)) % len(profile)]


@dataclass(frozen=True)
class Result:
    scores: Floats
    warmup: int  # scores before this index are not meaningful


def rolling_median_scores(x: Floats, train_n: int, period: int, seed: int) -> Result:
    dev = x - rolling_median(x, period)
    s = robust_sd(dev[period:train_n])
    scores = np.nan_to_num(np.abs(dev) / s, nan=0.0)
    return Result(scores, period)


def seasonal_profile_scores(x: Floats, train_n: int, period: int, seed: int) -> Result:
    r = residual(x, profile_of(x[:train_n], period))
    dev = r - rolling_median(r, period)
    s = robust_sd(dev[period:train_n])
    return Result(np.nan_to_num(np.abs(dev) / s, nan=0.0), period)


def seasonal_diff_scores(x: Floats, train_n: int, period: int, seed: int) -> Result:
    d = np.zeros_like(x)
    d[period:] = x[period:] - x[:-period]
    s = robust_sd(d[period:train_n])
    return Result(np.abs(d) / s, period)


def isolation_forest_scores(x: Floats, train_n: int, period: int, seed: int) -> Result:
    r = residual(x, profile_of(x[:train_n], period))
    s = robust_sd(r[:train_n] - np.median(r[:train_n]))
    z = (r - np.median(r[:train_n])) / s
    step = np.concatenate(([0.0], np.diff(z)))
    short = 12
    mean, sd = rolling_mean_std(z, short)
    feats = np.column_stack([z, step, mean, sd])
    forest = IsolationForest(n_estimators=100, random_state=seed, n_jobs=1)
    forest.fit(feats[period:train_n])
    return Result(-forest.score_samples(feats), period)


DETECTORS: dict[str, Callable[[Floats, int, int, int], Result]] = {
    "rolling_median": rolling_median_scores,
    "seasonal_profile": seasonal_profile_scores,
    "seasonal_diff": seasonal_diff_scores,
    "isolation_forest": isolation_forest_scores,
}

DESCRIPTIONS = {
    "rolling_median": "distance from the median of the last period, in robust standard deviations; ignores seasonality",
    "seasonal_profile": "residual from the median shape of one period, against its own recent median",
    "seasonal_diff": "difference from the same moment one period earlier; an anomaly echoes one period later",
    "isolation_forest": "scikit-learn isolation forest on the seasonal residual, its step, and its 12-point mean and spread",
}


def detect(
    name: str, x: Floats, train_n: int, period: int, seed: int = 0, quantile: float = 1.0
) -> tuple[Floats, float, NDArray[np.bool_]]:
    """Scores, threshold and flags of one detector. The threshold is the `quantile` of the clean-start scores."""
    if name not in DETECTORS:
        raise DetectorError(f"unknown method {name!r}; choose from {', '.join(DETECTORS)}")
    if not 0.5 <= quantile <= 1.0:
        raise DetectorError("quantile must be between 0.5 and 1")
    if not 3 * period <= train_n < len(x):
        raise DetectorError(f"train points must be between {3 * period} (three periods) and {len(x) - 1}")
    res = DETECTORS[name](x, train_n, period, seed)
    threshold = float(np.quantile(res.scores[res.warmup : train_n], quantile))
    flags = res.scores > threshold
    flags[: res.warmup] = False
    return res.scores, threshold, flags
