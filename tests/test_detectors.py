from __future__ import annotations

import numpy as np
import pytest

from tsanom import detectors as d
from tsanom import gen


def test_rolling_median_matches_brute_force() -> None:
    x = np.random.default_rng(0).normal(size=300)
    got = d.rolling_median(x, 25)
    assert np.isnan(got[:25]).all()
    for t in (25, 26, 100, 299):
        assert got[t] == pytest.approx(np.median(x[t - 25 : t]))  # excludes the point itself
    assert np.isnan(d.rolling_median(x[:20], 25)).all()


def test_rolling_median_across_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    x = np.random.default_rng(1).normal(size=500)
    whole = d.rolling_median(x, 10)
    monkeypatch.setattr(d, "CHUNK", 37)
    assert np.array_equal(d.rolling_median(x, 10), whole, equal_nan=True)


def test_rolling_mean_std_matches_brute_force() -> None:
    x = np.random.default_rng(2).normal(5, 2, size=200)
    mean, sd = d.rolling_mean_std(x, 12)
    for t in (0, 5, 11, 12, 150, 199):
        w = x[max(t - 11, 0) : t + 1]
        assert mean[t] == pytest.approx(w.mean()) and sd[t] == pytest.approx(w.std(), abs=1e-6)


def test_profile_and_residual() -> None:
    period = 10
    shape = np.arange(period, dtype=float)
    x = np.tile(shape, 6) + np.random.default_rng(3).normal(0, 0.01, 60)
    prof = d.profile_of(x[:40], period)
    assert np.allclose(prof, shape, atol=0.05)
    assert np.abs(d.residual(x, prof)).max() < 0.1
    with pytest.raises(d.DetectorError, match="three periods"):
        d.profile_of(x[:20], period)


def test_robust_sd() -> None:
    v = np.random.default_rng(4).normal(0, 3, 20000)
    assert d.robust_sd(v) == pytest.approx(3.0, rel=0.05)
    v[:50] = 1e6  # outliers do not move it
    assert d.robust_sd(v) == pytest.approx(3.0, rel=0.05)
    with pytest.raises(d.DetectorError, match="constant"):
        d.robust_sd(np.ones(10))
    with pytest.raises(d.DetectorError, match="no clean"):
        d.robust_sd(np.array([np.nan]))


@pytest.mark.parametrize("name", list(d.DETECTORS))
def test_nothing_above_threshold_in_the_clean_start(series: gen.Series, name: str) -> None:
    scores, thr, flags = d.detect(name, series.values, series.train_n, series.period, 1)
    assert not flags[: series.train_n].any()  # the threshold is the largest clean score
    assert thr > 0 and len(scores) == len(series.values)


@pytest.mark.parametrize("name", list(d.DETECTORS))
def test_scores_use_only_the_past(series: gen.Series, name: str) -> None:
    """Changing points after t must not change the score at t (a live monitor cannot see the future)."""
    x = series.values.copy()
    cut = series.train_n + 3000
    before = d.DETECTORS[name](x, series.train_n, series.period, 1).scores
    y = x.copy()
    y[cut:] += 500.0
    after = d.DETECTORS[name](y, series.train_n, series.period, 1).scores
    assert np.allclose(before[:cut], after[:cut])
    assert not np.allclose(before[cut:], after[cut:])


@pytest.mark.parametrize("name", list(d.DETECTORS))
def test_a_huge_spike_scores_highest(series: gen.Series, name: str) -> None:
    x = series.values.copy()
    spot = series.train_n + 5
    x[spot] += 1000.0
    scores, _, flags = d.detect(name, x, series.train_n, series.period, 1)
    assert flags[spot]
    if name != "isolation_forest":  # its score saturates: a huge point is no more isolated than a big one
        # (seasonal_diff echoes the spike a period later, with a different noise, so allow a little less)
        assert scores[spot] >= 0.9 * scores[series.train_n :].max()


def test_seasonal_diff_echoes_one_period_later(series: gen.Series) -> None:
    x = series.values.copy()
    spot = series.train_n + 5
    x[spot] += 1000.0
    _, _, flags = d.detect("seasonal_diff", x, series.train_n, series.period, 1)
    assert flags[spot] and flags[spot + series.period]
    _, _, flags2 = d.detect("seasonal_profile", x, series.train_n, series.period, 1)
    assert not flags2[spot + series.period]


def test_quantile_lowers_the_threshold(series: gen.Series) -> None:
    _, t1, f1 = d.detect("seasonal_profile", series.values, series.train_n, series.period, 1, 1.0)
    _, t2, f2 = d.detect("seasonal_profile", series.values, series.train_n, series.period, 1, 0.99)
    assert t2 < t1 and f2.sum() > f1.sum()


def test_isolation_forest_is_deterministic(series: gen.Series) -> None:
    a = d.detect("isolation_forest", series.values, series.train_n, series.period, 4)[0]
    b = d.detect("isolation_forest", series.values, series.train_n, series.period, 4)[0]
    assert np.array_equal(a, b)


def test_bad_inputs(series: gen.Series) -> None:
    x, n, p = series.values, series.train_n, series.period
    with pytest.raises(d.DetectorError, match="unknown method"):
        d.detect("magic", x, n, p)
    with pytest.raises(d.DetectorError, match="quantile"):
        d.detect("seasonal_diff", x, n, p, 0, 0.2)
    with pytest.raises(d.DetectorError, match="train points"):
        d.detect("seasonal_diff", x, p, p)
    with pytest.raises(d.DetectorError, match="train points"):
        d.detect("seasonal_diff", x, len(x), p)
    with pytest.raises(d.DetectorError, match="constant"):
        d.detect("seasonal_diff", np.ones(len(x)), n, p)
