from __future__ import annotations

import numpy as np

from tsanom import bench, detectors, gen
from tsanom.detectors import MAD_TO_SD


def mad_sd(v: np.ndarray) -> float:
    return float(MAD_TO_SD * np.median(np.abs(v - np.median(v))))


def test_detectors_agree_with_a_point_by_point_reference() -> None:
    """The vectorised detectors against plain loops written from the definitions in the docstring."""
    s = gen.generate(6000, 9, rate=2)
    x, n, p = s.values, s.train_n, s.period
    ref_rm = np.zeros(len(x))
    dev = np.full(len(x), np.nan)
    for t in range(p, len(x)):
        dev[t] = x[t] - np.median(x[t - p : t])
    sd = mad_sd(dev[p:n])
    ref_rm[p:] = np.abs(dev[p:]) / sd
    assert np.allclose(detectors.rolling_median_scores(x, n, p, 0).scores, ref_rm)

    prof = np.array([np.median(x[:n][k::p]) for k in range(p)])
    r = np.array([x[t] - prof[t % p] for t in range(len(x))])
    dev2 = np.full(len(x), np.nan)
    for t in range(p, len(x)):
        dev2[t] = r[t] - np.median(r[t - p : t])
    sd2 = mad_sd(dev2[p:n])
    ref_sp = np.zeros(len(x))
    ref_sp[p:] = np.abs(dev2[p:]) / sd2
    assert np.allclose(detectors.seasonal_profile_scores(x, n, p, 0).scores, ref_sp)

    d = np.zeros(len(x))
    for t in range(p, len(x)):
        d[t] = x[t] - x[t - p]
    ref_sd = np.abs(d) / mad_sd(d[p:n])
    assert np.allclose(detectors.seasonal_diff_scores(x, n, p, 0).scores, ref_sd)


def test_quality_pools_counts_and_matches_the_events() -> None:
    q = bench.quality(3, 20_000, 1, lambda _m: None)
    assert q["series"] == 3 and q["points"] == 20_000
    for name, m in q["methods"].items():  # type: ignore[attr-defined]
        assert m["events"] == 72  # 3 series x 4 kinds x 6 events
        assert sum(v["events"] for v in m["by_kind"].values()) == 72
        assert sum(v["detected"] for v in m["by_kind"].values()) == m["detected"]
        assert m["segments"] - m["false_alarms"] <= m["segments"]
        assert 0 <= m["recall"] <= 1 and 0.5 < m["auc"] <= 1, name


def test_the_seasonal_methods_beat_the_one_that_ignores_seasonality() -> None:
    m = bench.quality(4, 20_000, 1, lambda _m: None)["methods"]  # type: ignore[index]
    assert m["seasonal_profile"]["recall"] > m["rolling_median"]["recall"]
    assert m["seasonal_profile"]["auc"] > m["rolling_median"]["auc"]
    assert m["seasonal_diff"]["false_alarms_per_10k"] > m["seasonal_profile"]["false_alarms_per_10k"]


def test_timing_rows() -> None:
    rows = bench.timing([5000, 9000], 1, 1, lambda _m: None)
    assert [r["points"] for r in rows] == [5000, 9000]
    assert all(set(r["ms"]) == set(detectors.DETECTORS) and min(r["ms"].values()) > 0 for r in rows)  # type: ignore[attr-defined]
