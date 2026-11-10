from __future__ import annotations

import numpy as np

from tsanom import evaluate as ev
from tsanom.gen import Event


def flags_at(n: int, *points: int) -> np.ndarray:
    f = np.zeros(n, dtype=bool)
    f[list(points)] = True
    return f


def test_segments() -> None:
    assert ev.segments(flags_at(20)) == []
    assert ev.segments(flags_at(20, 3, 4, 5)) == [(3, 6)]
    assert ev.segments(flags_at(20, 3, 5)) == [(3, 6)]  # gap of one point merges
    assert ev.segments(flags_at(20, 3, 8)) == [(3, 4), (8, 9)]  # gap of four does not
    assert ev.segments(flags_at(20, 3, 7)) == [(3, 4), (7, 8)] or ev.segments(flags_at(20, 3, 7)) == [(3, 8)]
    assert ev.segments(flags_at(20, 0, 19)) == [(0, 1), (19, 20)]


def test_hits_misses_and_false_alarms() -> None:
    events = [Event("spike", 100, 101), Event("level_shift", 200, 230), Event("dropout", 300, 310)]
    flags = flags_at(400, 101, 215, 216, 350)  # 101 is one after the spike: inside the slack
    scores = np.zeros(400)
    scores[[100, 215, 350]] = 1.0
    m = ev.evaluate(events, flags, scores, first=50)
    assert (m.events, m.detected) == (3, 2)
    assert m.by_kind["spike"] == [1, 1] and m.by_kind["level_shift"] == [1, 1] and m.by_kind["dropout"] == [1, 0]
    assert m.segments == 3 and m.false_alarms == 1  # the flag at 350 touches nothing
    assert m.recall == 2 / 3 and m.precision == 2 / 3
    assert m.delays == [1, 15]


def test_slack_boundaries() -> None:
    events = [Event("spike", 100, 101)]
    for point, found in [(97, False), (98, True), (102, True), (103, False)]:
        m = ev.evaluate(events, flags_at(200, point), np.zeros(200), first=0)
        assert (m.detected == 1) is found, point


def test_flags_before_first_are_ignored() -> None:
    m = ev.evaluate([Event("spike", 100, 101)], flags_at(200, 10, 100), np.zeros(200), first=50)
    assert m.segments == 1 and m.false_alarms == 0


def test_point_level_scores() -> None:
    events = [Event("level_shift", 100, 110)]
    perfect = np.zeros(200)
    perfect[100:110] = 5.0
    m = ev.evaluate(events, np.zeros(200, dtype=bool), perfect, first=0)
    assert m.auc == 1.0 and m.ap == 1.0
    assert m.segments == 0 and np.isnan(m.precision) and m.recall == 0.0
    none = ev.evaluate([], np.zeros(200, dtype=bool), perfect, first=0)
    assert np.isnan(none.auc) and np.isnan(none.recall)
