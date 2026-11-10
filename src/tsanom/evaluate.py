"""Scoring a detector against the events that were injected.

Event level (what an operator cares about): an event counts as found if a flagged point falls within `slack` points of it;
a run of flagged points (runs closer than `gap` points are one) that touches no event is a false alarm.
Point level, independent of the threshold: ROC AUC and average precision of the scores against the point labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score, roc_auc_score

from tsanom.gen import KINDS, Event

SLACK = 2
GAP = 3


def segments(flags: NDArray[np.bool_], gap: int = GAP) -> list[tuple[int, int]]:
    """Runs of flagged points as (start, end exclusive); runs separated by fewer than `gap` unflagged points merge."""
    idx = np.flatnonzero(flags)
    if len(idx) == 0:
        return []
    cut = np.flatnonzero(np.diff(idx) > gap)
    starts = np.concatenate(([idx[0]], idx[cut + 1]))
    ends = np.concatenate((idx[cut] + 1, [idx[-1] + 1]))
    return [(int(a), int(b)) for a, b in zip(starts, ends, strict=True)]


@dataclass
class Metrics:
    events: int = 0
    detected: int = 0
    by_kind: dict[str, list[int]] = field(default_factory=lambda: {k: [0, 0] for k in KINDS})  # kind -> [events, found]
    segments: int = 0
    false_alarms: int = 0
    delays: list[int] = field(default_factory=list)
    auc: float = float("nan")
    ap: float = float("nan")

    @property
    def recall(self) -> float:
        return self.detected / self.events if self.events else float("nan")

    @property
    def precision(self) -> float:
        return (self.segments - self.false_alarms) / self.segments if self.segments else float("nan")


def evaluate(
    events: list[Event], flags: NDArray[np.bool_], scores: NDArray[np.float64], first: int, slack: int = SLACK
) -> Metrics:
    """Metrics over the points from `first` on (the test part)."""
    n = len(flags)
    f = flags.copy()
    f[:first] = False
    m = Metrics()
    flagged = np.flatnonzero(f)
    for e in events:
        m.events += 1
        m.by_kind[e.kind][0] += 1
        lo, hi = max(e.start - slack, 0), min(e.end + slack, n)
        hit = flagged[(flagged >= lo) & (flagged < hi)]
        if len(hit):
            m.detected += 1
            m.by_kind[e.kind][1] += 1
            m.delays.append(max(int(hit[0]) - e.start, 0))
    segs = segments(f)
    m.segments = len(segs)
    for a, b in segs:
        if not any(a < e.end + slack and b > e.start - slack for e in events):
            m.false_alarms += 1
    labels = np.zeros(n, dtype=bool)
    for e in events:
        labels[e.start : e.end] = True
    y, s = labels[first:], scores[first:]
    if 0 < int(y.sum()) < len(y):
        m.auc = float(roc_auc_score(y, s))
        m.ap = float(average_precision_score(y, s))
    return m
