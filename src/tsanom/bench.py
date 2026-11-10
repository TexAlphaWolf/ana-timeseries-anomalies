"""The comparison: quality over many generated series, and time over growing ones."""

from __future__ import annotations

import os
import platform
import statistics
import time
from collections.abc import Callable
from importlib import metadata

import numpy as np

from tsanom import __version__, detectors, evaluate, gen

QUALITY_POINTS = 20_000
TIMING_RATE = 1.0  # events per kind per 10,000 test points: keeps a million-point series placeable


def environment(command: str, **extra: object) -> dict[str, object]:
    return {
        "tsanom": __version__,
        "python": platform.python_version(),
        "numpy": metadata.version("numpy"),
        "scikit-learn": metadata.version("scikit-learn"),
        "system": f"{platform.system()} {platform.machine()}",
        "cpus": os.cpu_count(),
        "command": command,
        **extra,
    }


def quality(series_count: int, points: int, seed: int, progress: Callable[[str], None]) -> dict[str, object]:
    pooled: dict[str, evaluate.Metrics] = {m: evaluate.Metrics() for m in detectors.DETECTORS}
    aucs: dict[str, list[float]] = {m: [] for m in detectors.DETECTORS}
    aps: dict[str, list[float]] = {m: [] for m in detectors.DETECTORS}
    test_points = 0
    for i in range(series_count):
        progress(f"quality: series {i + 1} of {series_count}")
        s = gen.generate(points, seed + i)
        test_points += points - s.train_n
        for name in detectors.DETECTORS:
            scores, _, flags = detectors.detect(name, s.values, s.train_n, s.period, seed + i)
            m = evaluate.evaluate(list(s.events), flags, scores, s.train_n)
            p = pooled[name]
            p.events += m.events
            p.detected += m.detected
            p.segments += m.segments
            p.false_alarms += m.false_alarms
            p.delays += m.delays
            for k, (a, b) in m.by_kind.items():
                p.by_kind[k][0] += a
                p.by_kind[k][1] += b
            aucs[name].append(m.auc)
            aps[name].append(m.ap)
    methods: dict[str, object] = {}
    for name, p in pooled.items():
        methods[name] = {
            "events": p.events,
            "detected": p.detected,
            "recall": p.recall,
            "by_kind": {k: {"events": a, "detected": b} for k, (a, b) in p.by_kind.items()},
            "segments": p.segments,
            "false_alarms": p.false_alarms,
            "precision": p.precision,
            "false_alarms_per_10k": p.false_alarms / test_points * 10_000,
            "median_delay": statistics.median(p.delays) if p.delays else None,
            "auc": float(np.mean(aucs[name])),
            "ap": float(np.mean(aps[name])),
        }
    return {"series": series_count, "points": points, "test_points": test_points, "methods": methods}


def timing(sizes: list[int], repeat: int, seed: int, progress: Callable[[str], None]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for n in sizes:
        progress(f"timing: {n:,} points")
        s = gen.generate(n, seed, rate=TIMING_RATE)
        row: dict[str, float] = {}
        for name in detectors.DETECTORS:
            runs = []
            for _ in range(repeat):
                t0 = time.perf_counter()
                detectors.detect(name, s.values, s.train_n, s.period, seed)
                runs.append((time.perf_counter() - t0) * 1000)
            row[name] = statistics.median(runs)
        out.append({"points": n, "events": len(s.events), "ms": row})
    return out


def run_bench(
    series: int,
    sizes: list[int],
    repeat: int,
    seed: int,
    command: str,
    progress: Callable[[str], None] = lambda _m: None,
) -> dict[str, object]:
    t0 = time.perf_counter()
    q = quality(series, QUALITY_POINTS, seed, progress)
    t = timing(sizes, repeat, seed, progress)
    env = environment(command, seed=seed, repeat=repeat, seconds=round(time.perf_counter() - t0, 1))
    return {"environment": env, "quality": q, "timing": t}
