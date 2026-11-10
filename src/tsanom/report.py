"""Markdown and figures built from a benchmark result or an evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from tsanom import __version__
from tsanom.evaluate import segments
from tsanom.gen import KINDS, Event

COLORS = {
    "rolling_median": "#9a9a9a",
    "seasonal_profile": "#0072b2",
    "seasonal_diff": "#e69f00",
    "isolation_forest": "#009e73",
}


def pct(x: float | None) -> str:
    return "n/a" if x is None or x != x else f"{100 * x:.1f}%"


def ms(v: float) -> str:
    return f"{v:,.1f}" if v >= 10 else f"{v:,.2f}"


def render_markdown(bench: dict[str, Any]) -> str:
    env, q = bench["environment"], bench["quality"]
    out = [
        "# Benchmark",
        "",
        f"Run on {env['system']}, {env['cpus']} CPUs, Python {env['python']}, NumPy {env['numpy']}, scikit-learn "
        f"{env['scikit-learn']}; seed {env['seed']}; each timing is the median of {env['repeat']} run(s); "
        f"{env['seconds']} s; command `{env['command']}`.",
        "",
        f"## 1. Quality on {q['series']} generated series of {q['points']:,} points",
        "",
        f"Each series has {q['points'] - q['test_points'] // q['series']:,} clean points to learn from and "
        f"{q['test_points'] // q['series']:,} to check, with spikes, level shifts, variance bursts and dropouts "
        "injected at known places. Counts are pooled over all series.",
        "",
        "| Method | Events found | Recall | Precision (runs) | False alarms per 10,000 points | Median delay (points) | ROC AUC | Average precision |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, m in q["methods"].items():
        delay = "n/a" if m["median_delay"] is None else f"{m['median_delay']:g}"
        out.append(
            f"| {name} | {m['detected']} of {m['events']} | {pct(m['recall'])} | {pct(m['precision'])} "
            f"({m['segments'] - m['false_alarms']} of {m['segments']}) | {m['false_alarms_per_10k']:.2f} | {delay} "
            f"| {m['auc']:.3f} | {m['ap']:.3f} |"
        )
    out += [
        "",
        "## 2. Recall by kind of anomaly",
        "",
        "| Method | " + " | ".join(KINDS) + " |",
        "|---|" + "---|" * len(KINDS),
    ]
    for name, m in q["methods"].items():
        cells = [
            f"{pct(m['by_kind'][k]['detected'] / m['by_kind'][k]['events'])} ({m['by_kind'][k]['detected']} of {m['by_kind'][k]['events']})"
            for k in KINDS
        ]
        out.append(f"| {name} | " + " | ".join(cells) + " |")
    names = list(q["methods"])
    out += [
        "",
        "## 3. Time to score a series (ms)",
        "",
        "| Points | " + " | ".join(names) + " |",
        "|---|" + "---|" * len(names),
    ]
    for row in bench["timing"]:
        out.append(f"| {row['points']:,} | " + " | ".join(ms(row["ms"][n]) for n in names) + " |")
    return "\n".join(out) + "\n"


def bench_figure(bench: dict[str, Any], path: Path) -> None:
    q = bench["quality"]
    names = list(q["methods"])
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    width = 0.2
    x = np.arange(len(KINDS))
    for i, name in enumerate(names):
        vals = [
            100 * q["methods"][name]["by_kind"][k]["detected"] / q["methods"][name]["by_kind"][k]["events"]
            for k in KINDS
        ]
        a.bar(x + (i - 1.5) * width, vals, width, label=name, color=COLORS[name])
    a.set_xticks(x, [k.replace("_", " ") for k in KINDS])
    a.set_ylabel("events found (%)")
    a.set_ylim(0, 125)
    a.set_yticks([0, 20, 40, 60, 80, 100])
    a.set_title(f"Recall by kind, {q['series']} series of {q['points']:,} points")
    a.legend(loc="upper center", ncol=4, fontsize=8, frameon=False)
    sizes = [r["points"] for r in bench["timing"]]
    for name in names:
        b.plot(sizes, [r["ms"][name] for r in bench["timing"]], marker="o", label=name, color=COLORS[name])
    b.set_xscale("log")
    b.set_yscale("log")
    b.set_xticks(sizes, [f"{v:,}" for v in sizes])
    b.minorticks_off()
    b.set_xlabel("points in the series")
    b.set_ylabel("time to fit and score (ms)")
    b.set_title("Time against length")
    fig.suptitle(f"tsanom {__version__}", fontsize=10)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def detect_figure(
    values: NDArray[np.float64],
    flags: NDArray[np.bool_],
    train_n: int,
    path: Path,
    events: list[Event] | None = None,
    title: str = "",
    window: tuple[int, int] | None = None,
) -> None:
    lo, hi = window or (0, len(values))
    fig, ax = plt.subplots(figsize=(12, 3.8), constrained_layout=True)
    ax.plot(np.arange(lo, hi), values[lo:hi], lw=0.7, color="#444444")
    ax.axvspan(
        lo, min(train_n, hi), color="#dddddd", alpha=0.5, label="clean start (learning)"
    ) if train_n > lo else None
    for k, e in enumerate(events or []):
        if e.end > lo and e.start < hi:
            ax.axvspan(e.start - 0.5, e.end + 0.5, color="#d55e00", alpha=0.25, label="injected" if k == 0 else None)
    for a, b in segments(flags):
        if b > lo and a < hi:
            ax.plot([a, b], [float(np.max(values[lo:hi])) * 1.02] * 2, color="#0072b2", lw=3, solid_capstyle="butt")
    ax.set_title(title)
    ax.set_xlabel("point")
    ax.legend(loc="lower left", fontsize=8)
    fig.savefig(path, dpi=130)
    plt.close(fig)
