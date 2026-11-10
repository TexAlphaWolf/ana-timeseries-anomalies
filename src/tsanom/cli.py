"""Command line: generate, detect, evaluate, bench, render, methods."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from tsanom import __version__, bench, detectors, evaluate, gen, report


def parse_sizes(text: str) -> list[int]:
    try:
        sizes = [int(p.replace("_", "")) for p in text.split(",") if p.strip()]
    except ValueError as exc:
        raise gen.SeriesError(f"--sizes must be integers separated by commas ({exc})") from exc
    if not sizes or any(s < 1000 for s in sizes):
        raise gen.SeriesError("--sizes needs at least one size, each of 1,000 points or more")
    return sizes


def pick(method: str) -> list[str]:
    if method == "all":
        return list(detectors.DETECTORS)
    if method not in detectors.DETECTORS:
        raise detectors.DetectorError(f"unknown method {method!r}; choose from all, {', '.join(detectors.DETECTORS)}")
    return [method]


def train_points(args: argparse.Namespace, n: int) -> int:
    return int(args.train) if args.train else int(n * args.train_fraction)


def cmd_generate(args: argparse.Namespace) -> int:
    s = gen.generate(args.points, args.seed, args.period, args.train_fraction, args.rate)
    files = gen.write(s, Path(args.out))
    kinds = ", ".join(f"{sum(e.kind == k for e in s.events)} {k.replace('_', ' ')}" for k in gen.KINDS)
    print(f"written: {', '.join(str(f) for f in files)}")
    print(f"{args.points:,} points, the first {s.train_n:,} clean; {len(s.events)} anomalies injected ({kinds})")
    return 0


def cmd_detect(args: argparse.Namespace) -> int:
    x = gen.read_values(Path(args.file))
    n_train = train_points(args, len(x))
    names = pick(args.method)
    flagged_any = False
    rows: list[dict[str, Any]] = []
    for name in names:
        scores, thr, flags = detectors.detect(name, x, n_train, args.period, args.seed, args.quantile)
        segs = evaluate.segments(flags)
        flagged_any |= bool(segs)
        print(f"{name}: {len(segs)} anomalous runs, {int(flags.sum())} points above {thr:.2f}")
        for a, b in segs[: args.show]:
            peak = float(np.max(scores[a:b]))
            print(f"  points {a}-{b - 1} (score up to {peak:.1f})")
        if len(segs) > args.show:
            print(f"  ... and {len(segs) - args.show} more")
        rows.append({"method": name, "scores": scores, "flags": flags})
        if args.figure and len(names) == 1:
            Path(args.figure).parent.mkdir(parents=True, exist_ok=True)
            report.detect_figure(x, flags, n_train, Path(args.figure), title=f"{name}: {len(segs)} anomalous runs")
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        with (out / "flags.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(
                ["point", "value", *[f"{r['method']}_score" for r in rows], *[f"{r['method']}_flag" for r in rows]]
            )
            for i in range(len(x)):
                w.writerow(
                    [i, f"{x[i]:.4f}", *[f"{r['scores'][i]:.3f}" for r in rows], *[int(r["flags"][i]) for r in rows]]
                )
        print(f"written: {out / 'flags.csv'}")
    return 1 if flagged_any else 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    folder = Path(args.dir)
    events, meta = gen.read_events(folder / "events.json")
    x = gen.read_values(folder / "series.csv")
    n_train, period = int(meta["train_points"]), int(meta["period"])
    print(f"{len(x):,} points, {len(events)} injected anomalies; learning on the first {n_train:,}")
    for name in pick(args.method):
        scores, _, flags = detectors.detect(name, x, n_train, period, int(meta["seed"]), args.quantile)
        m = evaluate.evaluate(events, flags, scores, n_train)
        kinds = ", ".join(f"{k.replace('_', ' ')} {a[1]}/{a[0]}" for k, a in m.by_kind.items())
        print(
            f"{name}: found {m.detected}/{m.events} ({report.pct(m.recall)}), precision {report.pct(m.precision)} "
            f"({m.false_alarms} false alarms), AUC {m.auc:.3f}; {kinds}"
        )
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    sizes = parse_sizes(args.sizes)

    def progress(msg: str) -> None:
        print(msg, flush=True)

    result = bench.run_bench(args.series, sizes, args.repeat, args.seed, args.cmdline, progress)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "bench.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    (out / "bench.md").write_text(report.render_markdown(result), encoding="utf-8")
    report.bench_figure(result, out / "summary.png")
    print(f"written: {out / 'bench.json'}, {out / 'bench.md'}, {out / 'summary.png'}")
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    data = json.loads(Path(args.file).read_text(encoding="utf-8"))
    text = report.render_markdown(data)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"written: {args.output}")
    else:
        print(text, end="")
    if args.figure:
        report.bench_figure(data, Path(args.figure))
        print(f"written: {args.figure}")
    return 0


def cmd_methods(_: argparse.Namespace) -> int:
    for name, text in detectors.DESCRIPTIONS.items():
        print(f"{name}\n    {text}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tsanom", description="Anomaly detectors for time series, compared on injected anomalies."
    )
    p.add_argument("--version", action="version", version=f"tsanom {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def learning(sp: argparse.ArgumentParser) -> None:
        sp.add_argument(
            "--period", type=int, default=144, help="points in one daily cycle (default 144: every 10 minutes)"
        )
        sp.add_argument("--train", type=int, default=0, help="clean points at the start to learn from")
        sp.add_argument("--train-fraction", type=float, default=0.25, help="used when --train is not given")
        sp.add_argument(
            "--quantile",
            type=float,
            default=1.0,
            help="threshold: this quantile of the clean-start scores (1 = the largest)",
        )
        sp.add_argument("--method", default="all", help="all or one of: " + ", ".join(detectors.DETECTORS))
        sp.add_argument("--seed", type=int, default=0)

    g = sub.add_parser("generate", help="write a series with anomalies injected at known places")
    g.add_argument("--points", type=int, default=20_000)
    g.add_argument("--seed", type=int, default=1)
    g.add_argument("--period", type=int, default=144)
    g.add_argument("--train-fraction", type=float, default=0.25)
    g.add_argument("--rate", type=float, default=4.0, help="events of each kind per 10,000 points")
    g.add_argument("--out", default="tsanom-out")
    g.set_defaults(func=cmd_generate)

    d = sub.add_parser("detect", help="find anomalies in a CSV file with a 'value' column; exit 1 if any are found")
    d.add_argument("file")
    learning(d)
    d.add_argument("--show", type=int, default=10, help="runs listed per method")
    d.add_argument("--out", help="folder for flags.csv")
    d.add_argument("--figure", help="PNG of the series and the runs found (one method)")
    d.set_defaults(func=cmd_detect)

    e = sub.add_parser("evaluate", help="score the detectors on a folder written by 'generate'")
    e.add_argument("dir")
    e.add_argument("--method", default="all")
    e.add_argument("--quantile", type=float, default=1.0)
    e.set_defaults(func=cmd_evaluate)

    b = sub.add_parser("bench", help="quality on many generated series and time on growing ones")
    b.add_argument("--series", type=int, default=20)
    b.add_argument("--sizes", default="10000,100000,1000000")
    b.add_argument("--repeat", type=int, default=3)
    b.add_argument("--seed", type=int, default=1)
    b.add_argument("--out", default="tsanom-out")
    b.set_defaults(func=cmd_bench)

    r = sub.add_parser("render", help="rebuild the Markdown (and the figure) from bench.json")
    r.add_argument("file")
    r.add_argument("-o", "--output")
    r.add_argument("--figure")
    r.set_defaults(func=cmd_render)

    m = sub.add_parser("methods", help="list the detectors")
    m.set_defaults(func=cmd_methods)
    return p


def main(argv: list[str] | None = None) -> int:
    given = sys.argv[1:] if argv is None else argv
    args = build_parser().parse_args(given)
    args.cmdline = " ".join(given)
    try:
        return int(args.func(args))
    except (gen.SeriesError, detectors.DetectorError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
