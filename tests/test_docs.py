"""The documentation must say what the measurements say."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tsanom import cli

ROOT = Path(__file__).resolve().parent.parent
BENCH = json.loads((ROOT / "docs" / "bench.json").read_text(encoding="utf-8"))
README = (ROOT / "README.md").read_text(encoding="utf-8")
ANALYSIS = (ROOT / "docs" / "analysis.md").read_text(encoding="utf-8")
Q = BENCH["quality"]["methods"]
T = {r["points"]: r["ms"] for r in BENCH["timing"]}
KINDS = ("spike", "level_shift", "variance_burst", "dropout")


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def ms(v: float) -> str:
    return f"{v:,.1f}" if v >= 10 else f"{v:,.2f}"


def test_the_benchmark_is_what_the_text_says_it_is() -> None:
    env = BENCH["environment"]
    assert env["tsanom"] == "1.0.0" and env["seed"] == 1 and env["repeat"] == 5 and env["cpus"] == 12
    assert BENCH["quality"]["series"] == 20 and BENCH["quality"]["points"] == 20_000
    assert BENCH["quality"]["test_points"] == 300_000
    assert all(m["events"] == 480 for m in Q.values())
    assert all(v["events"] == 120 for m in Q.values() for v in m["by_kind"].values())
    assert sorted(T) == [10_000, 100_000, 1_000_000]


def test_bench_markdown_is_the_render_of_the_json(tmp_path: Path) -> None:
    out = tmp_path / "again.md"
    assert cli.main(["render", str(ROOT / "docs" / "bench.json"), "-o", str(out)]) == 0
    assert out.read_text(encoding="utf-8") == (ROOT / "docs" / "bench.md").read_text(encoding="utf-8")


def test_figure_exists() -> None:
    assert (ROOT / "docs" / "summary.png").stat().st_size > 10_000


@pytest.mark.parametrize("text", [README, ANALYSIS], ids=["README", "analysis"])
def test_no_scaffold_left(text: str) -> None:
    assert not re.search(r"work in progress|TODO|TBD|lorem|measurements pending|provisional", text, re.I)


@pytest.mark.parametrize("name", ["README.md", "docs/analysis.md"])
def test_local_links_resolve(name: str) -> None:
    path = ROOT / name
    for target in re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", path.read_text(encoding="utf-8")):
        if target.startswith(("http://", "https://")):
            continue
        assert (path.parent / target).exists(), f"{name}: broken link {target}"


def test_readme_headline_numbers() -> None:
    sp, rm, sd, iso = Q["seasonal_profile"], Q["rolling_median"], Q["seasonal_diff"], Q["isolation_forest"]
    assert f"{sp['detected']} of the {sp['events']} events ({pct(sp['recall'])})" in README
    assert f"precision of {pct(sp['precision'])}" in README
    assert f"{sp['false_alarms_per_10k']:.2f} false alarms per" in README
    assert f"({sp['by_kind']['spike']['detected']} of 120)" in README
    assert f"finds {pct(rm['recall'])} of the events" in README
    assert f"{rm['by_kind']['spike']['detected']} of 120 spikes" in README
    assert f"({pct(rm['by_kind']['spike']['detected'] / 120)})" in README
    assert f"{rm['by_kind']['level_shift']['detected']} of 120 level shifts" in README
    assert f"{sd['false_alarms_per_10k']:.2f} per 10,000" in README
    assert f"{sd['false_alarms_per_10k'] / sp['false_alarms_per_10k']:.1f} times" in README
    assert f"precision of {pct(sd['precision'])}" in README and f"finds {pct(sd['recall'])} of the events" in README
    assert f"({iso['auc']:.3f} against {sp['auc']:.3f})" in README
    assert f"finds {pct(iso['recall'])} of the events against {pct(sp['recall'])}" in README
    assert iso["auc"] == max(m["auc"] for m in Q.values())
    assert sp["recall"] == max(m["recall"] for m in Q.values())
    big = T[1_000_000]
    for name in ("seasonal_profile", "rolling_median", "isolation_forest", "seasonal_diff"):
        assert ms(big[name]) in README, name
    assert round(1_000_000 * 10 / 60 / 24 / 365.25) == 19


def test_analysis_tables_match_the_json() -> None:
    for name, m in Q.items():
        cells = [f"{m['by_kind'][k]['detected']} of 120" for k in KINDS]
        row = f"| {name} | " + " | ".join(cells) + f" | {m['detected']} of 480 ({pct(m['recall'])}) |"
        assert row in ANALYSIS, name
        far = f"| {name} | {m['segments']:,} | {m['segments'] - m['false_alarms']:,} | {m['false_alarms']:,} | "
        assert far + f"{m['false_alarms_per_10k']:.2f} | {pct(m['precision'])} |" in ANALYSIS, name
        assert f"| {name} | {m['auc']:.3f} | {m['ap']:.3f} | {pct(m['recall'])} |" in ANALYSIS, name
    for n, row in T.items():
        cells = " | ".join(
            f"{ms(row[x])} ms" for x in ("rolling_median", "seasonal_profile", "seasonal_diff", "isolation_forest")
        )
        assert f"| {n:,} | {cells} |" in ANALYSIS


def test_analysis_derived_claims() -> None:
    small, big = T[10_000], T[1_000_000]
    for name, want in {
        "rolling_median": 92.8,
        "seasonal_profile": 84.3,
        "seasonal_diff": 71.1,
        "isolation_forest": 22.8,
    }.items():
        assert round(big[name] / small[name], 1) == want, name
    assert f"{big['isolation_forest'] / big['seasonal_diff']:.0f} times the seasonal difference" in ANALYSIS
    assert all(m["median_delay"] == 0 for m in Q.values())
    assert min(Q["rolling_median"]["by_kind"][k]["detected"] for k in ("dropout",)) == 120


def test_readme_output_matches_a_real_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["generate", "--points", "20000", "--seed", "1", "--out", str(tmp_path)]) == 0
    capsys.readouterr()
    assert cli.main(["evaluate", str(tmp_path)]) == 0
    got = capsys.readouterr().out.strip().splitlines()
    block = re.search(r"```text\n(.*?)```", README, re.S)
    assert block is not None
    want = block.group(1).strip().splitlines()
    assert len(got) == len(want) == 5
    for a, b in zip(got, want, strict=True):
        if a.startswith("isolation_forest"):  # scikit-learn's trees can differ between versions: check the shape only
            assert b.startswith("isolation_forest: found ")
        else:
            assert a == b
