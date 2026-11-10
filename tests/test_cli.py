from __future__ import annotations

import json
from pathlib import Path

import pytest

from tsanom import cli, gen


@pytest.fixture(scope="module")
def folder(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("demo")
    assert cli.main(["generate", "--points", "20000", "--seed", "2", "--out", str(out)]) == 0
    return out


def test_generate_output(folder: Path, capsys: pytest.CaptureFixture[str]) -> None:
    capsys.readouterr()
    assert cli.main(["generate", "--points", "8000", "--rate", "2", "--out", str(folder / "b")]) == 0
    text = capsys.readouterr().out
    assert "8,000 points, the first 2,000 clean" in text and "anomalies injected" in text


def test_evaluate_prints_every_method(folder: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["evaluate", str(folder)]) == 0
    out = capsys.readouterr().out
    for name in ("rolling_median", "seasonal_profile", "seasonal_diff", "isolation_forest"):
        assert f"{name}: found" in out
    assert cli.main(["evaluate", str(folder), "--method", "seasonal_profile"]) == 0
    assert capsys.readouterr().out.count("found") == 1


def test_detect_exit_codes_and_files(folder: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fig, out = tmp_path / "new" / "dir" / "f.png", tmp_path / "o"  # folders that do not exist yet
    rc = cli.main(
        [
            "detect",
            str(folder / "series.csv"),
            "--method",
            "seasonal_profile",
            "--out",
            str(out),
            "--figure",
            str(fig),
            "--show",
            "2",
        ]
    )
    text = capsys.readouterr().out
    assert rc == 1 and "anomalous runs" in text and "... and" in text
    assert fig.stat().st_size > 5000
    head = (out / "flags.csv").read_text().splitlines()[0]
    assert head == "point,value,seasonal_profile_score,seasonal_profile_flag"
    assert b"\r" not in (out / "flags.csv").read_bytes()
    clean = tmp_path / "clean.csv"
    s = gen.generate(8000, 4)
    clean.write_text("value\n" + "\n".join(f"{v:.4f}" for v in s.values[: s.train_n]) + "\n")
    assert cli.main(["detect", str(clean), "--train", "1000", "--method", "seasonal_profile"]) == 0


def test_detect_all_methods(folder: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["detect", str(folder / "series.csv")]) == 1
    out = capsys.readouterr().out
    assert out.count("anomalous runs") == 4


def test_bench_and_render(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "b"
    assert cli.main(["bench", "--series", "2", "--sizes", "5000,8000", "--repeat", "1", "--out", str(out)]) == 0
    data = json.loads((out / "bench.json").read_text())
    assert data["environment"]["command"].startswith("bench --series 2")
    assert [r["points"] for r in data["timing"]] == [5000, 8000]
    assert (out / "summary.png").stat().st_size > 5000
    capsys.readouterr()
    md = tmp_path / "again.md"
    assert cli.main(["render", str(out / "bench.json"), "-o", str(md), "--figure", str(tmp_path / "x.png")]) == 0
    assert md.read_text() == (out / "bench.md").read_text()
    capsys.readouterr()
    assert cli.main(["render", str(out / "bench.json")]) == 0
    assert capsys.readouterr().out.startswith("# Benchmark")


def test_methods_lists_all(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["methods"]) == 0
    assert capsys.readouterr().out.count("\n    ") == 4


@pytest.mark.parametrize(
    "argv",
    [
        ["generate", "--points", "100"],
        ["detect", "no-such-file.csv"],
        ["bench", "--sizes", "abc"],
        ["bench", "--sizes", "10"],
        ["evaluate", "no-such-dir"],
    ],
)
def test_errors_exit_2(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(argv) == 2
    assert capsys.readouterr().err.startswith("error:")


def test_unknown_method_and_short_training(folder: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["detect", str(folder / "series.csv"), "--method", "magic"]) == 2
    assert "unknown method" in capsys.readouterr().err
    assert cli.main(["detect", str(folder / "series.csv"), "--train", "100"]) == 2
    assert "train points" in capsys.readouterr().err


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as e:
        cli.main(["--version"])
    assert e.value.code == 0 and "tsanom 1.0.0" in capsys.readouterr().out
