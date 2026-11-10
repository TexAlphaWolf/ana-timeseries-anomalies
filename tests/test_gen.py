from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tsanom import gen


def test_same_seed_same_series() -> None:
    a, b = gen.generate(8000, 5), gen.generate(8000, 5)
    assert np.array_equal(a.values, b.values) and a.events == b.events
    assert not np.array_equal(a.values, gen.generate(8000, 6).values)


def test_events_are_where_the_rules_put_them(series: gen.Series) -> None:
    n = len(series.values)
    assert series.train_n == n // 4
    assert {e.kind for e in series.events} == set(gen.KINDS)
    for k in gen.KINDS:  # the same number of each kind
        assert sum(e.kind == k for e in series.events) == sum(e.kind == "spike" for e in series.events)
    prev_end = -(10**9)
    for e in series.events:
        assert e.start >= series.train_n + series.period  # the clean start stays clean
        assert e.end <= n - series.period
        assert e.start - prev_end >= gen.MIN_GAP  # events do not touch
        prev_end = e.end
        assert {"spike": 1}.get(e.kind, e.length()) == e.length()
    assert all(5 <= e.length() <= 20 for e in series.events if e.kind == "dropout")
    assert all(20 <= e.length() <= 60 for e in series.events if e.kind in ("level_shift", "variance_burst"))


def test_anomalies_really_change_the_values() -> None:
    s = gen.generate(20000, 3)
    clean = np.random.default_rng(3)
    base = gen.base(20000, 144, clean)  # the same stream of numbers as the generator used first
    diff = s.values - base
    outside = ~s.labels()
    assert np.allclose(diff[outside], 0.0)
    for e in s.events:
        assert np.abs(diff[e.start : e.end]).max() > 0
    drops = [e for e in s.events if e.kind == "dropout"]
    assert all(s.values[e.start : e.end].max() <= 2.0 for e in drops)


def test_train_part_is_clean(series: gen.Series) -> None:
    assert not series.labels()[: series.train_n].any()


@pytest.mark.parametrize(
    ("kwargs", "text"),
    [
        ({"n": 20000, "period": 4}, "period"),
        ({"n": 20000, "train_fraction": 0.9}, "train_fraction"),
        ({"n": 1000}, "three periods"),
        ({"n": 20000, "rate": 0}, "rate"),
        ({"n": 20000, "rate": 400}, "do not fit"),
    ],
)
def test_bad_requests(kwargs: dict[str, float], text: str) -> None:
    with pytest.raises(gen.SeriesError, match=text):
        gen.generate(seed=1, **kwargs)  # type: ignore[arg-type]


def test_write_and_read_back(tmp_path: Path) -> None:
    s = gen.generate(6000, 2, rate=2)
    gen.write(s, tmp_path)
    assert np.allclose(gen.read_values(tmp_path / "series.csv"), s.values, atol=5e-5)
    events, meta = gen.read_events(tmp_path / "events.json")
    assert events == list(s.events) and meta["train_points"] == s.train_n and meta["period"] == 144
    lines = (tmp_path / "series.csv").read_text().splitlines()
    assert lines[0] == "ts,value,anomaly" and lines[1].startswith("2026-01-01T00:00:00,")
    assert sum(int(x.rsplit(",", 1)[1]) for x in lines[1:]) == int(s.labels().sum())
    assert json.loads((tmp_path / "events.json").read_text())["seed"] == 2


def test_read_values_formats(tmp_path: Path) -> None:
    (tmp_path / "plain.csv").write_text("1.5\n2.5\n3.5\n")
    assert gen.read_values(tmp_path / "plain.csv").tolist() == [1.5, 2.5, 3.5]
    (tmp_path / "single.csv").write_text("temp\n1\n2\n")  # one column with a header
    assert gen.read_values(tmp_path / "single.csv").tolist() == [1.0, 2.0]
    (tmp_path / "head.csv").write_text("a,b\n1,2\n")
    with pytest.raises(gen.SeriesError, match="no 'value' column"):
        gen.read_values(tmp_path / "head.csv")
    (tmp_path / "multi.csv").write_text("ts,Value\na,1\nb,2\n")
    assert gen.read_values(tmp_path / "multi.csv").tolist() == [1.0, 2.0]


@pytest.mark.parametrize(
    ("content", "text"),
    [("", "empty"), ("value\n1\nx\n", "not a number"), ("value\n1\n\n", "not a number"), ("value\n1\nnan\n", "nan")],
)
def test_bad_files(tmp_path: Path, content: str, text: str) -> None:
    f = tmp_path / "f.csv"
    f.write_text(content)
    with pytest.raises(gen.SeriesError, match=text):
        gen.read_values(f)
    with pytest.raises(gen.SeriesError, match="cannot read"):
        gen.read_values(tmp_path / "missing.csv")


def test_files_use_plain_line_ends(tmp_path: Path) -> None:
    """A shell script comparing the header line must not meet a carriage return, on any system."""
    gen.write(gen.generate(6000, 2, rate=2), tmp_path)
    raw = (tmp_path / "series.csv").read_bytes()
    assert b"\r" not in raw and raw.startswith(b"ts,value,anomaly\n")
