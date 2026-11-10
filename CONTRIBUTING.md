# Contributing

```bash
pip install -e ".[dev]"
ruff check . && ruff format --check . && mypy && pytest --cov
```

No service is needed. `TSANOM_TEST_POINTS` and `TSANOM_TEST_SEED` run the same tests on other generated series (CI also
uses 100,000 points with seed 11).

Rules that keep the project honest:

- **A new detector needs three things:** a function in `detectors.py` with the signature
  `(values, train_n, period, seed) -> Result`, an entry in `DETECTORS` and `DESCRIPTIONS`, and (in `tests/test_bench.py`) a
  point-by-point reference written from its definition. It must score point `t` using only points up to `t`:
  `tests/test_detectors.py` checks that for every detector.
- **A new kind of anomaly** goes in `KINDS` and `generate` (`gen.py`), with a line in `docs/method.md`, and a case in the
  tests of the generator.
- **Numbers in the README come from `docs/bench.json`**, and `tests/test_docs.py` checks them. After a new measurement
  (`tools/run_full.ps1`), copy `bench.json`, `bench.md` and `summary.png` into `docs/` and update the text if a number moved.
