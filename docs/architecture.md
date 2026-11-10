# Architecture

```
gen.py ──► series (values, events, train_n) ──► detectors.py ──► scores, threshold, flags ──► evaluate.py ──► Metrics
   │                                                                  │                                         │
   └─► series.csv, events.json                                        └─► flags.csv, figure                     └─► bench.py ─► bench.json ─► report.py ─► bench.md, summary.png
```

| Module | Role |
|---|---|
| `gen.py` | Makes a series with events injected at known positions; reads and writes the CSV and JSON files |
| `detectors.py` | The four detectors, the rolling helpers, and `detect()` which applies the threshold rule |
| `evaluate.py` | Turns flags into runs and compares them with the events; point-level AUC and AP |
| `bench.py` | Quality over many generated series, time over growing ones |
| `report.py` | Markdown from `bench.json` and the figures; the Markdown is a pure function of the JSON |
| `cli.py` | `generate`, `detect`, `evaluate`, `bench`, `render`, `methods`; exit codes 0, 1 (anomalies found) and 2 (error) |

Decisions:

- **A detector is a function** `(values, train_n, period, seed) -> scores`; adding one is adding an entry to `DETECTORS`
  and a description, and the tests, the CLI and the benchmark pick it up.
- **Causal scoring.** Rolling quantities look at the points before the one scored; learnt quantities (the seasonal shape,
  the scale, the forest) come from the clean start only.
- **`bench.json` is the source of truth.** `bench.md` and the figure are rebuilt from it by `tsanom render`, and a test
  checks that the committed Markdown is exactly that rebuild and that the README quotes the JSON.
- **Few dependencies:** NumPy, scikit-learn (only for the isolation forest and the two ranking metrics) and matplotlib.
  The rolling median is NumPy over a sliding window view, in chunks.
