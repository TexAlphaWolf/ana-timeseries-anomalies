# Method

How the series are made, how the detectors work and how they are scored. Everything here is in code and tested.

## The series

`gen.generate(n, seed)` builds, for each point `t`:

```
100 + 10 sin(2πt/P) + 4 sin(4πt/P + 0.7) + 5 sin(2πt/7P + 1.1) + 0.5 t/n + noise
```

`P` is the period (144 points by default: one point every ten minutes, one cycle a day), the last two terms before the
noise are a weekly cycle and a slow trend, and the noise is AR(1) with coefficient 0.5 and standard deviation 2. The first
25% of the points are clean. In the rest, for every kind, `rate` events per 10,000 points are injected (4 by default, at
least one), never closer than 400 points to each other and never in the first period after the clean start:

| Kind | What is changed | Length |
|---|---|---|
| spike | one point moves up or down by 5 to 10 noise standard deviations | 1 |
| level_shift | the level moves by 4 to 8 standard deviations, up or down | 20 to 60 |
| variance_burst | extra noise with 3 to 5 times the standard deviation | 20 to 60 |
| dropout | the value falls to between 0 and 2 (a sensor that reports nothing) | 5 to 20 |

The generator returns the exact start and end of every event; `series.csv` has an `anomaly` column and `events.json` the
list. Sizes are in units of the noise, so difficulty does not depend on the level.

## The detectors

Each detector is fitted on the clean start and then scores every point (higher is more anomalous) **using only the points
up to that one**, as a live monitor would; a test changes the future of a series and checks that earlier scores do not move.

- **rolling_median.** `|x_t − median(x_{t−P..t−1})| / s`, where `s` is the robust standard deviation (1.4826 times the
  median absolute deviation) of the same quantity on the clean start. It does not know the series has a cycle.
- **seasonal_profile.** The median shape of one period is learnt on the clean start (the median of every phase) and
  subtracted; the residual is compared with its own median over the last period, scaled by the robust deviation of that on
  the clean start.
- **seasonal_diff.** `|x_t − x_{t−P}| / s`: the same moment one period earlier. It needs no learnt shape, but a spike
  appears twice, at the spike and one period later.
- **isolation_forest.** scikit-learn's `IsolationForest` (100 trees, fixed seed, one thread) on four features of the
  seasonal residual: its standardised value, its first difference, and its mean and standard deviation over the last 12 points.

**One threshold rule for all:** the largest score seen on the clean start (`--quantile` lowers it). By construction nothing
is flagged in the clean start, and the threshold costs no tuning against the anomalies.

## The scoring

- **Event level**, what an operator cares about. A run of flagged points (runs less than 3 points apart are one) is a
  *detection*. An injected event is **found** if a flagged point falls within 2 points of it. A run that touches no event is
  a **false alarm**. Recall is events found over events; precision is runs touching an event over runs; the number of
  false alarms is also given per 10,000 points. Delay is the first flagged point minus the first point of the event.
- **Point level**, independent of the threshold: ROC AUC and average precision of the scores against the labels of the
  points inside events. It is stricter for long events, because a detector that flags only the edge of a level shift gets
  most of its points wrong.

The benchmark pools the counts over all series (so a method is not helped by a lucky series) and averages AUC and AP per
series. Times are the median of repeated runs of fit plus scoring.

## What is checked against what

- The vectorised detectors are compared with plain loops written from the definitions above
  (`tests/test_bench.py`).
- The events, their positions and the clean start are checked on every generated series (`tests/test_gen.py`).
- The scoring is checked on hand-made flags: boundaries of the slack, merging of runs, flags before the test part
  (`tests/test_evaluate.py`).
