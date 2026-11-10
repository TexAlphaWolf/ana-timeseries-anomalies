# Analysis of the benchmark

Source: [`bench.json`](bench.json), rendered in [`bench.md`](bench.md). One run in Docker on Linux x86_64 with 12 CPUs,
Python 3.14.10, NumPy 2.5.4 and scikit-learn 1.9.1. Quality: 20 generated series of 20,000 points (5,000 clean, 15,000
checked), 480 injected events (120 of each kind), seed 1. Time: one series each of 10,000, 100,000 and 1,000,000 points,
median of 5 runs. All four detectors use the same threshold: the largest score on the clean start.

## 1. Which method finds what

| Method | Spike | Level shift | Variance burst | Dropout | Found |
|---|---|---|---|---|---|
| rolling_median | 23 of 120 | 44 of 120 | 87 of 120 | 120 of 120 | 274 of 480 (57.1%) |
| seasonal_profile | 116 of 120 | 120 of 120 | 120 of 120 | 120 of 120 | 476 of 480 (99.2%) |
| seasonal_diff | 65 of 120 | 92 of 120 | 115 of 120 | 120 of 120 | 392 of 480 (81.7%) |
| isolation_forest | 104 of 120 | 99 of 120 | 119 of 120 | 115 of 120 | 437 of 480 (91.0%) |

- **Dropouts are easy** for everyone but the forest (the value falls by about 100 against a noise of 2): 100% for three
  methods, 95.8% for the forest.
- **The cycle is the difficulty.** The daily cycle moves the series by about 14 units on either side of its level, much
  more than the 10 to 20 units of a spike. A method that does not model it (`rolling_median`) has to set its scale on the
  swings, and what is smaller than the swings is lost: 19.2% of the spikes and 36.7% of the level shifts. Removing the
  learnt shape (`seasonal_profile`) leaves the noise, and the same anomalies stand out.
- **Variance bursts** are found by all methods in most cases because they are long and large (3 to 5 times the noise
  deviation over 20 to 60 points).

## 2. False alarms and precision

| Method | Runs flagged | Touching an event | False alarms | Per 10,000 points | Precision |
|---|---|---|---|---|---|
| rolling_median | 483 | 412 | 71 | 2.37 | 85.3% |
| seasonal_profile | 1,027 | 955 | 72 | 2.40 | 93.0% |
| seasonal_diff | 1,627 | 771 | 856 | 28.53 | 47.4% |
| isolation_forest | 1,009 | 894 | 115 | 3.83 | 88.6% |

The seasonal difference raises 28.53 false alarms per 10,000 points, 11.9 times the seasonal profile's. Two properties of
its definition account for it: an anomaly appears again one period later as the "same moment a period earlier" (a test
shows that echo), and the difference of two noisy points is noisier than one residual. This run does not separate how
much of the 856 each one explains.

Precision counts runs, not points: a long event flagged in several pieces gives several true runs, which is why the
flagged runs (1,027 for the seasonal profile) outnumber the 480 events.

## 3. Ranking against thresholding

| Method | ROC AUC | Average precision | Recall at the shared threshold |
|---|---|---|---|
| rolling_median | 0.693 | 0.335 | 57.1% |
| seasonal_profile | 0.909 | 0.766 | 99.2% |
| seasonal_diff | 0.835 | 0.336 | 81.7% |
| isolation_forest | 0.970 | 0.710 | 91.0% |

The isolation forest orders the points best (the highest AUC), yet the seasonal profile finds more events and has the
higher average precision. The threshold rule (the largest clean score) is applied to every method alike, and this run does not isolate why it
suits the seasonal profile better; a different threshold would change the event numbers (`--quantile` does that) but not
the AUC. The AUC is also computed point by point, so it
penalises a detector that flags only the first points of a long level shift.

The median delay is 0 points for all methods: when an event is found, it is found at its first point (or within the 2-point
slack). It says nothing about events not found.

## 4. Time

| Points | rolling_median | seasonal_profile | seasonal_diff | isolation_forest |
|---|---|---|---|---|
| 10,000 | 12.1 ms | 18.1 ms | 0.12 ms | 131.8 ms |
| 100,000 | 119.7 ms | 173.0 ms | 0.75 ms | 436.6 ms |
| 1,000,000 | 1,125.1 ms | 1,526.8 ms | 8.51 ms | 3,009.2 ms |

For 100 times the points, the rolling median costs 92.8 times, the seasonal profile 84.3 times, the seasonal difference
71.1 times and the isolation forest 22.8 times as much: the forest has a fixed cost (building 100 trees on the clean start)
that dominates a short series. At a million points the forest takes 354 times the seasonal difference. The two median
methods most likely spend their time in the rolling median, a sort-based operation over a window of one period
(144 points; this run does not profile it); the seasonal difference is one subtraction. A million points is 19 years at one point every ten minutes, scored in 1.5 s by the
best method here: speed does not choose between them for series of this size.

## 5. Caveats

- **One family of generated series, one seed range.** The ranking is for these series and anomaly sizes
  ([`limitations.md`](limitations.md)). Quality is pooled over 20 series; the spread between series is not reported.
- **The threshold rule is one choice.** Another rule moves precision and recall for every method.
- **12 CPUs and Docker on Windows.** The detectors here run on one thread (the forest is fixed to one), so the core count
  matters little, but absolute times include the virtual machine layer.
- **Timings are of fit plus scoring of a whole series**, not of live scoring point by point.
