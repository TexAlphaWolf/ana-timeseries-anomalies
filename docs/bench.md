# Benchmark

Run on Linux x86_64, 12 CPUs, Python 3.14.10, NumPy 2.5.4, scikit-learn 1.9.1; seed 1; each timing is the median of 5 run(s); 38.5 s; command `bench --series 20 --sizes 10000,100000,1000000 --repeat 5 --out /out`.

## 1. Quality on 20 generated series of 20,000 points

Each series has 5,000 clean points to learn from and 15,000 to check, with spikes, level shifts, variance bursts and dropouts injected at known places. Counts are pooled over all series.

| Method | Events found | Recall | Precision (runs) | False alarms per 10,000 points | Median delay (points) | ROC AUC | Average precision |
|---|---|---|---|---|---|---|---|
| rolling_median | 274 of 480 | 57.1% | 85.3% (412 of 483) | 2.37 | 0 | 0.693 | 0.335 |
| seasonal_profile | 476 of 480 | 99.2% | 93.0% (955 of 1027) | 2.40 | 0 | 0.909 | 0.766 |
| seasonal_diff | 392 of 480 | 81.7% | 47.4% (771 of 1627) | 28.53 | 0 | 0.835 | 0.336 |
| isolation_forest | 437 of 480 | 91.0% | 88.6% (894 of 1009) | 3.83 | 0 | 0.970 | 0.710 |

## 2. Recall by kind of anomaly

| Method | spike | level_shift | variance_burst | dropout |
|---|---|---|---|---|
| rolling_median | 19.2% (23 of 120) | 36.7% (44 of 120) | 72.5% (87 of 120) | 100.0% (120 of 120) |
| seasonal_profile | 96.7% (116 of 120) | 100.0% (120 of 120) | 100.0% (120 of 120) | 100.0% (120 of 120) |
| seasonal_diff | 54.2% (65 of 120) | 76.7% (92 of 120) | 95.8% (115 of 120) | 100.0% (120 of 120) |
| isolation_forest | 86.7% (104 of 120) | 82.5% (99 of 120) | 99.2% (119 of 120) | 95.8% (115 of 120) |

## 3. Time to score a series (ms)

| Points | rolling_median | seasonal_profile | seasonal_diff | isolation_forest |
|---|---|---|---|---|
| 10,000 | 12.1 | 18.1 | 0.12 | 131.8 |
| 100,000 | 119.7 | 173.0 | 0.75 | 436.6 |
| 1,000,000 | 1,125.1 | 1,526.8 | 8.51 | 3,009.2 |
