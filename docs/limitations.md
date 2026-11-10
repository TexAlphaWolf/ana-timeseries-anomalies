# Limitations

What this does not do, and where its numbers can mislead.

- **Synthetic series.** The comparison is on one family of generated series (a daily and a weekly cycle, a little trend,
  AR(1) noise, four kinds of anomaly). Real metrics have missing points, several seasons, changing variance and anomalies
  nobody labelled. The ranking of the methods here is not a ranking on your data.
- **Easy noise.** The noise is Gaussian and stable. A detector that depends on a stable scale (all four do) will do worse
  on heavy tails.
- **The injected sizes decide the result.** A spike of 5 to 10 standard deviations is found by most methods; at 2 to 3 it
  would not be. Sizes are in `gen.py` and in `docs/method.md`.
- **A clean start is assumed.** Every detector learns from the first part of the series, and the threshold is its largest
  score. If that part holds anomalies, the threshold rises and detection falls; if it is short, the threshold is
  unreliable (the code asks for at least three periods).
- **No retraining.** The seasonal shape is learnt once. A series whose shape drifts will drift away from it; the rolling
  median detector adapts but cannot see the cycle.
- **One season.** Only the period given (default one day) is modelled; the weekly cycle in the generator is absorbed by the
  rolling level of the seasonal detectors and shows up as noise to `seasonal_diff`.
- **Event scoring has a slack of 2 points and merges runs less than 3 apart.** Other choices change precision and recall a
  little; they are constants in `evaluate.py`.
- **Precision counts runs, not points.** A long event flagged in several pieces counts as several true runs.
- **Timings are one machine.** They come from one computer running Docker (on Windows, inside a virtual machine) with the
  cores and library versions in the report, and measure fit plus scoring of the whole series, not live, point by point.
- **Not a service.** It reads a file and writes a file; there is no stream, no alerting and no state between runs.
