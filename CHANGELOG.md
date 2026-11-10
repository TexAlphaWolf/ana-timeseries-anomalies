# Changelog

## 1.0.0

First release.

- Four detectors (rolling median, seasonal profile, seasonal difference, isolation forest), each scoring a point from the
  points before it, with one threshold rule: the largest score on the clean start.
- A generator of series with four kinds of anomaly (spike, level shift, variance burst, dropout) injected at known
  positions, with the events written next to the data.
- Scoring at event level (recall, precision, false alarms, delay) and at point level (ROC AUC, average precision).
- A benchmark over many series and up to a million points, rebuilt into Markdown and a figure from its JSON.
- A command line (`generate`, `detect`, `evaluate`, `bench`, `render`, `methods`) with exit codes for a pipeline.
- Docker image (no network, read-only, no capabilities); CI on Python 3.11 to 3.14, Linux, Windows and macOS.
