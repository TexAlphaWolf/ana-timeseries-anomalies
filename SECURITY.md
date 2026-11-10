# Security

## Model

tsanom is a command line program that **reads a CSV file of numbers, scores it, and writes files**. It starts no service,
opens no network connection and runs no code from its input.

- **What it reads:** a CSV with a `value` column (or one column of numbers), and the `series.csv` and `events.json` that
  `generate` wrote. Values are parsed as numbers; anything else is an error (exit code 2). Nothing in a file is executed or
  used to build a command.
- **What it writes:** only where the options say: `generate --out`, `detect --out` and `--figure`, `bench --out`,
  `render -o`.
- **The Docker image** runs as an unprivileged user and works with a read-only root filesystem (plus `--tmpfs /tmp` for
  matplotlib's cache), no capabilities and no network (`--network none`); see `tools/docker_e2e.sh`.
- **No real data** is in the repository: everything the tests and the benchmark use is generated from a seed.

## Reporting

If you find a vulnerability, please open a private security advisory on the GitHub repository (Security tab) rather than a
public issue.
