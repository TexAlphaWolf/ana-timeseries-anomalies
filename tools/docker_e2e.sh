#!/usr/bin/env bash
# End to end inside the image (read-only root, no capabilities, no network): a series with injected anomalies is
# generated into a volume, evaluated, searched, benchmarked, and the result files are checked. Needs Docker.
# Usage: tools/docker_e2e.sh [image]
set -euo pipefail
IMAGE="${1:-tsanom}"
VOL="tsanom-e2e-$$"
TMP="$(mktemp -d)"
trap 'docker volume rm -f "$VOL" >/dev/null 2>&1 || true; rm -rf "$TMP"' EXIT

[ "$(docker run --rm --entrypoint id "$IMAGE" -u)" != 0 ]   # not root
docker volume create "$VOL" >/dev/null
docker run --rm --user 0 -v "$VOL":/data --entrypoint chown "$IMAGE" 10001 /data

RUN=(docker run --rm --read-only --tmpfs /tmp --cap-drop ALL --security-opt no-new-privileges --network none -v "$VOL":/data "$IMAGE")
cat_file() { docker run --rm --entrypoint cat -v "$VOL":/data "$IMAGE" "$1" > "$2"; }

out="$("${RUN[@]}" generate --points 20000 --seed 1 --out /data/d)"
grep -q '20,000 points, the first 5,000 clean; 24 anomalies injected' <<<"$out"
cat_file /data/d/series.csv "$TMP/series.csv"
[ "$(head -n 1 "$TMP/series.csv")" = "ts,value,anomaly" ]
[ "$(wc -l < "$TMP/series.csv")" = 20001 ]

out="$("${RUN[@]}" evaluate /data/d)"
for m in rolling_median seasonal_profile seasonal_diff isolation_forest; do grep -q "^$m: found" <<<"$out"; done

# anomalies are in the file, so detect exits 1 and lists runs; the figure is written
set +e; out="$("${RUN[@]}" detect /data/d/series.csv --method seasonal_profile --out /data/o --figure /data/o/f.png)"; rc=$?; set -e
[ "$rc" = 1 ]
grep -q 'seasonal_profile: .* anomalous runs' <<<"$out"
cat_file /data/o/flags.csv "$TMP/flags.csv"
[ "$(head -n 1 "$TMP/flags.csv")" = "point,value,seasonal_profile_score,seasonal_profile_flag" ]
cat_file /data/o/f.png "$TMP/f.png"
[ "$(head -c 4 "$TMP/f.png" | tail -c 3)" = "PNG" ]

"${RUN[@]}" bench --series 2 --sizes 5000,9000 --repeat 1 --out /data/bench >/dev/null
cat_file /data/bench/bench.json "$TMP/bench.json"
cat_file /data/bench/bench.md "$TMP/bench.md"
grep -q '"series": 2' "$TMP/bench.json"
"${RUN[@]}" render /data/bench/bench.json -o /data/bench/again.md >/dev/null
cat_file /data/bench/again.md "$TMP/again.md"
cmp "$TMP/again.md" "$TMP/bench.md"                               # the report is rebuilt exactly from the JSON

set +e; "${RUN[@]}" detect /data/nope.csv >/dev/null 2>&1; rc=$?; set -e   # a missing file is exit code 2
[ "$rc" = 2 ]
echo "docker end-to-end: OK"
