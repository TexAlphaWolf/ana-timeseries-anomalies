# End to end inside the image, from Windows PowerShell: a series with injected anomalies is generated into a volume,
# evaluated, searched and benchmarked, and the result files are checked (read-only root, no capabilities, no network).
# Needs Docker Desktop.
# Usage: .\tools\docker_e2e.ps1 [-Image tsanom]
param([string]$Image = "tsanom")
$ErrorActionPreference = "Stop"

$id = Get-Random
$vol = "tsanom-e2e-$id"
$secure = @("--read-only", "--tmpfs", "/tmp", "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--network", "none", "-v", "${vol}:/data")

# Windows PowerShell 5.1 turns a program's message on stderr into an error under "Stop": the commands that print
# progress on stderr or are expected to fail run with errors not terminating, and only their exit code is looked at.
function Invoke-TsCode([int[]]$Allowed, [string[]]$TsArgs) {
    $old = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $out = docker run --rm @secure $Image @TsArgs 2>&1
    $code = $LASTEXITCODE
    $ErrorActionPreference = $old
    if ($Allowed -notcontains $code) { throw "tsanom $TsArgs exited with $code : $out" }
    ($out | Out-String)
}
function Invoke-Ts([string[]]$TsArgs) { Invoke-TsCode @(0) $TsArgs }
function Read-Volume($path) {
    $out = docker run --rm --entrypoint cat -v "${vol}:/data" $Image $path 2>$null
    if ($LASTEXITCODE -ne 0) { throw "cannot read $path" }
    ($out | Out-String)
}
function Assert-Contains($text, $needle) { if ($text -notmatch [regex]::Escape($needle)) { throw "expected '$needle' in: $text" } }

try {
    docker volume create $vol | Out-Null
    docker run --rm --user 0 -v "${vol}:/data" --entrypoint chown $Image 10001 /data
    if ($LASTEXITCODE -ne 0) { throw "could not prepare the volume" }

    Assert-Contains (Invoke-Ts @("generate", "--points", "20000", "--seed", "1", "--out", "/data/d")) "20,000 points, the first 5,000 clean; 24 anomalies injected"
    $series = Read-Volume "/data/d/series.csv"
    Assert-Contains $series "ts,value,anomaly"

    $eval = Invoke-Ts @("evaluate", "/data/d")
    foreach ($m in "rolling_median", "seasonal_profile", "seasonal_diff", "isolation_forest") { Assert-Contains $eval "${m}: found" }

    # anomalies are in the file, so detect exits 1 and lists runs; the figure is written
    $det = Invoke-TsCode @(1) @("detect", "/data/d/series.csv", "--method", "seasonal_profile", "--out", "/data/o", "--figure", "/data/o/f.png")
    Assert-Contains $det "seasonal_profile:"
    Assert-Contains $det "anomalous runs"
    Assert-Contains (Read-Volume "/data/o/flags.csv") "point,value,seasonal_profile_score,seasonal_profile_flag"

    Invoke-Ts @("bench", "--series", "2", "--sizes", "5000,9000", "--repeat", "1", "--out", "/data/bench") | Out-Null
    $bench = Read-Volume "/data/bench/bench.json"
    Assert-Contains $bench '"series": 2'
    $md = Read-Volume "/data/bench/bench.md"
    Invoke-Ts @("render", "/data/bench/bench.json", "-o", "/data/bench/again.md") | Out-Null
    if ((Read-Volume "/data/bench/again.md") -ne $md) { throw "the report rebuilt from the JSON is not identical" }

    Invoke-TsCode @(2) @("detect", "/data/nope.csv") | Out-Null
    Write-Host "docker end-to-end: OK"
}
finally {
    $ErrorActionPreference = "Continue"
    docker volume rm -f $vol 2>&1 | Out-Null
}
