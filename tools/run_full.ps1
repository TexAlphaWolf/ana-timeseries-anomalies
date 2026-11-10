# The full measurement on your own computer, in Docker (no network, read-only root, no capabilities), from Windows
# PowerShell: the four detectors are scored on generated series with injected anomalies (quality) and on series of
# growing length (time).
# Usage: .\tools\run_full.ps1 [-Series 20] [-Sizes 10000,100000,1000000] [-Repeat 5] [-Out results]
param(
    [int]$Series = 20,
    [string]$Sizes = "10000,100000,1000000",
    [int]$Repeat = 5,
    [string]$Out = "results"
)
# Windows PowerShell 5.1 turns anything a program writes on stderr into an error under "Stop": docker and tsanom print
# their progress there, so only the exit code is looked at.
$ErrorActionPreference = "Continue"
New-Item -ItemType Directory -Force -Path $Out | Out-Null
$outFull = (Resolve-Path -LiteralPath $Out).Path

docker build -t tsanom .
if ($LASTEXITCODE -ne 0) { Write-Host "docker build failed"; exit 1 }

Write-Host "== bench ($Series series, sizes $Sizes, $Repeat runs each) =="
docker run --rm --read-only --tmpfs /tmp --cap-drop ALL --security-opt no-new-privileges --network none -v "${outFull}:/out" tsanom bench --series $Series --sizes $Sizes --repeat $Repeat --out /out
if ($LASTEXITCODE -ne 0) { Write-Host "bench failed (exit code $LASTEXITCODE)"; exit 1 }
Write-Host "Done. Send me $Out\bench.json, $Out\bench.md and $Out\summary.png."
