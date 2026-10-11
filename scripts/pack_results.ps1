# Pack results/ into results_<hostname>_<yyyymmdd_HHMM>.zip
# Usage: pwsh -File scripts/pack_results.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Results = Join-Path $Root "results"
if (-not (Test-Path $Results)) {
    throw "No results/ directory at $Results"
}
$ts = Get-Date -Format "yyyyMMdd_HHmm"
$hostShort = $env:COMPUTERNAME.ToLowerInvariant()
$out = Join-Path $Root "results_${hostShort}_${ts}.zip"
if (Test-Path $out) { Remove-Item $out }
Compress-Archive -Path (Join-Path $Results "*") -DestinationPath $out
Write-Host "Packed -> $out"
