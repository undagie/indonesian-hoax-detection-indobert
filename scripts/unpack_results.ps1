# Unpack a results archive into ./results/, merging with existing files.
# Usage: pwsh -File scripts/unpack_results.ps1 -Archive .\results_host_20260520_1530.zip

param(
    [Parameter(Mandatory = $true)]
    [string]$Archive
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Dest = Join-Path $Root "results"
if (-not (Test-Path $Archive)) {
    throw "Archive not found: $Archive"
}
New-Item -ItemType Directory -Force -Path $Dest | Out-Null
Expand-Archive -Path $Archive -DestinationPath $Root -Force
Write-Host "Unpacked $Archive into $Dest"
