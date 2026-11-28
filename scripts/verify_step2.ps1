# Step 2 integration verification for MLOps Electricity Demand Forecasting.
# Copy this file into <repository>\scripts\verify_step2.ps1, then run from PowerShell.
[CmdletBinding()]
param(
    [string]$ApiBaseUrl = 'http://127.0.0.1:8000',
    [ValidateSet('auto', 'api', 'dashboard')]
    [string]$Source = 'dashboard'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $repoRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Python virtual environment not found at $pythonExe. Install the project dependencies first."
}

$apiUrl = $ApiBaseUrl.TrimEnd('/')
if ($apiUrl -notmatch '^https?://[^/?#]+(?::\d+)?$') {
    throw 'ApiBaseUrl must be an HTTP(S) origin without query strings or credentials.'
}

$oldSource = [Environment]::GetEnvironmentVariable('EIA_SOURCE', 'Process')
$oldApiUrl = [Environment]::GetEnvironmentVariable('INFERENCE_API_URL', 'Process')

try {
    $env:EIA_SOURCE = $Source
    $env:INFERENCE_API_URL = $apiUrl

    Write-Host '[1/3] Checking FastAPI /health...'
    $health = Invoke-RestMethod -Uri "$apiUrl/health" -TimeoutSec 12
    if ($health.status -ne 'ok' -or $health.model_loaded -ne $true) {
        throw 'FastAPI /health did not confirm a loaded model.'
    }

    Write-Host '[2/3] Checking MLflow connectivity and loaded model version...'
    $registry = Invoke-RestMethod -Uri "$apiUrl/mlflow-check" -TimeoutSec 15
    if ($registry.communication -ne 'success' -or -not $registry.loaded_model_version) {
        throw 'FastAPI /mlflow-check did not confirm an MLflow connection and a loaded model version.'
    }

    Write-Host '[3/3] Running one real EIA -> FastAPI inference cycle...'
    Push-Location $repoRoot
    try {
        $outputLines = & $pythonExe -m src.pipelines.run_online_inference
        $pythonExitCode = $LASTEXITCODE
    }
    finally {
        Pop-Location
    }

    $outputJson = $outputLines -join "`n"
    if ([string]::IsNullOrWhiteSpace($outputJson)) {
        throw "The inference CLI produced no JSON output (exit code: $pythonExitCode)."
    }
    try {
        $prediction = $outputJson | ConvertFrom-Json
    }
    catch {
        throw 'The inference CLI returned invalid JSON.'
    }

    if ($pythonExitCode -ne 0 -or $prediction.status -ne 'success') {
        $status = [string]$prediction.status
        $reason = [string]$prediction.message
        throw "Online inference did not succeed: $status. $reason"
    }

    $origin = [DateTimeOffset]::Parse($prediction.observation_timestamp)
    $target = [DateTimeOffset]::Parse($prediction.forecast_target_timestamp)
    $created = [DateTimeOffset]::Parse($prediction.prediction_created_at)
    if ($target - $origin -ne [TimeSpan]::FromHours(2)) {
        throw 'Forecast target is not exactly observation timestamp + 2 hours.'
    }
    if ($target -le $created) {
        throw 'Forecast target has already passed; result is not a live forecast.'
    }

    $y = [double]$prediction.predicted_demand
    if ([double]::IsNaN($y) -or [double]::IsInfinity($y)) {
        throw 'Prediction is not a finite numeric value.'
    }
    if ([string]$prediction.model_version -ne [string]$registry.loaded_model_version) {
        throw 'Reported model version differs from the version loaded by FastAPI.'
    }

    Write-Host ''
    Write-Host 'PASS: End-to-end online inference verified.' -ForegroundColor Green
    Write-Host "Source:        $($prediction.data_source)"
    Write-Host "Observation:   $($prediction.observation_timestamp)"
    Write-Host "Target (t+2h): $($prediction.forecast_target_timestamp)"
    Write-Host "Actual demand: $($prediction.latest_actual_demand)"
    Write-Host "Prediction:    $($prediction.predicted_demand)"
    Write-Host "Latency (ms):  $($prediction.inference_latency_ms)"
    Write-Host "Model URI:     $($prediction.model_uri)"
    Write-Host "Loaded version:$($prediction.model_version)"
}
finally {
    [Environment]::SetEnvironmentVariable('EIA_SOURCE', $oldSource, 'Process')
    [Environment]::SetEnvironmentVariable('INFERENCE_API_URL', $oldApiUrl, 'Process')
}
