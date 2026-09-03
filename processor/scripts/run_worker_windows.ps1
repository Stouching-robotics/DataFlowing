$ErrorActionPreference = 'Stop'

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$pythonPath = Join-Path $projectRoot '.venv-windows\Scripts\python.exe'
if (-not (Test-Path $pythonPath)) {
    throw "Windows environment not found. Run scripts\setup_windows.ps1 first."
}

function Read-DotEnvValue([string]$name) {
    $envFile = Join-Path $projectRoot '.env'
    if (-not (Test-Path $envFile)) { return '' }
    $line = Get-Content -LiteralPath $envFile | Where-Object { $_ -match "^$name=" } | Select-Object -First 1
    if (-not $line) { return '' }
    return (($line -split '=', 2)[1]).Trim().Trim('"').Trim("'")
}

if (-not $env:EGODATA_SERVER_URL) { $env:EGODATA_SERVER_URL = 'http://127.0.0.1:8000' }
if (-not $env:EGODATA_WORKER_ID) { $env:EGODATA_WORKER_ID = "windows-$env:COMPUTERNAME" }
if (-not $env:EGODATA_DEVICE) { $env:EGODATA_DEVICE = 'auto' }
if (-not $env:EGODATA_WORK_DIR) { $env:EGODATA_WORK_DIR = Join-Path $projectRoot 'data\tmp\worker' }

if (-not $env:EGODATA_WORKER_API_KEY) { $env:EGODATA_WORKER_API_KEY = Read-DotEnvValue 'WORKER_API_KEY' }
if (-not $env:EGODATA_WORKER_API_KEY) { throw 'Set EGODATA_WORKER_API_KEY or WORKER_API_KEY in .env before starting the worker.' }

Push-Location $projectRoot
try {
    & $pythonPath -m worker
} finally {
    Pop-Location
}
