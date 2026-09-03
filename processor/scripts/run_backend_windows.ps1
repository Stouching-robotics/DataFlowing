$ErrorActionPreference = 'Stop'

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$pythonPath = Join-Path $projectRoot '.venv-windows\Scripts\python.exe'
if (-not (Test-Path $pythonPath)) {
    throw "Windows environment not found. Run scripts\setup_windows.ps1 first."
}

$env:PYTHONPATH = $projectRoot
if (-not $env:HOST) { $env:HOST = '0.0.0.0' }
if (-not $env:PORT) { $env:PORT = '8000' }
$reloadArgs = @()
$reloadEnabled = $env:EGODATA_RELOAD -notin @('0', 'false', 'False', 'no', 'off')
if ($reloadEnabled) {
    # Watch code/web only; data, videos, logs and SFTP cache do
    # not cause a pointless backend restart.
    $reloadArgs = @(
        '--reload',
        '--reload-delay', '0.5',
        '--reload-dir', (Join-Path $projectRoot 'app'),
        '--reload-dir', (Join-Path $projectRoot 'web/templates'),
        '--reload-dir', (Join-Path $projectRoot 'web/static'),
        '--reload-dir', (Join-Path $projectRoot 'web/workflow-studio/src')
    )
}
Push-Location $projectRoot
try {
    & $pythonPath -m uvicorn app.main:app --host $env:HOST --port ([int]$env:PORT) @reloadArgs
} finally {
    Pop-Location
}
