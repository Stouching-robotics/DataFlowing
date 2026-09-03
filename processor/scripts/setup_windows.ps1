$ErrorActionPreference = 'Stop'

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$venvPath = Join-Path $projectRoot '.venv-windows'
$pythonPath = Join-Path $venvPath 'Scripts\python.exe'

if (-not (Test-Path $pythonPath)) {
    Write-Host "Creating Windows virtual environment: $venvPath"
    & python -m venv $venvPath
}

& $pythonPath -m pip install --upgrade pip
& $pythonPath -m pip install -r (Join-Path $projectRoot 'requirements-windows.txt')
Write-Host "Windows environment ready: $pythonPath"
