$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $projectRoot "venv310\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    throw "Missing virtualenv Python at $venvPython"
}

$env:PYTHONIOENCODING = "utf-8"
$env:YOLO_CONFIG_DIR = Join-Path $projectRoot ".runtime"

& $venvPython -m uvicorn fastapi_server:app --host 0.0.0.0 --port 8000 --reload
