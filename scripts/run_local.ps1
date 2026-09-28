param(
    [ValidateSet('smoke','formal','verify','math','tfrecord','tests')]
    [string]$Action = 'smoke'
)
$ErrorActionPreference = 'Stop'
$taskRepo = Split-Path -Parent $PSScriptRoot
$taskPythonCandidates = @(
    (Join-Path $taskRepo '.venv/Scripts/python.exe'),
    (Join-Path $taskRepo '../../work/har-env/Scripts/python.exe')
)
$taskPython = $taskPythonCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $taskPython) { throw 'Python environment not found. Follow README installation commands first.' }
$taskPython = (Resolve-Path -LiteralPath $taskPython).Path
$taskDataCandidates = @(
    (Join-Path $taskRepo 'data/raw/UCI HAR Dataset'),
    (Join-Path $taskRepo '../../work/uci-har/UCI HAR Dataset')
)
if ($env:HAR_DATA_ROOT) { $taskDataCandidates = @($env:HAR_DATA_ROOT) + $taskDataCandidates }
$taskData = $taskDataCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (($Action -notin @('math','tests')) -and (-not $taskData)) {
    throw 'Data not found. Run scripts/download_data.py or set HAR_DATA_ROOT.'
}
if ($taskData) { $taskData = (Resolve-Path -LiteralPath $taskData).Path }
Push-Location -LiteralPath $taskRepo
try {
    $taskStamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
    switch ($Action) {
        'smoke' { & $taskPython -m src.train --config configs/E02.json --data-root $taskData --smoke --runs-dir "runs/local_$taskStamp" }
        'formal' { & $taskPython -m src.experiments --data-root $taskData --runs-dir "runs/local_$taskStamp" }
        'verify' { & $taskPython -m src.verify_bundle --data-root $taskData --runs-dir runs/formal }
        'math' { & $taskPython -m src.math_checks }
        'tfrecord' { & $taskPython -m src.tfrecord_check --data-root $taskData }
        'tests' { & $taskPython -m pytest -q }
    }
    $taskExit = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $taskExit
