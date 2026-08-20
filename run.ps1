# MuleGuard AI — Windows launcher (PowerShell)
#
#   .\run.ps1            set up if needed, then run the full pipeline
#   .\run.ps1 -Setup     only build the environment
#   .\run.ps1 -Verify    only check the environment
#   .\run.ps1 -Stage 01_clean.py
#   .\run.ps1 -Serve     start the command-center UI at http://127.0.0.1:8000
#
#   .\run.ps1 -Dataset D:\theirs.csv      run on a dataset handed over live
#   .\run.ps1 -Dataset D:\theirs.csv -Full    full precision, slower
#
# Any dataset, not just the hackathon file. All optional:
#   $env:MULEGUARD_DATA    = "D:\other.csv"    dataset to run on
#   $env:MULEGUARD_DICT    = "D:\dict.xlsx"    data dictionary (.xlsx or .csv)
#   $env:MULEGUARD_TARGET  = "is_fraud"        name the target; auto-detected otherwise
#   $env:MULEGUARD_WORKDIR = "runs\other"      keep this run's artefacts separate
#   $env:MULEGUARD_REPEATS = "1"               faster cross-validation
#
# If PowerShell blocks this script:
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
# or run it as:  powershell -ExecutionPolicy Bypass -File .\run.ps1

[CmdletBinding()]
param(
    [switch]$Setup,
    [switch]$Verify,
    [switch]$Serve,
    [int]$Port = 8000,
    [string]$Stage,
    [string]$Dataset,          # run on a file handed over live
    [string]$Target,           # only if auto-detection cannot find it
    [switch]$Full              # full precision instead of demo speed
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

# A macOS/Linux .venv may be sitting in this folder (it syncs through OneDrive
# and its layout is bin/, not Scripts/). Keep the Windows environment separate so
# neither platform clobbers the other.
$VenvDir = Join-Path $Root ".venv-win"
$Py = Join-Path $VenvDir "Scripts\python.exe"

function Find-BasePython {
    foreach ($v in @("3.12", "3.11")) {
        try {
            $null = & py "-$v" -c "import sys" 2>$null
            if ($LASTEXITCODE -eq 0) { return @("py", "-$v") }
        } catch {}
    }
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) {
        $ver = & python -c "import sys;print(f'{sys.version_info.major}.{sys.version_info.minor}')"
        if ($ver -in @("3.11", "3.12")) { return @("python") }
        Write-Host "Found Python $ver. 3.11 or 3.12 is required (ML wheels lag on 3.13+)." -ForegroundColor Yellow
    }
    throw "No Python 3.11/3.12 found. Install from https://www.python.org/downloads/ and tick 'Add python.exe to PATH'."
}

function Initialize-Venv {
    if (-not (Test-Path $Py)) {
        $base = Find-BasePython
        Write-Host "Creating virtual environment in .venv-win ..." -ForegroundColor Cyan
        & $base[0] $base[1..($base.Count - 1)] -m venv $VenvDir
    }
    Write-Host "Installing dependencies ..." -ForegroundColor Cyan
    & $Py -m pip install --upgrade pip --quiet
    & $Py -m pip install -r (Join-Path $Root "requirements.txt") --quiet
    # pyarrow keeps the intermediate artefacts as parquet rather than fat CSVs.
    & $Py -m pip install pyarrow --quiet
    Write-Host "Environment ready." -ForegroundColor Green
}

if (-not (Test-Path $Py)) { Initialize-Venv }
if ($Setup) { Initialize-Venv; exit 0 }

$env:PYTHONIOENCODING = "utf-8"   # box-drawing and +/- survive the console

if ($Verify) { & $Py (Join-Path $Root "verify_env.py"); exit $LASTEXITCODE }

# ---------------------------------------------------------------------------
# Live demo: point it at somebody else's dataset and get a result in minutes.
#
#   .\run.ps1 -Dataset D:\judges.csv
#
# Writes everything under runs\<name>\ so the submission's own results are
# never touched, and runs in demo mode unless -Full is passed.
# ---------------------------------------------------------------------------
if ($Dataset) {
    if (-not (Test-Path $Dataset)) { throw "Dataset not found: $Dataset" }
    $Full1 = (Get-Item $Dataset).FullName
    $Name = [IO.Path]::GetFileNameWithoutExtension($Full1) -replace '[^A-Za-z0-9_-]', '_'
    $Work = "runs\$Name"

    $env:MULEGUARD_DATA = $Full1
    $env:MULEGUARD_WORKDIR = $Work
    if ($Target) { $env:MULEGUARD_TARGET = $Target }
    if (-not $Full) { $env:MULEGUARD_FAST = "1" }
    $env:PYTHONHASHSEED = "0"

    Write-Host ""
    Write-Host "  Dataset : $Full1" -ForegroundColor Cyan
    Write-Host "  Output  : $Work" -ForegroundColor Cyan
    Write-Host "  Mode    : $(if ($Full) {'FULL precision'} else {'DEMO speed'})" -ForegroundColor Cyan
    Write-Host "  Nothing in this run touches the submission's own results." -ForegroundColor DarkGray
    Write-Host ""

    $sw = [Diagnostics.Stopwatch]::StartNew()
    & $Py (Join-Path $Root "src\pipeline.py")
    $code = $LASTEXITCODE
    $sw.Stop()
    Write-Host ""
    Write-Host ("  Finished in {0:N0}s" -f $sw.Elapsed.TotalSeconds) -ForegroundColor Green
    if ($code -eq 0) {
        Write-Host "  Read $Work\reports\00_INTEGRITY.md first." -ForegroundColor Yellow
        Write-Host "  Then:  .\run.ps1 -Serve" -ForegroundColor Yellow
    }
    exit $code
}

if ($Serve) {
    # Bound to 127.0.0.1 deliberately: this is a local analyst tool that loads a
    # pickled model and exposes account-level risk data. It is not hardened for
    # exposure on a shared network.
    Write-Host "MuleGuard command center -> http://127.0.0.1:$Port" -ForegroundColor Green
    Write-Host "Press Ctrl+C to stop." -ForegroundColor DarkGray
    & $Py -m uvicorn app.server:app --host 127.0.0.1 --port $Port
    exit $LASTEXITCODE
}

if ($Stage) {
    & $Py (Join-Path $Root "src\$Stage")
    exit $LASTEXITCODE
}

& $Py (Join-Path $Root "src\pipeline.py")
exit $LASTEXITCODE
