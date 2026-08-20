#!/usr/bin/env bash
# MuleGuard AI — macOS / Linux launcher
#
#   ./run.sh            set up if needed, then run the full pipeline
#   ./run.sh setup      only build the environment
#   ./run.sh verify     only check the environment
#   ./run.sh stage 01_clean.py
#   ./run.sh serve      start the command-center UI at http://127.0.0.1:8000
#   ./run.sh dataset /path/theirs.csv        run on a dataset handed over live
#   MULEGUARD_FULL=1 ./run.sh dataset f.csv  full precision, slower
#
# Any dataset, not just the hackathon file. All optional:
#   MULEGUARD_DATA=/path/other.csv      dataset to run on
#   MULEGUARD_DICT=/path/dict.xlsx      data dictionary (.xlsx or .csv)
#   MULEGUARD_TARGET=is_fraud           name the target; auto-detected otherwise
#   MULEGUARD_WORKDIR=runs/other        keep this run's artefacts separate
#   MULEGUARD_REPEATS=1                 faster cross-validation
#
# e.g.  MULEGUARD_DATA=~/other.csv MULEGUARD_WORKDIR=runs/other ./run.sh
#
# Make it executable once:  chmod +x run.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

VENV="$ROOT/.venv"
PY="$VENV/bin/python"

find_base_python() {
  for v in python3.12 python3.11; do
    if command -v "$v" >/dev/null 2>&1; then echo "$v"; return; fi
  done
  if command -v python3 >/dev/null 2>&1; then
    ver="$(python3 -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    case "$ver" in
      3.11|3.12) echo python3; return ;;
      *) echo "Found Python $ver; 3.11 or 3.12 is required (ML wheels lag on 3.13+)." >&2 ;;
    esac
  fi
  echo "No suitable Python found. macOS: brew install python@3.12" >&2
  exit 1
}

setup_venv() {
  if [ ! -x "$PY" ]; then
    base="$(find_base_python)"
    echo "Creating virtual environment in .venv ..."
    "$base" -m venv "$VENV"
  fi
  echo "Installing dependencies ..."
  "$PY" -m pip install --upgrade pip --quiet
  "$PY" -m pip install -r "$ROOT/requirements.txt" --quiet
  "$PY" -m pip install pyarrow --quiet
  # LightGBM and XGBoost both fail at import on macOS without the OpenMP runtime.
  if [[ "$OSTYPE" == darwin* ]] && ! "$PY" -c "import lightgbm" >/dev/null 2>&1; then
    echo "LightGBM failed to import — on macOS this almost always means: brew install libomp" >&2
  fi
  echo "Environment ready."
}

[ -x "$PY" ] || setup_venv

export PYTHONIOENCODING=utf-8

case "${1:-run}" in
  setup)  setup_venv ;;
  verify) "$PY" "$ROOT/verify_env.py" ;;
  stage)  "$PY" "$ROOT/src/${2:?usage: ./run.sh stage <script.py>}" ;;
  run)    "$PY" "$ROOT/src/pipeline.py" ;;
  # Live demo: ./run.sh dataset /path/theirs.csv  [target_column]
  # Writes under runs/<name>/ so the submission's results are never touched.
  dataset)
          f="${2:?usage: ./run.sh dataset <file.csv> [target_column]}"
          [ -f "$f" ] || { echo "Dataset not found: $f" >&2; exit 1; }
          abs="$(cd "$(dirname "$f")" && pwd)/$(basename "$f")"
          name="$(basename "${f%.*}" | tr -cs 'A-Za-z0-9_-' '_')"
          export MULEGUARD_DATA="$abs"
          export MULEGUARD_WORKDIR="runs/$name"
          [ -n "${3:-}" ] && export MULEGUARD_TARGET="$3"
          [ -n "${MULEGUARD_FULL:-}" ] || export MULEGUARD_FAST=1
          export PYTHONHASHSEED=0
          echo ""
          echo "  Dataset : $abs"
          echo "  Output  : runs/$name"
          echo "  Mode    : ${MULEGUARD_FAST:+DEMO speed}${MULEGUARD_FULL:+FULL precision}"
          echo "  Nothing in this run touches the submission's own results."
          echo ""
          start=$(date +%s)
          "$PY" "$ROOT/src/pipeline.py"; code=$?
          echo ""
          echo "  Finished in $(( $(date +%s) - start ))s"
          [ $code -eq 0 ] && echo "  Read runs/$name/reports/00_INTEGRITY.md first, then: ./run.sh serve"
          exit $code ;;
  # Bound to 127.0.0.1 deliberately: this is a local analyst tool that loads a
  # pickled model and exposes account-level risk data. It is not hardened for
  # exposure on a shared network.
  serve)  port="${2:-8000}"
          echo "MuleGuard command center -> http://127.0.0.1:$port  (Ctrl+C to stop)"
          "$PY" -m uvicorn app.server:app --host 127.0.0.1 --port "$port" ;;
  *)      echo "usage: ./run.sh [setup|verify|run|serve [port]|dataset <file.csv> [target]|stage <script.py>]" >&2; exit 2 ;;
esac
