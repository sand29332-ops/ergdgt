#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD/python_quant:$PWD/bindings${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
exec .venv/bin/python python_quant/scripts/serve_dashboard.py \
  --synthetic --host 0.0.0.0 --port "${PORT:-3000}"
