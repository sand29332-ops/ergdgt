#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
if ! command -v c++ >/dev/null || ! python3 -c \
  'import pathlib, sys, sysconfig; sys.exit(not (pathlib.Path(sysconfig.get_path("include")) / "Python.h").is_file())'; then
  if command -v apt-get >/dev/null && [ "$(id -u)" -eq 0 ]; then
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      build-essential python3-dev python3-venv
  else
    echo "Install a C++20 compiler and development headers for python3 before setup." >&2
    exit 1
  fi
fi
python3 -m venv .venv
.venv/bin/python -m pip install -r python_quant/requirements.txt -r requirements-dev.txt

.venv/bin/cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DNEXUS_NATIVE_ARCH=OFF \
  -DNEXUS_ENABLE_CUDA=OFF \
  -DNEXUS_BUILD_PYBIND=ON \
  -DPython3_EXECUTABLE="$PWD/.venv/bin/python" \
  -Dpybind11_DIR="$(.venv/bin/python -m pybind11 --cmakedir)"
.venv/bin/cmake --build build --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-2}"
