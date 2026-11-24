#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
git submodule update --init --depth 1
uv sync --locked

python="$(uv run --no-sync python -c 'import sys; print(sys.executable)')"
build_dir="${KINCPP_BUILD_DIR:-submodules/kincpp/build}"
# kincpp gives VIRTUAL_ENV precedence over Python3_EXECUTABLE.
env -u VIRTUAL_ENV cmake -S submodules/kincpp -B "$build_dir" \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.10 \
  -DPython3_EXECUTABLE="$python" \
  -Dpybind11_DIR="$("$python" -m pybind11 --cmakedir)"
cmake --build "$build_dir" --target kincpp --parallel 2
