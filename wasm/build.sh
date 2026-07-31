#!/usr/bin/env bash
# Compile the same C core to WebAssembly for in-browser inference.
# Requires the Emscripten SDK: https://emscripten.org/docs/getting_started
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v emcc >/dev/null; then
  echo "emcc not found. Activate the emsdk first:" >&2
  echo "  source /path/to/emsdk/emsdk_env.sh" >&2
  exit 1
fi

[ -f core/echo_model.h ] || python3 train/train.py

emcc core/echo.c wasm/echo_wasm.c \
  -Icore \
  -O3 -ffast-math -msimd128 \
  -s WASM=1 \
  -s MODULARIZE=1 \
  -s EXPORT_NAME=EchoModule \
  -s ENVIRONMENT=web \
  -s ALLOW_MEMORY_GROWTH=1 \
  -s EXPORTED_RUNTIME_METHODS='["ccall","cwrap"]' \
  -s EXPORTED_FUNCTIONS='["_echo_wasm_init","_echo_wasm_push","_echo_wasm_state","_echo_wasm_confidence","_echo_wasm_feature","_echo_wasm_vote","_echo_wasm_valid","_echo_wasm_enrolling","_echo_wasm_ready","_echo_wasm_baselined","_echo_wasm_enroll_progress","_echo_wasm_baseline","_echo_wasm_stage_baseline","_echo_wasm_commit_baseline","_echo_wasm_model_id","_malloc","_free"]' \
  -o web/echo.js

echo "built web/echo.js + web/echo.wasm"
