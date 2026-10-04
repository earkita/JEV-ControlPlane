#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
model_dir="$HOME/ai/models/openjev/OpenJev-27B-Q4_K_M"
model_file="${OPENJEV27B_MODEL:-$model_dir/OpenJev-Q4_K_M.gguf}"
server_bin="${LLAMA_SERVER_BIN:-$HOME/ai/llama.cpp-prebuilt/llama-b11393/llama-server}"
server_port="${LLAMA_SERVER_PORT:-8091}"
gpu_layers="${LLAMA_N_GPU_LAYERS:-999}"
context="${LLAMA_CONTEXT:-8192}"

if [[ ! -f "$model_file" ]]; then
  printf 'Model missing: %s\nSee README.md for the pinned hf download command.\n' "$model_file" >&2
  exit 1
fi
if [[ ! -x "$server_bin" ]]; then
  printf 'llama-server missing: %s\nSee README.md for build instructions.\n' "$server_bin" >&2
  exit 1
fi
if [[ "$gpu_layers" == 999 ]] && command -v nvidia-smi >/dev/null 2>&1; then
  free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')"
  if [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib < 19000 )); then
    printf 'Only %s MiB GPU memory is free; stop the other model server before full GPU offload.\n' "$free_mib" >&2
    exit 1
  fi
fi

"$server_bin" -m "$model_file" -ngl "$gpu_layers" -c "$context" -np 1 \
  --host 127.0.0.1 --port "$server_port" &
server_pid=$!
cleanup() {
  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
}
trap cleanup EXIT

ready=0
for ((i=0; i<180; i++)); do
  if curl -fsS "http://127.0.0.1:$server_port/health" >/dev/null 2>&1; then
    ready=1
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    printf 'llama-server exited before becoming ready.\n' >&2
    exit 1
  fi
  sleep 1
done
if [[ "$ready" -ne 1 ]]; then
  printf 'llama-server did not become ready within 180 seconds.\n' >&2
  exit 1
fi

OPENJEV_CONFIG="$repo_dir/config/serve-openjev27b.json" \
  MODEL="$model_file" \
  BACKEND_URL="http://127.0.0.1:$server_port" \
  "$repo_dir/launch.sh" "$@"
