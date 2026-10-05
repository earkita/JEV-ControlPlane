#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
runtime_dir="${WINNOW_RUNTIME_DIR:-$HOME/ai/winnow-inference-reference}"
model_file="${WINNOW_MODEL:-/mnt/ai/models/classifiers/Winnow-12B/Winnow-12B-Q8_0.gguf}"
projector_file="${WINNOW_MMPROJ:-/mnt/ai/models/classifiers/Winnow-12B/mmproj-Winnow-12B.gguf}"
server_bin="${WINNOW_SERVER_BIN:-$runtime_dir/.build-gcc13/bin/winnow-server}"
server_port="${WINNOW_SERVER_PORT:-8091}"
context="${WINNOW_CONTEXT:-8192}"
batch="${WINNOW_BATCH:-4096}"
ubatch="${WINNOW_UBATCH:-4096}"

if ! [[ "$batch" =~ ^[0-9]+$ && "$ubatch" =~ ^[0-9]+$ ]] || (( ubatch < 1 || ubatch > batch )); then
  printf 'WINNOW_BATCH and WINNOW_UBATCH must be positive integers with UBATCH <= BATCH.\n' >&2
  exit 1
fi

if [[ ! -f "$model_file" && -f "/mnt/ai/models/classifiers/Winnow-12B/gguf/Winnow-12B-Q8_0.gguf" ]]; then
  model_file="/mnt/ai/models/classifiers/Winnow-12B/gguf/Winnow-12B-Q8_0.gguf"
fi
if [[ ! -f "$projector_file" && -f "/mnt/ai/models/classifiers/Winnow-12B/gguf/mmproj-Winnow-12B.gguf" ]]; then
  projector_file="/mnt/ai/models/classifiers/Winnow-12B/gguf/mmproj-Winnow-12B.gguf"
fi

for file in "$model_file" "$projector_file" "$server_bin"; do
  if [[ ! -f "$file" ]]; then
    printf 'Missing Winnow artifact: %s\n' "$file" >&2
    exit 1
  fi
done

if command -v nvidia-smi >/dev/null 2>&1; then
  free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')"
  if [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib < 16000 )); then
    printf 'Only %s MiB GPU memory free; stop the other model server first.\n' "$free_mib" >&2
    exit 1
  fi
fi

python3 "$runtime_dir/scripts/serve.py" \
  --model "$model_file" --mmproj "$projector_file" --server "$server_bin" \
  --context "$context" --decision-context "$context" \
  --batch "$batch" --ubatch "$ubatch" \
  --cache q8_0 --decision-parallel 2 --chat-parallel 1 \
  --memory exclusive --host 127.0.0.1 --port "$server_port" &
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
    printf 'Winnow server exited before becoming ready.\n' >&2
    exit 1
  fi
  sleep 1
done
if [[ "$ready" -ne 1 ]]; then
  printf 'Winnow server did not become ready in 180 seconds.\n' >&2
  exit 1
fi

OPENJEV_CONFIG="$repo_dir/config/serve-winnow12b.json" \
  MODEL="$model_file" PROJECTOR_PATH="$projector_file" \
  BACKEND_URL="http://127.0.0.1:$server_port" MAX_CONTEXT="$context" \
  "$repo_dir/launch.sh" "$@"
