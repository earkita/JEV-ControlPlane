#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
config_path="${OPENJEV_CONFIG:-$repo_dir/config/serve.json}"

if [[ -x "$repo_dir/.venv/bin/openjev-semif" ]]; then
  cli="$repo_dir/.venv/bin/openjev-semif"
elif command -v openjev-semif >/dev/null 2>&1; then
  cli="$(command -v openjev-semif)"
else
  printf 'openjev-semif is not installed; see README.md\n' >&2
  exit 1
fi

exec "$cli" serve --config "$config_path" "$@"
