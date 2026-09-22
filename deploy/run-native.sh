#!/usr/bin/env bash
set -euo pipefail
script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == "v41" ]]; then
    shift
    exec bash "$script_directory/run-v41-native.sh" "$@"
fi
exec "${PYTHON:-python3}" "$script_directory/native.py" "$@"
