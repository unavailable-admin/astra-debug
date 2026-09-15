#!/usr/bin/env bash
# Use the validated environment and resolve all runtime paths from the repository.
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ASTRABOT_PYTHON:-/kairos_vepfs_volc/embodied/fuzuoyi/anaconda3/envs/kairos_3_1_pre/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  printf 'Python interpreter not executable: %s\nSet ASTRABOT_PYTHON to your Python executable.\n' "$PYTHON_BIN" >&2
  exit 1
fi

cd -- "$PROJECT_DIR"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export OPENAI_PROXY="${OPENAI_PROXY-http://127.0.0.1:18888}"

if (( $# == 0 )); then
  set -- run --word ACE --speed 1.0 --max-skills 4
fi

exec "$PYTHON_BIN" -m astrabot "$@"
