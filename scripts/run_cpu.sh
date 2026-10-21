#!/usr/bin/env bash
set -euo pipefail
source /workspace/tools/setup.sh
cd /workspace/work/tig/098-qwen3-embedding-vs-embeddinggemma
export EXP_ROOT="$PWD"
export HF_HOME="$PWD/.cache/huggingface"
export PYTHONPATH="$PWD/.deps"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
exec /workspace/tools/mlenv/bin/python -u scripts/run_eval.py --cpu
