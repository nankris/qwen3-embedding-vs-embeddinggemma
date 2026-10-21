#!/usr/bin/env bash
set -euo pipefail
source /workspace/tools/setup.sh
cd /workspace/work/tig/098-qwen3-embedding-vs-embeddinggemma
# Existing session only. Creation and final stop are explicit lifecycle steps.
colab upload -s b6-098 scripts/run_eval.py /content/run_eval.py
colab exec -s b6-098 -f scripts/launch.py
