#!/usr/bin/env bash
set -euo pipefail
source /workspace/tools/setup.sh
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# Existing session only. Creation and final stop are explicit lifecycle steps.
colab upload -s embedding-comparison scripts/run_eval.py /content/run_eval.py
colab exec -s embedding-comparison -f scripts/launch.py
