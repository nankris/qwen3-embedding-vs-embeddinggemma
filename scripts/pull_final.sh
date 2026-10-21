#!/usr/bin/env bash
set -euo pipefail
source /workspace/tools/setup.sh
cd /workspace/work/tig/098-qwen3-embedding-vs-embeddinggemma
colab exec -s b6-098 -f scripts/snapshot.py --timeout 60
colab download -s b6-098 /content/exp098_snapshot.tar.gz results/resume_snapshot.tar.gz
/workspace/tools/mlenv/bin/python -S scripts/merge_snapshot.py
