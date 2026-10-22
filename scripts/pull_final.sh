#!/usr/bin/env bash
set -euo pipefail
source /workspace/tools/setup.sh
cd "$(dirname "${BASH_SOURCE[0]}")/.."
colab exec -s embedding-comparison -f scripts/snapshot.py --timeout 60
colab download -s embedding-comparison /content/embedding-comparison_snapshot.tar.gz results/resume_snapshot.tar.gz
/workspace/tools/mlenv/bin/python -S scripts/merge_snapshot.py
