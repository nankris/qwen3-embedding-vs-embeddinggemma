#!/usr/bin/env bash
set -euo pipefail
source /workspace/tools/setup.sh
cd "$(dirname "${BASH_SOURCE[0]}")/.."
colab exec -s embedding-comparison -f scripts/snapshot.py
colab download -s embedding-comparison /content/embedding-comparison_snapshot.tar.gz results/colab_snapshot.tar.gz
tar -xzf results/colab_snapshot.tar.gz
