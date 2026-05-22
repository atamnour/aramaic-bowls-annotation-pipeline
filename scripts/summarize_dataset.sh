#!/usr/bin/env bash
set -euo pipefail

python -m aramaic_bowls_pipeline.summarize_dataset \
  --root outputs/processed_dataset \
  --out_dir outputs/dataset_summary
