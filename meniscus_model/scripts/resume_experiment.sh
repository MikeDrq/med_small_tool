#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

python -m engine.train_module2 \
  --config configs/module2_swin.yaml \
  --resume checkpoints/paper_reproduction/module2/last.pt

if [ -f checkpoints/paper_reproduction/module3_f1/last.pt ]; then
  python -m engine.train_module3 \
    --config configs/module3_bucket_swin.yaml \
    --resume checkpoints/paper_reproduction/module3_f1/last.pt
else
  python -m engine.train_module3 --config configs/module3_bucket_swin.yaml
fi

bash scripts/validate_all.sh
