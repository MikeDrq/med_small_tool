#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

python -m engine.train_module3 --config configs/module3_bucket_swin.yaml
python -m engine.validate_module3 \
  --config configs/module3_bucket_swin.yaml \
  --checkpoint checkpoints/paper_reproduction/module3_f1/best.pt
