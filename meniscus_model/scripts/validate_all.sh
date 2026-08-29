#!/usr/bin/env bash
set -euo pipefail

INDEX="${1:-data/paper_reproduction/test.jsonl}"
OUTPUT_DIR="${2:-validation_outputs/paper_reproduction/test_set}"
BOOTSTRAP="${3:-1000}"
TEST_ROIS="${TEST_ROIS:-rois/paper_reproduction/test}"

python -m engine.validate_module1 \
  --config configs/module1_meniscovit_cop.yaml \
  --checkpoint checkpoints/paper_reproduction/module1/best.pt \
  --index "${INDEX}" \
  --bootstrap "${BOOTSTRAP}"
python scripts/generate_rois.py \
  --config configs/module1_meniscovit_cop.yaml \
  --index "${INDEX}" \
  --module1-checkpoint checkpoints/paper_reproduction/module1/best.pt \
  --output-dir "${TEST_ROIS}" \
  --use-gt-boxes false \
  --slice-selection-usage val \
  --crop-size 224 \
  --save-visualizations false \
  --storage-dtype float16
python -m engine.validate_module2 \
  --config configs/module2_swin.yaml \
  --checkpoint checkpoints/paper_reproduction/module2/best.pt \
  --roi-index "${TEST_ROIS}/index.jsonl" \
  --bootstrap "${BOOTSTRAP}"
python -m engine.validate_module3 \
  --config configs/module3_bucket_swin.yaml \
  --checkpoint checkpoints/paper_reproduction/module3_f1/best.pt \
  --roi-index "${TEST_ROIS}/index.jsonl" \
  --bootstrap "${BOOTSTRAP}"
python -m engine.validate_module4 \
  --config configs/pipeline.yaml \
  --index "${INDEX}" \
  --roi-cache-dir "${TEST_ROIS}" \
  --bootstrap "${BOOTSTRAP}"
python -m engine.validate_pipeline --config configs/pipeline.yaml --index "${INDEX}" --output-dir "${OUTPUT_DIR}" --bootstrap "${BOOTSTRAP}"
