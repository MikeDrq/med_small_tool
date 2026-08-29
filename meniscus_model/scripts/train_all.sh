#!/usr/bin/env bash
set -euo pipefail

MODULE1_CONFIG="${MODULE1_CONFIG:-configs/module1_meniscovit_cop.yaml}"
MODULE2_CONFIG="${MODULE2_CONFIG:-configs/module2_swin.yaml}"
MODULE3_CONFIG="${MODULE3_CONFIG:-configs/module3_bucket_swin.yaml}"
PIPELINE_CONFIG="${PIPELINE_CONFIG:-configs/pipeline.yaml}"
TRAIN_INDEX="${TRAIN_INDEX:-data/paper_reproduction/train.jsonl}"
VAL_INDEX="${VAL_INDEX:-data/paper_reproduction/val.jsonl}"
MODULE1_CHECKPOINT="${MODULE1_CHECKPOINT:-checkpoints/paper_reproduction/module1/best.pt}"
TRAIN_ROIS="${TRAIN_ROIS:-rois/paper_reproduction/train}"
VAL_ROIS="${VAL_ROIS:-rois/paper_reproduction/val}"
PIPELINE_OUTPUT_DIR="${PIPELINE_OUTPUT_DIR:-validation_outputs/paper_reproduction/val_pipeline}"
PIPELINE_BOOTSTRAP="${PIPELINE_BOOTSTRAP:-1000}"
RUN_PIPELINE_VALIDATION="${RUN_PIPELINE_VALIDATION:-1}"

python -m engine.train_module1 --config "${MODULE1_CONFIG}"
python scripts/generate_rois.py \
  --config "${MODULE1_CONFIG}" \
  --index "${TRAIN_INDEX}" \
  --module1-checkpoint "${MODULE1_CHECKPOINT}" \
  --output-dir "${TRAIN_ROIS}" \
  --use-gt-boxes true \
  --slice-selection-usage uniform \
  --crop-size 224 \
  --save-visualizations false \
  --storage-dtype float16
python scripts/generate_rois.py \
  --config "${MODULE1_CONFIG}" \
  --index "${VAL_INDEX}" \
  --module1-checkpoint "${MODULE1_CHECKPOINT}" \
  --output-dir "${VAL_ROIS}" \
  --use-gt-boxes false \
  --slice-selection-usage val \
  --crop-size 224 \
  --save-visualizations false \
  --storage-dtype float16
python -m engine.train_module2 --config "${MODULE2_CONFIG}"
python -m engine.train_module3 --config "${MODULE3_CONFIG}"

if [[ "${RUN_PIPELINE_VALIDATION}" == "1" ]]; then
  python -m engine.validate_pipeline \
    --config "${PIPELINE_CONFIG}" \
    --index "${VAL_INDEX}" \
    --output-dir "${PIPELINE_OUTPUT_DIR}" \
    --bootstrap "${PIPELINE_BOOTSTRAP}"
fi
