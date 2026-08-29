#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: scripts/infer_case.sh CASE_ID /path/sagittal /path/coronal /path/output [tibia_width]" >&2
  exit 1
fi

CASE_ID="$1"
SAGITTAL_DIR="$2"
CORONAL_DIR="$3"
OUTPUT_DIR="$4"
TIBIA_WIDTH="${5:-}"

CMD=(
  python -m engine.infer_pipeline
  --config configs/pipeline.yaml
  --case-id "${CASE_ID}"
  --sagittal-dicom-dir "${SAGITTAL_DIR}"
  --coronal-dicom-dir "${CORONAL_DIR}"
  --output-dir "${OUTPUT_DIR}"
)

if [[ -n "${TIBIA_WIDTH}" ]]; then
  CMD+=(--tibia-width "${TIBIA_WIDTH}")
fi

"${CMD[@]}"
