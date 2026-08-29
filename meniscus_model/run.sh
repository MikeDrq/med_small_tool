# python -m scripts.convert_labeled_dataset \
#     --dicom-root ../data_full/dicom/10299 \
#     --dicom-root ../data_full/dicom/10312 \
#     --dicom-root ../data_full/dicom/10424 \
#     --dicom-root ../data_full/dicom/10593 \
#     --annotation-root ../data_full/json \
#     --output-jsonl data/paper_reproduction_full/index_all.jsonl \
#     --report-json data/paper_reproduction_full/conversion_report.json \
#     --box-format center-wh \
#     --bucket-handle-code t3 \
#     --scan-workers 16

# python -m scripts.split_index \
#     --input-jsonl data/paper_reproduction_full/index_all.jsonl \
#     --output-dir data/paper_reproduction_full \
#     --group-key patient_group_id \
#     --seed 42

# python -m scripts.audit_paper_dataset \
#     data/paper_reproduction_full/train.jsonl \
#     data/paper_reproduction_full/val.jsonl \
#     data/paper_reproduction_full/test.jsonl \
#     --annotation-root ../data_full/json \
#     --output data/paper_reproduction_full/audit_splits.json

python -m engine.train_module1 \
    --config configs/module1_meniscovit_cop_full.yaml

python scripts/generate_rois.py \
    --config configs/module1_meniscovit_cop_full.yaml \
    --index data/paper_reproduction_full/train.jsonl \
    --module1-checkpoint checkpoints/paper_reproduction_full/module1/best.pt \
    --output-dir rois/paper_reproduction_full/train \
    --use-gt-boxes true \
    --slice-selection-usage uniform \
    --crop-size 224 \
    --storage-dtype float16 \
    --num-workers 4

python scripts/generate_rois.py \
    --config configs/module1_meniscovit_cop_full.yaml \
    --index data/paper_reproduction_full/val.jsonl \
    --module1-checkpoint checkpoints/paper_reproduction_full/module1/best.pt \
    --output-dir rois/paper_reproduction_full/val \
    --use-gt-boxes false \
    --slice-selection-usage val \
    --crop-size 224 \
    --storage-dtype float16 \
    --num-workers 4

python -m engine.train_module2 \
    --config configs/module2_swin_full.yaml

python -m engine.train_module3 \
    --config configs/module3_bucket_swin_full.yaml