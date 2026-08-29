python -m engine.validate_module1 \
    --config configs/module1_meniscovit_cop_full.yaml \
    --checkpoint checkpoints/paper_reproduction_full/module1/best.pt \
    --index data/paper_reproduction_full/test.jsonl \
    --bootstrap 1000

python scripts/generate_rois.py \
    --config configs/module1_meniscovit_cop_full.yaml \
    --index data/paper_reproduction_full/test.jsonl \
    --module1-checkpoint checkpoints/paper_reproduction_full/module1/best.pt \
    --output-dir rois/paper_reproduction_full/test \
    --use-gt-boxes false \
    --slice-selection-usage val \
    --crop-size 224 \
    --save-visualizations false \
    --storage-dtype float16 \
    --num-workers 4

python -m engine.validate_module2 \
    --config configs/module2_swin_full.yaml \
    --checkpoint checkpoints/paper_reproduction_full/module2/best.pt \
    --roi-index rois/paper_reproduction_full/test/index.jsonl \
    --bootstrap 1000

python -m engine.validate_module3 \
    --config configs/module3_bucket_swin_full.yaml \
    --checkpoint checkpoints/paper_reproduction_full/module3_f1/best.pt \
    --roi-index rois/paper_reproduction_full/test/index.jsonl \
    --bootstrap 1000

python -m engine.validate_pipeline \
    --config configs/pipeline_full.yaml \
    --index data/paper_reproduction_full/test.jsonl \
    --output-dir validation_outputs/paper_reproduction_full/test_set \
    --bootstrap 1000