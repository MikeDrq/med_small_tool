# Module 1 Postprocess Sweep

Sorted by `f1@0.45`, matching the manuscript's max-F1 rule and IoU>0.45 TP definition.

| rank | experiment | score_threshold | topk | nms | precision@0.45 | recall@0.45 | f1@0.45 | final preds | f1@0.3 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | paper_reproduction | 0.2 | all | 0.3 | 0.8556 | 0.8235 | 0.8392 | 11527 | 0.9014 |
| 2 | paper_reproduction | 0.1 | all | 0.3 | 0.7745 | 0.8711 | 0.8200 | 13470 | 0.8911 |
| 3 | paper_reproduction | 0.3 | all | 0.3 | 0.8988 | 0.6919 | 0.7819 | 9218 | 0.8262 |
| 4 | paper_reproduction | 0.05 | all | 0.3 | 0.6940 | 0.8771 | 0.7749 | 15133 | 0.8435 |
| 5 | paper_reproduction | 0.03 | all | 0.3 | 0.6261 | 0.8786 | 0.7312 | 16803 | 0.7961 |
| 6 | paper_reproduction | 0.01 | all | 0.3 | 0.4327 | 0.8828 | 0.5808 | 24428 | 0.6321 |
