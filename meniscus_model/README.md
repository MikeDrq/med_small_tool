# Meniscus MRI paper reproduction

本仓库复现当前目录中的论文与补充材料所描述的四模块半月板 MRI 流程。默认配置面向单张 H100 80GB，并明确区分“论文公开参数”和“因原文缺少代码而做的工程推断”。

## 本次修复的核心问题

旧结果差的首要原因不是单纯的调参，而是标签和模型契约错误：

- 源标注 `t10.menicustype` 的 `t1/t2` 表示“损伤/变性”，旧转换器却将其当成“内/外侧”，造成 Module 2 标签系统性错位。
- 病灶侧别现在通过同一 DICOM instance 上的 `t10` 病灶框与 `t9` 半月板框几何匹配恢复；不会再从 `menicustype` 猜侧别。
- 旧代码把异常侧的所有切片都标为 abnormal；现在只标记病灶框实际重叠的半月板框，保留切片级异常定位监督。
- 划分由 case 随机切分改为 `patient_group_id` 分组，避免同一患者的重复 MRI 泄漏到不同 split。
- DICOM 强度预处理改为补充材料的 `WindowCenter ± WindowWidth/2` 截断公式，并保留缺失窗宽窗位时的 percentile fallback。
- Module 1 从 `224/patch16/单尺度 8 层` 改为论文 Figure 8/9 的四阶段双平面 UniFormer：`512×512`、维度 `[64,128,320,512]`、深度 `[3,4,8,3]`、初始 `3×4×4 stride 2×4×4`、每阶段 RCPE 与 `k=32` cross-plane attention。
- Module 1 bbox 的旧 `softplus` 偏移被错误地按整幅图解释，初始化时大量框直接贴边；现在按 feature-grid cell 单位解码。
- 旧 objectness focal loss 被全部约 32K 个密集 token 直接平均，真实 batch 中正样本损失从 `1.13` 被稀释到 `0.0045`；现在按正样本数归一化（保留 `all_tokens` 兼容开关），恢复有效的定位梯度。
- Module 2/3 从两个小型独立分支改为论文 Figure 10 的单个标准 Video Swin-T，参数量与论文一致，并使用本地 Kinetics-400 初始化。
- 论文增强（随机翻转、±30°旋转、±0.2 亮度、σ=0.02 噪声）以每个 volume 一次 `p=0.5` 总开关接入；Module 1 的 bbox 和 DICOM 世界坐标网格会与空间变换同步更新。
- Module 3 只在 ground-truth tear 半月板中训练和验证，匹配论文流程的条件分支，避免模型靠“正常 vs 撕裂”捷径虚高结果。
- 原文只写了 bootstrap 1,000 次，并未说明重采样单位。当前采用更保守的 `patient_group_id` 整组抽样；同一患者的重复 MRI、双侧半月板和多个 token/框不会被拆开。Module 1 的 pooled token PR-AUC 保留点估计，其 CI 报告病例宏平均 PR-AUC，避免对数百万 token 伪重复抽样。
- Module 1 的验证集 max-F1 扫描在论文规定的 `IoU>0.45` 下选出 `score_threshold=0.2`，与 Figure S1 一致；推理、ROI 生成和单模块验证共用同一 NMS/阈值。

## 与论文参数的对应

| 模块 | 论文 | 当前默认实现 |
|---|---:|---:|
| MeniscoViT-CoP 参数量 | 51.3M | 52.39M（含检测/质量头） |
| Module 1 输入 | 全序列，512×512 | `T=32`，512×512；T 为未公开推断值 |
| Module 1 训练 | 15 epochs，batch 8 | 15 epochs，`2×grad_accum 4` |
| Module 2 参数量/训练 | 27.8M / 40 epochs | 27.85M / 40 epochs，`8×grad_accum 2` |
| Module 3 参数量/训练 | 27.8M / 25 epochs | 27.85M / 25 epochs，`8×grad_accum 2` |
| 优化器 | AdamW, lr 1e-4, warmup 2, cosine | 一致 |
| weight decay | Module 1: .005；Module 2/3: .05 | 一致 |
| Module 4 | `beta=w2/w1 > .4` | 规则一致；`w1` 由 `seg/` 冠状位 U2Net 自动生成 |

仍不能从论文中唯一确定、因而没有伪装成“完全一致”的部分：

- 论文没有公开固定序列长度、ROI 尺寸、预训练来源、检测 target/loss、NMS 或缺失切片 ROI 填充方法。当前的 `T=32`、ROI `224×224`、Kinetics-400 初始化、anchor-free target、bbox-quality head、`NMS=0.3` 和 nearest fill 都是工程选择。
- Module 1 为控制 H100 上的显存和计算量，使用了 4,096-token cross-attention 上限和 factorized global attention；这两点未在原文出现，也使参数量从论文的 51.3M 变为 52.39M。
- Module 2 正文描述为“两平面分别池化后拼接特征”，Figure 10 却画成“两平面 channel concatenate 后进入一个 Video Swin”。当前按 Figure 10 和 27.8M 参数量实现，并补了第三个均值通道以加载标准三通道预训练权重。
- 原文没有说明 Module 2/3 训练时使用 GT ROI 还是 Module 1 预测 ROI。当前训练使用 GT ROI、验证/测试使用预测 ROI，因此存在论文无法核实的 train-test ROI domain gap。
- 原文没有说明 bootstrap 单位、Module 1 detection specificity 的 TN 定义、PR-AUC 是 trapezoidal AUC 还是 average precision，也没有说明侧别/异常指标是在 GT-assigned token 还是匹配后的检测框上计算。当前实现分别采用患者分组 bootstrap、average precision 和 GT-assigned positive token；检测 specificity 不凭空构造。
- Figure S1 中盘状半月板阈值图示疑似为 0.25，但 Methods 明确写 `beta>0.4`；当前以 Methods 的 0.4 为准。

论文训练集有 6,681 例，并使用三个独立测试集。当前目录有 3,859 个标注文件（3,858 个有效 study），但只有 **1,736** 个 study 同时有可读 sagittal/coronal DICOM；2,122 个有效标注没有可用的完整双平面影像，实际覆盖率为 **45.0%**。可用数据按患者分组划为 train/val/test = `1215/260/261`，患者组交集均为 0。当前也没有论文的三个独立外部测试集，因此不能把本地 split 指标直接等同于论文报告值。详细证据见 `data/paper_reproduction/audit_splits.json`。

## 源标注映射

- `t9.position=t1/t2`：medial/lateral。
- `t9.structure=t1`：discoid；`t2`：non-discoid。
- `t10.menicustype=t1/t2`：tear/degeneration。
- `t10.lesiontype.t3`：当前作为 bucket-handle。这个私有枚举未在论文中公开，转换报告会明确记录该假设，可用 `--bucket-handle-code` 修改。

## 环境

本机已使用 `shine` 环境验证：PyTorch 2.5.1 + CUDA 12.4、torchvision 0.20.1、pydicom 3.0.2。

```bash
conda activate shine
cd /mnt/aoss/dongruiqi/small_tool/meniscus_model
pip install -r requirements.txt
```

官方 UniFormer-S 与 torchvision Video Swin-T 权重保存在被 git 忽略的 `pretrained/`：

```bash
mkdir -p pretrained
curl -L -o pretrained/uniformer_small_k400_16x8.pth \
  https://huggingface.co/Sense-X/uniformer_video/resolve/main/uniformer_small_k400_16x8.pth
curl -L -o pretrained/swin3d_t-7615ae03.pth \
  https://download.pytorch.org/models/swin3d_t-7615ae03.pth
```

期望 SHA-256：

- UniFormer-S：`d5fd7b0c49ee6a5422ef5d0c884d962c742003bfbd900747485eb99fa269d0db`；329 个非分类头 tensor 会严格映射到两个平面分支，共加载 658 个 tensor、缺失 0。
- Swin3D-T：`7615ae035996b65eb38dad437ae533d2dfcd36f9f89d28c0f0fa7bfb8e6b3130`。

## 数据准备

从当前工作区的导出数据生成索引：

```bash
python -m scripts.convert_labeled_dataset \
  --dicom-root ../data/dicom/10299 \
  --dicom-root ../data/json/10299 \
  --annotation-root ../data/json/10299 \
  --output-jsonl data/paper_reproduction/index_all.jsonl \
  --report-json data/paper_reproduction/conversion_report.json \
  --box-format center-wh \
  --bucket-handle-code t3 \
  --scan-workers 16

python -m scripts.split_index \
  --input-jsonl data/paper_reproduction/index_all.jsonl \
  --output-dir data/paper_reproduction \
  --group-key patient_group_id \
  --seed 42

python -m scripts.audit_paper_dataset \
  data/paper_reproduction/train.jsonl \
  data/paper_reproduction/val.jsonl \
  data/paper_reproduction/test.jsonl \
  --annotation-root ../data/json/10299 \
  --output data/paper_reproduction/audit_splits.json
```

当前审计结果：三个 split 的 `patient_group_overlap=0`、`invalid_boxes=0`、标签逻辑冲突为 0；23,041 个 lesion box 中只有 1 个无法与同 instance 的半月板框匹配。

## 训练与验证

H100 论文尺寸冒烟测试：

```bash
python scripts/smoke_paper_models.py --module1-backward --module1-batch-size 2
python -m scripts.smoke_real_module1_step --config configs/module1_meniscovit_cop.yaml
```

真实 DICOM 实测：Module 1 batch 2 的完整优化步骤为 1.92 秒、峰值 39.06 GiB，91/91 个原始框精确映射；修复后的初始 `loss_objectness=1.1311`，总损失有限。真实 ROI 的 Module 2 batch 8 完整反向为 0.70 秒、峰值 20.09 GiB。

### 当前完整训练结果

Module 1 已完成 15 epochs，best 为 epoch 15。验证集阈值扫描按论文的 `IoU>0.45` 最大 F1 规则选择 `0.2`，得到 P/R/F1=`0.8556/0.8235/0.8392`；按 242 个患者组 bootstrap 1,000 次的 95% CI 分别为 `(0.8371,0.8727)`、`(0.8054,0.8426)`、`(0.8208,0.8566)`。原先 `0.05` 的 F1 仅为 `0.7749`。完整扫描保存在 `outputs/paper_reproduction/module1_threshold_sweep/`。

Module 2 已完成 40 epochs。现有按总体 accuracy 保存的 checkpoint 为 `0.7750`，但 degeneration F1 仅 `0.1778`；配置已改为按 `macro_f1` 保存后续重训的 best，避免多数类掩盖 degeneration。Module 3 已完成 25 epochs，best 为 epoch 6，overall ROC-AUC=`0.9262`、F1=`0.5714`。这些均为本地内部验证结果，不能与论文的独立测试集直接等同。

依次运行完整流程：

```bash
bash scripts/train_all.sh
```

或分步运行：

```bash
python -m engine.train_module1 --config configs/module1_meniscovit_cop.yaml

python scripts/generate_rois.py \
  --config configs/module1_meniscovit_cop.yaml \
  --index data/paper_reproduction/train.jsonl \
  --module1-checkpoint checkpoints/paper_reproduction/module1/best.pt \
  --output-dir rois/paper_reproduction/train \
  --use-gt-boxes true --slice-selection-usage uniform \
  --crop-size 224 --storage-dtype float16 \
  --save-visualizations false

python scripts/generate_rois.py \
  --config configs/module1_meniscovit_cop.yaml \
  --index data/paper_reproduction/val.jsonl \
  --module1-checkpoint checkpoints/paper_reproduction/module1/best.pt \
  --output-dir rois/paper_reproduction/val \
  --use-gt-boxes false --slice-selection-usage val \
  --crop-size 224 --storage-dtype float16 \
  --save-visualizations false

python -m engine.train_module2 --config configs/module2_swin.yaml
python -m engine.train_module3 --config configs/module3_bucket_swin.yaml
bash scripts/validate_all.sh
```

GT ROI 默认使用 uniform slice selection，避免旧的 label-chunk 模式用相同 `case_id.pt` 重复覆盖。ROI 以 float16 缓存，加载时恢复为 float32；裁剪已按整个 slice batch 向量化。本机挂载的 DICOM 存储在 DataLoader 多进程下会等待，因此默认可靠值是 `--num-workers 0`；本地 SSD 数据可显式提高该值。

## Module 4

论文 Module 4 依赖先前研究的分割模型得到胫骨宽度 `w1`。当前数据没有 tibia mask，但同级 `seg/` 目录提供了三分类冠状位 U2Net 代码与权重（0=背景、1=股骨、2=胫骨）。当前实现按以下顺序取得 `w1`：

- 传入 `module4_labels.tibia_width`；
- 传入 `.npy/.pt` tibia mask；
- 自动运行 `seg/save_weights/model_cor_multiclass_2024424_best.pth` 并只测量类别 2；
- 单病例推理仍可用 `--tibia-width` 显式覆盖自动分割。

端到端运行 `engine.infer_pipeline` 时，Module 1 冠状位外侧半月板框给出 `w2`，U2Net 给出 `w1`，最终严格应用论文规则 `beta=w2/w1 > 0.4`。Figure 11 明确把 `w1`、`w2` 画在同一冠状位切片上，因此当前先保留 `w1` 不低于病例最大平台宽度 95% 的近中央层，默认再取其中最大的真实 `w2` 并同层计算 beta；这会排除只显示单侧髁、导致 `w1` 偏小的边缘层。95% 是原文未披露的工程选择。fallback/填充框不参与解剖测量，U2Net mask 每层只保留最大连通域。分割权重没有随附原训练集上的定量验证结果，所以其迁移可靠性仍需用少量人工 tibia mask 校验。

病例级选层支持 `max_w2`、`max_w1`、`median_beta`、`q25_beta` 和 `central3_median`。默认仍为旧行为 `max_w2`。只能在验证集选择策略；测试集用于锁定规则后的最终一次评估。一次U2Net推理同时比较GT框和预测框：

```bash
python -m scripts.audit_module4_strategies \
  --config configs/pipeline_full.yaml \
  --index data/paper_reproduction_full/val.jsonl \
  --gt-roi-cache-dir rois/paper_reproduction_full/val_gt_module4 \
  --pred-roi-cache-dir rois/paper_reproduction_full/val \
  --output-dir outputs/module4_ab/strategy_sweep \
  --strategies max_w2 max_w1 median_beta q25_beta central3_median \
  --plateau-fractions 0.90 0.95 1.00 \
  --bootstrap 0
```

逐层宽度保存在 `per_slice_widths.jsonl`；修改策略后可通过 `--reuse-width-cache --width-cache ...` 跳过DICOM读取和U2Net，并把1000次bootstrap结果写入另一个目录。`selected_strategy.json` 按验证集预测框ROC-AUC给出候选策略。锁定后也可使用 `validate_module4 --aggregation-strategy ... --plateau-fraction ... --output-dir ... --bootstrap 1000` 计算最终CI。

## 测试

```bash
python -m pytest -q
```

当前完整测试集：116 passed。主要入口见 `configs/`、`engine/`、`scripts/train_all.sh` 和 `scripts/validate_all.sh`。
