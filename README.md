# 3DDP

Official implementation of **"Parameterization-Based Dataset Distillation for 3D Point Cloud Classification"** (ICLR 2026).

[[Paper]](https://iclr.cc/virtual/2026/poster/10009568)

We synthesize a small set of point clouds that faithfully preserves the information of a full training set, enabling networks trained only on the distilled data to match or approach full-data accuracy.

---

## Method overview

**Adaptive 3D Shape Morphing** represents each class with *anchor chunks* that are linearly combined via learned softmax weights to produce morphed training samples, drastically reducing the number of stored parameters while expanding effective training diversity.

The distillation objective matches per-partition intermediate features between real and synthetic data, weighted by a uniformity score that penalises unevenly distributed chunks (*fReshape_nodup* loss).

---

## Installation

Requires a machine with a CUDA 12.1-compatible GPU and [conda](https://docs.conda.io/).

```bash
git clone git@github.com:yimjae0/point-distill.git
cd point-distill
bash install.sh
```

`install.sh` creates a conda environment called `point-distill` with Python 3.10, PyTorch 2.2.2 + CUDA 12.1, and builds the two required CUDA extensions (`pointops`, `emd_`).

---

## Data preparation

Download the datasets and set environment variables to point to them before running:

| Dataset | Environment variable | Default path |
|---|---|---|
| ModelNet40 / ModelNet10 | `MODELNET40_ROOT` | `/root/dataset/ModelNet40` |
| ScanObjectNN | `SONN_ROOT` | `/root/dataset/ScanObjectNN/main_split_nobg` |
| ShapeNet | `SHAPENET_ROOT` | `/root/dataset/ShapeNetv2/PointCloud` |
| OmniObject3D | `OMNI_ROOT` | `/root/dataset/OmniObject3D` |

Example:
```bash
export MODELNET40_ROOT=/data/ModelNet40
```

---

## Reproducing Table 1

```bash
conda activate point-distill
bash scripts/run_table1.sh 0   # 0 = GPU id
```

This runs all 15 experiments (5 datasets × IPC ∈ {1, 3, 10}) sequentially on a single GPU and saves results to `./result/`.

### Key hyperparameters

| Argument | Description | Default |
|---|---|---|
| `--ipc` | Synthetic samples per class | 1 |
| `--npoints` | Points per anchor chunk | 255 (252 for ModelNet10) |
| `--num_morph` | Morphed samples per IPC slot | 4 (16 for ModelNet10) |
| `--Iteration` | Distillation iterations | 2000 |
| `--init` | Initialisation strategy (`real` / `noise`) | `real` |
| `--num_eval` | Evaluation runs to average | 10 |
| `--epoch_eval_train` | Epochs for evaluation training | 500 |

### Single experiment

```bash
conda activate point-distill
export PYTHONPATH="$(pwd)/extensions:$PYTHONPATH"

python scripts/train.py \
  --dataset MODELNET40 \
  --model PointNet \
  --ipc 1 \
  --npoints 255 \
  --num_morph 4 \
  --Iteration 2000 \
  --num_eval 10 \
  --epoch_eval_train 500 \
  --init real \
  --batch_real 8 \
  --batch_train 8 \
  --lr_net 0.01
```

---

## Repository structure

```
point-distill/
├── install.sh                   # one-shot environment setup
├── scripts/
│   ├── train.py                 # main distillation script
│   └── run_table1.sh            # Table 1 reproduction
├── src/point_distill/
│   ├── data/datasets.py         # dataset loaders
│   ├── distill/losses.py        # M3D loss
│   ├── models/factory.py        # backbone factory
│   └── ops/
│       ├── pc_ops.py            # FPS, EMD alignment, uniformity score
│       └── training.py          # evaluation loop
└── extensions/
    ├── pointops/                # CUDA: KNN, ball query, FPS, grouping
    └── emd_/                    # CUDA: Earth Mover's Distance
```

---

## Supported backbones

`PointNet`, `PointNetPlusPlus`, `DGCNN`, `PointConvDensityClsSsg`, `PointTransformerCls`

Pass via `--model`.


---

## Citation

```bibtex
@inproceedings{pointdistill2026,
  title     = {Parameterization-Based Dataset Distillation for 3D Point Cloud Classification},
  booktitle = {ICLR},
  year      = {2026},
}
```
