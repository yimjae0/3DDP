# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Parameterization-Based Dataset Distillation for 3D Point Clouds (ICLR 2026).  
Compresses large point-cloud datasets into small synthetic sets that preserve model training quality.

## Setup

```bash
# 1. Install Python package (editable)
pip install -e .

# 2. Compile CUDA extensions (required for FPS, kNN, EMD)
cd extensions/pointops && python setup.py install && cd ../..
cd extensions/emd_     && python setup.py install && cd ../..
```

Set dataset paths via environment variables (or edit `src/point_distill/data/datasets.py`):
```bash
export MODELNET40_ROOT=/path/to/ModelNet40
export SONN_ROOT=/path/to/ScanObjectNN/main_split_nobg
export SHAPENET_ROOT=/path/to/ShapeNetv2/PointCloud
export OMNI_ROOT=/path/to/OmniObject3D
```

## Running

```bash
# Main distillation (paper's method)
python scripts/train.py \
  --dataset MODELNET40 --model PointNet --ipc 1 \
  --layer_label fReshape_nodup --mode fReshape_nodup \
  --npoints 255 --num_morph 4 --Iteration 2000

# Cross-architecture evaluation
python scripts/train.py --eval_mode CrossArchi ...
```

## Architecture

```
src/point_distill/
  data/datasets.py     — dataset loaders (ModelNet40/10, ScanObjectNN, ShapeNet, OmniObject3D)
  models/factory.py    — get_network(), get_eval_pool()
  ops/pc_ops.py        — FPS, EMD align/merge, normalization, uniformity score, voxel sort
  ops/training.py      — epoch(), evaluate_synset(), get_loops(), seed utils
  distill/losses.py    — M3DLoss (MMD + multi-scale RBF), match_loss, RFF-MMD

extensions/            — vendored point-cloud backbones + CUDA kernels
  pointnet/            — PointNet, PointNet++, DGCNN, PointConv
  point_transformer/   — Point Transformer
  pointops/            — CUDA: ball query, kNN, FPS, grouping (needs compilation)
  emd_/                — CUDA: Earth Mover's Distance (needs compilation)

scripts/
  train.py             — main distillation loop

configs/
  default.yaml         — all hyperparameters with descriptions
```

## Key Concepts

**Adaptive 3D Shape Morphing**: each synthetic sample is `args.samples` anchor chunks
(`pointcloud_syn`) blended by softmax weights (`alpha_syn`) to produce `args.num_morph`
morphed variants per IPC slot.

**Uniformity-Aware Matching Loss** (`layer_label=fReshape_nodup`):
penalty = `exp(-1000 * (u_div - u_real)²)` scales each partition's M3DLoss contribution.

**`layer_label` modes**:
- `fReshape_nodup` — paper's final method (uniformity-weighted per-partition MMD)
- `fReshape_nodup_shapenet` — ShapeNet variant with accumulated loss
- `fReshape_wo_penalty` — ablation without uniformity penalty
- `DownSample` — FPS-downsampled feature matching
- `Interpolate` — interpolated feature matching

## Important Bug Fixed

The original `main_dwkim_final_tmp.py` had a variable-shadowing bug in the evaluation loop
(line 269): `alpha_syn[c * ...]` used the outer `c` loop variable instead of `cls`,
causing every class to use class-0's alpha weights. This is fixed in `scripts/train.py`.
