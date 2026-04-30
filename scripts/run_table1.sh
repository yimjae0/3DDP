#!/bin/bash
# Reproduces Table 1 from the paper.
# 5 datasets × IPC {1, 3, 10} = 15 experiments, all evaluated on PointNet.
#
# Usage:
#   bash scripts/run_table1.sh [GPU_ID]
#   e.g., CUDA_VISIBLE_DEVICES=0 bash scripts/run_table1.sh
#   or    bash scripts/run_table1.sh 0

GPU=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU

CMD="python scripts/train.py \
  --model PointNet \
  --init real \
  --Iteration 2000 \
  --num_eval 10 \
  --num_exp 1 \
  --batch_real 8 \
  --batch_train 8 \
  --lr_net 0.01 \
  --num_morph 4 \
  --epoch_eval_train 500"

echo "=========================================="
echo "Table 1 Reproduction — GPU $GPU"
echo "=========================================="

# ---------- ModelNet10 (npoints=252, per paper) ----------
for IPC in 1 3 10; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ModelNet10 IPC=$IPC"
  $CMD --dataset MODELNET10 --ipc $IPC --npoints 252
done

# ---------- ModelNet40 ----------
for IPC in 1 3 10; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ModelNet40 IPC=$IPC"
  $CMD --dataset MODELNET40 --ipc $IPC --npoints 255
done

# ---------- ShapeNet ----------
for IPC in 1 3 10; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ShapeNet IPC=$IPC"
  $CMD --dataset shapenet --ipc $IPC --npoints 255
done

# ---------- ScanObjectNN ----------
for IPC in 1 3 10; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ScanObjectNN IPC=$IPC"
  $CMD --dataset scanobjectnn --ipc $IPC --npoints 255
done

# ---------- OmniObject3D ----------
for IPC in 1 3 10; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] OmniObject3D IPC=$IPC"
  $CMD --dataset omni --ipc $IPC --npoints 255
done

echo "=========================================="
echo "All Table 1 experiments done."
echo "Results saved to ./result/"
echo "=========================================="
