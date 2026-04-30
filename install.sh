#!/bin/bash
# Full environment setup for point-distill
# Requirements: conda, CUDA 12.1 driver already installed on the machine
#
# Usage:
#   bash install.sh

set -e

ENV_NAME="point-distill"

echo "============================================"
echo "  point-distill environment setup"
echo "  CUDA 12.1 / Python 3.10 / PyTorch 2.1.2"
echo "============================================"

# ---------- 1. conda environment ----------
echo "[1/5] Creating conda environment: $ENV_NAME"
conda create -n $ENV_NAME python=3.10 -y

# conda activate는 스크립트 안에서 직접 안 됨 — conda run으로 대신 실행
RUN="conda run -n $ENV_NAME --no-capture-output"

# ---------- 2. PyTorch (CUDA 12.1) ----------
echo "[2/5] Installing PyTorch 2.1.2 with CUDA 12.1"
$RUN pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu121

# ---------- 3. GCC (CUDA 확장 빌드용) ----------
# 시스템 GCC가 오래됐을 때 conda-forge에서 설치
echo "[3/5] Installing GCC 12 via conda-forge (CUDA extension build)"
conda install -n $ENV_NAME -c conda-forge gcc=12 gxx=12 -y

# ---------- 4. Python 의존성 ----------
echo "[4/5] Installing Python dependencies"
$RUN pip install -r requirements.txt
$RUN pip install -e .

# ---------- 5. CUDA 확장 빌드 ----------
echo "[5/5] Building CUDA extensions"

# pointops
echo "  -> pointops"
$RUN bash -c "cd extensions/pointops && python setup.py install"

# emd_
echo "  -> emd_"
$RUN bash -c "cd extensions/emd_ && python setup.py install"

echo ""
echo "============================================"
echo "  Done! Activate the environment with:"
echo "    conda activate $ENV_NAME"
echo ""
echo "  Then run Table 1 experiments:"
echo "    bash scripts/run_table1.sh 0    # GPU 0"
echo "============================================"
