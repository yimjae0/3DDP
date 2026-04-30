#!/bin/bash
# Setup script for point-distill
# Requirements: conda installed, CUDA 12.1 driver, system g++ (apt install build-essential)
#
# Usage: bash install.sh

set -e

ENV_NAME="point-distill"
RUN="conda run -n $ENV_NAME --no-capture-output"

echo "======================================================"
echo "  point-distill | Python 3.10 | PyTorch 2.2.2 | cu121"
echo "======================================================"

# 1. conda environment
echo "[1/5] Creating conda env: $ENV_NAME (Python 3.10)"
conda create -n $ENV_NAME python=3.10 -y

# 2. PyTorch 2.2.2 + CUDA 12.1
echo "[2/5] Installing PyTorch 2.2.2 + CUDA 12.1"
$RUN pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
    --index-url https://download.pytorch.org/whl/cu121

# 3. h5py and numpy via conda (handles HDF5 C library; pins numpy to 1.x for CUDA ext ABI)
echo "[3/5] Installing h5py and numpy via conda-forge"
conda install -n $ENV_NAME -c conda-forge h5py "numpy<2" -y

# 4. Python dependencies
echo "[4/5] Installing Python packages"
$RUN pip install scipy scikit-learn tqdm pandas
$RUN pip install -e .

# 5. CUDA extensions (uses system g++, no conda linker issues)
echo "[5/5] Building CUDA extensions"
echo "  -> pointops"
$RUN bash -c "cd extensions/pointops && python setup.py install"
echo "  -> emd_"
$RUN bash -c "cd extensions/emd_ && python setup.py install"

echo ""
echo "======================================================"
echo "  Done!"
echo "  Activate : conda activate $ENV_NAME"
echo "  Run      : bash scripts/run_table1.sh 0"
echo "======================================================"
