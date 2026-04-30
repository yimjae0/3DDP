#!/bin/bash
# Setup script for point-distill
# Requirements: conda installed, CUDA 12.1 driver on the machine
#
# Usage: bash install.sh

set -e

ENV_NAME="point-distill"
RUN="conda run -n $ENV_NAME --no-capture-output"

echo "======================================================"
echo "  point-distill | Python 3.8 | PyTorch 2.2.2 | cu121"
echo "======================================================"

# 1. conda environment
echo "[1/5] Creating conda env: $ENV_NAME (Python 3.8)"
conda create -n $ENV_NAME python=3.8 -y

# 2. PyTorch 2.2.2 + CUDA 12.1
#    (2.2.x: last version supporting Python 3.8, no pkg_resources dependency)
echo "[2/5] Installing PyTorch 2.2.2 + CUDA 12.1"
$RUN pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
    --index-url https://download.pytorch.org/whl/cu121

# 3. System-level packages via conda
#    gcc/gxx : CUDA extension build (avoids old system GCC issue)
#    h5py    : requires HDF5 lib — conda handles the C dependency
#    numpy<2 : CUDA extensions compiled against NumPy 1.x API
echo "[3/5] Installing gcc, h5py, numpy, libxcrypt via conda-forge"
# libxcrypt: provides crypt.h required by Python 3.8 headers on modern Linux
conda install -n $ENV_NAME -c conda-forge gcc=12 gxx=12 h5py "numpy<2" libxcrypt -y

# 4. Python dependencies
echo "[4/5] Installing Python packages"
$RUN pip install scipy scikit-learn tqdm pandas
$RUN pip install -e .

# 5. CUDA extensions
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
