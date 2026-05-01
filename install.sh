#!/bin/bash
# Setup script for point-distill
# Requirements: conda, CUDA 12.1 driver
#
# Usage: bash install.sh

set -e

ENV_NAME="point-distill"
RUN="conda run -n $ENV_NAME --no-capture-output"

echo "======================================================"
echo "  point-distill | Python 3.10 | PyTorch 2.2.2 | cu121"
echo "======================================================"

# 1. conda environment
echo "[1/6] Creating conda env: $ENV_NAME (Python 3.10)"
conda create -n $ENV_NAME python=3.10 -y

# 2. PyTorch 2.2.2 + CUDA 12.1
echo "[2/6] Installing PyTorch 2.2.2 + CUDA 12.1"
$RUN pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
    --index-url https://download.pytorch.org/whl/cu121

# 3. gcc 12 + h5py + numpy via conda-forge
#    gcc: system gcc is too old for PyTorch extensions (need >= 9)
#    h5py: requires HDF5 C lib — conda handles it
#    numpy<2: CUDA extensions compiled against NumPy 1.x ABI
echo "[3/6] Installing gcc=12, h5py, numpy<2 via conda-forge"
conda install -n $ENV_NAME -c conda-forge gcc=12 gxx=12 h5py "numpy<2" -y

# 4. Fix conda linker compat on Debian/Ubuntu
#    conda's ld wrapper looks for /lib64/libpthread.so.0 (RHEL path),
#    but on Debian/Ubuntu it lives in /lib/x86_64-linux-gnu/
echo "[4/6] Fixing linker paths for Debian/Ubuntu"
if [ ! -e /lib64/libpthread.so.0 ] && [ -f /lib/x86_64-linux-gnu/libpthread.so.0 ]; then
    mkdir -p /lib64
    ln -sf /lib/x86_64-linux-gnu/libpthread.so.0 /lib64/libpthread.so.0
fi
_nsa=$(find /usr/lib -name "libpthread_nonshared.a" 2>/dev/null | head -1)
if [ -n "$_nsa" ] && [ ! -e /usr/lib64/libpthread_nonshared.a ]; then
    mkdir -p /usr/lib64
    ln -sf "$_nsa" /usr/lib64/libpthread_nonshared.a
fi

# 5. Python dependencies
echo "[5/6] Installing Python packages"
$RUN pip install scipy scikit-learn tqdm pandas "numpy<2"
$RUN pip install -e .

# 6. CUDA extensions
echo "[6/6] Building CUDA extensions"
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
