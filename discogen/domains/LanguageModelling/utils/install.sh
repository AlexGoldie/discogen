#!/usr/bin/env bash
set -e

pip install -r requirements.txt

# Ensure CUDA toolkit (nvcc) exists
if ! command -v nvcc &>/dev/null; then
    echo "nvcc not found. Please install CUDA toolkit first."
    echo "Ubuntu example:"
    echo "sudo apt install nvidia-cuda-toolkit"
    exit 1
fi

# Detect CUDA installation path
CUDA_HOME=$(dirname $(dirname $(which nvcc)))

export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export CPATH="$CUDA_HOME/include:${CPATH:-}"
export LIBRARY_PATH="$CUDA_HOME/lib64:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# Torch library path
TORCH_LIB=$(python -c 'import torch, os; print(os.path.join(torch.__path__[0], "lib"))')
export LD_LIBRARY_PATH="$TORCH_LIB:${LD_LIBRARY_PATH:-}"

# Force build from source
export CAUSAL_CONV1D_FORCE_BUILD=TRUE
export MAMBA_FORCE_BUILD=TRUE
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;10.0+ptx"

echo "CUDA_HOME=$CUDA_HOME"
echo "nvcc: $(which nvcc)"
nvcc --version

# Faster compilation
pip install ninja

# Build against installed torch
pip install "causal-conv1d>=1.4.0" --no-build-isolation --no-cache-dir
pip install "mamba-ssm>=2.0.0" --no-build-isolation --no-cache-dir
