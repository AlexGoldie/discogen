#!/usr/bin/env bash
set -e

pip install -r requirements.txt

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

pip install flash-attn==2.6.3 --no-build-isolation
