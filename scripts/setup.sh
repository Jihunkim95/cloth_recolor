#!/usr/bin/env bash
# Setup ClothSplat dependencies (conda env + 4DGaussians fork + CUDA env).
# Usage:  scripts/setup.sh  [env_name=clothsplat]
#
# Prerequisites:
#   - conda (miniconda / anaconda)
#   - CUDA 12.8+ toolkit (for sm_100 Blackwell; older GPUs use CUDA 12.1+)
#   - git
set -euo pipefail
ENV="${1:-clothsplat}"
HERE="$(cd "$(dirname "$0")"/.. && pwd)"
ROOT="$(dirname "$HERE")"

# 0) sibling 4DGaussians fork (hustvl/4DGaussians + our cloth_logit mods)
if [ ! -d "$ROOT/4DGaussians" ]; then
  echo "[setup] cloning sibling 4DGaussians/ ..."
  git clone https://github.com/hustvl/4DGaussians.git "$ROOT/4DGaussians"
  echo "[setup] WARN: this is upstream 4DGaussians. Apply our cloth_logit patches"
  echo "         (scene/gaussian_model.py, gaussian_renderer/__init__.py,"
  echo "          utils/soft_cal.py) — see docs/PATCHES_TO_4DGAUSSIANS.md"
fi

# 1) conda env
if ! conda env list | grep -q "^$ENV "; then
  echo "[setup] creating conda env: $ENV (python 3.11)"
  conda create -y -n "$ENV" python=3.11
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"

# 2) CUDA toolkit inside env (needed for gsplat JIT compile to sm_100)
echo "[setup] installing CUDA toolkit + nvcc inside env"
conda install -y -c nvidia 'cuda-toolkit=12.8' 'cuda-nvcc=12.8' || true

# 3) PyTorch (cu128 build)
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 4) gsplat (CUDA JIT)
pip install gsplat==1.5.3

# 5) 4DGaussians requirements
pip install -r "$ROOT/4DGaussians/requirements.txt" || true
pip install mmengine open3d plyfile lpips pytorch_msssim matplotlib opencv-python

# 6) SAM3 + DINOv3 (HuggingFace)
pip install 'transformers>=4.40' accelerate safetensors

# 7) viser (interactive viewer)
pip install viser

echo
echo "[setup] DONE. Activate env and source env vars:"
echo "    conda activate $ENV"
echo "    source scripts/env.sh   # exports CUDA_HOME, CPATH, LD_LIBRARY_PATH"
echo
echo "Next: scripts/download_data.sh    # fetch D-NeRF Blender data"
