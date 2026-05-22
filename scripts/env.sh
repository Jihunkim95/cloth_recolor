# source this BEFORE running any train/recolor/viewer script.
# Exports CUDA env vars so gsplat's cloth_logit 2nd-pass JIT compile succeeds.
# Usage:  source scripts/env.sh
[ -z "${CONDA_PREFIX:-}" ] && { echo "[env] activate your conda env first"; return 1; }
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:$PATH"
export CPATH="$CONDA_PREFIX/targets/x86_64-linux/include:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_runtime/include:${CPATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
echo "[env] CUDA_HOME=$CUDA_HOME"
