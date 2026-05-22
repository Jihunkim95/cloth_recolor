#!/usr/bin/env bash
# Stage 3: Per-Gaussian projection supervision (exp010 breakthrough).
# Loads baseline 4DGS + SAM3 cache, computes per-Gaussian soft_target via
# projection, then trains cloth_logit with BCE (2000 Adam steps).
#
# Usage:  scripts/cloth_logit.sh <scene>  [cache_dir=cache/sam3_dnerf]  [gpu=0]
set -euo pipefail
SCENE="${1:?usage: $0 <scene> [cache_dir] [gpu]}"
CACHE_DIR="${2:-cache/sam3_dnerf}"
GPU="${3:-0}"
HERE="$(cd "$(dirname "$0")"/.. && pwd)"
BASE_CKPT="$HERE/3_output/$SCENE/ckpt_baseline_hardBCE"
OUT="$HERE/3_output/$SCENE/ckpt_exp010_per_gaussian"

if [ ! -d "$BASE_CKPT/point_cloud/iteration_20000" ]; then
  echo "[err] baseline ckpt missing — run scripts/train_baseline.sh $SCENE first"
  exit 1
fi
rm -rf "$OUT"
echo "[stage 3] cloth_logit training for $SCENE (cache=$CACHE_DIR)"
CUDA_VISIBLE_DEVICES=$GPU python "$HERE/2_process/per_gaussian_supervision.py" \
  --base-ckpt "$BASE_CKPT" --base-iter 20000 \
  --sam3-cache "$HERE/$CACHE_DIR" --scene "$SCENE" \
  --out "$OUT" --iters 2000 --lr 0.05
echo "[done] Stage 3 → $OUT"
echo "Next: scripts/recolor.sh $SCENE  [hue=220]  [threshold=0.7]"
