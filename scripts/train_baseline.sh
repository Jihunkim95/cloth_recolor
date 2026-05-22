#!/usr/bin/env bash
# Stage 1: Train baseline 4DGS (no cloth_logit supervision, RGB only).
# Output: ckpts at 3_output/<scene>/ckpt_baseline_hardBCE/iteration_20000/
#
# Usage:  scripts/train_baseline.sh <scene>  [gpu=0]
#   scene : jumpingjacks | standup | hellwarrior | mutant | hook | bouncingballs | lego | trex
set -euo pipefail
SCENE="${1:?usage: $0 <scene> [gpu]}"
GPU="${2:-0}"
HERE="$(cd "$(dirname "$0")"/.. && pwd)"
ROOT="$(dirname "$HERE")"
FOURDGS="$ROOT/4DGaussians"
OUT="$HERE/3_output/$SCENE/ckpt_baseline_hardBCE"
mkdir -p "$OUT"

if [ ! -f "$OUT/point_cloud/iteration_20000/point_cloud.ply" ]; then
  echo "[stage 1] training baseline 4DGS for $SCENE on GPU $GPU"
  cd "$FOURDGS"
  CUDA_VISIBLE_DEVICES=$GPU python train.py \
    -s "$FOURDGS/data/dnerf/data/$SCENE" \
    --expname "$SCENE_baseline" \
    --configs "$FOURDGS/arguments/dnerf/$SCENE.py"
  mv "$FOURDGS/output/${SCENE}_baseline" "$OUT" || true
else
  echo "[stage 1] baseline ckpt already exists: $OUT"
fi
echo "[done] Stage 1 → $OUT"
echo "Next: scripts/sam3_cache.sh $SCENE"
