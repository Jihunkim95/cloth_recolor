#!/usr/bin/env bash
# Stage 2: cache SAM3 masks for one D-NeRF scene.
# Output: cache/sam3_dnerf/<scene>/masks.npz
#
# Usage:  scripts/sam3_cache.sh <scene>  [prompt='clothing']  [gpu=0]
#   For multi-prompt union, use comma-separated prompts:
#     scripts/sam3_cache.sh jumpingjacks "orange hoodie,sweater,hoodie,jacket"
set -euo pipefail
SCENE="${1:?usage: $0 <scene> [prompt] [gpu]}"
PROMPT="${2:-clothing}"
GPU="${3:-0}"
HERE="$(cd "$(dirname "$0")"/.. && pwd)"
ROOT="$(dirname "$HERE")"

if [[ "$PROMPT" == *","* ]]; then
  echo "[stage 2] union SAM3 cache (prompts: $PROMPT)"
  OUT="$HERE/cache/sam3_union_$SCENE"
  CUDA_VISIBLE_DEVICES=$GPU python "$HERE/2_process/sam3_cache_union.py" \
    --root "$ROOT/4DGaussians/data/dnerf/data" \
    --scene "$SCENE" --prompts "$PROMPT" --out "$OUT"
else
  echo "[stage 2] single-prompt SAM3 cache (prompt: $PROMPT)"
  OUT="$HERE/cache/sam3_dnerf"
  CUDA_VISIBLE_DEVICES=$GPU python "$HERE/2_process/sam3_cache.py" \
    --root "$ROOT/4DGaussians/data/dnerf/data" \
    --scenes "$SCENE" --prompts "$PROMPT" --out "$OUT"
fi
echo "[done] Stage 2 mask cache → $OUT/$SCENE/masks.npz"
echo "Next: scripts/cloth_logit.sh $SCENE"
