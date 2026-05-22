#!/usr/bin/env bash
# Stage 4: Recolor cloth Gaussians via SH_DC HSV swap, render PNG panels + PLY.
#
# Usage:  scripts/recolor.sh <scene>  [hue=220]  [threshold=0.7]  [gpu=0]
#   D-NeRF sweet spot: threshold 0.7 (per-Gaussian projection, bimodal dist)
#   N3V sweet spot:    threshold 0.3 (spatial filter, compressed dist)
set -euo pipefail
SCENE="${1:?usage: $0 <scene> [hue=220] [threshold=0.7] [gpu=0]}"
HUE="${2:-220}"
THRESH="${3:-0.7}"
GPU="${4:-0}"
HERE="$(cd "$(dirname "$0")"/.. && pwd)"
CKPT="$HERE/3_output/$SCENE/ckpt_exp010_per_gaussian"
OUT="$HERE/3_output/$SCENE/recolor_h${HUE}_t${THRESH}"

if [ ! -d "$CKPT/point_cloud/iteration_20000" ]; then
  echo "[err] cloth_logit ckpt missing — run scripts/cloth_logit.sh $SCENE first"
  exit 1
fi
rm -rf "$OUT"
echo "[stage 4] recolor $SCENE hue=$HUE threshold=$THRESH"
CUDA_VISIBLE_DEVICES=$GPU python "$HERE/2_process/recolor.py" \
  --ckpt-dir "$CKPT" --iter 20000 \
  --hue $HUE --threshold $THRESH --num-frames 4 --save-ply \
  --min-sat 0.6 --min-val 0.7 --out "$OUT"
echo "[done] Stage 4 → $OUT"
echo "      panels: $OUT/panel_*.png"
echo "      recolored PLY: $OUT/point_cloud_recolored_*.ply"
echo "Next: scripts/viewer.sh $SCENE   # interactive viewer"
