#!/usr/bin/env bash
# Full pipeline demo on jumpingjacks scene (~1 hour total).
# Stage 1 (baseline 4DGS) → Stage 2 (SAM3 cache) → Stage 3 (cloth_logit) → Stage 4 (recolor)
# Usage:  scripts/run_jumpingjacks.sh  [gpu=0]
set -euo pipefail
GPU="${1:-0}"
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "============ STAGE 1: baseline 4DGS (40 min) ============"
"$HERE/train_baseline.sh" jumpingjacks $GPU

echo "============ STAGE 2: SAM3 union mask cache (3 min) ============"
"$HERE/sam3_cache.sh" jumpingjacks "orange hoodie,sweater,hoodie,jacket,long sleeve top" $GPU

echo "============ STAGE 3: per-Gaussian cloth_logit (5 min) ============"
"$HERE/cloth_logit.sh" jumpingjacks cache/sam3_union_jumpingjacks $GPU

echo "============ STAGE 4: recolor hue=220 (blue, 1 min) ============"
"$HERE/recolor.sh" jumpingjacks 220 0.7 $GPU

echo
echo "============ DONE — open viewer ============"
echo "  $HERE/viewer.sh jumpingjacks"
echo
echo "Output:"
RECOLOR_DIR=$(dirname "$HERE")/3_output/jumpingjacks/recolor_h220_t0.7
ls $RECOLOR_DIR
