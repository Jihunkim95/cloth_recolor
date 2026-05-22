#!/usr/bin/env bash
# Launch interactive viewer (viser, port 8080).
# Open browser at http://localhost:8080 (with SSH tunnel -L 8080:localhost:8080
# if running on remote).
#
# Usage:  scripts/viewer.sh <scene>  [ckpt_subdir=ckpt_exp010_per_gaussian]  [port=8080]  [gpu=0]
set -euo pipefail
SCENE="${1:?usage: $0 <scene> [ckpt_subdir] [port] [gpu]}"
CKPT_SUBDIR="${2:-ckpt_exp010_per_gaussian}"
PORT="${3:-8080}"
GPU="${4:-0}"
HERE="$(cd "$(dirname "$0")"/.. && pwd)"
CKPT="$HERE/3_output/$SCENE/$CKPT_SUBDIR"

if [ ! -d "$CKPT/point_cloud" ]; then
  echo "[err] ckpt missing: $CKPT"
  exit 1
fi

# D-NeRF ckpts use iter 20000, others 14000
ITER=20000
[ ! -d "$CKPT/point_cloud/iteration_20000" ] && ITER=14000

echo "[viewer] $CKPT iter=$ITER port=$PORT GPU=$GPU"
echo "         open http://localhost:$PORT/"
CUDA_VISIBLE_DEVICES=$GPU python "$HERE/4_viewer/viewer.py" \
  --ckpt-dir "$CKPT" --iter $ITER --port $PORT
