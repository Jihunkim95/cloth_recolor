#!/usr/bin/env bash
# Offline render a trained 4D-GS ckpt to per-frame PNG + mp4 video.
# Usage:  ./render_video.sh <ckpt_dir>

set -e
CKPT="${1:?usage: $0 <ckpt_dir>}"
cd /NHNHOME/WORKSPACE/0526040060_B/research/4DGaussians
source /home/bjh0309/miniconda3/etc/profile.d/conda.sh && conda activate zcar

# 4DGaussians/render.py expects --model_path pointing at the ckpt dir
python render.py \
  --model_path "$CKPT" \
  --skip_train --skip_video      # output: <CKPT>/test/ours_14000/{renders,gt}/*.png
echo "[done] renders at $CKPT/test/ours_14000/"
