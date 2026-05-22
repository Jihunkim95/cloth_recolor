#!/usr/bin/env bash
# Download D-NeRF Blender data (8 scenes) to ../4DGaussians/data/dnerf/data/
# Usage:  scripts/download_data.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")"/.. && pwd)"
ROOT="$(dirname "$HERE")"
DATA="$ROOT/4DGaussians/data/dnerf/data"
mkdir -p "$DATA"

if [ ! -d "$DATA/jumpingjacks" ]; then
  echo "[data] downloading D-NeRF data (~530MB) to $DATA"
  cd "$DATA"
  # D-NeRF official: https://www.dropbox.com/s/0bf6fl0ye2vz3vr/data.zip
  wget -q --show-progress https://www.dropbox.com/s/0bf6fl0ye2vz3vr/data.zip
  unzip -q data.zip
  rm data.zip
fi

echo "[data] D-NeRF scenes present:"
ls -1 "$DATA"
echo
echo "Next: scripts/run_jumpingjacks.sh   # full pipeline demo"
