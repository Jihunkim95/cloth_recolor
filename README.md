# ClothSplat — Text-Driven Garment Recoloring in 4D Gaussian Splatting

Mask-aware 4D Gaussian Splatting that learns a per-Gaussian `cloth_logit` from
SAM3 text-prompt masks, then HSV-swaps the hue of cloth Gaussians to recolor
garments across time and viewpoint — **with no SAM3 inference at runtime**
(30× faster than the naive baseline).

```
[D-NeRF video] ─┐
                ├─→ 4DGS (Stage 1) ─→ ckpt_baseline
[SAM3 prompt] ──┘                          │
                                           ▼
                            per-Gaussian projection (Stage 3, exp010)
                                           │
                                           ▼
                                     cloth_logit ∈ ℝ^N
                                           │
                                  HSV swap (Stage 4)
                                           ▼
                                     recolored garment
```

## Quick start (from fresh clone)

```bash
# 1. clone (+ sibling 4DGaussians fork required)
git clone <THIS_REPO> cloth_recolor
cd cloth_recolor

# 2. install (conda env "clothsplat" + CUDA 12.8 + gsplat + SAM3 + DINOv3)
./scripts/setup.sh

# 3. activate + env vars (gsplat needs CUDA_HOME exported)
conda activate clothsplat
source ./scripts/env.sh

# 4. fetch D-NeRF data (~530 MB)
./scripts/download_data.sh

# 5. full pipeline on jumpingjacks (~1 hour on B200)
./scripts/run_jumpingjacks.sh
# → 3_output/jumpingjacks/recolor_h220_t0.7/   (panels + recolored.ply)

# 6. interactive viewer
./scripts/viewer.sh jumpingjacks
# open http://localhost:8080/ (SSH tunnel -L 8080:localhost:8080 if remote)
```

## Pipeline (4 stages)

| Stage | Script | Output | Time (B200) |
|---|---|---|---|
| 1. **Baseline 4DGS** train (RGB only) | `scripts/train_baseline.sh <scene>` | `ckpt_baseline_hardBCE/` | ~40 min |
| 2. **SAM3 mask cache** (text prompt) | `scripts/sam3_cache.sh <scene> [prompt]` | `cache/sam3_*/<scene>/masks.npz` | 1-5 min |
| 3. **Per-Gaussian projection** + BCE | `scripts/cloth_logit.sh <scene>` | `ckpt_exp010_per_gaussian/` | ~5 min |
| 4. **HSV recolor** + render panels | `scripts/recolor.sh <scene> [hue] [τ]` | `recolor_h<hue>_t<τ>/{panels,ply}` | ~1 min |

Each script is **idempotent** (skips if output exists) and prints the next
command to run.

## Repository layout

```
cloth_recolor/
├── scripts/                    # one-shot entry points (this README references these)
│   ├── setup.sh                # conda env + CUDA + 4DGaussians fork
│   ├── env.sh                  # gsplat CUDA env vars (source this every shell)
│   ├── download_data.sh        # D-NeRF Blender data
│   ├── train_baseline.sh       # Stage 1
│   ├── sam3_cache.sh           # Stage 2 (single or multi-prompt union)
│   ├── cloth_logit.sh          # Stage 3 (exp010 breakthrough)
│   ├── recolor.sh              # Stage 4
│   ├── viewer.sh               # interactive viser viewer
│   └── run_jumpingjacks.sh     # full pipeline demo
├── 2_process/                  # python implementation
│   ├── per_gaussian_supervision.py        # exp010 D-NeRF Stage 3
│   ├── per_gaussian_supervision_multiclass.py  # exp025 K=N instances
│   ├── exp027_soft_cal.py     # memory-based negative mining
│   ├── recolor.py             # HSV swap + render panels
│   ├── sam3_cache.py          # SAM3 mask cache (per-scene)
│   ├── sam3_cache_union.py    # multi-prompt union variant
│   ├── exp021_efficiency.py   # baseline vs ours wall-time
│   └── edge_metric.py         # boundary IoU + Chamfer metric
├── 4_viewer/
│   └── viewer.py              # viser-based interactive 3D viewer
├── vault/                     # Obsidian research log + experiment notes
│   ├── 001연구_가이드북.md ~ 010BACKUP_MANIFEST.md
│   └── results/exp001~027.md   # detailed per-experiment notes
├── docs/                      # paper outline, REPORT, QnA
└── 3_output/<scene>/          # trained ckpts + recolor results (gitignored)
```

**Sibling 4DGaussians fork** (cloned by `scripts/setup.sh`):
- `4DGaussians/scene/gaussian_model.py` — `_cloth_logit (N, K)` parameter
- `4DGaussians/gaussian_renderer/__init__.py` — gsplat 2nd-pass for cloth_logit
- `4DGaussians/utils/soft_cal.py` — DINOv3 + memory bank (used by exp027)

## Key contributions

| # | Contribution | Reference |
|---|---|---|
| 1 | **Per-Gaussian projection supervision** (bypasses alpha-composit averaging) | `vault/results/exp010_per_gaussian_projection.md` |
| 2 | **Spatial outlier filter** (training-time MAD) for cluttered scenes | `vault/results/exp015_n3v_train_time_spatial_filter.md` |
| 3 | **30× inference speedup** (SAM3 removed from runtime) | `vault/results/exp021_efficiency.md` |
| 4 | **Multi-instance K=N** (cloth/shoes/hat with per-class hue) | `vault/results/exp025_multiclass.md` |
| 5 | **Boundary IoU + Chamfer metric** (SAM3 as pseudo-GT) | `vault/results/exp024_edge_metric.md` |
| 6 | **Memory-based negative mining calibration** | `vault/results/exp027_soft_cal.md` |

## Headline results (D-NeRF jumpingjacks)

| Method | Inference / frame | Edge IoU ↑ | Chamfer ↓ (px) |
|---|---|---|---|
| baseline (4DGS + per-frame SAM3) | 180.2 ms | 0.121 | 12.06 |
| **ours (exp010 per-Gaussian projection)** | **6.1 ms (30×)** | **0.253 (2.1×)** | **6.40 (½)** |

Multi-instance demo (`vis/SUMMARY_meeting/exp025_v2_jumpingjacks_union.png`):
hoodie → red / shorts → green / shoes → blue with one model, no inference-time SAM3.

## Multi-class example

```bash
# K=3 (hoodie/shorts/shoes) on jumpingjacks
python 2_process/per_gaussian_supervision_multiclass.py \
  --base-ckpt 3_output/jumpingjacks/ckpt_baseline_hardBCE --base-iter 20000 \
  --scene jumpingjacks \
  --sam3-caches cache/sam3_exp025_jumpingjacks_hoodie_union,cache/sam3_exp025_jumpingjacks_shorts,cache/sam3_exp025_jumpingjacks_shoes \
  --class-names hoodie,shorts,shoes \
  --out 3_output/jumpingjacks/ckpt_multiclass \
  --spatial-filter 3.0

# recolor target=0 (hoodie) red
python 2_process/recolor.py \
  --ckpt-dir 3_output/jumpingjacks/ckpt_multiclass --iter 20000 \
  --hue 0 --threshold 0.4 --per-class-bce --target-class 0 \
  --num-frames 4 --out 3_output/jumpingjacks/recolor_hoodie_red
```

## Limitations

- **Dynamic multi-view with cluttered background (Neural 3D Video)**: tested
  but post-hoc cloth/non-cloth separation fails because the deformation MLP
  couples cloth Gaussians with "trail" background Gaussians during 4DGS Stage 1
  training. Documented in `vault/results/exp014~020`. Main paper results use
  D-NeRF (8 scenes) and 4D-DRESS (8 takes) where this issue is absent.
- **SAM3 single-prompt reliability**: for some prompts SAM3 fails on most
  frames (e.g., "hoodie" fails 79% of frames on jumpingjacks). Workaround:
  multi-prompt union via `scripts/sam3_cache.sh <scene> "p1,p2,p3"`.
- **DINOv3 calibration (exp027 flavor b)**: single-frame DINOv3 patch
  features insufficiently discriminate skin-vs-cloth boundary on
  similar-color scenes. Multi-frame aggregation left to future work.

## Citation

```bibtex
@misc{clothsplat2026,
  title = {ClothSplat: Text-Driven Garment Recoloring in 4D Gaussian Splatting},
  author = {Jihun Kim and others},
  year = {2026},
  note = {Manuscript in preparation}
}
```

## See also

- `vault/008Viewer_사용법.md` — detailed viewer guide
- `vault/009Meeting_Response_2026-05-15.md` — current experimental status
- `vault/010BACKUP_MANIFEST.md` — what's in `backup_dnerf_essentials.zip`
- `docs/REPORT.md` — full ablation findings (exp001~027)
