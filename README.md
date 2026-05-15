# ClothSplat — Text-Driven Garment Recoloring in 4D Gaussian Splatting

*(가제, 2026-05-13 확정. 자세한 paper outline: `vault/005Paper_outline.md`.)*

Mask-aware 4D-GS that learns a per-Gaussian `cloth_logit` channel under SAM 3
supervision plus memory-based **soft-label calibration**, then HSV-swaps the hue
of cloth Gaussians to recolor clothing across time and viewpoint.

## Pipeline (Input → Process → Output → Viewer)

```
1_input/dnerf/<scene>/                  ← D-NeRF Blender data (symlink)
        │
        ▼
2_process/                              ← scripts
   sam3_cache.py    : SAM 3 mask caching
   train.sh         : 4DGaussians + cloth_logit + soft-cal training
   recolor.py/.sh   : HSV recolor + PLY/PNG export
        │
        ▼
3_output/<scene>/
   ckpt_baseline_hardBCE/               ← 4D-GS ckpt (cloth_logit, no soft-cal)
   ckpt_softcal_best/                   ← 4D-GS ckpt (best soft-cal variant)
   ckpt_softcal_sweepbest/              ← 4D-GS ckpt (best of hyperparam sweep)
   recolor_baseline/                    ← 16 PNG panels + recolored.ply
   recolor_softcal_best/
   recolor_softcal_sweepbest/
        │
        ▼
4_viewer/
   viewer.py        : interactive viser/nerfview viewer (D-NeRF + multipleview)
   render_video.sh  : offline per-frame PNG + mp4 via 4DGaussians/render.py
```

## Quick start

### Train one scene (clothing-aware 4D-GS)

```bash
CUDA_VISIBLE_DEVICES=0 ./2_process/train.sh jumpingjacks both
# → 3_output/jumpingjacks/ckpt_softcal_<variant>_<MMDD_HHMM>/
```

### Recolor a trained ckpt

```bash
./2_process/recolor.sh 3_output/jumpingjacks/ckpt_softcal_best 220 0.2
# → recolor_*_h220_t0.2/{panel_*.png, recolored.ply, summary.json}
```

### View interactively

```bash
python 4_viewer/viewer.py \
  --ckpt-dir 3_output/jumpingjacks/ckpt_softcal_best \
  --port 8080
# open http://<host>:8080/
```

### Offline render (PNG + mp4)

```bash
./4_viewer/render_video.sh 3_output/jumpingjacks/ckpt_softcal_best
# → <ckpt>/test/ours_14000/{renders,gt}/*.png + video_rgb.mp4
```

## What's in this repo

| dir | purpose |
|---|---|
| `1_input/` | symlink to D-NeRF dataset (`4DGaussians/data/dnerf/data/`) |
| `2_process/` | training + recolor scripts |
| `3_output/` | 9 trained ckpts (3 scenes × 3 variants) + recolor results |
| `4_viewer/` | interactive viewer + offline render |
| `cache/sam3_dnerf/` | per-scene SAM 3 mask cache (`<scene>/masks.npz` + `meta.json`) |
| `docs/` | `REPORT.md` — full ablation findings |
| `archive/` | legacy 4D-DRESS attempts + sweep ckpts (122 GB, kept for provenance) |

## Engine

The actual 4D-GS implementation is `../4DGaussians/` (hustvl/4DGaussians + our
mods: `_cloth_logit` parameter in `scene/gaussian_model.py`, gsplat 2nd-pass in
`gaussian_renderer/__init__.py`, `utils/soft_cal.py`).
This repo is the project layer that drives it for the recolor task.

## Headline result

| scene | hard-BCE PSNR | best soft-cal PSNR | Δ | cloth boundary |
|---|---|---|---|---|
| jumpingjacks | 33.91 | **34.08** (variant b) | +0.17 | over-cover 27.3% → 1.32% (target SAM3 1.93%) |
| standup | 35.70 | 34.99 (variant a) | -0.71 | 53.1% → 36.0% (target 50%) |
| hellwarrior | 28.28 | **28.92** (both) | +0.64 | 58.5% → 52.4% (target 60%) |

Soft-cal collapses the hard-BCE 0/1 polarization (jumpingjacks: 17% ambiguous
→ 6.5% near-cloth, with 97% of Gaussians correctly placed in mid-low range).
See `docs/REPORT.md` for full numbers.
# cloth_recolor
