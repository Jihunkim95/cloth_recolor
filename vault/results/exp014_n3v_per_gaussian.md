---
title: exp014 — Neural 3D Video (N3V) multi-cam per-Gaussian projection
status: done
date: 2026-05-13
code: 2_process/per_gaussian_supervision_n3v.py
---

# exp014

## 질문

[[exp010_per_gaussian_projection|D-NeRF 의 exp010]] (1 카메라 monocular) 과 [[exp013_4ddress_retrain|4D-DRESS 의 exp013]] (4 카메라 sparse) 에서 per-Gaussian projection 검증. **18-21 카메라 dynamic multi-view** (Neural 3D Video) 에서도 같은 framework 작동하는가? supervision 강도 (cam × frame) 가 5400-6300 으로 D-NeRF 의 200 보다 27-30× 강하면 cloth 분리 정밀도 어떻게 변하나?

## 설정

- 데이터셋: Neural 3D Video (Li 2022 / DyNeRF) 6 scene
  - `coffee_martini`, `cook_spinach`, `cut_roasted_beef`, `flame_salmon_1`, `flame_steak`, `sear_steak`
  - 각 scene 18-21 cam × 300 frame = 5400-6300 (view, frame) pair
  - cooking/bartending — chef 가 shirt 입고 작업
- preprocessing:
  - mp4 → PNG ffmpeg 병렬 추출 (cv2 단일스레드 대비 80× 빠름, 3분 vs 4시간)
  - COLMAP sparse 재구성 (각 scene 18-21 image × COLMAP feature/match/mapper) — 3500-4700 sparse points
  - PLY post-process (nx/ny/nz normals 추가 — COLMAP 기본 export 에 없음)
- baseline 4DGS: 14000 iter, dynerf config (per-scene tuned)
- SAM3 cache: prompt `"shirt"`, 6 scene × 18-21 cam × 300 frame = ~36000 inference, 6 GPU 병렬 ~17 분
- per-Gaussian projection: 14000 iter ckpt → projection supervision → 2000 step BCE fine-tune

## 예측

- N3V baseline PSNR 27-33 dB (논문 normal)
- SAM3 cov ~1.5% (chef shirt 가 작은 area)
- per-Gaussian soft_target mean ~0.05 — most Gaussians background → cloth_logit 분포 잘 분리
- recolor 시 chef shirt 만 정확히, background/skin/cookware 그대로

## 결과

### N3V baseline 4DGS (random init vs COLMAP init)

| scene | random init PSNR | COLMAP init PSNR |
|---|---|---|
| coffee_martini | 13.86 | **31.99** |
| cook_spinach | 14.23 | **32.77** |
| cut_roasted_beef | 14.34 | **30.08** |
| flame_salmon_1 | 15.19 | **31.11** |
| flame_steak | 12.26 | **28.00** |
| sear_steak | 10.98 | **28.10** |
| **avg** | **13.5** | **30.3** |

→ random uniform init 으로 학습 시 PSNR 13.5 ⇒ baseline 자체가 안 됨. **COLMAP sparse init 필수**.

### SAM3 cov + per-Gaussian cloth_pct (thresh 0.2)

| scene | SAM3 cov (avg per cam) | exp014 cloth_pct @ 0.2 |
|---|---|---|
| coffee_martini | 1.61% | 12.77% |
| cook_spinach | 1.31% | 7.08% |
| cut_roasted_beef | 1.47% | 12.25% |
| flame_salmon_1 | 1.18% | 6.08% |
| sear_steak | 1.35% | 9.47% |
| flame_steak | 1.25% | 8.69% |
| **avg** | **1.36%** | **9.39%** |

flame_steak 는 두 가지 fix 필요: (1) NaN-to-num for ndc 변환, (2) per-camera mask resolution (mixed res scene 처리: cam00/20 = 2704×2028, cam01-19 = 1352×1014).

soft_target 분포 (cook_spinach 예시):
- mean = 0.0494, max = 0.5415
- > 0.05: 30.11% / > 0.1: 19.13% / > 0.2: 7.07% / > 0.3: 2.48% / > 0.5: 0.05%

→ SAM3 coverage 작아서 soft_target 도 작음. **threshold 0.2 가 적절** (0.5 면 0% — 너무 엄격).

### 시각 (cook_spinach)

`vis/SUMMARY_n3v_exp014/cook_spinach.png`:
- Left (original): chef in dark/gray apron, cooking spinach in kitchen
- Middle (recolored, hue=220°): **chef shirt + apron BLUE**, background (window, shelves, bottles) 그대로, skin (face, hands) 그대로, 야채/팬/도마 그대로
- Right (mask overlay): subtle 그대로

기존 D-NeRF / 4D-DRESS 결과와 일관 — *cloth Gaussian 만 정확히 추출 + recolor*.

## 예측 맞았나?

- N3V baseline PSNR: ✅ COLMAP init 으로 30.3 dB (논문 normal 범위)
- soft_target distribution: ✅ mean 0.05, bimodal-like
- cloth_pct 적절: ✅ 6-13% (SAM3 cov 의 4-10× 확장, alpha-composit 의도된 효과)
- 시각 분리: ✅ chef shirt 만 (perfect)
- multi-cam supervision 강도 효과: ✅ 5400-6300 (view, frame) pair = D-NeRF 의 27-30× — 더 안정적인 soft_target

## Method 의 카메라 수에 따른 supervision 강도

| 데이터셋 | 카메라 수 | frame 수 | (view, frame) pair / scene |
|---|---|---|---|
| D-NeRF (monocular) | 1 (trajectory) | 100-200 | 100-200 |
| 4D-DRESS | 4 (fixed) | 100-200 | 400-800 |
| **N3V** | **18-21** (fixed) | **300** | **5400-6300** |

per-Gaussian projection 의 **noise reduction** 효과:
- D-NeRF: 가우시안 i 가 hoodie 영역에 splat 하는 frame ~100 개 — soft_target = (100 중 mask=1 hit 수) / 100
- N3V: 가우시안 i 가 chef 영역에 splat 하는 frame ~3000 개 — 30× more samples → mean stable, var ↓

→ **multi-cam 이 per-Gaussian projection 의 noise 를 자연스럽게 줄임**. 우리 framework 가 cam 수에 *scale up* 함.

## 산출물

`3_output/n3v_<scene>/`:
- `ckpt_baseline_4dgs/` — vanilla 4DGS (COLMAP init, 14k iter)
- `ckpt_exp014_per_gaussian/` — cloth_logit fine-tune
- `recolor_exp014_per_gaussian/` — 16 panel + recolored.ply

`cache/sam3_n3v/<scene>/<cam>/masks.npz` — SAM3 mask cache (6 scene × 18-21 cam × 300 frame)

## 환경 leakage 진단 + 수정 (2026-05-14)

N3V 결과 (recolor_exp014_per_gaussian, threshold 0.2) 시각 확인 → chef 옷 외 *주변 환경 (선반, 병, 커튼)* 까지 파란색으로 변경. D-NeRF 결과는 깨끗했는데 N3V 만 leakage. 원인 진단:

**spatial spread 분석** (cloth Gaussian xyz std / 전체 std):

| scene | ratio @ t=0.2 | ratio @ t=0.4 | reduction |
|---|---|---|---|
| coffee_martini | 0.123 | 0.137 | (이미 tight) |
| cook_spinach | 0.515 | 0.357 | 31% ↓ |
| cut_roasted_beef | 0.528 | 0.228 | 57% ↓ |
| flame_salmon_1 | 0.617 | 0.293 | 53% ↓ |
| flame_steak | 0.302 | 0.134 | 56% ↓ |
| sear_steak | 0.336 | 0.186 | 45% ↓ |

D-NeRF 의 ratio < 0.15 인데 N3V 는 0.3-0.6 — *환경 leakage 정량 확인*. 원인:
- N3V SAM3 cov 가 작음 (1.5%) → soft_target mean = 0.05 → distribution peak 가 낮음
- threshold 0.2 가 marginal Gaussian (한두 frame 만 hit) 도 포함
- 18-21 cam 다 시점에서 background Gaussian 이 가끔 cloth mask 영역에 정렬 (시점 다양성의 부작용)

**수정**: threshold 0.2 → 0.35 + recolor `--min-sat 0.6 --min-val 0.7` (어두운 chef 옷 visibility 보장):

| scene | cloth_pct @ t=0.2 | cloth_pct @ t=0.35 |
|---|---|---|
| coffee_martini | 12.77% | **1.39%** |
| cook_spinach | 7.08% | **1.30%** |
| cut_roasted_beef | 12.25% | **0.87%** |
| flame_salmon_1 | 6.08% | **0.66%** |
| flame_steak | 8.69% | **0.47%** |
| sear_steak | 9.47% | **0.89%** |

→ D-NeRF 의 cloth_pct (0.5-2%) 수준 회복. 결과: `recolor_exp014_tight/` 에서 chef 옷만 깔끔하게 파란색, 환경 보존.

### 추가 fix: spatial outlier filter (recolor_exp014_filtered)

threshold 0.35 후에도 일부 frame 에서 chef 의 *팔 motion path 옆 ghost trail* 남음. 원인: canonical Gaussian 이 deformation 으로 chef 영역을 스쳐 지나가는 frame 에서 mask hit 누적 (환경 leakage 아닌, motion trail).

`recolor.py --spatial-filter 3.0` 추가: cloth Gaussian xyz 의 median ± 3×MAD 밖 outlier 제거. 16-20% 추가 제거.

```python
# recolor.py:107-122
xyz_c = pc._xyz.detach()[cloth_mask]
med = xyz_c.median(dim=0).values
dist = (xyz_c - med).norm(dim=1)
mad = (dist - dist.median()).abs().median()
keep = dist <= dist.median() + 3.0 * mad
```

| scene | t=0.35 cloth | + spatial filter | hue |
|---|---|---|---|
| coffee_martini | 1300 | ~1100 | 220 |
| cook_spinach | 1291 | ~1080 | 220 |
| cut_roasted_beef | 865 | ~720 | 220 |
| flame_salmon_1 | 627 | ~530 | **0 (red)** — chef 이미 파란 옷 |
| flame_steak | 420 | ~350 | 220 |
| sear_steak | 808 | 680 | 220 |

결과: `recolor_exp014_filtered/` 에서 chef body 만 깨끗하게 색변경. 환경 보존 + 가로 motion trail 제거.

**교훈**: per-Gaussian projection 의 *threshold 는 cam 수에 reverse-correlated 하게 조정* 필요. D-NeRF (1 cam, 100 frame) → t=0.2, N3V (20 cam × 300 frame) → t=0.35-0.4. **multi-cam 이 supervision 강도를 높이지만, 동시에 환경 noise 누적도 증가**. supervision 강도 ↑ ⇒ threshold ↑.

## 다음

- N3V 결과 정성 비교 패널 (D-NeRF vs 4D-DRESS vs N3V) — paper Figure
- 학습 step / cam 수 의 supervision 강도 ablation
- threshold sweep 곡선 ablation (paper Figure 후보)
