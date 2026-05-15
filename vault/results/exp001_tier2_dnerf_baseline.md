---
title: exp001 — Tier 2 D-NeRF baseline (cloth_logit + hard-BCE)
status: done
date: 2026-05-10
code: ~/research/4DGaussians/train.py + cloth_recolor/2_process/
---

# exp001

## 질문

D-NeRF 데이터셋 3 씬 (jumpingjacks, standup, hellwarrior) 에 [[../notes/cloth_logit_channel]] + [[../notes/sam3_mask_supervision]] hard-BCE 만 추가한 단순 [[../notes/4d_gaussian_splatting]] 학습이 RGB 품질을 깨뜨리지 않는가? 그리고 cloth mask 를 가우시안 단위로 attribute 화 가능한가?

(4D-DRESS 4-view 학습이 novel-view 에서 깨지는 것을 확인한 후 데이터셋을 D-NeRF 로 옮긴 직후의 baseline.)

## 설정

- 데이터셋: D-NeRF `jumpingjacks` (200 frame), `standup` (150), `hellwarrior` (100)
- 모델: 4DGaussians (HexPlane) + per-Gaussian `_cloth_logit` Parameter
- supervision: per-pixel BCE between gsplat 2nd-pass cloth_logit_image vs SAM3 mask
- λ_cloth = 0.1, cloth_warmup = 3000, cloth_lr = 1e-2
- iter = 14000 (논문 baseline 과 동일 schedule)
- GPU: B200 sm_100 ×3 (씬 1개당 GPU 1)

## 예측

- RGB PSNR 은 baseline 4DGS (논문 33-36 dB) 와 비슷하게 유지될 것
- cloth_logit 는 SAM3 mask 비율 근처 (jj ~2%, su ~50%, hw ~60%) 로 수렴할 것
- 옷이 SH-HSV swap 으로 색 바뀌는 것 시각 확인 가능할 것

## 결과

| scene | N (Gaussians) | cloth_pct (>0.5) | test PSNR | SAM3 coverage |
|---|---|---|---|---|
| jumpingjacks | 24,964 | 27.3% | 33.91 | 1.93% |
| standup      | 27,420 | 53.1% | 35.70 | ~50% |
| hellwarrior  | 41,308 | 58.5% | 28.28 | ~60% |

NaN 1회 발생 → 자동 재실행 (os.execv) 으로 회복. λ_cloth=0.5 가 너무 강해서 발생, λ=0.1 + warmup=3000 으로 안정화.

[[../notes/hsv_recolor_trick]] 으로 hue=220° 적용 → blue 옷 시각 확인 (특히 standup 의 vest 영역).

## 예측 맞았나?

- RGB PSNR: ✅ baseline 범위 유지 (33-36 dB)
- cloth_pct: ⚠️ jumpingjacks 27% >> SAM3 1.93% (over-cover) — hard BCE 가 SAM3 mask 를 직접 모방 못 하고 가우시안 단위 반올림 한 것. 추정: alpha-composite 후 픽셀이 mask 안에 있으면 그 픽셀을 그리는 모든 가우시안에 흘러가 모두 cloth 로 학습됨.
- 시각: ✅ 색 변경 작동, 그러나 boundary 가 over-aggressive (vest 만 바꾸려 했는데 셔츠/바지/부츠까지)

## 다음

[[exp002_softcal_ablation]] — over-cover 문제를 해결하기 위해 [[../notes/soft_label_calibration]] 도입.
