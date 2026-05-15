---
title: exp005 — D-NeRF 8씬 전체 커버리지 확장
status: done
date: 2026-05-11
code: /tmp/extend_3output.sh + 4DGaussians/train.py
---

# exp005

## 질문

[[exp001_tier2_dnerf_baseline]]·[[exp002_softcal_ablation]] 의 옷 3 씬에 더해, [[../notes/4d_gaussian_splatting]] 논문이 사용한 D-NeRF 8 씬 모두 (bouncingballs, hook, lego, mutant, trex 추가) 에서 우리 파이프라인이 동작하는가? cloth_pct (over-cover) 패턴이 일반화되는가?

목적: `3_output/` 에 8 씬 일관된 baseline+soft-cal 비교 자산 확보 → 논문 figure/table 준비.

## 설정

- 데이터셋: D-NeRF 8 씬 전부 (옷 3 + 비-옷 5)
- 새로 추가된 5 씬:
  - SAM3 prompt: bouncingballs="ball", hook="puppet body", lego="lego figure body", mutant="monster body and armor", trex="dinosaur"
  - SAM3 coverage: 6.55 / 11.39 / 5.7 / 9.86 / 4.26 %
- 학습: baseline (`--soft-cal none`) + soft-cal both (`--soft-cal both`), 14000 iter, default hyperparam
- bouncingballs/hook/lego/mutant 일부는 [[exp004_softcal_extra_scenes]] 의 archive ckpt 재사용
- trex baseline + trex soft-cal both + mutant baseline 은 새로 학습 (3 jobs, ~5분)
- recolor: thresh=0.2, hue=220° (8 GPU 병렬)

## 예측

- baseline 모든 씬에서 RGB PSNR 정상 학습 (D-NeRF 는 모두 안정 데이터셋)
- baseline cloth_pct 가 [[exp001_tier2_dnerf_baseline]] 처럼 over-cover (50-90% 정도)
- soft-cal 이 cloth_pct 를 SAM3 GT 쪽으로 끌어내림 (10-30 %p 정도)

## 결과

### cloth_pct (recolor at thresh=0.2)

| scene | SAM3 GT | baseline | soft-cal best | Δ (baseline−softcal) |
|---|---|---|---|---|
| bouncingballs | 6.55% | **100.0%** | 77.62% | -22.4 |
| hook | 11.39% | **100.0%** | 96.54% | -3.5 |
| lego | 5.70% | **100.0%** | 86.97% | -13.0 |
| mutant | 9.86% | **100.0%** | 95.04% | -5.0 |
| trex | 4.26% | **100.0%** | 93.54% | -6.5 |

### 새로 학습된 PSNR

| scene | variant | PSNR |
|---|---|---|
| trex | baseline (none) | 33.17 |
| trex | soft-cal both | 33.02 |
| mutant | baseline (none) | 36.97 |

(bouncingballs/hook/lego/mutant 의 baseline+soft-cal 은 [[exp004_softcal_extra_scenes]] 결과 재사용)

## 예측 맞았나?

- 8 씬 학습 모두 안정: ✅
- baseline cloth_pct over-cover: ⚠️ **예상보다 훨씬 심함** — 5 씬 모두 **100%** 로 collapse. SAM3 mask 가 6-11 %p 인데 cloth_logit 이 모든 가우시안에 1 로 박힘.
  - 추정 원인: alpha-composit 에 의해 mask 안 픽셀을 그리는 모든 가우시안이 + gradient 받음 + soft-cal 없으니 멈출 신호 없음 → 모두 cloth 로 수렴
  - 옷 3 씬 (jumpingjacks/standup/hellwarrior) 은 baseline 27/53/58 % 였는데 이건 학습 epoch / opacity reset 등 schedule 차이로 collapse 가 막혔을 뿐 본질은 동일 over-cover
- soft-cal 효과: ⚠️ **부분적 보정** (3-22 %p 감소) — 옷 씬에서 봤던 극적 변화 (27%→1.32%) 만큼은 아님. SAM3 prompt 가 비-옷 attribute 라 mask 자체가 모호해서 calibration signal 도 약함

## 다음

- 8 씬 통일된 정량 표 + recolor 패널 비교 → 논문 Figure 1 후보 ([[../notes/hsv_recolor_trick]] 의 시각 차이 확보)
- 새 5 씬 전용 hyperparam sweep (현재 default 만 — λ_cal 더 높이면 cloth_pct collapse 완화될지)
- baseline 의 cloth_pct=100% collapse 를 *학습 schedule* (cloth_warmup, opacity_reset_interval) 변동으로 막아볼 수 있는지 별도 실험
