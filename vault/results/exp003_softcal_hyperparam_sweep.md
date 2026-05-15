---
title: exp003 — λ_cal × mem_size hyperparam sweep on best variant
status: done
date: 2026-05-10
code: cloth_recolor/softcal_overnight.sh (Phase C)
---

# exp003

## 질문

[[exp002_softcal_ablation]] 의 best variant 가 씬마다 달랐다. 각 씬 best variant 위에서 λ_cal · mem_size 를 흔들면 더 나은 hyperparameter 가 있는가?

## 설정

- best variant per scene (from exp002):
  - jumpingjacks → b
  - standup → a
  - hellwarrior → both
- grid:
  - λ_cal ∈ {0.05, 0.2}  (default 0.1)
  - mem_size ∈ {500, 2000}  (default 1000)
- 4 configs × 3 scenes = 12 jobs, 8 GPU 큐
- 학습 14000 iter, 다른 hyperparam 은 exp002 와 동일

## 예측

- λ_cal 이 너무 작으면 calibration 효과 없음, 너무 크면 RGB 학습 방해
- mem_size 가 너무 작으면 noisy easy sample, 너무 크면 distance 분별력 떨어짐
- 기본값 (0.1, 1000) 근처가 최적일 것

## 결과

### scene 별 best of sweep

| scene (variant) | best λ_cal | best mem_size | PSNR | exp002 PSNR | Δ |
|---|---|---|---|---|---|
| jumpingjacks (b) | **0.2** | **2000** | 35.09 | 34.08 | +1.01 |
| standup (a) | **0.2** | **2000** | 36.97 | 34.99 | +1.98 |
| hellwarrior (both) | **0.05** | **500** | 29.05 | 28.92 | +0.13 |

### 전체 결과 표 (PSNR)

| scene (var) | λ=0.05 mem=500 | λ=0.05 mem=2000 | λ=0.2 mem=500 | λ=0.2 mem=2000 |
|---|---|---|---|---|
| jumpingjacks (b) | 35.06 | 35.04 | 35.05 | **35.09** |
| standup (a) | 36.73 | 36.71 | 36.95 | **36.97** |
| hellwarrior (both) | **29.05** | 28.93 | 28.88 | 28.83 |

## 예측 맞았나?

- 부분적으로. jumpingjacks/standup 은 **λ_cal=0.2 (default 보다 큼)** 에서 더 좋음. hellwarrior 는 λ=0.05 (default 보다 작음). → "λ 클수록 좋다" 도 "작을수록 좋다" 도 아님 (씬 의존)
- mem_size 효과는 미미 (대부분 < 0.1 dB 차이) — 1000 정도면 충분, 2000 으로 늘리는 비용 > 이득

## 다음

[[exp004_softcal_extra_scenes]] — 추가 D-NeRF 씬 (lego, bouncingballs, hook) 으로 일반화 테스트.
