---
title: exp006 — jumpingjacks 8-way grid (prompt × λ_cal × warmup)
status: done
date: 2026-05-11
code: /tmp/jj_grid.sh + jj_apply.sh
---

# exp006

## 질문

[[exp002_softcal_ablation]] 에서 jumpingjacks 가 가장 어려운 씬 (작은 캐릭터 + hoodie 주름 + female body silhouette) 으로 판명. SAM3 prompt / λ_cal / cloth_warmup 세 변수를 동시에 8 GPU 풀가동 grid 로 흔들면 더 나은 config 가 있는가?

## 설정

- 데이터셋: D-NeRF jumpingjacks (200 frames)
- 8 GPU 동시, 각 14000 iter
- 변수:
  - **prompt**: `shirt` (default) / `hoodie` / `orange hoodie` / `jacket`
  - **soft-cal**: `b` (current best) / `both`
  - **λ_cal**: 0.1 (default) / 0.5 / 1.0
  - **cloth_warmup**: 500 / 1000 / 3000 (default)
- 평가: PSNR + cloth_pct at thresh=0.2 (현재 default)

## 예측

- SAM3 prompt 구체화 (`orange hoodie`) → coverage ↑ → 학습 더 잘 됨
- λ_cal ↑ → calibration 강도 ↑ → cloth_pct 가 SAM3 GT 에 더 가까워짐
- cloth_warmup ↓ → 조기 supervision → 분포 형성 빠름

## 결과

### SAM3 prompt 재캐싱 결과 (coverage)

| prompt | coverage | 비고 |
|---|---|---|
| `shirt` (기존) | 1.93% | baseline |
| `orange hoodie` | **3.88%** | 색+종류 명시, 2× 향상 |
| `hoodie` | 0.66% | 너무 추상적, hood 부위만 |
| `jacket` | 0.25% | 거의 안 잡힘 |

### 학습 결과 (8 configs, 14k iter, 약 8 분)

| GPU | expname | prompt | sc | λ_cal | warmup | PSNR | cloth_pct (th 0.2) |
|---|---|---|---|---|---|---|---|
| 0 | jj_g0_p_orange_hoodie | orange hoodie | both | 0.1 | 3000 | 35.03 | 82.69% |
| 1 | jj_g1_p_hoodie | hoodie | both | 0.1 | 3000 | 35.19 | 83.72% |
| 2 | jj_g2_p_jacket | jacket | both | 0.1 | 3000 | 35.03 | 90.44% |
| 3 | jj_g3_p_shirt_b | shirt | b | 0.1 | 3000 | 35.03 | **100.00%** |
| 4 | jj_g4_lcal0_5 | shirt | both | **0.5** | 3000 | 35.19 | 91.90% |
| 5 | jj_g5_lcal1_0 | shirt | both | **1.0** | 3000 | 34.82 | **67.22%** ← cloth_pct best |
| 6 | jj_g6_warmup1000 | shirt | both | 0.1 | 1000 | 34.91 | 87.82% |
| 7 | jj_g7_warmup500 | shirt | both | 0.1 | 500 | 34.94 | 83.72% |

PSNR 차이 미미 (34.82 ~ 35.19, 약 0.37 dB).
cloth_pct 는 67% ~ 100% 까지 큰 차이.

### Best combo 별도 실험 (orange hoodie + λ_cal=1.0)

가장 정확한 prompt + 가장 강한 calibration 결합:

| variant | PSNR | cloth_pct @0.2 | cloth_pct @0.5 |
|---|---|---|---|
| orange hoodie baseline | 34.91 | **100.00%** | 64.29% |
| orange hoodie + soft-cal both + λ=1.0 | 34.85 | 86.38% | 26.93% |

### 이전 결과와 비교 (thresh 0.5 기준, SAM3 GT = 1.93%/3.88%)

| 조합 | cloth_pct | GT 와 차이 |
|---|---|---|
| `shirt` + soft-cal b (exp002 best) | **1.32%** | **−0.61 %p** (under-cover) |
| `orange hoodie` + soft-cal both/λ1.0 | 26.93% | +23.05 %p (over-cover) |

## 예측 맞았나?

- prompt 구체화: ✅ coverage 2× 향상 (1.93% → 3.88%)
- λ_cal ↑ → cloth_pct 보정: ✅ shirt prompt 에서 λ=0.1 → λ=1.0 로 87.8% → 67.2% 감소 (15 %p 보정)
- 그러나 **prompt 구체화 + λ_cal ↑ 동시 적용 시 의외의 over-cover 강화**: ❌
  - SAM3 신호 ↑ → BCE 가 더 많은 가우시안에 cloth gradient 흘림 → soft-cal 의 보정 능력 한계 초과
  - 결과: cloth_pct 가 SAM3 GT (3.88%) 의 **7×** 인 26.93% 로 over-cover

## 결정

**`shirt` prompt + soft-cal b (exp002 결과) 가 jumpingjacks 의 SAM3 GT 에 가장 근접** (1.32% vs GT 1.93%, 0.61 %p 차이).
canonical `3_output/jumpingjacks/ckpt_softcal_best` 를 [[exp002_softcal_ablation]] 의 softcalv2_jumpingjacks_b 로 **복원**.

## 다음

- single-prompt + hard-BCE 의 supervision 강화는 over-cover 를 가속함이 확인됨 → multi-prompt union (e.g., `shirt` ∪ `orange hoodie`) 또는 **multi-class cloth_logit** 으로 cloth 의 *내부 구조* 를 명시하는 방향
- iterative SAM3 refinement (학습된 cloth_logit 을 SAM3 점 prompt 로 재공급) — 단일 prompt 의 한계 우회
- 또는 더 강한 λ_cal (1.5, 2.0) 단독 sweep 으로 calibration scaling 한계 측정
