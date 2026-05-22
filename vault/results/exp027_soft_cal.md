---
title: exp027 — Memory-based negative mining calibration (9페이지 core)
status: done (mixed)
date: 2026-05-15
code: 2_process/exp027_soft_cal.py
---

# exp027

## 질문

미팅 [[../006Meeting_2026-05-15|2026-05-15]] 9페이지 원래 plan — easy sample memory + feature distance 기반 soft-label calibration 이 *under-cover / over-cover* 문제를 fix 할 수 있는가?

## 9페이지 공식 (구현)

1. Easy sample 식별 (FIFO memory, K=1000):
   - cloth: `soft_target > 0.95` top-K (highest cloth probability)
   - non-cloth: `soft_target < 0.05` bottom-K (lowest cloth probability)
2. Feature distance:
   - (a) Gaussian param: `[xyz, scale, rot, opacity, SH_DC]` (14-d, z-score normalized) L2
   - (b) DINOv3 ViT-S/16 patch token (384-d): render → patch grid → Gaussian-projected pixel lookup
3. Calibration (`cal_strength = 0.45`):
   - cloth-side (raw > 0.5): `target = 1 − α`
   - non-cloth-side: `target = 0 + α`
   - `α = cal_strength · σ((d − d_ref) / τ)`, where d_ref/τ from hard-sample distance distribution
4. BCE on calibrated `target_cal` (2000 iter, lr 0.05)

## 설정

- D-NeRF 우선 (사용자 지시: N3V 무시)
- 2 scene: **jumpingjacks** (sam3_union, exp010 baseline cloth_pct 37% — 이미 깨끗) + **standup** (sam3_mc_cloth, cloth_pct 64% — middle mass 큰 noisy 분포)
- 4 flavor: none / a / b / both — 4 GPU 병렬

## 결과 1: jumpingjacks (clean baseline)

### `target_cal` 분포 (raw soft_target → 4 flavor)

| flavor | mean | >0.5 | >0.7 | >0.9 |
|---|---|---|---|---|
| none (raw exp010) | 0.397 | 9263 | 8695 | **6102** |
| a (Gauss param) | 0.473 | 9263 | 8455 | **3834** (−37%) |
| b (DINOv3) | 0.472 | 9263 | 9263 | 6880 |
| both | 0.472 | 9263 | 9111 | 4464 |

→ flavor **(a) 가 가장 aggressive** — high-confidence cloth 의 37% 를 [0.5, 0.9] 범위로 push (outlier "demote"). flavor (b) DINOv3 는 거의 push 안 함.

### 시각 (`/tmp/exp027_jj_grid.png`, threshold 0.5)

4 flavor 모두 hoodie 깔끔 blue, 동일해 보임. **threshold 0.5 에선 calibration 효과 invisible** (cloth_pct@0.5 = 37.16% 동일).

→ jumpingjacks 처럼 baseline 이 이미 깨끗하면 calibration 의 visible improvement 없음.

## 결과 2: standup (noisy baseline)

### Raw distribution

- raw>0.5 = 12,680 (46%), raw>0.7 = **0** ← middle mass 큼, 절대 confident 없음

→ baseline (none) recolor @ t=0.7 = 5% (under-cover, vest 만)

### Calibration 후 (`cal_strength=0.45`)

| flavor | cal>0.5 | cal>0.7 | cal>0.9 |
|---|---|---|---|
| none | 12,680 | **0** | 0 |
| a | 12,680 | **10,388** | 1,040 |
| b | 12,680 | 10,775 | 0 |
| both | 12,680 | 10,504 | 0 |

→ middle-mass cloth-side Gaussian 의 82-85% 가 cal > 0.7 로 push.

### 시각 (`/tmp/exp027_su_grid.png`, threshold 0.7)

| flavor | 결과 |
|---|---|
| **none** (baseline) | **vest 만 blue** (under-cover, pants/arms 그대로) |
| a | vest + pants + **arms (skin) 도 blue** (over-cover into skin) |
| b | vest + pants + arms blue (over-cover) |
| both | vest + pants + arms blue (over-cover) |

→ calibration 이 vest+pants 까지 정상 cover 확장 (positive) BUT *arms (skin)* 까지 잘못 cover (negative).

## 핵심 통찰

### Calibration mechanism

미팅 plan 의 의도: **distance 멀수록 target 을 0.5 쪽으로 (애매하게)**. 구현 정확:
- d small (typical) → α small → cloth-side cal ≈ 1
- d large (outlier) → α 큰 (≤ 0.45) → cloth-side cal ≈ 0.55

### 작동 시나리오 vs 한계

**작동**:
- jumpingjacks: distribution shift (a flavor 가 high-conf 37% demote) — quantitative
- standup: middle-mass 를 high 쪽으로 push 해서 pants 까지 cover 확장

**한계**:
- arms (skin) 가 cloth-side 분류된 후, feature 가 z-score normalized space 에서 vest 와 *충분히 close* → α 작음 → cal 높음 → BCE 가 confident cloth 로 학습
- 즉 *intra-class feature similarity* 가 너무 강할 때 calibration discrimination 부족

### 왜 DINOv3 (b) 도 비슷한 결과?

- DINOv3 는 visually arm 과 vest 를 다르게 봐야 하나, render 의 patch (16×16) 해상도 한계로 arm-vest boundary 부근 patch 가 mixed
- DINOv3 patch lookup 이 단일 frame (mid frame) 으로만 했음 — multi-frame aggregation 하면 개선 가능

## paper 활용

**Ablation table** (jumpingjacks): none / a / b / both 의 cal>0.9 비율 변화 → algorithm 의 distribution shift 효과 정량 입증

**Discussion**: calibration 은 *baseline 의 distribution 특성* 에 따라 다른 효과 — clean bimodal (jumpingjacks) 엔 visible improvement 없음, middle-mass (standup) 엔 coverage extension 있지만 boundary discrimination 약함

**Future work**: (1) multi-frame DINOv3 patch aggregation, (2) class-specific feature weighting (color-dominant for skin/cloth separation), (3) mutual exclusion between cloth/skin classes

## 산출물

- `3_output/jumpingjacks/ckpt_exp027_{none,a,b,both}/` — 4 ckpts + soft_target_raw.npy + soft_target_cal.npy + calib_meta.json
- `3_output/standup/ckpt_exp027_{none,a,b,both}/` — 4 ckpts
- `3_output/{scene}/recolor_exp027_{flavor}/` — 8 recolor 결과
- `/tmp/exp027_jj_grid.png`, `/tmp/exp027_su_grid.png` — 시각 비교
- `utils/soft_cal.py` (재사용) — feat_a, DinoV3PatchExtractor, SoftCalMemory 기존 구현

## 다음 시도 후보

- `cal_strength` 더 크게 (0.45 → 0.55+) → cloth-side 최저 0.5 보다 낮춰서 threshold cross 가능
- Iterative refinement: 학습 중 N step 마다 memory 재구성 (현재는 single-pass)
- 9페이지 plan 의 *원래 의도* (BCE 학습 *중* 매 iter calibration) 으로 재구현 — exp007-009 의 tier2 pipeline 으로 통합
- 다른 scene (D-NeRF mutant, hellwarrior 등) 에서 generalization
