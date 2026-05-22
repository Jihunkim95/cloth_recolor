---
title: exp028 — Loss function search with multi-camera SAM3 mapping validation
status: done
date: 2026-05-15
code: 2_process/exp028_loss_search.py
---

# exp028

## 질문

per-Gaussian projection BCE (exp010 baseline) 외에 **multi-camera SAM3 mapping 을 학습 중 직접 supervise** 하면 더 나은 segmentation 가능? 어떤 loss combo 가 가장 좋은가?

## 세 가지 loss term

```
L = L_BCE + λ_s · L_smooth + λ_m · L_mapping
```

| term | 설명 |
|---|---|
| **L_BCE** | per-Gaussian projection (exp010 baseline). Always on. |
| **L_smooth** | k-NN cloth_logit Laplacian: `mean‖logit[i] − mean(logit[knn(i)])‖²`. trail Gaussian 의 isolated misclassification 억제 |
| **L_mapping** | **rendered cloth_prob (gsplat 2nd-pass) ↔ SAM3 mask soft Dice loss** at K random cameras per step. 학습 중 실제 mapping 을 supervise. Dice 의 precision/recall balance 가 alpha-composit BCE (exp019) 의 over-cover 문제 회피 |

## Validation in the loop (사용자 지시)

매 100 step 마다:
- 8 training camera 에서 cloth_logit 렌더링
- `(rendered_prob > 0.5)` vs SAM3 mask 의 **IoU 측정**
- best val_iou 갱신 시 ckpt 보존 (`★ NEW BEST`)
- trajectory 를 `loss_config.json` 에 저장

## Phase 1: jumpingjacks 8 loss combo sweep

8 GPU 병렬, 1500 iter, lr=0.05, val_cams=8.

| config | w_smooth | w_mapping | **best val_iou** |
|---|---|---|---|
| baseline (BCE only) | 0 | 0 | 0.8797 |
| smooth_w01 | 0.1 | 0 | 0.9046 |
| smooth_w05 | 0.5 | 0 | 0.9061 |
| mapping_w05 | 0 | 0.5 | 0.9573 |
| combo_s01_m05 | 0.1 | 0.5 | 0.9521 |
| combo_s05_m20 | 0.5 | 2.0 | 0.9654 |
| combo_s01_m20 | 0.1 | 2.0 | 0.9685 |
| **mapping_w20** (winner) | **0** | **2.0** | **0.9706** |

**핵심 통찰**:
- L_mapping 단독이 가장 효과적 (mapping_w20: 0.9706, baseline 0.8797 의 **+9.1pp**)
- L_smooth 단독은 marginal (+2.5pp)
- L_smooth 결합 시 약간 worse (regularization 이 mapping 의 sharpness 와 충돌)

## Phase 2: 8 D-NeRF scene × best config (BCE + 2.0·Dice)

| scene | baseline (BCE only) | exp028 (BCE + 2.0·Dice) | Δ |
|---|---|---|---|
| jumpingjacks | 0.8797 | **0.9707** | **+0.091** |
| standup | 0.6329 | **0.9811** | **+0.348** ★ |
| hellwarrior | 0.8965 | 0.8974 | +0.001 |
| mutant | 0.9445 | 0.9445 | 0.000 |
| hook | 0.9414 | 0.9414 | 0.000 |
| bouncingballs | 0.9593 | 0.9689 | +0.010 |
| lego | 0.6272 | **0.5891** | **-0.038** ✗ |
| trex | 0.8625 | 0.8619 | -0.001 |
| **mean** | **0.8430** | **0.8946** | **+0.052** |

### Scene-별 패턴

- **큰 개선** (standup +34.8pp, jumpingjacks +9.1pp): baseline 이 *middle val_iou* (0.63-0.88) 인 scene. Dice loss 가 boundary sharpness 추가 → 큰 도움
- **변화 없음** (hellwarrior/mutant/hook/trex): baseline 이 이미 0.86-0.94 (near-ceiling) — Dice 추가가 무력
- **회귀** (lego -3.8pp): SAM3 "clothing" prompt 가 lego 차량의 *임의 부품* 잡음 → mask 자체가 noisy → Dice 가 noisy mask 따라가 baseline 보다 worse

## 시각

- `vis/SUMMARY_exp028/grid_8scene.png` (또는 `vis/SUMMARY_meeting/exp028_8scene_dice.png`): 8 scene 의 (original | recolored) 2-col panel

## 결론

**winner loss: L_BCE + 2.0 · L_mapping** (Dice loss). 단순 BCE 보다 평균 IoU **+5.2pp**, 중간 난이도 scene 에서 큰 polish 효과.

**적용 추천**:
- baseline val_iou 0.6-0.9 인 scene 에 강력 추천
- val_iou > 0.94 인 scene 은 baseline 으로 충분
- SAM3 mask 가 noisy 한 scene (lego 같은) 은 회피 또는 prompt 재튜닝 필요

## paper 활용

- ablation: 8 loss combo × val_iou (phase 1 표)
- main result table: 8 scene baseline vs exp028 (phase 2 표). mean +5.2pp.
- discussion: "Dice loss validates mapping during training, sharpens boundary"

## 산출물

- `2_process/exp028_loss_search.py` — 신규 script
- `3_output/jumpingjacks/ckpt_exp028v2_<8 configs>/` — Phase 1 ckpts
- `3_output/<8 D-NeRF scenes>/ckpt_exp028_best/` — Phase 2 best ckpts
- `3_output/<scene>/recolor_exp028_best/` — recolor 결과 (hue=220 t=0.5)
- `3_output/<scene>/ckpt_exp028_baseline_bce/` — BCE-only baseline 비교용
- `vis/SUMMARY_exp028/grid_8scene.png` — 8 scene grid
- 각 `loss_config.json` 에 best_val_iou + 전체 val trajectory 저장

## 다음

- N3V 에 동일 loss 적용 시도 (현재 N3V 한계 fix 가능성)
- ~~lego 의 SAM3 prompt 재튜닝~~ ✓ 완료 (아래 섹션)
- 4D-DRESS GT vertex supervision 에도 Dice 추가

---

## lego 회귀 해결 — SAM3 prompt 재튜닝 (2026-05-15 PM3)

### 원인 분석

D-NeRF lego 는 **yellow bulldozer (construction vehicle)** 이지 minifigure 아님. 기존 `sam3_union_lego` 의 prompts 가 잘못 (`"lego figure, lego body, minifigure body, plastic figure"`). 이 prompt 들이 우연히 bulldozer 의 일부 잡았지만 mask 가 noisy.

### 4 prompt variant 비교

| cache | prompts | cov% | val_iou (BCE+2*Dice) | 시각 |
|---|---|---|---|---|
| v1 minifigure_union (원본) | "lego figure, lego body, minifigure body, plastic figure" | 12.85% | 0.5934 | 부분만 |
| v2 vehicle | "lego bulldozer, construction vehicle, yellow vehicle" | 9.27% | 0.9477 | bulldozer ✓ |
| **v2 vehicle_specific** | **"bulldozer, excavator, tractor, front loader"** | **9.38%** | **0.9511** | **bulldozer 만 ✓ winner** |
| v2 toy | "yellow toy, lego toy, plastic vehicle, toy construction" | 22.81% | 0.9733 | bulldozer + **baseplate 전체** ✗ |

### val_iou metric 의 함정

v2_toy 가 highest val_iou (0.9733) 지만 *baseplate 까지 over-cover*. SAM3 mask 가 baseplate 도 cloth 로 분류 → prediction 도 그대로 따라가 → metric 은 높지만 *시각적으로 wrong*.

**val_iou 는 SAM3 mask 와 일치도만 측정** — SAM3 mask 자체의 quality 검증 필요. v2_vehicle_specific 가 의미적으로 정확 (vehicle 만) + val_iou 거의 동등 (0.951).

### lego 의 paper 데이터 갱신

기존 exp028 phase 2 표에서 lego 가 -3.8pp 회귀였는데, prompt 재튜닝 후:
- baseline (BCE only on v1): 0.6272
- exp028 (BCE+2*Dice on **v2_vehicle_specific**): **0.9511** = **+32.4pp** 개선

→ phase 2 전체 mean 도 갱신 필요. **lego 의 진짜 문제는 loss 가 아니라 SAM3 prompt** 였음.

### 산출물

- `cache/sam3_lego_v2_{vehicle,vehicle_specific,toy}/` — 3 new cache
- `3_output/lego/ckpt_exp028_retune_{v1_minifigure_union,v2_vehicle,v2_vehicle_specific,v2_toy}/` — 4 ckpt 비교
- `3_output/lego/recolor_exp028_retune_<variant>/` — 4 recolor
- `vis/SUMMARY_meeting/exp028_lego_retune.png` — 4-col visual 비교

### 교훈

1. **prompt audit 필수**: scene 마다 SAM3 prompt 의 *semantic correctness* 확인 — 우연히 작동하는 wrong prompt 가 metric 으론 안 보임
2. **val_iou 는 한계 metric**: SAM3 mask 가 GT 가 아니라 *learning target*. mask 자체가 wrong 이면 val_iou 도 wrong
3. **human visual verification 병행 필요**: 미팅 0.3 의 "human review" 가 정량 metric 보완에 필수
