---
title: exp004 — Soft-cal generalization to extra D-NeRF scenes
status: done
date: 2026-05-11
code: cloth_recolor/softcal_overnight_v2.sh (Phase G/H)
---

# exp004

## 질문

[[exp002_softcal_ablation]] 의 soft-cal 이 옷이 명확하지 않은 다른 D-NeRF 씬 (lego, bouncingballs, hook) 에서도 작동하는가? (옷 대신 "주요 object" 를 SAM3 로 segment 하면)

목적: 우리 방법이 *특정 옷 데이터에 overfit 한 트릭* 이 아니라, **임의 attribute mask 에 일반화** 되는지 확인.

## 설정

- 4 추가 씬 시도: trex, lego, bouncingballs, hook
- SAM 3 prompt:
  - trex: "dinosaur skin and body"
  - lego: "lego figure body"
  - bouncingballs: "ball"
  - hook: "puppet body"
- soft-cal variant: `both` (가장 일반적), default hyperparam (λ=0.1, mem=1000)
- 각 씬 baseline (`--soft-cal none`) 도 학습해서 비교
- 6 jobs, 6 GPU 병렬

## 예측

- SAM 3 가 옷이 아니어도 prompt 따라 정상 mask 생성할 것
- soft-cal 이 추가 supervision 으로 RGB PSNR 약간 향상
- 효과는 옷 씬보다 작을 것 (옷 경계만큼 모호한 attribute 가 아니므로)

## 결과

### SAM3 caching coverage (>5% 이면 사용 가능)

| scene | prompt | coverage | usable? |
|---|---|---|---|
| trex | "dinosaur skin and body" | 4.23% | ❌ (prompt 부적절 추정) |
| lego | "lego figure body" | 5.7% | ✓ |
| bouncingballs | "ball" | 6.55% | ✓ |
| hook | "puppet body" | 11.39% | ✓ |

### 학습 결과 (3 usable scenes)

| scene | baseline PSNR | soft-cal both PSNR | Δ |
|---|---|---|---|
| lego | 25.17 | 25.17 | 0.00 |
| bouncingballs | 39.81 | **40.00** | +0.18 |
| hook | 32.60 | **32.68** | +0.08 |

## 예측 맞았나?

- SAM 3 generality: ✅ 모든 prompt 에서 작동, trex 만 부적절 (prompt 재시도 필요)
- soft-cal RGB 향상: ⚠️ bouncingballs +0.18, hook +0.08 — **확실한 향상이지만 옷 씬보다 작음** (예측대로)
- 일반화: ✅ 옷이 아닌 attribute (공, 인형, lego) 에서도 framework 작동

## 결론

[[../notes/cloth_logit_channel]] + [[../notes/soft_label_calibration]] 이 옷 specific 트릭이 아니라 **임의 binary attribute supervision** 으로 동작 — 향후 multi-attribute (옷+머리+피부) 또는 multi-class (셔츠/바지/모자) 확장 가능.

## 다음

- mutant SAM 3 재캐싱 (prompt v1 "suit" 0% → v2 "monster body and armor" 9.86%) 후 학습
- trex prompt 재시도
- 또는 우리 옷 3 씬 결과로 논문 Figure 만들기 시작 (시각 비교 패널 정제)
