---
title: 실험 설계 2026-05-15 (미팅 피드백 반영)
date: 2026-05-15
tags: [experiment, plan, soft-label, calibration]
---

# 실험 설계 (미팅 [[006Meeting_2026-05-15]] 피드백 반영)

## 전체 목표

현재까지 구현 (per-Gaussian projection + spatial filter) 을 baseline 으로,
**미팅에서 제시된 5 가지 새 axis** 를 단계적으로 추가하며 *2D → 3D label mapping* 의 design space 를 체계적으로 탐색.

## 현재 구현 자산 (재사용)

- `per_gaussian_supervision.py` (D-NeRF), `_n3v.py`, `_4ddress_gt.py`
- `recolor.py` (threshold + spatial filter + min_sat/val)
- `sam3_cache.py`, `sam3_cache_n3v.py` (single + union prompt)
- 학습된 ckpt: D-NeRF 8 × baseline+exp010, 4D-DRESS 8 × baseline+exp013, N3V 6 × baseline+exp015
- vault: exp001-020 노트

## 실험 sprint

### Phase 1 (즉시 시작 — 코드/데이터 재사용)

#### exp021: Pipeline 효율성 측정 (미팅 0.2)

**질문**: baseline (매번 SAM3) vs 우리 방법 (train 1회, infer cloth label) wall-time?

**설정**:
- 1 scene (jumpingjacks) 의 200 frame
- baseline: 각 frame `4DGS render → SAM3 detect "hoodie" → HSV swap masked pixels → finalize` 의 평균 시간
- 우리: 각 frame `4DGS render with cloth_logit → HSV swap > threshold → finalize` 의 평균 시간
- 측정: per-frame wall-time, SAM3 부분 cost 분리

**산출**: 시간 비교 표 + paper 의 efficiency claim 데이터

#### exp022: Threshold 매핑 ablation (미팅 0.3+α)

**질문**: per-Gaussian soft_target 의 binary 화 threshold?

**설정**:
- D-NeRF jumpingjacks ckpt_exp010 + N3V sear_steak ckpt_exp015 + 4D-DRESS Take 1 ckpt_exp013 각각
- threshold 6 단계 sweep: 0.0 / 0.1 / 0.3 / 0.5 / 0.7 / 0.9
- 각 threshold 별 recolor panel 생성 + cloth_pct 통계

**산출**: threshold-cloth_pct 표 + 시각 비교 grid + 권장 threshold 결정

### Phase 2 (1-2일 — soft_target 수정)

#### exp023: Grid-based soft_target (미팅 0.3)

**질문**: center pixel lookup vs window 평균?

**현재**: `soft_target_i += mask[py, px]` (1 pixel)
**변경**: `soft_target_i += mask[py-k:py+k+1, px-k:px+k+1].mean()` (window 평균)

**설정**:
- window size: k=0 (현재 baseline), k=1 (3x3), k=2 (5x5), k=4 (9x9)
- jumpingjacks + sear_steak (D-NeRF + N3V)
- 각 window 별 soft_target distribution 분석 + recolor 결과

**산출**: window size ablation 표 + boundary smoothness 비교

#### exp024: Edge metric 정의 + 정성 figure (미팅 0.3)

**질문**: GT 없이 cloth 분리 품질 metric?

**설정**:
- Sobel edge (RGB 원본) 와 sigmoid gradient (prediction) 의 boundary alignment 측정
- Metric 1: edge IoU @ tau (RGB edge ∩ prediction edge / union)
- Metric 2: boundary distance (각 RGB edge pixel 의 nearest prediction edge 까지 거리 평균)
- 정성: 옷 boundary 확대 figure 5장 (8 scene 중 대표)
- Human review setup (3 person × 0/1 vote × 16 panel)

**산출**: edge metric implementation + 4 baseline 비교 (no-filter, spatial, exp019 joint, exp015 main)

### Phase 3 (1주 — multi-object + 4D)

#### exp025: Multi-object N=3 × M=2 scene (미팅 0.3+α)

**질문**: cloth 외 (신발, 모자) 등 multiple instance label 동시 학습 가능한가?

**설정**:
- N=3 type: `["shirt", "shoes", "hat"]`
- M=2 scene: jumpingjacks (D-NeRF), sear_steak (N3V)
- SAM3 3개 prompt 별로 mask cache (multi-prompt union 이 아니라 *분리* 저장)
- K=3 multi-class cloth_logit 학습 (각 Gaussian 이 어느 instance class 인지 softmax)

**산출**: 3-class confusion matrix + 각 class 별 recolor 가능 검증 + 멀티 application 검증 (셔츠 빨강, 모자 노랑)

#### exp026: 4D timeframe grid + optical flow (미팅 0.3+α)

**질문**: 4D 축으로 label mapping 확장?

**설정 옵션 A** (단순 timeframe grid):
- frame 을 T_grid 개의 bucket 으로 분할 (예: 300 frame → 30 bucket × 10 frame)
- bucket 마다 soft_target 별도 계산 → `soft_target[i, bucket]`
- 학습 시 bucket 평균 또는 weighted

**설정 옵션 B** (optical flow 기반):
- 인접 frame 의 optical flow 추정 (RAFT 등)
- flow 따라 mask 를 propagate → temporal smoothness
- 같은 Gaussian 의 시간별 soft_target 이 flow 와 일치하는지 검증

→ Option A 부터 시도 (구현 1일), 효과 보이면 Option B 추가

### Phase 4 (1주+ — 9페이지 원본)

#### exp027: Memory-based negative mining (9페이지)

**질문**: easy sample memory 로 soft-label calibration 효과?

**설정**:
- 학습 iteration 마다:
  1. Easy sample 식별: `sigmoid(cloth_logit) > 0.95` (cloth easy), `< 0.05` (not-cloth easy)
  2. 각 class top-1000 점 메모리 저장 (FIFO)
  3. 모든 학습 point 의 feature distance 계산:
     - (a) Gaussian param distance: `(xyz, scale, rot, opacity, SH)` 의 nearest easy point 까지 L2
     - (b) Image feature distance: `DINOv3(rendered_patch)` 의 nearest easy patch 까지 cosine
  4. distance 클수록 soft-label `α, β` 크게:
     - `C=1 → 1 − β · f(x)`
     - `C=0 → 0 + α · f(x)`

**Method 이름**: "Memory-based negative mining for 4D Gaussian Splatting instance label"

**설정**:
- jumpingjacks + sear_steak 에서 (a), (b), (a+b) 3 가지 ablation
- baseline: exp015 (no calibration)
- 평가: edge metric (exp024) + 정성 figure

**산출**: 9페이지 core contribution + paper 의 핵심 실험

## 작업 순서 (sprint)

| Day | 작업 | 산출물 |
|---|---|---|
| Day 1 (오늘) | **exp021** (효율성) + **exp022** (threshold) | baseline 표 + threshold 권장값 |
| Day 2 | **exp023** (grid soft_target) | window ablation |
| Day 3-4 | **exp024** (edge metric + figure) | metric impl + 정성 figure |
| Day 5-7 | **exp025** (multi-object) | 3-class 검증 |
| Day 8-10 | **exp026** (4D label) | timeframe grid 결과 |
| Day 11-17 | **exp027** (memory mining) | 9페이지 core |
| Day 18-21 | paper 작성 | draft v1 |

## 평가 기준 (각 exp)

1. **수치** (가능한 곳): cloth_pct, edge IoU, training time
2. **정성**: 옷 boundary 확대 figure, baseline 대비 차이
3. **failure mode**: 실패 case 도 기록 (limitation section 용)

## 관련 노트

- [[006Meeting_2026-05-15]] — 미팅 원본
- [[results/exp010_per_gaussian_projection]] — current Stage 3 baseline
- [[results/exp015_n3v_train_time_spatial_filter]] — current calibration
- [[005Paper_outline]] — paper 구조
