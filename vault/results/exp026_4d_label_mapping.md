---
title: exp026 — 4D timeframe grid label mapping
status: done (negative)
date: 2026-05-15
code: 2_process/per_gaussian_supervision_4d.py
---

# exp026

## 질문

미팅 [[../006Meeting_2026-05-15|2026-05-15]] 0.3+α — 현재 `soft_target` 이 frame 1장 기준의 mean. **4D 축으로 늘려서** time bucket 별 cloth label 학습하면 시간에 따라 cloth ↔ background 가 바뀌는 경우 더 정확?

## 설정

- N3V sear_steak (ckpt_baseline_4dgs, 14000 iter, N=90,543)
- SAM3 cache: union (shirt+apron, cov 3.76%)
- B = 10 buckets, n_frames=300 → 30 frame per bucket
- cloth_logit shape: **(N, B) = (90543, 10)**
- Per-bucket BCE: soft_target[i, b] = mean mask hit for Gaussian i in frames [b·30, (b+1)·30)
- 2000 iter, lr 0.05

## 결과

### Per-bucket soft_target distribution

| bucket | mean | >0.5 count |
|---|---|---|
| 0 | 0.111 | 5,305 |
| 1 | 0.118 | 6,515 |
| 2 | 0.117 | 6,370 |
| 3 | 0.115 | 5,617 |
| 4 | 0.119 | 6,017 |
| 5 | 0.118 | 5,983 |
| 6 | 0.118 | 6,069 |
| 7 | 0.124 | 6,550 |
| 8 | 0.124 | 6,486 |
| 9 | 0.122 | 6,247 |

→ **mean 0.111-0.124 (편차 0.013), >0.5 count 5305-6550 (편차 ±10%)**. bucket 마다 매우 유사.

### Per-bucket cloth_pct@0.5 (학습 후)

| bucket | cloth_pct |
|---|---|
| 0 | 5.9% |
| 1 | 7.2% |
| 2 | 7.0% |
| 3 | 6.2% |
| 4 | 6.6% |
| 5 | 6.6% |
| 6 | 6.7% |
| 7 | 7.2% |
| 8 | 7.2% |
| 9 | 6.9% |

→ **5.9-7.2% 범위 (≈1% 변동)** — 거의 동일.

## 예측 맞았나?

- 가정: bucket 별로 cloth Gaussian 이 다를 것 (chef 동작에 따라)
- 실제: ✗ bucket 별 거의 동일 → chef 옷 자체는 시간 불변, deformation 만 바뀜
- → 시간 차원 라벨이 의미 없음 (negative)

## 핵심 통찰

**4D label mapping 이 유용한 경우** (이 실험에선 해당 안 됨):
- 같은 Gaussian 이 cloth ↔ background 으로 *의미적으로* 전환 (e.g., 사람이 옷을 벗는 video)
- Object 가 frame 마다 들어왔다 나갔다 (e.g., 캐릭터 추가/삭제 video)
- 시간에 따라 *다른 옷* 입은 시나리오 (e.g., morphing avatar)

**chef 시나리오 한계**:
- chef 가 한 outfit 유지 → cloth Gaussian set 은 시간 불변
- deformation MLP 가 *그 Gaussian set* 을 시간에 따라 움직임
- per-bucket cloth_logit 는 같은 정보 10× 중복 학습

→ **timeframe grid 는 시나리오 dependent**. chef-style 정적 outfit 데이터엔 K=1 (단일 시간 불변 라벨) 가 sufficient.

## paper 활용

- ablation: "4D bucket label adds no signal for outfit-stable scenes; reserved for outfit-changing future scenarios" 한 줄
- limitation: dataset 의 outfit dynamic 한계 명시

## 산출물

- `3_output/n3v_sear_steak/ckpt_exp026_4d/` — K=10 ckpt (cloth_logit shape (N, 10))
- viewer 에서 `target class slider 0-9` 로 bucket 전환 가능 — 거의 동일 결과

## 다음

- exp027: memory-based negative mining (9페이지 원래 plan)
- (시간 후) outfit-changing dataset 발굴 시 4D bucket 재시도
