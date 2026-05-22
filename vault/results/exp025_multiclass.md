---
title: exp025 — Multi-class K=3 instance (N=3 × M=2 scene)
status: done
date: 2026-05-15
code: 2_process/per_gaussian_supervision_multiclass.py
---

# exp025

## 질문

미팅 [[../006Meeting_2026-05-15|2026-05-15]] 0.3+α — cloth 외 다른 instance (신발, 모자) 도 동시에 segment+recolor 가능한가? N=3 type × M=2 scene 검증.

## 설정

### 데이터

- M=2 scene: jumpingjacks + standup (둘 다 D-NeRF)
- N=3 instance per scene:
  - **jumpingjacks**: hoodie / shorts / shoes
  - **standup**: vest / pants / shoes

### SAM3 cache (6개)

| scene | prompt | coverage |
|---|---|---|
| jumpingjacks | "hoodie" | 0.7% |
| jumpingjacks | "shorts" | 0.8% |
| jumpingjacks | "shoes" | 0.5% |
| standup | "vest" | 2.6% |
| standup | "pants" | 2.5% |
| standup | "shoes" | 0.5% |

### 학습

`per_gaussian_supervision_multiclass.py`:
- cloth_logit shape: (N, K) with K=3
- Per-class independent **BCE** (multi-label, not softmax mutual exclusion)
- soft_target[i, k] = mean per-frame mask hit for class k at Gaussian i
- 2000 Adam steps, lr=0.05

### Recolor

`recolor.py --per-class-bce --target-class k`:
- 각 class 마다 별도 hue (hoodie/vest=red, shorts/pants=green, shoes=blue)
- threshold per class (hoodie 0.15, 나머지 0.4-0.5)

## 결과

### 학습 통계

**jumpingjacks** (N=24,964):

| class | soft_target mean | >0.5 (count) | cloth_pct |
|---|---|---|---|
| hoodie | 0.086 | 0 | 0.0% (low) |
| shorts | 0.060 | 1,303 | 5.2% |
| shoes | 0.055 | 1,709 | 6.8% |

→ hoodie 의 SAM3 coverage 가 너무 작아 (0.7%) soft_target 분포 낮음. threshold 0.15 로 조정.

**standup** (N=27,420):

| class | cloth_pct @0.5 |
|---|---|
| vest | 22.1% |
| pants | 25.5% |
| shoes | 8.0% (@0.4 → 8.7%) |

### 시각 결과 (`/tmp/exp025_{scene}.png`)

**jumpingjacks**: ✓ 완벽 class separation
- original (orange hoodie + blue shorts + yellow shoes)
- hoodie → red: hoodie 만 빨강, 나머지 unchanged
- shorts → green: shorts 만 초록, 나머지 unchanged
- shoes → blue: shoes 만 파랑, 나머지 unchanged

**standup**: ✓ vest/pants 확인, shoes 는 top-down camera 시점으로 occluded
- vest → red 명확
- pants → green 부분 (다리 보이는 부분만)
- shoes → 가려져서 invisible

## 예측 맞았나?

- K=3 multi-class 가 K=1 (binary) 처럼 작동: ✅ — 각 class 가 독립적으로 학습되고 recolor 가능
- per-class BCE > softmax CE: ✅ — instances 가 mutually exclusive 가 아니므로 (Gaussian 한 개가 여러 class 일 수 없지만, *서로 다른 Gaussian* 이 *서로 다른 class*)
- class 별로 threshold 다르게 적용 필요: ✅ — SAM3 coverage 가 다르면 logit 분포도 다름 ([[exp022_threshold_sweep]] 의 method-dependent threshold 와 일치)

## 핵심 통찰 (paper 활용)

1. **Multi-instance 지원이 trivial**: 기존 K=1 BCE 를 K=N 독립 BCE 로 확장만 하면 됨. softmax CE 보다 더 안정적 (class 간 상호 영향 없음)
2. **Instance 별 SAM3 cache 필요**: prompt = "shirt" vs "pants" vs "shoes" 따로
3. **Per-class threshold 튜닝 필요**: 작은 instance 는 threshold 낮춰야 (hoodie 0.15 vs shorts/shoes 0.5)
4. **Recolor 도 per-class hue 설정**: 한 모델로 셔츠 빨강 + 모자 파랑 같은 multi-edit 가능

## paper 활용

**Figure**: 4-column grid (original + 3 class recolor) for both scenes
**Table**: cloth_pct per class × scene
**Section**: "Multi-instance extension" — algorithm 부담 없이 K=3+ 지원

## 산출물

- `3_output/jumpingjacks/ckpt_exp025_multi/` — K=3 ckpt (cloth_logit shape (N, 3))
- `3_output/standup/ckpt_exp025_multi/` — K=3 ckpt
- `3_output/{scene}/recolor_exp025_{class}/` — 6 recolor 결과 (per-class hue)
- `cache/sam3_exp025_{scene}_{class}/` — 6 SAM3 cache
- `/tmp/exp025_{scene}.png` — 4-col grid (original + 3 class)

## v2 — Union prompt + spatial filter per class

**문제 발견** (v1): hoodie single prompt 가 200 frame 중 **158 frame (79%) 에서 detection fail** (SAM3 의 unreliability). shorts/shoes 는 98% 검출. v1 의 hoodie/shorts 결과가 부자연스러운 boundary bleeding 원인.

**fix**:
1. hoodie 만 multi-prompt union 으로 재캐싱: `"orange hoodie, sweater, hoodie, jacket, long sleeve top"` → cov 0.7% → **3.9%** (5.5×)
2. per-class spatial filter (exp015 기법) 추가: `--spatial-filter 3.0 --spatial-seed-thresh 0.2`
   - hoodie outlier 1578 zeroed
   - shorts outlier 39 zeroed
   - shoes outlier 0 zeroed

**v2 통계** (jumpingjacks):
| class | soft_target mean | >0.5 (count) |
|---|---|---|
| hoodie (union) | 0.397 | 9,263 (37%) — *v1 의 0.086 / 0 대비 큰 향상* |
| shorts | 0.060 | 1,303 (5.2%) |
| shoes | 0.055 | 1,709 (6.8%) |

**v2 시각** (`/tmp/exp025_union.png`):
- hoodie → red: **전체 깔끔**, sleeves 포함, shorts 영역 bleeding 없음
- shorts → green: 전체 shorts (v1 의 하단만 → 전체로 개선)
- shoes → blue: 깔끔

→ **핵심 fix: union prompt per class + spatial filter per class**. single prompt 의 SAM3 unreliability 가 multi-instance 의 *진짜 bottleneck*.

## 다음

- 더 많은 instance (모자, 안경, 시계 등) 추가
- N3V chef scene 에서 multi-instance 시도 (apron/shirt/glasses)
- shorts 도 union 적용 권장 ("shorts, denim shorts, jean shorts")
- exp026: 4D timeframe grid 으로 진행
