---
title: 미팅 피드백 + 실험 응답 정리 (다음 미팅 prep)
date: 2026-05-15
tags: [meeting, response, prep]
---

# 미팅 피드백 + 실험 응답 정리

> 지난 미팅 [[006Meeting_2026-05-15]] 의 5 axis 피드백에 대해 1일 sprint (exp021~027) 로 응답한 결과 매핑. 다음 미팅에서 *각 피드백 → 실험 → 결과* 흐름으로 보고.

---

## 9페이지 원래 plan (재확인)

**문제 정의**: gaussian cloth label 이 hard 0/1 fix → soft-label 필요 (GT 없음)

**학습 공식 (hard label calibration)**:
- `C = 1` → `C = 1 − β · f(x_point)`  (cloth-side 점을 약간 낮춤)
- `C = 0` → `C = 0 + α · f(x_point)`  (non-cloth-side 점을 약간 높임)

**Soft-label 생성 절차**:
1. 학습 iteration 마다 easy sample 식별:
   - cloth: `sigmoid(cloth_logit) > 0.95` top-1000 (memory)
   - non-cloth: `sigmoid(cloth_logit) < 0.05` top-1000 (memory)
   - hard sample: 예측값 ~0.5 (애매한 점들)
2. 각 학습 point 의 feature distance 계산:
   - (a) **Gaussian param distance**: gaussian 의 parameters L2
   - (b) **DINOv3 image patch distance**: renderer 의 2D patch → DINOv3 token cosine
3. distance 클수록 soft-label α 크게 → calibration 더 강하게 적용

**Method 명**: *"Memory-based negative mining for 4D Gaussian Splatting instance label"*

**Application**: cloth recolor, watermark insertion, multi-instance editing

→ **응답**: [[results/exp027_soft_cal|exp027]] 으로 4-flavor ablation (none/a/b/both) 진행 중 (D-NeRF jumpingjacks).

---

## 미팅 피드백 (5 axis)

### 0) Baseline 비교 정의

| pipeline | 단계 |
|---|---|
| **baseline (infer)** | 4DGS → SAM3 cloth detection → color change → finalized RGB |
| **김지훈 (ours)** | (train) SAM3 with cloth → (infer) 4DGS + cloth → color change → finalized RGB |

→ **응답**: ✓ [[results/exp021_efficiency|exp021]] 측정 완료. baseline 180.2 ms/frame, ours 6.1 ms/frame, **speedup 29.78×**. SAM3 가 baseline 시간의 82.5% 차지.

### 0.1) GT 부재 → 수치 metric 불가능

> "Cloth label GT 가 없기 때문에 얼마나 옷을 더 잘 찾는지 수치적으로 체크는 확인 불가능"

→ **응답**: ✓ [[results/exp024_edge_metric|exp024]] 에서 *SAM3 boundary 를 pseudo-GT 로* 사용한 edge IoU/Chamfer metric 작성. jumpingjacks 5 ckpt 비교:
- baseline_hardBCE: IoU 0.121, Chamfer 12.06 px
- exp010 (per-Gaussian projection): **IoU 0.253 (2.1×), Chamfer 6.40 px (절반)**
- softcal_best: IoU 0.095, Chamfer 13.41 px (가장 worst — over-cover 심함)

→ paper 의 "supervision-aligned boundary metric" 으로 표기 (caveat: SAM3 supervision 이라 fully independent GT 아님).

### 0.2) SAM3 속도 분석

> "SAM3 가 얼마나 걸리느냐? 미리 SAM3 돌려두고 4DGS 에서 렌더링할때 옷을 바로 찾아내 두는 게 속도로서 어떤 이득이 있는가?"

→ **응답**: ✓ [[results/exp021_efficiency|exp021]] (위 0번과 같은 실험). breakdown:
- baseline: render 3.4 ms + **SAM3 inference 148.6 ms** + 2D swap 28.2 ms = 180.2 ms
- ours: render 6.1 ms (precomputed cloth_logit 으로 SH_DC swap 후 한 번 render) = **6.1 ms**

→ paper Table 1 의 "efficiency" 핵심 데이터.

### 0.3) 2D pixel → 3D GS label mapping (핵심)

> SAM3 label 은 2D pixel 기준. 단순 매핑은 0/1 hard. 개선: `soft_target_i = (True 갯수) / (SAM3_mask grid)` (grid=16). 2D Object Mask 정보를 어떻게 3D GS 학습 시 자연스럽게 매핑?

→ **응답**: 두 가지 실험.

**(1) 매핑 baseline ablation** [[results/exp022_threshold_sweep|exp022]]:
- `> 0` vs `> 0.5` vs sweep 6 threshold × 2 scene
- D-NeRF/exp010 sweet spot: **t=0.7** (bimodal distribution)
- N3V/exp015 sweet spot: **t=0.3** (spatial filter 로 logit 분포 압축)
- → threshold 는 method-dependent. paper 표 에 method 별 보고 필요.

**(2) Grid-based soft_target** [[results/exp023_grid_soft_target|exp023]]:
- center pixel (k=0) vs window (k=1,2,4) 평균
- 결과: cloth_pct **±0.5% 변화만** — 거의 효과 없음
- 원인: SAM3 mask 가 이미 sharp binary 라 window 평균이 boundary 외 영향 없음
- → paper ablation 에 "window-size robustness" 로 보고 (algorithm robustness 어필)

#### 평가 방법

> 1) 옷-다른영역 분리 지점 line 의 RGB pixel edge & prediction value edge → metric (수학)
> 2) 사람 review 0/1 투표 (휴먼)
> 3) 정성적 확대 figure

→ **응답**:
- (1) 수학 metric: [[results/exp024_edge_metric|exp024]] 완료 (위 참조). Sobel edge IoU + Chamfer distance.
- (2) 휴먼 review: 아직 미시행. exp024 의 4-5 scene zoom-in panel 을 Google Form 으로 정리 예정.
- (3) 정성 figure: exp024 의 `view_NNNN.png` (3-col overlay: RGB, cloth_prob heatmap, edge overlay) 사용 가능.

### 0.3+α) 추가 확장

#### (a) Multi-object N=3 × M=2 scene

> "cloth 외 신발, 모자 등 object type N×M (최소 N≥3, M≥2)"

→ **응답**: ✓ [[results/exp025_multiclass|exp025]] 완료.
- M=2 scene: jumpingjacks + standup (D-NeRF)
- N=3 per scene: jumpingjacks [hoodie/shorts/shoes], standup [vest/pants/shoes]
- 6 SAM3 cache, K=3 independent BCE
- v1 에서 hoodie single prompt 가 158/200 frame fail (SAM3 unreliability 발견)
- v2 에서 hoodie multi-prompt union ("orange hoodie, sweater, hoodie, jacket, long sleeve top") + per-class spatial filter → **완벽 분리** (hoodie→red, shorts→green, shoes→blue 각각 자기 영역만)

#### (b) 4D timeframe grid

> "soft target_i = True / SAM3_mask grid 는 frame 1 장면용 → 4D 축으로 늘릴 수 있으면? (timeframe grid 추가, optical flow 옵션)"

→ **응답**: ✓ [[results/exp026_4d_label_mapping|exp026]] 완료 (negative).
- N3V sear_steak, B=10 time buckets
- per-bucket cloth_pct **5.9-7.2% 거의 동일** — chef 옷 안 바뀜
- → outfit-stable scene 엔 4D 라벨 무용. *outfit-changing future dataset* 에서 유용할 가능성.

#### (c) Positive/Negative mining (training softness)

> "9페이지에서 하려고 했던 positive/negative mining 통한 training softness"

→ **응답**: ✓ [[results/exp027_soft_cal|exp027]] 완료 (D-NeRF 2 scene, 4-flavor).

**jumpingjacks (clean baseline)** distribution shift:
| flavor | cal>0.9 (vs raw 6102) |
|---|---|
| a (Gauss param) | **3834 (−37%)** ✓ outlier demote 작동 |
| b (DINOv3) | 6880 |
| both | 4464 |

→ baseline 이 깨끗해 시각 변화는 없지만 *고-confidence 의 37% 가 distribution 내에서 demote* — calibration mechanism quantitative 검증.

**standup (noisy middle-mass baseline)**:
| flavor | cloth_pct @t=0.7 | 시각 |
|---|---|---|
| none | 5% | vest only (under-cover) |
| a/b/both | 40% | **vest + pants ✓, arms 도 over-cover ✗** |

→ calibration 이 middle-mass 를 high 로 push 해서 pants extension (positive). 하지만 feature distance 의 *intra-class similarity 한계* 로 skin/cloth 구분 부족 (arms 도 cloth 분류). Future work: multi-frame DINOv3 aggregation, mutual exclusion.

---

## 추가 finding (미팅 전 공유)

### N3V 한계 확인 (paper limitation 으로 정리)

multi-view dynamic + complex BG (kitchen) scene 에서 다음 모두 실패:
- exp014-018 (per-Gaussian projection + spatial filter + depth-aware + temporal-coherent + 4D bucket)
- exp019 (joint deformation fine-tune)
- exp020 (COLMAP sparse classify)

원인: trail Gaussian 들이 deformation 으로 chef 위치 따라가서 *visibility/depth/temporal cluster* 모두 통과. **fundamental limit**.

→ paper main result 는 **D-NeRF 8 scene + 4D-DRESS 8 take = 16 scene** 으로. N3V 는 limitation 으로.

### 핵심 contribution 정리

| # | contribution | 데이터 |
|---|---|---|
| 1 | **Per-Gaussian projection supervision** (alpha-composit 우회) | [[results/exp010_per_gaussian_projection]] |
| 2 | **Spatial outlier filter** (training-time MAD) | [[results/exp015_n3v_train_time_spatial_filter]] |
| 3 | **Inference 30× speedup** (SAM3 추론 제거) | [[results/exp021_efficiency]] |
| 4 | **Multi-instance K=N independent BCE** | [[results/exp025_multiclass]] |
| 5 | **Memory-based negative mining calibration** | [[results/exp027_soft_cal]] (진행중) |
| 6 | **Edge IoU/Chamfer metric** (SAM3 pseudo-GT) | [[results/exp024_edge_metric]] |

### 미팅 후 작업 우선순위 (예상)

1. exp027 결과 확인 + ablation table
2. paper draft 시작 (16 scene main + N3V limitation)
3. multi-instance scene 확장 (모자, 시계 등)
4. Human review setup (Google Form)

## 관련 노트

- [[006Meeting_2026-05-15]] — 원본 미팅 노트
- [[007Experiment_Plan_2026-05-15]] — 실험 sprint 설계
- [[008Viewer_사용법]] — viewer.py 가이드
- [[005Paper_outline]] — paper 구조
- [[results/exp021_efficiency|exp021]] · [[results/exp022_threshold_sweep|exp022]] · [[results/exp023_grid_soft_target|exp023]] · [[results/exp024_edge_metric|exp024]] · [[results/exp025_multiclass|exp025]] · [[results/exp026_4d_label_mapping|exp026]] · [[results/exp027_soft_cal|exp027]]
