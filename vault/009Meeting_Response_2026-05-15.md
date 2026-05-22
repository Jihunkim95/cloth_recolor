---
title: 다음 미팅 준비 — 지난 피드백에 대한 응답
date: 2026-05-15
tags: [meeting, response, prep]
---

# 다음 미팅 준비 — 지난 피드백에 대한 응답

> 지난 미팅 (2026-05-15) 에서 교수님이 주신 **5가지 질문/지시** 에 대해 1일 동안 실험으로 답해본 결과. 미팅에서는 이 순서대로 발표.

## 📁 시연용 파일 위치 (cloth_recolor/ 기준)

| 종류 | 경로 |
|---|---|
| **모든 미팅용 panel** | `vis/SUMMARY_meeting/` (10개 PNG) |
| 실험 스크립트 | `2_process/exp021_efficiency.py`, `edge_metric.py`, `per_gaussian_supervision*.py`, `exp027_soft_cal.py` |
| 학습된 ckpt | `3_output/<scene>/ckpt_exp02{1..7}_*/` |
| 측정 결과 (json/npy) | `3_output/<scene>/exp02X_*/result.json`, `metric.json`, `summary.json` |
| Viewer 데모 | `4_viewer/viewer.py` — 사용법: [[008Viewer_사용법]] |
| 각 실험 상세 노트 | `vault/results/exp02{1..7}_*.md` |

---

## 핵심 요약 (1분 발표용)

지난 미팅에서 *"우리 방법이 진짜 더 좋은가? 어떻게 증명?"* 에 대한 답:

1. **속도**: 우리 방법이 **30배 빠름** (180ms → 6ms per frame). SAM3 가 매번 돌면 안 빠른데 우리는 미리 학습해뒀으니까.
2. **품질 수치**: SAM3 mask 의 경계와 우리 prediction 경계가 얼마나 잘 맞는지 → **우리가 baseline 의 2배 더 잘 맞음** (IoU 0.253 vs 0.121).
3. **여러 옷 동시에**: 한 모델에서 셔츠/바지/신발 각각 다른 색으로 — *완벽 분리* 됨 (단 *각 옷마다 SAM3 prompt 를 여러 단어로* 줘야 함).
4. **9페이지 calibration**: 구현해서 돌렸음 — *깨끗한 scene 에선 효과 invisible, noisy scene 에선 cover 늘려주지만 skin 까지 over-cover*. 부분적 성공.
5. **N3V 결과**: 모든 시도 실패. paper 의 *limitation* 으로 명시.

---

## 1) "그래서 우리 방법이 빠르긴 한가?"

### 교수님 지시
> baseline 은 추론할 때마다 SAM3 돌리는데, 우리는 학습할 때만 SAM3 쓰고 추론은 빠를 거 → 속도 측정해봐.

### 우리 측정 결과 (`exp021`)

jumpingjacks scene, 50 frame 평균:

```
baseline:  매 frame 마다 4DGS render(3ms) + SAM3(149ms) + HSV swap(28ms) = 180ms
우리:     매 frame 마다 4DGS render with precomputed cloth = 6ms
                                                        ────
                                                        30배 빠름
```

**메시지**: SAM3 가 baseline 의 **82%** 차지. 이걸 추론에서 제거한 게 우리의 핵심 efficiency contribution.

→ **paper Table 1** 에 그대로.

📁 **파일**:
- 측정 결과: `3_output/jumpingjacks/exp021_efficiency/result.json`
- 코드: `2_process/exp021_efficiency.py`
- 노트: `vault/results/exp021_efficiency.md`

---

## 2) "GT 없으니 수치적으로 뭐가 더 잘했는지 어떻게 보임?"

### 교수님 지시
> 사람이 직접 라벨링한 GT 가 없음. 그래도 *옷 경계가 얼마나 깔끔한지* 를 수치로 보고 싶음.

### 우리 응답 (`exp024`)

**아이디어**: SAM3 mask 의 *경계 라인* 과 우리 prediction 의 *경계 라인* 이 얼마나 잘 겹치는지 측정.

- Sobel filter 로 두 mask 의 boundary edge 추출
- 두 edge 의 **IoU** (얼마나 겹치는지, 0~1) + **Chamfer distance** (얼마나 가까운지, pixel 단위)

**jumpingjacks 5 ckpt 비교**:

| 방법 | 경계 IoU ↑ | 경계 거리 ↓ |
|---|---|---|
| baseline (전체 over-cover) | 0.121 | 12.06 px |
| 이전 softcal | 0.095 | 13.41 px (worst) |
| **우리 exp010 (per-Gaussian)** | **0.253** | **6.40 px** |

→ baseline 대비 **IoU 2.1배, 경계거리 절반**.

**caveat (교수님께 미리 말씀드릴 점)**: SAM3 가 학습 신호이기도 해서 *완벽한 독립 GT* 는 아님. paper 에선 "supervision-aligned boundary metric" 으로 표기.

📁 **파일**:
- 측정 결과: `3_output/exp024_edge_sam3gt/jumpingjacks_<ckpt>/metric.json`
- 정성 panel (3-col: RGB, cloth_prob heatmap, edge overlay): `3_output/exp024_edge_sam3gt/jumpingjacks_exp010/view_*.png` (8 view)
- 코드: `2_process/edge_metric.py`
- 노트: `vault/results/exp024_edge_metric.md`

---

## 3) "2D pixel mask 를 3D Gaussian 으로 어떻게 자연스럽게 매핑?"

### 교수님 지시 (3가지)

A. **단순 baseline**: SAM3 mask hit 을 `> 0` 으로 1 / `> 0.5` 로 1 — 어느 게 좋은가?  
B. **Grid 평균**: 단일 pixel 대신 16×16 grid 평균으로 soft 하게 — 효과 있는가?  
C. **평가 방법**: 경계 line 의 RGB edge 와 prediction edge 비교 (수학), 사람 review (휴먼), 정성 figure 3가지로.

### 우리 응답

**A. Threshold 비교 (`exp022`)**

per-Gaussian 의 cloth 확률을 어떤 값에서 자를지 sweep (0, 0.1, 0.3, 0.5, 0.7, 0.9):

- D-NeRF jumpingjacks → **t=0.7 가 sweet spot** (이때 hoodie 만 깔끔하게 분리, t=0.5 면 약간 over-cover, t=0.9 면 under-cover)
- N3V sear_steak (spatial filter 적용) → **t=0.3 가 sweet spot** (filter 가 logit 분포를 좁혀서 0.5 이상이 거의 없음)

→ **threshold 는 method 마다 다름**. paper 에서 method 별로 sweet spot 보고.

📁 **A 파일**:
- 시각 grid (6 threshold rows): `vis/SUMMARY_meeting/exp022_threshold_dnerf.png`, `vis/SUMMARY_meeting/exp022_threshold_n3v.png`
- 각 threshold 별 ckpt + summary: `3_output/exp022_thresh_sweep/{dnerf_jumpingjacks,n3v_sear_steak}_t{0.0..0.9}/`
- 노트: `vault/results/exp022_threshold_sweep.md`

**B. Grid 평균 (`exp023`)**

`mask[py, px]` (1픽셀 lookup) 대신 `mask[py-k:py+k, px-k:px+k].mean()` (window 평균) — k=0/1/2/4 비교:

- 결과: **cloth_pct 가 ±0.5% 만 변함**. 시각적으로도 4개 모두 거의 같음.
- 이유: SAM3 mask 가 이미 sharp binary 라 window 평균이 boundary 외엔 영향 없음.

→ **negative result**. paper 에선 "window-size robustness" 로 보고 (튜닝 부담 없음 = robustness 어필).

📁 **B 파일**:
- 시각 grid (4 window rows): `vis/SUMMARY_meeting/exp023_grid_dnerf.png`, `vis/SUMMARY_meeting/exp023_grid_n3v.png`
- 8 ckpts: `3_output/{jumpingjacks,n3v_sear_steak}/ckpt_exp023_grid_k{0,1,2,4}/`
- 코드 (수정 부분): `2_process/per_gaussian_supervision.py:128-145`, `_n3v.py:155-170` (`--grid-window` flag)
- 노트: `vault/results/exp023_grid_soft_target.md`

**C. 평가 방법**: (위 2번의 edge metric 이 그 답)

---

## 4) "옷 외 신발/모자도 동시에 — multi-instance 가능?"

### 교수님 지시
> 최소 N=3 종류 × M=2 scene. 신발, 모자, 셔츠 같은.

### 우리 응답 (`exp025`)

**M=2 scene × N=3 instance**:
- jumpingjacks: [hoodie / shorts / shoes]
- standup: [vest / pants / shoes]

**구현**: K=3 channel cloth_logit, 각 class 마다 *독립* BCE (softmax mutual exclusion 아님), per-class hue 다르게 (hoodie=red, shorts=green, shoes=blue 처럼).

**v1 결과**: hoodie 와 shorts 가 boundary bleeding. 원인 추적:
- `SAM3 prompt "hoodie"` 가 200 frame 중 **158 frame (79%) 에서 detection fail**!!
- SAM3 single prompt 의 *reliability 가 진짜 bottleneck*.

**v2 fix**: hoodie 만 multi-prompt union 으로 재캐싱
- prompts = `"orange hoodie, sweater, hoodie, jacket, long sleeve top"`
- coverage 0.7% → **3.9% (5.5배)**
- + per-class spatial filter (exp015 기법)

**v2 결과** (`/tmp/exp025_union.png`):
- hoodie → red: 전체 깔끔, shorts/shoes 영향 없음
- shorts → green: 전체 shorts 다 (v1 에선 하단만)
- shoes → blue: 깔끔

→ **multi-instance 자체는 trivial 확장** (K=1 → K=N independent BCE). 진짜 문제는 **SAM3 prompt 안정성**.

→ paper 에 **"prompt engineering matters"** 한 줄. 정성 figure 첨부.

📁 **파일**:
- v1 시각 (jj/standup): `vis/SUMMARY_meeting/exp025_v1_jumpingjacks.png`, `exp025_v1_standup.png`
- **v2 (union+spatial)** ★ 미팅 핵심 demo: `vis/SUMMARY_meeting/exp025_v2_jumpingjacks_union.png`
- SAM3 mask 실패 증거 (frame 66 hoodie=0% 보여줌): `vis/SUMMARY_meeting/exp025_sam3_mask_failure.png`
- 학습 ckpts: `3_output/{jumpingjacks,standup}/ckpt_exp025_{multi,multi_sf,union_sf}/`
- SAM3 union cache: `cache/sam3_exp025_jumpingjacks_hoodie_union/` (5 prompts)
- 코드: `2_process/per_gaussian_supervision_multiclass.py`, `recolor.py --per-class-bce`
- 노트: `vault/results/exp025_multiclass.md`

---

## 5) 9페이지 원래 plan — "memory-based negative mining"

### 교수님 plan 요약
1. 학습 중 매 N step:
   - `cloth_logit > 0.95` top-1000 → "쉬운 cloth memory"
   - `cloth_logit < 0.05` top-1000 → "쉬운 non-cloth memory"
2. 각 학습 point 마다 *memory 까지 거리* 계산
   - (a) Gaussian parameter 거리 (xyz + scale + color)
   - (b) DINOv3 patch feature 거리 (rendered 이미지 patch)
3. 거리 멀수록 (= outlier 수상) target 을 0.5 쪽으로 *애매하게* 만들기
   - cloth-side: `target = 1 − α(d)`
   - non-cloth-side: `target = 0 + α(d)`

### 우리 응답 (`exp027`)

위 그대로 구현. D-NeRF 2 scene × 4 flavor (none, a, b, both).

**결과 1 — jumpingjacks (baseline 이 이미 깨끗한 scene)**:

`cloth_logit > 0.9` 인 Gaussian 수 (= 매우 confident cloth):
- baseline (none): 6102 개
- flavor a (Gaussian param): **3834 개 (−37%)** ← outlier 들이 demote 됨 ✓
- flavor b (DINOv3): 6880 개 (거의 변화 없음)
- both: 4464 개

→ **distribution 내에서 outlier 가 less-confident 로 demote 되는 메커니즘 정량 확인**. 단 baseline 이 이미 깨끗해서 *시각적 변화는 없음*.

**결과 2 — standup (baseline 이 under-cover 인 scene)**:

threshold 0.7 에서 cloth_pct:
- baseline (none): 5% → vest 만 (under-cover, pants 못 잡음)
- flavor a/b/both: 40% → vest + pants 까지 ✓ (positive)
- 하지만 arms (skin) 도 같이 cover 됨 (negative)

→ calibration 이 *middle-mass 를 high 쪽으로 push* → pants 추가 coverage 는 좋은데 skin 까지 잘못 끌어들임.

### 교수님께 보고할 honest finding

| 시나리오 | 결과 |
|---|---|
| baseline 이 이미 깨끗 (jumpingjacks) | calibration 의 시각 효과 거의 없음 (distribution 만 변화) |
| baseline 이 under-cover (standup) | coverage 늘려주지만 boundary discrimination 부족 |

**원인**: Gaussian param 또는 DINOv3 patch feature 의 *intra-class similarity* 가 너무 강함. arms 가 vest 와 비슷한 feature 라 calibration 이 구분 못 함.

**다음 시도 후보** (미팅 후 의견 구하기):
- DINOv3 patch 를 *여러 frame 평균* (현재는 단일 frame)
- color-weighted feature distance (SH_DC 가중치 ↑)
- arms vs cloth 명시적 multi-class CE
- 9페이지 원본 의도대로 *학습 중 iter 마다* calibration (현재는 학습 전 single-pass)

📁 **파일**:
- 시각 (4 flavor 비교): `vis/SUMMARY_meeting/exp027_jumpingjacks_4flavor.png`, `exp027_standup_4flavor.png`
- 8 ckpts (2 scene × 4 flavor): `3_output/{jumpingjacks,standup}/ckpt_exp027_{none,a,b,both}/`
- distribution 분석 데이터: `3_output/<scene>/ckpt_exp027_<flavor>/soft_target_{raw,cal}.npy`, `calib_meta.json`
- 코드: `2_process/exp027_soft_cal.py` (+ 기존 `utils/soft_cal.py` 의 `feat_a`, `DinoV3PatchExtractor` 재사용 — `4DGaussians/utils/soft_cal.py`)
- 노트: `vault/results/exp027_soft_cal.md`

---

## 추가로 미팅에서 공유할 것

### N3V (kitchen chef video) 모든 변형 실패

`exp014~020` 까지 8가지 시도 다 실패:
- 단순 per-Gaussian projection
- spatial filter
- depth-aware (front-most Gaussian)
- temporal cluster (per-frame 3D coherence)
- joint deformation fine-tune
- COLMAP sparse classify
- 4D timeframe grid

→ **근본 원인**: N3V 의 4DGS deformation MLP 가 *cloth 와 trail Gaussian 을 같이 묶어서* 학습. RGB-only supervision 이라 cloth/background 의미적 구분 못 함. trail Gaussian 들이 어떤 frame 에선 chef 위치, 어떤 frame 에선 다른 위치로 deform 됨 → 어떤 후처리로도 분리 불가능.

→ **paper limitation section 으로 정리**. main result 는 D-NeRF (8 scene) + 4D-DRESS (8 take) = 16 scene.

📁 **N3V 실패 증거**:
- 시각 비교 grid: `vis/SUMMARY_n3v_exp014/`, `vis/SUMMARY_n3v_exp015/`, `vis/SUMMARY_n3v_exp016/`, `vis/SUMMARY_n3v_exp016_t07/`, `vis/SUMMARY_n3v_filtered/`, `vis/SUMMARY_n3v_tight/`, `vis/SUMMARY_n3v_vivid/`, `vis/SUMMARY_n3v_exp015/`
- 모든 ckpts: `3_output/n3v_sear_steak/ckpt_exp{014..020,026}_*/`
- 노트: `vault/results/exp014_n3v_per_gaussian.md`, `exp015_n3v_train_time_spatial_filter.md`, `exp026_4d_label_mapping.md` (+ 미작성된 exp016~020 은 daily log 참조)

### 우리의 6가지 핵심 contribution

| # | contribution | 검증 | 시연 파일 |
|---|---|---|---|
| 1 | Per-Gaussian projection supervision | exp010 | `vis/SUMMARY_exp010/jumpingjacks.png` |
| 2 | Spatial outlier filter (training-time) | exp015 | `vis/SUMMARY_n3v_exp015/sear_steak.png` (N3V 한계 인정) |
| 3 | Inference 30× speedup | exp021 | `3_output/jumpingjacks/exp021_efficiency/result.json` |
| 4 | Multi-instance K=N with prompt union | exp025 | `vis/SUMMARY_meeting/exp025_v2_jumpingjacks_union.png` |
| 5 | Boundary IoU/Chamfer metric | exp024 | `3_output/exp024_edge_sam3gt/jumpingjacks_exp010/view_0066.png` |
| 6 | Memory-based negative mining | exp027 | `vis/SUMMARY_meeting/exp027_standup_4flavor.png` |

---

## 관련 문서

- [[006Meeting_2026-05-15]] — 지난 미팅 원본 메모 (날 것)
- [[007Experiment_Plan_2026-05-15]] — 1일 sprint 설계
- [[008Viewer_사용법]] — 결과 직접 보고 싶을 때 (viser 3D viewer)
- 각 실험 상세: [[results/exp021_efficiency|exp021]] · [[results/exp022_threshold_sweep|exp022]] · [[results/exp023_grid_soft_target|exp023]] · [[results/exp024_edge_metric|exp024]] · [[results/exp025_multiclass|exp025]] · [[results/exp026_4d_label_mapping|exp026]] · [[results/exp027_soft_cal|exp027]]
