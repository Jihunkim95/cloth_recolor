---
title: HSV recolor trick (SH_DC hue swap)
layer: 1+2
date: 2026-05-11
---

# HSV recolor trick

> 학습된 [[cloth_logit_channel]] 만 있으면 옷 색을 바꾸는 데 *최적화 불필요*. layer 1+2 (HSV 자체는 정의상 layer 0/1 — RGB 와 1:1 변환, 우리가 옷에만 적용하는 건 layer 2 결정).

## 핵심

```python
# 1. 학습된 PLY 로드 — _features_dc 가 SH degree 0 항 (시점 무관 베이스 색)
pc.load_ply(...)

# 2. cloth Gaussian 골라내기
cloth_mask = sigmoid(pc._cloth_logit) > τ          # τ ∈ [0.2, 0.5]

# 3. SH_DC → RGB → HSV → hue swap → RGB → SH_DC
rgb = pc._features_dc[..., 0] * C0 + 0.5           # SH 정의상 RGB ≈ DC * C0 + 0.5
hsv = rgb_to_hsv(rgb)
hsv[..., 0] = target_hue / 360.0                   # H 만 교체
new_rgb = hsv_to_rgb(hsv).clip(0, 1)
pc._features_dc[cloth_mask, ..., 0] = (new_rgb - 0.5) / C0   # 다시 SH_DC

# 4. 평소처럼 4DGS render — 옷만 새 색
```

여기서 `C0 = 1 / (2 * sqrt(π)) ≈ 0.282` (SH degree 0 normalization).

## 학습 시점 vs 편집 시점 — 어디서 SH_DC 만 만지나

| 단계 | SH_DC | SH_rest | 비고 |
|---|---|---|---|
| **학습** ([[4d_gaussian_splatting]] / `train.py`) | Adam 학습 (lr = feature_lr) | Adam 학습 (lr = feature_lr / **20**) | 두 그룹 모두 학습. `iteration % 1000 == 0` 마다 `active_sh_degree++` 로 점진적 활성화 (0→1→2→3) |
| **편집 (recolor)** (`recolor.py`) | **HSV swap 으로 덮어씀** | **손대지 않음** | cloth_logit > τ 인 가우시안에만 적용 |

→ 학습 중에는 모델이 *베이스 색 + 시점 의존 광택 패턴* 을 모두 데이터로부터 학습.
→ 편집 중에는 학습된 광택 패턴 그대로 두고 베이스만 바꿈 → 새 색 위에 *원래 광택* 이 자연스럽게 얹힘.

## SH 와 "DC" 가 무엇인가

3DGS/4DGS 의 색은 시점 $d$ 의 함수로 SH (spherical harmonics) 표현:

$$
c(d) = \underbrace{c_0 \cdot Y_0^0}_{\text{view-independent}} + \sum_{l \ge 1} \sum_{m=-l}^{l} c_{lm} \, Y_l^m(d)
$$

| degree $l$ | 항 개수 (per RGB channel) | 의미 |
|---|---|---|
| **0 ("DC")** | **1** | 베이스 색 (방향 무관) |
| 1 | 3 | 부드러운 방향성 |
| 2 | 5 | sharp 한 방향성 |
| 3 | 7 | specular highlight |

"**DC**" = Direct Current = 0 차 항 = 시점 무관 베이스 색 (Fourier/SH 관용어).
RGB 3 채널 → SH_DC = (R_dc, G_dc, B_dc) = **3 float** per Gaussian.

3DGS 코드 변환 공식 (Y_0^0 = 1/(2√π) 정규화 + 0.5 offset 으로 RGB ∈ [0,1] 매핑):

```python
C0 = 0.28209479177387814          # = 1 / (2 * sqrt(pi))
rgb   = sh_dc * C0 + 0.5          # SH 공간 → RGB 공간
sh_dc = (rgb - 0.5) / C0          # 역변환
```

## 정확히 어디를 바꾸나 (가우시안 1개당 48 float 중 3 개)

```
가우시안 i 의 색 정보 (총 48 float):
┌─────────────────────────────────────────────┐
│ _features_dc   = [R_dc, G_dc, B_dc]         │  ★ 3 float — 우리가 바꿈
├─────────────────────────────────────────────┤
│ _features_rest                              │
│   degree 1: 9  (3×3)                       │
│   degree 2: 15 (5×3)                       │  → 45 float — 그대로 둠
│   degree 3: 21 (7×3)                       │
└─────────────────────────────────────────────┘

방향 d 로 본 최종 색:
c(d) = (변경된 SH_DC) · Y_0^0  +  (원본 SH_rest) · Y_l^m(d)
       └── 새 베이스 색 ──┘    └── 원래 광택/그림자 패턴 ──┘
```

## 왜 SH_DC 만 바꾸는가 (직관)

| 만약 | 결과 |
|---|---|
| **SH_DC 만 swap** ✅ | 베이스 색 = 파랑, view-dependent 변화 (highlight, glint) 는 *원래 modulation 패턴* 그대로 → 천이 파란 옷처럼 자연스럽게 빛남 |
| SH_DC + SH_rest 모두 swap | highlight 색까지 파랑 → "광원이 파란색" 같은 어색함 |
| SH_rest 만 swap | 베이스는 그대로, highlight 만 이상함 (말이 안 됨) |

비유: 옷에 "파란 페인트" 바르는 것 vs "옷+조명 전부" 파랑으로 바꾸는 것. 우리는 페인트만 바꾸고 싶음.

## 시간/시점 일관성

[[4d_gaussian_splatting]] 의 deformation MLP 는 가우시안 위치/모양만 변형 → 색은 그대로.
SH_DC 만 한 번 바꾸면 모든 t · 모든 시점에서 자동으로 옷이 새 색.
→ **시간 일관성·시점 일관성 자유** (optimization 불필요).

이 점이 [[../notes/4d_gaussian_splatting]] 위에 [[cloth_logit_channel]] 박은 가장 큰 이득.

## τ (threshold) 선택

`cloth_logit` 분포가 [[soft_label_calibration]] 으로 부드러워졌기 때문에 τ=0.5 면 너무 엄격:

| τ | jumpingjacks (cloth_pct) | 효과 |
|---|---|---|
| 0.5 | 0% (`both` 변형) | 한 픽셀도 안 바뀜 |
| 0.3 | ~5% | 옷 일부 |
| **0.2** | ~30% | 옷 대부분, 약간 over |
| 0.1 | ~60% | over-cover |

기본 `recolor.sh` 에서 τ=0.2 사용.

## 산출물

```
3_output/<scene>/recolor_<variant>/
├── panel_*.png           ← 16 패널 (orig | recolored | mask overlay)
├── recolored.ply         ← cloth Gaussian SH_DC 가 영구 baked 된 정적 PLY (canonical)
└── summary.json          ← N, n_cloth, hue, threshold
```

`recolored.ply` + 같은 폴더의 `deformation.pth` 를 함께 로드해야 시간 변형까지 확인 가능 (정적 PLY 만 외부 viewer 띄우면 canonical 자세만 보임 — [[4d_gaussian_splatting]] 의 PLY 단독 한계).

## 깰 수 있는 부분

- HSV 만 바꿈 — texture / pattern (꽃무늬 → 줄무늬) 은 불가, generative model (e.g., diffusion) 결합 필요
- target hue 1 개 — gradient (밑→위 색 변화) 같은 spatial pattern 도 가능 (가우시안 위치 기반)
- SH degree 0 만 — material change (matte→glossy) 는 SH_rest 도 같이 학습해야 함
