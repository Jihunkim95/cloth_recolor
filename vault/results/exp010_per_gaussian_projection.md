---
title: exp010 — Per-Gaussian SAM3 projection supervision (alpha-composit 우회)
status: done
date: 2026-05-12
code: 2_process/per_gaussian_supervision.py
---

# exp010

## 질문

기존 모든 실험 (exp001-009) 은 픽셀 단위 BCE/CE 로 cloth_logit 을 학습 — gsplat alpha-composit 으로 *한 가우시안이 여러 픽셀에 기여, 한 픽셀이 여러 가우시안에 기여* → SAM3 mask 가 깔끔해도 가우시안 단위 라벨이 over/under cover.

→ **픽셀 supervision 우회**: 각 가우시안을 모든 학습 카메라에 직접 projection 해서 SAM3 mask 값 평균 → per-Gaussian soft target → BCE 직접 학습. 효과가 나오는가?

## 설정

- 데이터셋: D-NeRF 8 씬 전부
- base ckpt: `3_output/<scene>/ckpt_baseline_hardBCE` iter **20000** (RGB·deformation 학습 완료된 상태)
- SAM3 mask: `cache/sam3_union_<scene>` (multi-prompt union, [[exp008_multiprompt_union]])
- per-Gaussian soft target 계산:
  ```
  for each Gaussian g:
    for each train frame i (i=0..T-1):
      g_t = deformation_MLP(g.xyz, time_i)
      px, py = project(g_t, camera_i)
      if in_view(g_t, camera_i):
        target_i = SAM3_mask[i, py, px]   # 0 or 1
    soft_target_g = mean(target_i)
  ```
- 학습: cloth_logit 만 (다른 모든 parameter freeze). BCE(sigmoid(cloth_logit), soft_target). 2000 step, Adam lr=0.05.

## 예측

- per-Gaussian projection 은 alpha-composit 평균 효과 우회 → cloth_pct 가 SAM3 GT coverage 근처로 깨끗하게 수렴
- soft_target 분포가 bimodal (~0 또는 ~1, ambig 영역 적음) — 한 가우시안이 매 frame 대부분 같은 클래스로 splat 되므로
- 시각: face/skin/배경 그대로, 옷만 깔끔하게 recolor

## 결과

### 8 씬 정량 (cloth_pct @ thresh 0.7, base iter=20000)

| scene | cloth_pct | SAM3 union cov | 의미 |
|---|---|---|---|
| jumpingjacks | **34.93%** | 3.90% | hoodie 만 정확 분리 (face/shorts/shoes 제외) |
| standup | 64.53% | 6.55% | vest+shirt+pants — 노출 피부(face/hands) 제외 |
| hellwarrior | 94.85% | 11.09% | full body armor — 캐릭터 = 갑옷 자체 |
| mutant | 94.22% | 10.18% | monster body+armor — 캐릭터 = 몸 |
| hook | 95.05% | 12.20% | puppet body — 캐릭터 = 몸 |
| bouncingballs | 26.88% | 6.55% | 공 자체 (배경 분리) |
| lego | 28.17% | 12.85% | lego body |
| trex | 42.75% | 4.31% | dinosaur 몸 |

cloth_pct 가 cov 보다 큰 이유: 픽셀 단위 cov 는 2D 영역, Gaussian 단위 cloth_pct 는 *3D 가우시안의 평균 visibility-weighted cov*. 두 metric 의 차이는 정상.

### 시각 검증

**`vis/SUMMARY_exp010/<scene>.png`** — 8 씬 모두 2-row 비교 패널 (K=1 baseline vs exp010).

대표 결과:
- **jumpingjacks** (가장 어려운 씬): hoodie 만 정확 파랑, face/머리/jean shorts/yellow shoes 그대로. threshold 0.5/0.7/0.85 모두 동일 (bimodal target).
- **standup**: orange vest 만 깔끔 파랑, 안전모/회색셔츠/부츠/바지 원본. K=1 baseline 은 vest 영역 색 얼룩.
- **hellwarrior/mutant/hook**: 캐릭터 = "옷" 인 씬, cloth_pct 95% 가 캐릭터 전체 = 올바름. 갑옷이 회색이라 hue swap 효과가 시각적으로 미세.

기존 K=1/K=2/K=3 모두에서 못 했던 *깔끔한 옷 분리* 가 처음으로 달성됨.

### 비교 표 (jumpingjacks)

| 방법 | cloth_pct @ thresh 0.5 | 시각 |
|---|---|---|
| K=1 hard-BCE baseline | 100.0% | 전체 over-cover |
| K=1 union + soft-cal | 91.7% | 거의 전체 over-cover |
| K=3 multi-class | 16.55% | hoodie 일부만 (under-cover) |
| K=2 union (no weight) | 59.1% | face+hoodie 모두 (over-cover) |
| **exp010 per-Gaussian** | **37.0%** | **hoodie 만 ✓** |

## 예측 맞았나?

- alpha-composit 우회: ✅ 시각 결과로 직접 검증 — face 영역 가우시안이 처음으로 "옷 아님" 으로 학습됨
- soft_target bimodal: ✅ 통계 (mean=0.397, in [0.3,0.7]=7.7%) — 92.3% 의 가우시안이 high-confidence (>0.7) 또는 low-confidence (<0.3)
- 시각: ✅ threshold 둔감, 깔끔한 옷 분리

## 결론

**픽셀 supervision → per-Gaussian supervision 전환이 단일 큰 breakthrough**.

기존 우리 framework 의 모든 한계 (K-class softmax 의 sparse-class 학습 실패, soft-cal 의 보정 한계, class weighting 의 saturation) 는 모두 alpha-composit pixel-level CE 의 근본 한계에서 발생. 가우시안 단위 직접 supervision 으로 한 번에 해결.

## Method (논문용 정리)

1. **Stage 1** — Vanilla 4DGS 학습 (RGB + deformation), cloth_logit 무시. PSNR 정상 ~33-36 dB.
2. **Stage 2** — Trained 4DGS 고정. 각 가우시안 g 에 대해, 모든 학습 frame i 에 projection → SAM3 mask 값 평균 → per-Gaussian soft target $t_g \in [0, 1]$.
3. **Stage 3** — cloth_logit 만 학습 (다른 parameter freeze):
   $$L = \text{BCE}(\sigma(\text{cloth\_logit}_g), t_g)$$
   2000 Adam steps, lr=0.05. 약 10초.
4. **Stage 4** — Recolor: $\sigma(\text{cloth\_logit}_g) > \tau$ 인 가우시안의 SH_DC HSV-swap.

## 다음

- 8 씬 정량 표 완성
- HyperNeRF 실제 monocular 데이터에 적용
- Hue gradient·texture pattern 같은 advanced edit 시도
- 4D-DRESS (multi-view) 데이터에 backport — 더 강한 soft target 가능 (frame 당 4 view × 200 frame = 800 supervision)

## 부가설명 — Stage 1~4 상세

### Stage 1: Vanilla 4DGS 학습 (cloth_logit 무시)

표준 4DGS 학습:
- 학습 파라미터: `xyz, scale, rotation, opacity, SH_DC, SH_rest, deformation MLP`
- Loss: RGB photometric only (L1 + SSIM)
- **14000 iter** (또는 20000) — base 4DGS RGB·geometry 학습
- → 결과: 깔끔한 4DGS ckpt. `cloth_logit` 은 random 초기값 그대로 (학습 X)

### Stage 2: per-Gaussian soft target 계산 (학습 X, 단순 lookup)

```python
# 입력: trained 4DGS ckpt, SAM3 mask T장 (frame 별)
for each Gaussian g_i (i = 1..N):
    accum, count = 0, 0
    for each frame f_j (j = 1..T):
        t_j   = frame j 의 시간
        cam_j = frame j 의 camera pose

        # deformation MLP 로 시간 t_j 의 위치 계산
        g_xyz_deformed = deformation_MLP(g_i.xyz_canonical, t_j)

        # camera_j 로 image plane 에 projection
        px, py = project(g_xyz_deformed, cam_j)

        if (px, py) is in image:
            # SAM3 mask 값 lookup (0 또는 1)
            mask_value = SAM3_mask_j[py, px]   # ∈ {0, 1}
            accum += mask_value
            count += 1

    soft_target_i = accum / count   # ∈ [0, 1]
```

→ N 개 가우시안마다 1개 scalar (cloth 일 확률):
- `soft_target_i ≈ 1`: 모든 frame 에서 cloth 픽셀에 splat 됨 → hoodie/cloth 가우시안
- `soft_target_i ≈ 0`: 모든 frame 에서 non-cloth 픽셀에 splat 됨 → skin/배경 가우시안

### Stage 3: cloth_logit 학습 ([[../notes/adam_optimizer|Adam]] + [[../notes/bce_loss|BCE]], 다른 모든 parameter freeze)

**별도 2000 Adam step** — Stage 1 의 14000 iter 와 무관 (다른 파라미터, 다른 loss).

```python
optimizer = Adam([cloth_logit], lr=0.05)
for step in range(2000):                          # ← Stage 3 의 2000 step (cloth_logit 만)
    pred = sigmoid(cloth_logit).squeeze()         # (N,) ∈ [0, 1]
    L = BCE(pred, soft_target)                    # binary cross-entropy
    L.backward()
    optimizer.step()
```

BCE 공식:

$$
\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \left[ t_i \log \sigma(\ell_i) + (1 - t_i) \log (1 - \sigma(\ell_i)) \right]
$$

#### 기호 해독

| 기호 | 의미 | 값 범위 |
|---|---|---|
| $\mathcal{L}$ | loss (얼마나 틀렸는지) | $\geq 0$ (0 이 perfect) |
| $N$ | 가우시안 개수 | 보통 25,000 ~ 200,000 |
| $i$ | 가우시안 index | 1, 2, ..., $N$ |
| $t_i$ | soft_target (정답) | $\in [0, 1]$ — Stage 2 결과, 고정 |
| $\ell_i$ | cloth_logit (학습 변수) | $\in (-\infty, +\infty)$ — Adam 으로 업데이트 |
| $\sigma$ | sigmoid 함수 | $\sigma(\ell) = 1/(1+e^{-\ell}) \in (0, 1)$ |
| $\log$ | 자연 로그 (ln) | |

$\sigma$ 가 logit 을 확률로 변환: $\sigma(0)=0.5$, $\sigma(+\infty)=1$, $\sigma(-\infty)=0$.

#### 가우시안 1 개의 contribution

```
loss_i = t_i · log σ(ℓ_i)     +     (1 - t_i) · log (1 - σ(ℓ_i))
         ────────────────────         ──────────────────────────
         "t_i=1 일 때만 살아남음"     "t_i=0 일 때만 살아남음"
```

- $t_i = 1$ (확실 cloth): `loss_i = log σ(ℓ_i)` — 예측 σ=1 이면 $\log 1 = 0$, σ=0 이면 $-\infty$
- $t_i = 0$ (확실 not-cloth): `loss_i = log (1 - σ(ℓ_i))` — 예측 σ=0 이면 0, σ=1 이면 $-\infty$
- $t_i = 0.5$ (애매): 양쪽 term 의 절반씩 합산

#### $\sum$ 과 $\frac{1}{N}$ 의 역할

- $\sum_{i=1}^{N}$ = 모든 가우시안에 대해 합산
- $\frac{1}{N}$ = 평균 → 가우시안 수 N 이 100 이든 100,000 이든 loss 크기가 비슷한 범위에 유지 (학습률 튜닝에 유리)

#### 왜 앞에 $-$ 가 붙는가

**$\log$ 가 확률에 적용되면 항상 음수가 나옴. $-$ 로 부호 뒤집어 *양수 loss* 를 만듦.**

직관 (확률 $p \in (0, 1)$):
- $\log 1 = 0$ (완벽)
- $\log 0.5 = -0.693$
- $\log 0.1 = -2.30$
- $\log 0.01 = -4.61$
- $\log 0 = -\infty$ (완전 틀림)

→ 대괄호 안 raw 값은 항상 $\leq 0$ (완벽이면 0, 틀릴수록 큰 음수).
**부호 뒤집기 전**: 음수 = "맞춤", 더 음수 = "더 틀림" (직관과 반대 — 최소화 문제로 다루기 어색).
**부호 뒤집기 후**: 0 = "맞춤", 양수 = "틀림" (직관 일치, 최소화 = 학습).

#### 숫자로 검증

가우시안 1 개, $N=1$ 가정:

| $t_i$ (정답) | $\sigma(\ell_i)$ (예측) | 대괄호 내부 raw | $-$ 후 $\mathcal{L}$ |
|---|---|---|---|
| 1 | 0.99 | $1 \cdot \log 0.99 + 0 = -0.01$ | **0.01** (잘 맞춤, 작은 loss) |
| 1 | 0.50 | $\log 0.5 = -0.693$ | **0.693** (애매, 중간 loss) |
| 1 | 0.01 | $\log 0.01 = -4.61$ | **4.61** (완전 틀림, 큰 loss) |
| 0 | 0.01 | $0 + \log 0.99 = -0.01$ | **0.01** (잘 맞춤) |
| 0 | 0.99 | $0 + \log 0.01 = -4.61$ | **4.61** (완전 틀림) |
| 0.7 | 0.7 | $0.7\log 0.7 + 0.3\log 0.3 = -0.61$ | **0.61** (soft target 의 entropy floor) |

규칙: 잘 맞춤 → $\mathcal{L} \approx 0$, 틀림 → $\mathcal{L}$ 큼.

#### 정보이론 해석 (선택)

이 식은 두 Bernoulli 분포 사이의 **cross-entropy**:
- $P_t$: 정답 분포 (확률 $t_i$)
- $P_{\hat p}$: 예측 분포 (확률 $\sigma(\ell_i)$)

$$
H(P_t, P_{\hat p}) = -\sum_x P_t(x) \log P_{\hat p}(x)
$$

= "정답 분포를 예측 분포로 인코딩할 때 필요한 평균 비트수". 작을수록 두 분포 가까움.
$-$ 는 *원래 cross-entropy 정의에 박혀 있는 것* — log 의 부호 보정.

#### Backprop 식 (gradient, 실제 학습 흐름)

위 식은 *loss 값* (forward). 실제 학습에 흐르는 건 그 **gradient**:

$$
\frac{\partial \mathcal{L}}{\partial \ell_i} = \frac{1}{N}\bigl(\sigma(\ell_i) - t_i\bigr)
$$

##### 유도

$p_i = \sigma(\ell_i)$ 라 두면, sigmoid 의 미분:
$$
\frac{d\sigma}{d\ell} = \sigma(\ell)\,(1-\sigma(\ell)) = p(1-p)
$$

각 term 의 $\ell_i$ 에 대한 미분:
- $\frac{d}{d\ell_i}[\log p_i] = \frac{1}{p_i} \cdot p_i(1-p_i) = (1-p_i)$
- $\frac{d}{d\ell_i}[\log(1-p_i)] = \frac{-1}{1-p_i} \cdot p_i(1-p_i) = -p_i$

대괄호 안 미분:
$$
\frac{d}{d\ell_i}\bigl[t_i\log p_i + (1-t_i)\log(1-p_i)\bigr]
= t_i(1-p_i) + (1-t_i)(-p_i)
= t_i - p_i
$$

전체 식 ($-$, $\frac{1}{N}$ 합쳐):
$$
\boxed{\;\frac{\partial \mathcal{L}}{\partial \ell_i} = \frac{1}{N}(p_i - t_i) = \frac{1}{N}\bigl(\sigma(\ell_i) - t_i\bigr)\;}
$$

##### 직관 — "예측 − 정답"

gradient 가 *예측값 − 정답* 이라는 매우 단순한 형태. 이게 BCE + sigmoid 조합의 *수학적 매력*.

| $t_i$ | $\sigma(\ell_i)$ | grad = $\sigma - t$ | 해석 |
|---|---|---|---|
| 1 | 0.99 | -0.01 | 거의 맞음, $\ell$ 살짝 ↑ |
| 1 | 0.30 | -0.70 | 많이 틀림, $\ell$ 크게 ↑ |
| 0 | 0.05 | +0.05 | 거의 맞음, $\ell$ 살짝 ↓ |
| 0 | 0.80 | +0.80 | 많이 틀림, $\ell$ 크게 ↓ |

[[../notes/adam_optimizer|Adam]] 이 이 gradient 받아 $\ell_i \leftarrow \ell_i - \text{lr} \cdot \text{grad}$ 로 업데이트.

##### Loss 식 vs Backprop 식

| | Loss 식 (forward) | Backprop 식 (backward) |
|---|---|---|
| 무엇 | loss 값 (scalar) | gradient (per Gaussian) |
| 식 | $-\frac{1}{N}\sum[t\log p + (1-t)\log(1-p)]$ | $\frac{1}{N}(\sigma(\ell_i) - t_i)$ |
| 역할 | 평가 ("지금 얼마나 틀렸나") | 학습 ("어느 방향으로 가야 덜 틀리나") |
| 코드 | `F.binary_cross_entropy(pred, target)` | `loss.backward()` 가 자동 (autograd) |

PyTorch 에서는 `loss.backward()` 호출 시 autograd 가 이 미분식을 자동 적용 — 직접 손으로 미분 작성 안 함.

→ **200~500 step 안에 수렴** (1D 문제라 빠름). `sigmoid(cloth_logit)` 가 `soft_target` 에 fit.

### Stage 4: HSV recolor — [[../notes/hsv_recolor_trick]]

```python
τ = 0.5  # threshold (hyperparameter, viewer 에서 slider 로 조정 가능)
cloth_mask = sigmoid(cloth_logit) > τ
# cloth_mask = bool tensor (N,)

for g_i where cloth_mask[i] = True:
    rgb = sh_dc_to_rgb(g_i.SH_DC)
    hsv = rgb_to_hsv(rgb)
    hsv.H = 220 / 360                 # 새 hue
    g_i.SH_DC = rgb_to_sh_dc(hsv_to_rgb(hsv))
```

## 학습 step / iter 표기 명확화

stage 별 학습 대상과 step 수가 다르므로 *총 iter* 단일 숫자가 모호. 본 연구의 정확한 표현:

| Stage | 학습 대상 | step 수 | 비고 |
|---|---|---|---|
| 1 (4DGS) | xyz, scale, rot, opacity, SH, deformation MLP | **14000 iter** (또는 20000) | photometric L1+SSIM |
| 2 (soft_target) | — (학습 X, projection lookup) | 0 | constant 생성만 |
| 3 (cloth_logit) | **cloth_logit 만** (1 scalar per Gaussian) | **2000 step** | Adam + BCE, 나머지 freeze |

### 폴더/파일 naming convention

저장 폴더 이름은 **Stage 1 의 iter** 기준 유지:

```
3_output/<scene>/ckpt_exp010_per_gaussian/point_cloud/iteration_14000/
                                                       └─ Stage 1 의 14000 iter 기준
                                                          (cloth_logit 은 Stage 3 의 2000 step 추가 학습됨)
```

이유: viewer / `recolor.py` 의 default `--iter 14000` 과 호환 + baseline 과 동일 lookup 으로 공정 비교.

### 논문/발표 시 표현

"총 16000 iter" 같은 모호 표현 대신 분리해서 적기:

> "Stage 1 (vanilla 4DGS) **14000 iter** + Stage 3 (cloth_logit fine-tune) **2000 step**, base 4DGS 의 다른 파라미터는 모두 freeze."

이 표현이 가장 명확.

---

## 표기 정정 — δ, τ 등이 어디 등장하나

| 기호 | exp010 에 있나? | 어디 나옴? |
|---|---|---|
| `L` (loss) | ✓ BCE 형태로 | Stage 3 |
| `BCE` | ✓ | Stage 3 |
| `δ` (delta) | ✗ | (exp002 도 없음. 혼동?) |
| `τ` (tau) | △ Stage 4 의 *threshold* 로 등장 (학습 X) | Stage 4 |
| `cloth_logit` | ✓ 학습 변수 (Stage 3, 4) | (Stage 3, 4) |

[[exp002_softcal_ablation|exp002 (soft-cal)]] 와 혼동 주의:

- exp002 의 $\alpha(d) = 0.5\,\sigma((d - d_\text{ref})/\tau)$ 는 *soft-label calibration* 식 (distance 기반 라벨 부드럽게). exp010 에선 안 씀.
- exp010 의 `soft_target` 은 **단순 projection 평균** — calibration 식 없음, 직접 lookup.

## 핵심 차별점 한 줄

> **exp002**: 픽셀 단위 BCE + 추가 calibration → 픽셀 supervision 의 *alpha-composit averaging* 문제 남음.
>
> **exp010**: projection 평균으로 per-Gaussian soft_target 직접 만들고 BCE → 픽셀 supervision **우회**. 가우시안 단위 직접 supervision.
