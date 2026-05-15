---
title: Binary Cross-Entropy (BCE) loss
layer: 1
date: 2026-05-12
---

# Binary Cross-Entropy (BCE)

> 이진 분류 (class 0/1) 의 표준 loss. 정보이론의 cross-entropy 를 이진 case 로 좁힌 것. layer 1 — Shannon entropy 에서 자연스럽게 유도됨.

## 정의

target $t \in [0, 1]$ (보통 {0, 1}), 모델 예측 확률 $\hat p \in (0, 1)$:

$$
\text{BCE}(\hat p, t) = -\big[\, t \log \hat p + (1 - t) \log (1 - \hat p) \,\big]
$$

## 직관 (값 표)

| $t$ | $\hat p$ | BCE | 의미 |
|---|---|---|---|
| 1 | 1 | 0 | perfect predict cloth |
| 1 | 0.5 | 0.693 | uncertain, 페널티 |
| 1 | 0.01 | 4.605 | "cloth 인데 not-cloth 라 우김" — 큰 페널티 |
| 1 | 0 | ∞ | 정반대 — 무한 loss |
| 0.5 | 0.5 | 0.693 | uncertain target 에 uncertain pred — fair |
| 0 | 0 | 0 | perfect predict not-cloth |

## sigmoid 와 함께 (logits → 확률)

모델은 보통 *logit* $\ell \in \mathbb R$ 을 출력하고, $\hat p = \sigma(\ell) = 1/(1+e^{-\ell})$ 로 변환:

$$
\text{BCEWithLogits}(\ell, t) = -\big[\, t \log \sigma(\ell) + (1-t) \log(1 - \sigma(\ell)) \,\big]
$$

수치 안정성을 위해 PyTorch 는 `F.binary_cross_entropy_with_logits(logit, target)` 권장.

## 정보이론 해석

두 Bernoulli 분포 $P_t(X=1) = t$, $P_{\hat p}(X=1) = \hat p$ 사이의 cross-entropy:

$$
H(P_t, P_{\hat p}) = \sum_{x \in \{0,1\}} P_t(x) (-\log P_{\hat p}(x))
$$

= "target 분포의 정보를 prediction 분포로 인코딩할 때 필요한 평균 비트수".

## 수렴 동작

- $t \in \{0, 1\}$ (hard label): 최소 BCE = 0
- $t \in (0, 1)$ (soft label): 최소 BCE = $H(t) = -t \log t - (1-t)\log(1-t)$ (entropy floor)
  - 예: $t = 0.397$ 라면 BCE 의 최소값은 0.671 — 더 못 줄임.
  - [[../results/exp010_per_gaussian_projection|exp010]] 에서 BCE 가 0.28 까지만 내려간 이유 = soft target 의 entropy floor.

## 우리 연구에서 사용처

| 위치 | $t$ | $\hat p$ |
|---|---|---|
| [[../results/exp001_tier2_dnerf_baseline|exp001]] cloth supervision | hard SAM3 mask (per-pixel 0/1) | rendered cloth_logit_image (per-pixel sigmoid) |
| [[../results/exp002_softcal_ablation|exp002]] soft-cal aux | $C_i \pm \alpha(d)$ (per-Gauss soft) | sigmoid(cloth_logit_i) |
| [[../results/exp010_per_gaussian_projection|exp010]] per-Gaussian | projection 평균 (per-Gauss soft ∈ [0,1]) | sigmoid(cloth_logit_i) |

## 깰 수 있는 부분

- BCE 외 다른 손실 — Focal loss (불균형 class), Dice loss (mask IoU 최적), MSE on probability
- weighted BCE — $\text{BCE} = -[w_+ t \log \hat p + w_- (1-t)\log(1-\hat p)]$ 로 양/음 class 가중 (드물게 cloth 가 sparse 일 때 효과)
- CE (cross-entropy, K-class) — [[../results/exp007_multiclass_K3|multi-class]] 에서 사용
