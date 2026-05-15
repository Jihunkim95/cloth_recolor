---
title: Soft-label calibration via memory-based negative mining
layer: 2
date: 2026-05-11
---

# Soft-label calibration

> [[sam3_mask_supervision]] 의 hard 0/1 라벨이 옷 경계 모호성을 무시하는 문제를 풀기 위한 layer 2 trick. 이 연구의 핵심 기여.

## 동기

옷 경계는 인간도 모호 — 후드 안쪽 머리, 목/카라 경계, 신발/바지 경계 등.
SAM 3 가 hard 0/1 라벨로 분류하면 `cloth_logit` 분포가 **인공적으로 양극화**:

| | hard-BCE 학습 후 | 가설 |
|---|---|---|
| jumpingjacks | 55% σ<0.1, 13% σ>0.9, **17% ambig** (0.3-0.7) | 더 ambig 해야 자연스럽다 |
| standup | 30% / 42% / **17% ambig** | 같음 |
| hellwarrior | 20% / 37% / **22% ambig** | 같음 |

## 아이디어

라벨 $C_i \in \{0, 1\}$ 을 다음과 같이 **calibrate** :

$$
\text{target}_i = \begin{cases} 0 + \alpha(d_i) & \text{if } C_i = 0 \\ 1 - \alpha(d_i) & \text{if } C_i = 1 \end{cases}
$$

여기서:
- $\alpha(d) = 0.5 \cdot \sigma\!\left(\frac{d - d_{\text{ref}}}{\tau}\right) \in [0, 0.5]$
- $d_i$ = 가우시안 $i$ 의 feature 가 자기 클래스 **easy memory** 에서 가장 가까운 점까지의 거리
- 멀수록 → 라벨 신뢰도 낮음 → α↑ → target 이 0.5 쪽으로 soften

## Memory-based negative mining

매 K=500 iter 마다:
1. 모든 가우시안 cloth_prob = σ(cloth_logit) 평가
2. **easy_cloth memory** = top-1000 highest cloth_prob (확실히 옷)
3. **easy_noncloth memory** = bottom-1000 (확실히 비옷)
4. **hard sample 거리** 의 median = $d_{\text{ref}}$, std = $\tau$ (자동 추정)

학습 step 에서: 4096 가우시안 샘플링 → 자기 클래스 memory 와 nearest distance → α(d) → target → BCE.

## Feature 두 종류 (a/b/both ablation)

### (a) Gaussian-param feature

각 가우시안의 자체 parameter 14-dim:
```
feat_a(g) = z_score( concat(xyz, log_scale, quat, opacity_logit, sh_dc) )
```
- 빠름 (학습 시간 무시할 수준)
- 가우시안 공간에서 이웃 = 비슷한 위치/모양 → 옷 클래스 spatial 일관성 부여

### (b) DINOv3 patch feature

렌더된 2D 이미지에서 가우시안이 투영되는 patch 의 DINOv3 ViT-S 토큰:
```
feat_b(g) = DINOv3_patch_token( project(g.xyz, cam) )  # 384-dim
```
- 약 +10% 학습 시간
- semantic 공간에서 이웃 = 비슷한 visual appearance → 옷 클래스 semantic 일관성

(처음 시도한 ViT-B (`vitb16-pretrain`) 는 license 미허가, ViT-S 사용)

### (both)

두 distance 의 α 평균 — spatial + semantic 결합.

## hybrid loss (iii)+(i) 디자인

```
L_total = L_rgb + L_reg
        + λ_BCE * BCE(rendered_cloth_image, sam3_mask)         ← 기존 per-pixel BCE 유지
        + λ_cal * BCE(σ(cloth_logit_i), target_i)              ← 새로 추가, per-Gaussian
```

per-Gaussian C_i derivation: canonical 가우시안 평균을 학습 카메라 ≤16개에 투영 → SAM3 mask 값 평균 → 0.5 threshold. 1000 iter 마다 1회 갱신 (매 step 아님).

## 결과 (from [[../results/exp002_softcal_ablation|exp002]])

cloth_logit 분포가 종 모양으로 변함:

| | hard-BCE | best soft-cal |
|---|---|---|
| jumpingjacks 가우시안 분포 | 55% σ<0.1, 13% σ>0.9 | **97% in 0.1-0.5** (모두 "uncertain non-cloth" 로 정정) |
| standup ambig (0.3-0.7) | 17% | **71%** |
| hellwarrior ambig | 22% | **78%** |

PSNR 도 향상: jumpingjacks +0.17, hellwarrior +0.64 ([[../results/exp002_softcal_ablation]] 참고).

## 깰 수 있는 부분

- α 함수 형태 (sigmoid 고정) — 학습 가능한 작은 MLP 로 교체 가능
- d_ref/τ 자동 추정 — sample 별 adaptive 하면 더 좋을 수 있음
- memory size K=1000 — scene scale 에 따라 적응적
- hybrid 의 λ_BCE / λ_cal balance — 학습 진행에 따라 동적 조정 (BCE→0, cal→1)
