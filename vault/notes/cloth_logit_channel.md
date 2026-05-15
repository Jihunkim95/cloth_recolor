---
title: per-Gaussian cloth_logit channel
layer: 2
date: 2026-05-11
---

# per-Gaussian `cloth_logit` channel

> 우리가 4DGS 가우시안 한 개당 추가한 1차원 학습 가능 라벨. layer 2.

## 동기

기존 [[4d_gaussian_splatting]] 의 가우시안 attribute = **위치** (xyz) + **shape** (scale, rotation) + **opacity** + **색** (SH_DC + SH_rest) — *의미 정보는 없음*. "이 가우시안이 옷인지" 를 모르므로 옷만 선택적으로 편집 불가.
편집을 위해서는 가우시안 단위 의미 라벨이 필요.

## 정의

각 가우시안 $g_i$ 에 스칼라 $\ell_i \in \mathbb{R}$ 추가. 옷 확률 = $\sigma(\ell_i)$.

```python
# scene/gaussian_model.py
self._cloth_logit = nn.Parameter(torch.randn(N, 1) * 0.01)
```

학습 가능. Adam group 으로 lr=1e-2 (다른 가우시안 attribute 와 동일 schedule).

## supervision

per-pixel BCE — [[sam3_mask_supervision]] 참고.

```
L_cloth = BCE( render2D(cloth_logit), SAM3_mask )
```

여기서 `render2D` 는 gsplat 2nd-pass — 가우시안의 cloth_logit 을 alpha-composit.

## 추가로 [[soft_label_calibration]]

per-Gaussian aux loss — 옷 경계 모호성 처리.

## 편집 (HSV recolor) 시 사용

```python
cloth_mask = sigmoid(cloth_logit) > τ  # τ=0.2 ~ 0.5
gaussians[cloth_mask].sh_dc = HSV_swap(gaussians[cloth_mask].sh_dc, hue=220)
```

여기서 `sh_dc` 는 SH degree 0 항 = 시점 무관 베이스 RGB. [[hsv_recolor_trick]] 참고.

## 깰 수 있는 부분

- 1차원 라벨 (옷/비옷 binary 만) — multi-class (셔츠/바지/모자/...) 로 확장 가능
- BCE supervision 만 — multi-task (color, material) 동시 학습 가능
- Threshold τ 가 hard cutoff — soft masking (probabilistic edit) 로 그라데이션 가능
