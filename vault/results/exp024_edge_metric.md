---
title: exp024 — Edge metric (boundary alignment, GT-free)
status: done
date: 2026-05-15
code: 2_process/edge_metric.py
---

# exp024

## 질문

미팅 [[../006Meeting_2026-05-15|2026-05-15]] 0.3 — cloth label GT 없는 상황에서 *boundary 품질* 을 수치적으로 측정?

## 두 metric 변형

### (1) RGB edge alignment — **misleading**

```python
e_rgb = Sobel(rendered RGB)      # body silhouette edges
e_pred = Sobel(sigmoid(cloth_logit))   # prediction edges
IoU = |e_rgb ∩ e_pred ∩ band| / |e_rgb ∪ e_pred ∩ band|
```

**문제**: 100% over-cover (baseline_hardBCE) 인 경우 prediction edge = body silhouette = RGB edge → IoU 가 *부당하게 높음*.

### (2) SAM3 boundary as pseudo-GT — **권장**

```python
e_sam3 = Sobel(SAM3 mask)         # hoodie boundary (true cloth)
e_pred = Sobel(sigmoid(cloth_logit))
IoU = |e_sam3 ∩ e_pred ∩ band(SAM3_dilated)| / |e_sam3 ∪ e_pred ∩ band|
```

**caveat**: SAM3 mask 가 학습 supervision 이므로 *순수 independent GT* 아님. "pseudo-GT" / "supervision-aligned boundary metric" 으로 표기 권장.

## 결과 (jumpingjacks, 8 view, SAM3-as-GT)

| ckpt | cloth_pct@τ | **IoU↑** | **Chamfer↓ (px)** | 비고 |
|---|---|---|---|---|
| baseline_hardBCE | 100% | 0.121 | 12.06 | catastrophic over-cover |
| softcal_best | ~92% | 0.095 | 13.41 | 가장 차이 큼 (over-cover + boundary blur) |
| **exp010** (per-Gauss) | 34.9% @ 0.7 | **0.253** | **6.40** | **best** |
| exp023_k0 (window 1×1) | 34.9% | 0.253 | 6.40 | exp010 와 동일 |
| exp023_k4 (window 9×9) | 34.8% | 0.225 | 7.18 | window 크면 약간 worse |

### 핵심 통찰

- **exp010 IoU 가 baseline 의 2.1×**, **Chamfer 의 절반** (12 → 6 px)
- exp010 = exp023_k0 (둘 다 window 0) 동일 → window 가 정말 효과 없음 ([[exp023_grid_soft_target|exp023]] 결론 정량 재확인)
- window k=4 면 IoU 약간 하락 (window 평균이 boundary 를 blur 시켜 prediction edge 가 weak)

## N3V 적용 시 제약

| ckpt | IoU | 문제 |
|---|---|---|
| n3v_baseline_4dgs | 0.000 | cloth_logit = 0 init → sigmoid = 0.5 → edge = 0 |
| n3v_exp015 | 0.000 | spatial filter 로 logit 너무 binary → sigmoid gradient ≈ 0 |
| n3v_exp019_joint | 0.099 | 큰 leakage |

**원인**: N3V exp015 의 cloth_logit 분포가 매우 sharp (대부분 < -5 또는 > 5) → sigmoid 미분 0 → Sobel edge 검출 안 됨.

**해결안** (미적용): pred = sigmoid(cloth_logit) 대신 `(sigmoid(cloth_logit) > τ).astype(float)` 의 boundary 를 추출 (binary mask 의 Sobel).

## 산출물

- `3_output/exp024_edge_sam3gt/jumpingjacks_<ckpt>/` — view_NNNN.png + metric.json
  - 각 view 의 3-column overlay: (RGB, cloth_prob heatmap, edge overlay [red=SAM3, green=pred, yellow=both])
- 5 ckpt × 8 view = 40 panel

## 예측 맞았나?

- exp010 가 baseline 보다 numerically 더 좋음: ✅ IoU 2.1×, Chamfer 0.5×
- window k 영향: ✅ 미미 ([[exp023_grid_soft_target|exp023]] 와 일치)
- SAM3-GT 가 RGB-edge 보다 정확: ✅ RGB-edge 로 baseline 이 best 였던 misleading 현상 해소

## paper 활용

**Table**: D-NeRF 8 scene 평균 IoU / Chamfer 비교 (baseline / softcal / exp010 / exp015)
- main result: "ours achieves 2.1× better boundary IoU vs hard-BCE baseline, 50% lower Chamfer distance"
- caveat: "metric uses SAM3 boundary as supervision-aligned proxy (not independent GT)"

**Figure**: 정성 zoom-in (`view_0000.png` 등)
- 옷 boundary 부근만 crop 해서 baseline vs ours 비교

## 다음

- 8 D-NeRF scene 모두 확장 (현재 jumpingjacks 만)
- N3V 의 binary sigmoid 문제 해결 → pred mask 의 boundary 추출 변형
- Human review setup (16 panel × Google Form)
