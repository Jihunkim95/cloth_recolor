---
title: exp021 — Pipeline 효율성 측정 (baseline vs ours)
status: done (1차)
date: 2026-05-15
code: 2_process/exp021_efficiency.py
---

# exp021

## 질문

미팅 [[../006Meeting_2026-05-15|2026-05-15]] 0.2 — baseline (매번 SAM3) vs ours (train 1회 + cloth label) 의 inference wall-time 차이?

## 설정

- scene: jumpingjacks (D-NeRF), N=24,964 Gaussians
- ckpt: `ckpt_exp010_per_gaussian` (iter 20000)
- frame: 50개 (warm-up 제외 평균)
- GPU: B200 (CUDA_VISIBLE_DEVICES=6)
- threshold: 0.7
- target hue: 220° (blue)

### 파이프라인

**baseline** (매 frame):
1. 4DGS render with 원본 SH_DC → RGB image
2. SAM3 (`facebook/sam3`, prompt="hoodie") on rendered image → 2D mask
3. HSV swap on masked pixels → final RGB

**ours** (매 frame):
1. cloth Gaussian (`sigmoid(cloth_logit) > τ`) 의 SH_DC 미리 HSV swap (1회)
2. 4DGS render with swapped SH_DC → final RGB

## 결과

| metric | baseline | ours |
|---|---|---|
| **avg / frame** | **180.2 ms** | **6.1 ms** |
| render | 3.4 ms | 6.1 ms |
| SAM3 inference | 148.6 ms | — (precomputed) |
| 2D HSV swap | 28.2 ms | — (3D swap precomputed) |
| **speedup** | — | **29.78×** |

→ baseline 의 *82% 시간이 SAM3 inference*. SAM3 를 추론에서 제거하면 **30× 빠름**.

## 예측 맞았나?

- speedup ≥ 5×: ✅ (29.78×)
- SAM3 가 bottleneck: ✅ (149/180 = 82.5%)
- ours 의 render time 이 baseline 보다 약간 큼 (3.4 → 6.1 ms): SH_DC swap 으로 인한 GPU 메모리 이동? — 큰 영향 없음

## 다음

- 더 많은 scene 으로 확장 (standup, sear_steak, 4D-DRESS 1 take)
- 표준편차 (5+ scene 평균 ± std) 측정 → paper Table 1
- 4DGS render resolution sweep (현재 D-NeRF 800×800)
- SAM3 batching 가능성 (다중 frame 한 번에) — baseline 의 fairness 보정 ablation

## 산출물

`3_output/jumpingjacks/exp021_efficiency/result.json`:
```json
{
  "ours_avg_ms": 6.1,
  "baseline_avg_ms": 180.2,
  "speedup": 29.78,
  "cloth_pct": 34.93
}
```
