---
title: exp023 — Grid-based soft_target (window 평균)
status: done (negative)
date: 2026-05-15
code: per_gaussian_supervision.py / _n3v.py --grid-window
---

# exp023

## 질문

미팅 [[../006Meeting_2026-05-15|2026-05-15]] 0.3 — center pixel lookup (`mask[py, px]`) 대신 *window 평균* (`mask[py-k:py+k, px-k:px+k].mean()`) 으로 soft_target 을 만들면 boundary 가 부드러워질까?

## 설정

- 2 scene × 4 window = 8 학습
  - D-NeRF jumpingjacks (sam3_union, exp010 baseline)
  - N3V sear_steak (sam3_n3v_union, spatial filter 3.0)
- window k ∈ {0 (current 1×1), 1 (3×3), 2 (5×5), 4 (9×9)}
- 학습 2000 iter, lr 0.05
- 8 GPU 병렬

## 구현

```python
# per_gaussian_supervision.py / _n3v.py
if args.grid_window > 0:
    k = args.grid_window
    offs = torch.arange(-k, k+1, device=device)
    dy, dx = torch.meshgrid(offs, offs, indexing="ij")
    py_w = (py.unsqueeze(-1) + dy.flatten().unsqueeze(0)).clamp(0, H-1)
    px_w = (px.unsqueeze(-1) + dx.flatten().unsqueeze(0)).clamp(0, W-1)
    v = mask[py_w, px_w].float().mean(dim=-1)
else:
    v = mask[py, px]
```

## 결과

### D-NeRF jumpingjacks (t=0.7 recolor)

| k | window | n_cloth | cloth_pct |
|---|---|---|---|
| 0 | 1×1 | 8,721 | 34.9% |
| 1 | 3×3 | 8,725 | 35.0% |
| 2 | 5×5 | 8,715 | 34.9% |
| 4 | 9×9 | 8,676 | 34.8% |

### N3V sear_steak (t=0.3 recolor, spatial filter)

| k | window | n_cloth | cloth_pct |
|---|---|---|---|
| 0 | 1×1 | 14,845 | 16.4% |
| 1 | 3×3 | 14,849 | 16.4% |
| 2 | 5×5 | 14,849 | 16.4% |
| 4 | 9×9 | 14,835 | 16.4% |

→ **window size 가 cloth_pct 에 거의 영향 없음** (±0.5%).

### 시각

`/tmp/exp023_dnerf_jumpingjacks.png`, `/tmp/exp023_n3v_sear_steak.png`:
- 4 rows (k=0/1/2/4) 가 시각적으로 *구별 불가능*
- hoodie / chef body 의 boundary 가 동일

## 예측 맞았나?

- 기대: window 가 클수록 boundary 가 부드러워짐 → 더 자연스러운 분리
- 실제: ✗ 거의 변화 없음

**원인**:
1. SAM3 mask 가 **이미 sharp binary**. boundary 가 1-2 pixel wide
2. window 평균이 영향 주는 곳: **boundary 근처 Gaussian 들**만
3. 그러나 boundary Gaussian 은 *어차피 threshold 로 분리*. 평균값이 0.3 → 0.4 가 되어도 threshold 0.7 이면 동일하게 cut

## 핵심 통찰

grid 가 의미 있는 경우 (미적용):
- **저해상도 mask** (예: 50×50)
- **fuzzy mask** (예: hair, fur, transparency)
- **multi-scale detail** (예: 작은 모자 vs 큰 셔츠)

ClothSplat 의 chef shirt/apron scenario 에선 단일 pixel 으로 충분.

## paper 활용

negative result 로 보고. ablation table 에 "window k ablation: no significant effect" 로 1줄. 그 대신 *알고리즘의 *robustness*는* 보여줌 — 정확한 hyperparameter 튜닝 불필요.

## 산출물

- `3_output/jumpingjacks/ckpt_exp023_grid_k{0,1,2,4}/` — 4 ckpts
- `3_output/n3v_sear_steak/ckpt_exp023_grid_k{0,1,2,4}/` — 4 ckpts
- `3_output/{scene}/recolor_exp023_k{k}/` — 8 recolor 결과
- `/tmp/exp023_{scene}.png` — 시각 grid

## 다음

- 더 fuzzy mask 환경에서 retry: e.g., hair, fur, partial occlusion
- 다음 exp024 (edge metric) 으로 진행 — *수치적으로* boundary 차이 측정
