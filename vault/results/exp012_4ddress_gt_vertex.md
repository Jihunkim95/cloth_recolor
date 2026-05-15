---
title: exp012 — 4D-DRESS GT vertex label per-Gaussian supervision
status: done
date: 2026-05-12
code: 2_process/per_gaussian_supervision_4ddress_gt.py
---

# exp012

## 질문

[[exp010_per_gaussian_projection]] (D-NeRF + SAM3) 의 *per-Gaussian projection supervision* 을 4D-DRESS 에 적용 시도 [[exp011_4ddress_sam3]] 했지만 SAM3 noise + greenscreen 배경 흡수로 실패. **4D-DRESS 가 *직접 제공하는* vertex semantic label** (Semantic/labels/ — per-vertex 6-class 라벨) 을 사용하면 deeper supervision 가능한가?

## 설정

- 8 4D-DRESS takes: 00122/00123/00127/00129/00148 (Inner), 00148/00170 (Inner), 00169/00190 (Outer)
- base ckpt: 기존 학습된 tier2 4DGS (RGB + deformation MLP)
- supervision pipeline:
  ```
  for each frame t:
    load mesh-fXXXXX.pkl (V_t shape (n_vt, 3))
    load label-fXXXXX.pkl (L_t shape (n_vt,)) — int 0..5
  for each Gaussian g:
    for each frame t:
      g_t = deformation_MLP(g.xyz, time_t)
      d_min, i_nn = kNN(g_t, V_t)
      if d_min < max_dist (0.05): label_t = L_t[i_nn]
      else:                       label_t = 0   # background
    label_dist_g = histogram(label_t for all frames) / T   # (6,)
    soft_target_g = label_dist_g[target_label]
  ```
- target_label: Inner=3 (upper cloth), Outer=5 (outer garment)
- train cloth_logit only, freeze else, BCE, 2000 iters

## 예측

- SAM3 우회 → noise 없음 → soft_target bimodal (cloth vs not)
- cloth_pct 가 mesh 의 target_label vertex 비율에 수렴 (Inner ~5-10%, Outer ~8-9%)
- 시각: jacket/shirt 영역만 정확히 recolor

## 결과

### per-take label class 0..5 분포 + cloth_pct

| take | label 0 | 1 | 2 | 3 | 4 | 5 | target | cloth_pct@0.5 |
|---|---|---|---|---|---|---|---|---|
| 00122 Inner Take2 | 86.8% | 1.2% | 2.4% | 6.9% | 2.7% | 0% | 3 | 6.5% |
| 00123 Inner Take1 | 83.2% | 1.1% | 2.0% | 5.6% | 8.2% | 0% | 3 | 5.3% |
| 00127 Inner Take10 | 86.6% | 1.6% | 2.8% | 5.7% | 3.3% | 0% | 3 | 4.7% |
| 00129 Inner Take1 | 90.0% | 1.3% | 2.1% | 2.8% | 3.8% | 0% | 3 | 2.6% |
| 00148 Inner Take1 | 82.9% | 1.4% | 2.5% | 13.3% | 0% | 0% | 3 | 14.7% |
| 00169 Outer Take12 | 81.1% | 0.8% | 1.0% | 0.5% | 7.8% | 8.7% | 5 | 8.1% |
| 00170 Inner Take1 | 84.3% | 2.2% | 4.2% | 5.4% | 3.9% | 0% | 3 | 4.9% |
| 00190 Outer Take10 | 84.4% | 1.5% | 1.3% | 1.3% | 3.1% | 8.4% | 5 | 6.8% |

label 의미 (vertex 분포 + Inner/Outer 차이로 추정):
- 0: body/skin/hair (대부분)
- 1: shoes
- 2: hair
- 3: upper cloth (shirt) — Inner takes 의 main cloth
- 4: lower cloth (pants)
- 5: outer garment (jacket) — Outer takes 에만 존재

### 시각

`3_output/<take>/recolor_exp012_gt_adaptive/`. 00169 Outer Take12 의 경우 right shoulder/chest 영역이 정확히 blue 로 변경. 다만 jacket 이 *검은색* (V≈0) 이라 HSV hue swap 만으로는 안 보이고, `--min-sat 0.6 --min-val 0.6` 강제 필요.

### 도전 과제

- **distance threshold 0.05 필수** — 없으면 배경 Gaussian 이 mesh 의 nearest vertex 라벨 흡수 → cloth_pct 60%+ 의 가짜 over-cover
- **검정/회색 의류 visualization**: hue swap 효과 없음, min_sat/min_val 강제 필요 → recolor.py 에 옵션 추가
- **occluded vertex**: jacket 의 내측/뒤쪽 vertex 도 label 5 → cloth Gaussian 으로 학습. 정면 view 에서는 보이지 않아 visual mismatch.

## 예측 맞았나?

- SAM3 우회 → noise 없음: ✅ exp011 의 0.0-3.8% (저성공) → exp012 의 5-15% (적절)
- bimodal soft_target: ✅ class 0 이 80%+ (확실히 not-cloth), target class 가 5-15% (확실히 cloth)
- 시각 정확: ⚠️ 부분 성공 — cloth Gaussian 식별은 정확하나 검정 옷 + occlusion 때문에 visual coverage 가 mask coverage 보다 적음

## Method 의 핵심 contribution 확인

**(1) mask-agnostic per-Gaussian projection supervision** — SAM3 (D-NeRF) 와 GT vertex label (4D-DRESS) 모두에서 동일 framework 로 작동.
**(2) GT mask 사용 시 upper-bound 검증** — D-NeRF SAM3 결과가 GT supervision 과 비슷한 수준에 도달함을 별도 metric 으로 보일 수 있음.

## 다음

- 다른 target_label (예: 4 = pants) 도 시도해서 multi-garment recolor 데모
- min_val 강제가 hue swap 본질을 깨므로, *base color 가 dark 인 경우의 정직한 visualization* 방법 모색 (e.g., 흰 줄무늬 overlay)
- 4D-DRESS Outer outfit (color 가 있는 garment) take 추가 — 검정/회색 cloth 의존 한계 극복
- 전체 32 subject 확장 — extracted 데이터 사용 가능
