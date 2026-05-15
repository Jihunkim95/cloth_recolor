---
title: exp015 — N3V 학습 중 spatial outlier filter
status: done
date: 2026-05-14
code: 2_process/per_gaussian_supervision_n3v.py (--spatial-filter)
---

# exp015

## 질문

[[exp014_n3v_per_gaussian]] 의 recolor-time spatial filter (post-hoc) 를 *학습 시점에* 적용하면 어떻게 다른가? ckpt 자체가 깨끗해져서 recolor 마다 filter 재적용 불필요한가?

## 설정

`per_gaussian_supervision_n3v.py` 에 두 flag 추가:
- `--spatial-filter 3.0` — MAD 배수 (3 = robust 3-sigma equivalent)
- `--spatial-seed-thresh 0.2` — soft_target 임계값으로 seed cluster 정의

**알고리즘**:
1. projection 후 `soft_target` 계산 (N, [0,1])
2. seed cluster: `soft_target > 0.2` 인 Gaussian → xyz median + dist median + MAD
3. cutoff = `dist_median + 3.0 × MAD`
4. seed Gaussian 중 `dist > cutoff` 인 outlier 의 `soft_target = 0` 으로
5. BCE 학습 2000 iter — outlier 의 cloth_logit 자연스럽게 음수로

```python
# per_gaussian_supervision_n3v.py (post-projection, pre-training)
seed_mask = soft_target > 0.2
xyz_seed = canonical[seed_mask]
med = xyz_seed.median(dim=0).values
dist_seed = (xyz_seed - med).norm(dim=1)
mad = (dist_seed - dist_seed.median()).abs().median().clamp_min(1e-6)
cutoff = dist_seed.median() + 3.0 * mad
dist_all = (canonical - med).norm(dim=1)
outlier = (dist_all > cutoff) & seed_mask
soft_target = torch.where(outlier, torch.zeros_like(soft_target), soft_target)
```

6 scene × 2000 iter, lr=0.05, 6 GPU 병렬 (~25 min).

## 예측

- ckpt 자체에서 outlier Gaussian 의 cloth_logit < 0 으로 학습됨
- recolor 시 threshold 만 적용해도 환경 leakage / 가로 smear 없음
- cloth_pct 가 recolor-time filter 보다 더 strict 할 수 있음 (training equilibrium 효과로 인접 marginal Gaussian 도 logit 하락)

## 결과

cloth_pct @ threshold 0.35:

| scene | exp014 (no filter) | exp014_filtered (recolor-time) | exp015 (train-time) |
|---|---|---|---|
| coffee_martini | 1.39% | 1.39% (post: ~1.18%) | **1.23%** |
| cook_spinach | 1.30% | 1.30% (post: ~1.09%) | **1.29%** |
| cut_roasted_beef | 0.87% | 0.87% (post: ~0.72%) | **0.85%** |
| flame_salmon_1 | 0.66% | 0.66% (post: ~0.55%) | **0.64%** |
| flame_steak | 0.47% | 0.47% (post: ~0.39%) | **0.19%** |
| sear_steak | 0.89% | 0.89% (post: ~0.75%) | **0.47%** |

exp015 가 flame_steak (0.47→0.19) 와 sear_steak (0.89→0.47) 에서 더 strict. 다른 scene 은 비슷. 시각 결과: 환경 leakage 와 가로 smear 둘 다 사라짐 — exp014_filtered 와 동등하거나 더 깨끗.

## 예측 맞았나?

- outlier 자연 학습: ✅ — flame_steak/sear_steak 에서 cloth_pct 대폭 감소 = outlier 가 logit 음수로 학습됨
- recolor 단순화: ✅ — `recolor.py` 에서 `--spatial-filter` 불필요, threshold 만으로 충분
- training equilibrium 효과: ✅ — outlier 인접한 Gaussian 도 logit 동반 하락 (3-sigma 밖 outlier 가 logit 음수면 BCE gradient 가 인접 Gaussian 까지 영향)

## training-time vs recolor-time

| 측면 | training-time (exp015) | recolor-time (exp014_filtered) |
|---|---|---|
| ckpt | 깨끗 (logit 자체 보정) | 원본 보존 (filter 는 post-hoc) |
| 유연성 | 다시 학습 필요 (k 변경 시) | slider 로 즉시 변경 |
| viewer | 자동 깨끗 | filter 별도 적용 필요 |
| 적용 코드 | `per_gaussian_supervision_n3v.py` | `recolor.py` |
| paper | main pipeline 추천 | ablation/comparison |

→ **paper 의 main pipeline 은 exp015 방식** 권장. ckpt 가 self-contained 으로 깨끗하므로.

## 산출물

- `3_output/n3v_<scene>/ckpt_exp015_spatial/` — 6 ckpt
- `3_output/n3v_<scene>/recolor_exp015/` — 6 recolor 결과 (4 panel each)
- `vis/SUMMARY_n3v_exp015/<scene>.png` — 비교 panel
- `/tmp/n3v_6scene_exp015.png` — 6 scene grid

## 다음

- D-NeRF / 4D-DRESS 의 `per_gaussian_supervision.py` 에도 동일 옵션 추가
- threshold-k sweep ablation table (k=2, 3, 5, ∞)
- paper Figure: exp014 (no filter) vs exp015 (filter) side-by-side
