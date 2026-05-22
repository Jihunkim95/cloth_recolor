---
title: exp022 — Threshold 매핑 ablation (binary 화)
status: done
date: 2026-05-15
code: 2_process/recolor.py --threshold (sweep)
---

# exp022

## 질문

미팅 [[../006Meeting_2026-05-15|2026-05-15]] 0.3+α — per-Gaussian `sigmoid(cloth_logit)` 를 binary cloth/non-cloth 으로 자를 때 **threshold** 영향?

- 단순 `> 0` 이면 모두 cloth
- `> 0.5` 이면 sigmoid 의 confident half 만
- 그 사이의 sweep 으로 best threshold 찾기

## 설정

- 2 scene × 6 threshold = 12 runs
  - D-NeRF jumpingjacks (`ckpt_exp010_per_gaussian`, iter 20000, N=24,964)
  - N3V sear_steak (`ckpt_exp015_spatial`, iter 14000, N=90,543)
- threshold: 0.0 / 0.1 / 0.3 / 0.5 / 0.7 / 0.9
- recolor: hue 220° (blue), min_sat=0.6, min_val=0.7
- 6 GPU 병렬 (각 GPU 1 run)

## 결과

### D-NeRF jumpingjacks (exp010 — *per-Gaussian projection 만, spatial filter 없음*)

| threshold | n_cloth | cloth_pct | 시각 |
|---|---|---|---|
| 0.0 | 24,964 | 100.0% | 전체 파랑 (catastrophic) |
| 0.1 | 14,549 | 58.3% | hoodie + shorts (over-cover) |
| 0.3 | 10,656 | 42.7% | hoodie + 약간 다리 |
| 0.5 | 9,276 | 37.2% | hoodie + 미세 bleeding |
| **0.7** | **8,721** | **34.9%** | **✓ hoodie 깔끔, 다른 부분 그대로** |
| 0.9 | 6,159 | 24.7% | hoodie 일부 누락 (under-cover) |

→ **sweet spot t=0.7**. 0.5/0.7/0.9 모두 hoodie 추출하지만, 0.7 이 face/shorts/shoes 와 가장 잘 분리.

### N3V sear_steak (exp015 — *per-Gaussian projection + spatial filter*)

| threshold | n_cloth | cloth_pct | 시각 |
|---|---|---|---|
| 0.0 | 90,543 | 100.0% | 전체 파랑 (catastrophic) |
| 0.1 | 14,885 | 16.4% | chef body + 환경 bleeding 큼 |
| **0.3** | **1,075** | **1.2%** | **✓ chef apron 위주, 환경 보존** |
| 0.5 | 19 | 0.02% | 거의 없음 |
| 0.7 | 0 | 0.0% | 0 |
| 0.9 | 0 | 0.0% | 0 |

→ **sweet spot t=0.3**. exp015 의 spatial filter 로 logit 분포가 좁아져 0.5 이상은 거의 없음.

## 예측 맞았나?

- threshold 가 dataset/method 마다 다름: ✅ — D-NeRF 의 *bimodal distribution* 은 t=0.7, N3V exp015 의 *압축된 distribution* 은 t=0.3
- baseline `> 0` 은 의미 없음: ✅ — 모두 100% over-cover (sigmoid logit 의 초기값이 0 이면 시작부터 0.5)
- `> 0.5` 가 항상 best 는 아님: ✅ — N3V 에선 0.5 이상이 텅 빔

## 핵심 통찰 (paper 활용)

**threshold 는 method dependent**:
- per-Gaussian projection 만 (D-NeRF/exp010): logit 의 *bimodal* → 0.5-0.7 권장
- spatial filter 후 (N3V/exp015): logit 이 *negative skew* → 0.2-0.4 권장

ablation table 의 "default" threshold 선택 시 method 별로 보고 필요.

## 산출물

- `3_output/exp022_thresh_sweep/{scene}_t{value}/` — 각 run 의 panel + summary.json
- `/tmp/exp022_dnerf_jumpingjacks.png` — D-NeRF 6 threshold grid
- `/tmp/exp022_n3v_sear_steak.png` — N3V 6 threshold grid

## 다음

- 다른 D-NeRF scene (standup, hellwarrior, ...) 도 sweep — generalization 확인
- 4D-DRESS exp013 ckpt 도 sweep (GT vertex 라 또 다른 분포 예상)
- `exp023` 의 grid soft_target 에도 같은 threshold ablation 반복
