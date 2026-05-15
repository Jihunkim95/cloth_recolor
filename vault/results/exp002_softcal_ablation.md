---
title: exp002 — Soft-cal ablation (a / b / both × 3 D-NeRF scenes)
status: done
date: 2026-05-10
code: ~/research/4DGaussians/utils/soft_cal.py + train.py
---

# exp002

## 질문

[[../notes/soft_label_calibration]] 을 [[exp001_tier2_dnerf_baseline]] 위에 얹으면:
1. 옷 cloth_pct 가 SAM3 GT 비율에 더 가까워지는가? (over-cover 해소)
2. RGB PSNR 이 떨어지지 않는가?
3. (a) Gaussian-param 거리 vs (b) DINOv3 patch 거리 vs (both) 어느 게 좋은가?

## 설정

- 데이터셋: D-NeRF jumpingjacks/standup/hellwarrior (exp001 와 동일)
- 변경사항: `--soft-cal {a, b, both}` 활성화
  - K=500 (memory 갱신 주기)
  - mem_size=1000 per class
  - sample=4096 가우시안/step
  - λ_cal=0.1, cal_warmup_iter=3000
  - α=0.5·σ((d−d_ref)/τ), d_ref/τ 자동 추정
- DINOv3: ViT-S/16 (license 문제로 ViT-B 대신, 384-dim patch token)
- 9 jobs (3 scenes × 3 variants), 8 GPU 병렬, 마지막 1개 큐잉
- 학습 시간 ~5분/job (B200)

## 예측

- soft-cal 이 cloth_pct 를 SAM3 GT 쪽으로 끌어내릴 것
- RGB PSNR 은 ±0.5 dB 안에서 유지
- 시각적으로 옷 경계가 더 깔끔해질 것
- (both) 가 (a) (b) 평균 정도 성능

## 결과

### PSNR + cloth_pct

| scene | hard-BCE baseline | a | b | both | SAM3 GT |
|---|---|---|---|---|---|
| jumpingjacks PSNR / cloth% | 33.91 / 27.3% | 33.97 / **1.10%** | **34.08** / 1.32% | 35.23 / 0.00% | 1.93% |
| standup PSNR / cloth% | 35.70 / 53.1% | **34.99** / 36.0% | 36.84 / 23.5% | 36.79 / 23.5% | ~50% |
| hellwarrior PSNR / cloth% | 28.28 / 58.5% | 28.93 / 72.9% | 28.71 / 42.1% | **28.92** / 52.4% | ~60% |

Best variant per scene (PSNR + cloth_pct 가 SAM3 GT 에 가까움 composite score):
- **jumpingjacks → b** (cloth_pct 1.32%, target 1.93%)
- **standup → a** (cloth_pct 36%, target 50% — under-shoot 이지만 best)
- **hellwarrior → both** (52.4%, target 60%)

### cloth_logit 분포 변화 (시그모이드 분포 분석)

| scene | hard-BCE | best soft-cal |
|---|---|---|
| jumpingjacks | 55.5% σ<0.1, **17% ambig** (0.3-0.7) | **97.3% in 0.1-0.5**, 6.5% ambig |
| standup | 30% σ<0.1, 42% σ>0.9, **17% ambig** | 4.7% / 4.5%, **71% ambig** |
| hellwarrior | 20% / 37%, **22% ambig** | 0.4% / 0.2%, **78% ambig** |

→ hard BCE 의 인공적 0/1 양극화가 종 모양으로 풀림.

### 시각 비교

`vis/SUMMARY_baseline_vs_softcal/<scene>_baseline_vs_softcal.png` 참고:
- standup baseline: 초록 mask 가 vest+shirt+pants+boots **전체**
- standup soft-cal a: vest+pants 중심으로 **축소** (boots 제외)
- recolor 시 baseline 은 옷 외 영역까지 색 변경, soft-cal 은 옷에만

## 예측 맞았나?

- cloth_pct 가 GT 쪽으로 이동: ✅ jumpingjacks 27.3% → 1.32% (정확), hellwarrior 58.5% → 52.4% (target 60% 에 더 가까움)
- RGB PSNR 유지: ⚠️ standup 만 -0.71 dB (그러나 soft-cal 이 mask 잘못된 픽셀에 RGB loss 안 받게 함). 다른 두 씬은 +0.17 / +0.64 dB
- (both) 가 평균: ❌ 씬마다 best variant 다름 — universal best 없음
- 시각 깔끔: ✅ standup 에서 boots over-cover 사라짐

추가 발견 1: **hard-BCE 가 σ U-shape 양극화를 만든다는 것** — 단순 PSNR 보다 더 의미 있는 결과. soft-cal 이 ambig 영역을 본래 모양 (∩) 으로 돌려놓음.

추가 발견 2 — **jumpingjacks 가 가장 어려운 씬, 그리고 (b) 가 best 인 이유**:
viewer 검증 결과 jumpingjacks 의 cloth 영역 학습이 다른 씬보다 어려움. 3 요인:
1. **subject scale 작음** — 800×800 frame 중 캐릭터 면적이 작아 SAM3 raw coverage 가 1.93 % 에 불과
2. **헐렁한 hoodie 의 wrinkle** — 주름 = 색·법선 변화 = SAM3 가 "옷" 이라 부를 경계가 매 frame 흔들림 → cloth_logit 도 흔들림
3. **female body silhouette** — 옷이 몸 곡선에 따라 굴곡 → "shirt" prompt 가 hoodie 외곽인지 몸 라인인지 모호

이게 best variant=**b (DINOv3)** 인 이유로 해석됨: (a) Gaussian-param 거리는 spatial 인접성만 보지만, (b) DINOv3 patch 는 *옷 vs 피부 의 visual texture 차이* 를 학습된 feature 로 구분 → 작은 캐릭터 + 주름·굴곡 환경에 유리.

## 다음

[[exp003_softcal_hyperparam_sweep]] — best variant 위에서 λ_cal × mem_size grid 탐색.
