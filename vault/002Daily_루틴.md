---
title: Daily 루틴
pinned: true
---

# Daily 루틴

> 매일 아침 이 파일을 연다. 3가지를 하고 닫는다. 5분이면 끝난다.

---

## 1. 어제 한 줄 (1분)

아래에 오늘 날짜를 쓰고, 어제 한 것을 **한 줄**로 적는다.
일주일이 지나면 지난 주 것은 지운다. 쌓이면 안 연다.

### 이번 주 기록

- 2026-02-16 (월): exp1 실험 및 PBR 렌더링에서는 hemisphere를 사용한순간 굴절율과 왜곡을 제외한것으로 확인
- 2026-02-17 (화): Layer0 맥스웰 방정식의 유도과정에서 법칙과 맥스웰 방정식에서 파생된 반사법칙, 스넬법칙, 프네렐 방정식
- 2026-02-20 (금): Fresnel equations복습과 BRDF 읽기
- 2026-02-23 (월): 3DGS-DR 코드 분석 및 정리중에 용어가 헷갈려서 '렌더링_파이프라인' 공부중
- 2026-05-10 (일): D-NeRF 3씬 cloth_logit hard-BCE baseline ([[results/exp001_tier2_dnerf_baseline]]) → soft-cal a/b/both ablation ([[results/exp002_softcal_ablation]]) → λ×mem 스윕 ([[results/exp003_softcal_hyperparam_sweep]])
- 2026-05-11 (월): soft-cal 일반화 검증 lego/bouncingballs/hook ([[results/exp004_softcal_extra_scenes]]); cloth_recolor/ 폴더 정리 (4D-DRESS 잔재 archive); viewer D-NeRF 호환 추가; Obsidian vault 노트 5+4개 작성
- 2026-05-11 (월) PM: D-NeRF 8씬 전체 커버리지 확장 ([[results/exp005_dnerf_full_8scene_coverage]]) — 새 5씬 baseline 모두 cloth_pct=100% collapse 발견, soft-cal 이 부분 보정
- 2026-05-11 (월) PM2: jumpingjacks 8-way grid ([[results/exp006_jumpingjacks_8way_grid]]) — 8 GPU 풀가동 prompt/λ_cal/warmup 동시 sweep. "orange hoodie" prompt 가 SAM3 coverage 2× 향상시키지만 BCE supervision 강화로 over-cover 가속 → 기존 `shirt`+soft-cal b 복원
- 2026-05-11 (월) PM3: Multi-class K=2/3 8씬 학습 ([[results/exp007_multiclass_K3.md|exp007]]) — collapse 해소 (100% → 18-50%) 성공. 그러나 SAM3 coverage < 10% 이라 class 0 이 너무 좁아 "other" 가 hoodie 흡수. 다음: multi-prompt union 으로 cloth coverage 확장
- 2026-05-12 (화): exp008/009 overnight (multi-prompt union, class weight 3/10/30/100, K=3, 30k iter) — cloth_pct 일부 개선 but face 까지 over-cover 지속. exp010 [[results/exp010_per_gaussian_projection]] — **per-Gaussian projection supervision (alpha-composit 우회)** 으로 *드디어 깔끔한 옷 분리 달성*. jumpingjacks 의 hoodie 가 face 영향 없이 파랑으로 변경. 8씬 모두 적용 완료. 핵심 breakthrough.
- 2026-05-12 (화) PM: exp011 4D-DRESS SAM3 시도 → noise 로 실패. exp012 [[results/exp012_4ddress_gt_vertex]] — **4D-DRESS GT vertex semantic label** 로 supervision (SAM3 우회). distance threshold 0.05 로 배경 제외. 8 takes 모두 cloth_pct 2-15% (적절). 검정 jacket 의 visualization 제한은 min_sat/min_val 강제로 보완. mask-agnostic framework 검증.
- 2026-05-12 (화) PM2: exp013 — exp012 결과가 흐릿했던 원인이 4DGS baseline 의 sparse-view 한계로 판명. 4DGS 8 takes 60k iter + 적극 densification 재학습 (N 30k→170k, PSNR 30→34+). exp012 GT supervision 재적용. **jacket 영역만 깔끔하게 분리** 됨 (배경 흡수 해결). D-NeRF [[results/exp010_per_gaussian_projection]] 가 main, 4D-DRESS 는 supplement validation.
- 2026-05-13 (수): **Neural 3D Video** ([[results/exp014_n3v_per_gaussian]]) — 6 scene 18-21 카메라 multi-view dynamic. mp4→PNG ffmpeg 병렬 추출 (3분), COLMAP sparse init (PSNR 14→30dB), SAM3 cache "shirt", per-Gaussian projection (5400-6300 view/frame pair). cook_spinach 의 chef shirt 정확히 파란색 recolor 검증 — multi-cam supervision 이 noise 자연 감소. 5/6 scene 성공 (flame_steak CUDA assert retry).
- 2026-05-14 (목): N3V 환경 leakage + 가로 motion trail 진단. cloth Gaussian xyz spread/scene xyz spread = 0.3-0.6 (D-NeRF 의 0.15 대비). 원인: low SAM3 cov 에서 threshold 0.2 가 marginal Gaussian 포함. **fix 1**: threshold 0.2→0.35 (cloth_pct 9.4%→0.9%). **fix 2**: spatial outlier filter (median ± 3×MAD, recolor.py 에 `--spatial-filter` 옵션 추가) — 추가 16-20% 제거. **fix 3**: flame_salmon_1 chef 이미 파란 옷이라 hue=0 (red) 변경. viewer.py 에 N3V 카메라 branch 추가 — `cam00/images/` 감지로 Neural3D_NDC_Dataset 로딩.
- 2026-05-14 (목) PM: [[results/exp015_n3v_train_time_spatial_filter|exp015]] — spatial filter 를 **학습 중** 적용 (recolor-time 후처리 → soft_target seed cluster 의 MAD outlier 의 target=0). ckpt 자체가 깨끗해져 recolor 마다 filter 재적용 불필요. flame_steak/sear_steak 에서 cloth_pct 추가 50% 감소 (training equilibrium 으로 인접 marginal Gaussian 도 logit 동반 하락). paper main pipeline 으로 채택.
- 2026-05-15 (금): 미팅 ([[006Meeting_2026-05-15]]) — 9페이지 (soft-label calibration) 재논의. 피드백: baseline vs 우리 방법 비교, GT 부재로 수치 metric 불가능, SAM3 속도 분석, **2D pixel → 3D GS label mapping** 이 핵심 (soft_target = True/grid). 4D 축 확장 + multi-object (N≥3 type × M≥2 scene) + edge metric/human review 평가. 9페이지 원래 plan 의 *training softness via positive/negative mining* 는 그대로 유효.
- 2026-05-15 (금) PM: [[007Experiment_Plan_2026-05-15|실험 sprint 설계]] (exp021~027). **[[results/exp021_efficiency|exp021]]** — pipeline 효율성: baseline (SAM3 매 frame) 180ms vs ours 6.1ms = **30× speedup** (SAM3 가 82% 차지). **[[results/exp022_threshold_sweep|exp022]]** — threshold sweep 2 scene × 6 t: D-NeRF/exp010 sweet spot t=0.7 (bimodal), N3V/exp015 sweet spot t=0.3 (spatial filter 로 logit 분포 압축). threshold 는 method dependent — paper 표 에 method 별 보고 필요. **[[results/exp023_grid_soft_target|exp023]]** — grid soft_target (window k=0/1/2/4): cloth_pct ±0.5% 변화 (negative). 원인: SAM3 mask 가 sharp binary 라 window 평균이 boundary 외 무영향. paper ablation 에 robustness 로 보고. **[[results/exp024_edge_metric|exp024]]** — boundary IoU/Chamfer metric (SAM3-as-pseudo-GT 변형이 RGB-edge 보다 정확). jumpingjacks 5 ckpt 비교: exp010 IoU=0.253 가 baseline 0.121 의 2.1×, Chamfer 6.4 vs 12.1 px 절반. softcal_best 가 가장 worse (over-cover). N3V 는 spatial filter sigmoid 가 binary 라 metric 추출 불가 (limitation). **[[results/exp025_multiclass|exp025]]** — multi-class K=3 instance (N=3 × M=2 scene): jumpingjacks [hoodie/shorts/shoes] + standup [vest/pants/shoes]. 6 SAM3 cache, per-class independent BCE, per-class hue recolor. v1 에서 hoodie 158/200 frame SAM3 fail 발견 → v2 에서 hoodie union prompt + spatial filter per class 적용 후 완벽 분리. union prompt 가 multi-instance 의 진짜 bottleneck. **[[results/exp026_4d_label_mapping|exp026]]** — 4D timeframe grid (K=10 bucket) N3V sear_steak: bucket 별 cloth_pct 5.9-7.2% 거의 동일 (negative). chef 가 outfit 안 바뀌니 시간 라벨 의미 없음. outfit-changing dataset 에선 유용할 수 있음. **viewer 사용법** [[008Viewer_사용법]] 정리. **[[results/exp027_soft_cal|exp027]]** — 9페이지 memory-based negative mining + soft-label calibration (D-NeRF only). 4 flavor (none/a/b/both, mem_k=1000, cal_strength=0.45) × 2 scene. jumpingjacks: distribution shift 정량 확인 (a flavor 가 high-conf 37% demote) but baseline 이미 깨끗해 visible 변화 없음. standup: middle-mass push 로 pants 추가 cover (positive) but arms (skin) 까지 over-cover (boundary discrimination 약함). 다음 미팅 발표용 [[009Meeting_Response_2026-05-15]] 정리.

---

## 2. 오늘 딱 한 가지 (2분)

**오늘 끝낼 것:** ___

고르는 법: [[003연구_대시보드]]를 열어서 맨 위 미완료 항목을 가져온다.
대시보드에 할 것이 없으면 → 가이드북 Part 2를 본다.

---

## 3. 90분 타이머 켜고 시작

하면 안 되는 것:
- 새 논문 읽기 (따로 시간 잡는다)
- 노트 정리 (그건 작업이 아니다)
- 시스템 개선 (지금 시스템으로 충분하다)

90분이 끝나면 → 내일 "어제 한 줄"에 쓸 내용이 있으면 성공이다.

---

## 금요일 추가: 주간 점검 (10분)

이번 주 기록 5줄을 읽는다. 한 가지만 답한다:

> **"이번 주에 논문 제출에 직접 기여한 것은?"**

답이 없으면 → [[003연구_대시보드]]의 우선순위를 재점검한다.

답을 여기에 한 줄로 적는다:

- W08 (02/17~21): 
- W09 (02/24~28): 
- W10 (03/03~07): 

---

## 규칙 (4개, 더 안 만든다)

1. **한 번에 한 가지.** 두 개 동시에 하면 둘 다 안 끝난다.
2. **90분은 협상 불가.** 컨디션 나빠도 앉는다.
3. **실패해도 적는다.** "실패함"이라고 쓰면 된다. 안 쓰면 반복한다.
4. **이 파일을 고치는 데 시간 쓰지 않는다.**
