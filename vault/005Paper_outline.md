---
title: Paper outline — ClothSplat
pinned: true
date: 2026-05-13
---

# ClothSplat: Text-Driven Garment Recoloring in 4D Gaussian Splatting

## Title 선정 근거 (2026-05-13)

7 후보 중 선택:
- **ClothSplat** = method name (짧고 기억 쉬움)
- **Text-Driven** = SAM3 text-prompt 사용 contribution 명시
- **Garment Recoloring** = 명확한 application
- **in 4D Gaussian Splatting** = 적용 도메인

[[001연구_가이드북]] Part 3 (CVPR papers 뼈대) 기반으로 골격 짜기.

## Abstract (5–6 문장 골격)

문장 별 채우기:

1. **이 분야는 ~가 중요하다.**
   - 4D Gaussian Splatting 으로 동적 장면 reconstruction 이 가능해졌으나, 학습된 4D 장면의 *부분 편집* (특히 의류 색 변경) 은 여전히 open.

2. **기존 방법들은 ~가 문제다.**
   - GaussianEditor 류는 매 편집마다 SDS optimization 필요 (분/시간), 시간·시점 일관성 별도 제약 필요.
   - 픽셀 단위 supervision 은 alpha-composit averaging 으로 옷 경계 over/under cover.

3. **우리는 ~를 제안한다.**
   - **ClothSplat**: 가우시안 단위 `cloth_logit` channel 추가 + **per-Gaussian projection supervision** (alpha-composit 우회) + HSV SH_DC swap.

4. **구체적으로 ~를 한다.**
   - Stage 1: vanilla 4DGS 학습, Stage 2: 각 가우시안을 모든 학습 frame 에 projection 해서 SAM3 mask 값 평균으로 soft target 계산, Stage 3: BCE 로 cloth_logit 만 fine-tune (geometry freeze).

5. **실험 결과 ~만큼 좋아졌다.**
   - D-NeRF 8 씬에서 SAM3 text prompt 만으로 옷 영역 분리 (정량: 가우시안의 cloth_pct 가 SAM3 GT coverage 와 일치).
   - 4D-DRESS 8 takes 에서 GT vertex label 로 검증 (upper bound).
   - 한 번 학습 후 *무한 recolor* (재학습 불필요).

6. **(선택) 코드 공개.**

## Introduction 단락 흐름

1. **배경** — dynamic 4D Gaussian Splatting 의 발전 (Wu et al. 2023 등) → editing 욕구.
2. **기존 방법 요약** — text-driven 3DGS editing (GaussianEditor, Edit-DyGS), per-pixel BCE supervision 등.
3. **한계** — alpha-composit averaging 문제 (per-pixel CE 에서 over/under cover), SDS 비용, 시간 일관성 어려움.
4. **우리 아이디어** — per-Gaussian 직접 supervision: projection 으로 가우시안마다 target 만들어 픽셀 supervision 우회.
5. **Contribution** (bullet):
    - **mask-agnostic per-Gaussian projection supervision** — alpha-composit averaging 우회한 깔끔한 cloth 분리
    - **text-driven zero-shot supervision via SAM3** — annotation 없이 작동
    - **3D-aware editing (SH_DC swap)** — 한 번 학습 후 시간/시점 일관 recolor 무한
    - **8 D-NeRF + 8 4D-DRESS** 정량/정성 실험으로 mask-agnostic 검증

## Related Work 카테고리

1. **4D Gaussian Splatting** — Wu et al. 2023, Yang et al. 2024, ...
2. **3D/4D scene editing** — GaussianEditor (Chen 2024), Edit-DyGS, NeRF-Editing
3. **Mask supervision in neural fields** — N3F, FFD, Decompose & Render
4. **Foundation models for segmentation** — SAM (Kirillov 2023), SAM 2 (2024), SAM 3 (2024)

각 카테고리 마지막에 "그런데 ~은 아직 안 했다. 우리가 한다."

## Method (4 subsections)

### 4.1 Pipeline overview (Figure 1 — 4-stage flow)

`Vanilla 4DGS → per-Gaussian projection → BCE fine-tune → HSV recolor`

### 4.2 per-Gaussian `cloth_logit` channel ([[notes/cloth_logit_channel]])

각 가우시안에 1-d 학습 가능 logit 추가. PLY 에 컬럼 1 개 더.

### 4.3 Per-Gaussian projection supervision ([[results/exp010_per_gaussian_projection]])

- formula: $t_i = \frac{1}{|\text{visible}_i|}\sum_{j \in \text{visible}_i} \text{mask}_j[\text{proj}(g_i, j)]$
- alpha-composit 우회 메커니즘 설명
- loss: $\mathcal{L} = -\frac{1}{N}\sum_i [t_i \log\sigma(\ell_i) + (1-t_i)\log(1-\sigma(\ell_i))]$
- backprop: $\partial\mathcal{L}/\partial\ell_i = (\sigma(\ell_i) - t_i)/N$
- Stage 1 freeze + Stage 3 만 cloth_logit 학습

### 4.4 HSV SH_DC recolor ([[notes/hsv_recolor_trick]])

- `cloth_mask = sigmoid(cloth_logit) > τ`
- cloth Gaussian 의 SH_DC 만 RGB → HSV → H swap → RGB → SH_DC
- SH_rest 유지 (광택 패턴 보존)
- 시간/시점 일관성 자동

## Experiments (5 subsections)

### 5.1 Setup
- 데이터셋: D-NeRF (8 scenes), 4D-DRESS (8 takes)
- baseline: 4DGaussians (Wu 2023) RGB only
- mask source: SAM3 text-prompt (D-NeRF), GT vertex label (4D-DRESS)
- 메트릭: cloth_pct vs target coverage, PSNR (RGB), 시각 정성 비교

### 5.2 정량 — cloth localization

표:
- D-NeRF: 8 씬 × (baseline cloth_pct=100% over-cover) vs (exp010 cloth_pct ≈ SAM3 GT)
- 4D-DRESS: 8 takes × cloth_pct vs vertex GT
- ablation: hard BCE / soft-cal / multi-class / **per-Gaussian projection**

### 5.3 정성 — recolor 시각 비교

- baseline (over-cover) vs ours (깔끔)
- side-by-side panel 16 frame × N scene
- video — 시간 일관성 demo

### 5.4 Ablation

- mask source: SAM3 single / multi-prompt union / GT vertex
- thresh τ : 0.2 / 0.5 / 0.7
- Stage 1 iter: 14k / 20k / 60k

### 5.5 Limitations
- 검정/회색 옷 → hue swap visual 약함 (S, V 보강 필요)
- 4D-DRESS 4-view sparse baseline RGB 품질 한계
- multi-class (K>1) 는 mask coverage 가 sparse 면 학습 어려움

## Conclusion (4 문장)

1. ClothSplat 제안 — per-Gaussian projection supervision 으로 4D-GS 의 cloth 분리.
2. D-NeRF + 4D-DRESS 에서 깔끔한 recolor 달성, mask source agnostic.
3. 한계 = 검정 옷 + sparse baseline.
4. 향후 = texture pattern edit, hue gradient, generative model 결합.

## Figure 후보

1. **Figure 1 — pipeline overview** (Method 앞)
2. **Figure 2 — alpha-composit averaging problem 설명** (Method 4.3)
3. **Figure 3 — D-NeRF 8 씬 정성 비교** (Experiments)
4. **Figure 4 — 4D-DRESS GT vertex label 검증** (Experiments)
5. **Figure 5 — Ablation: SAM3 prompt 별 cloth_pct** (Experiments)
6. **Figure 6 — Recolor 시간 일관성 demo** (Experiments)

## 작성 순서 (가이드북 Part 3 권장)

1. Experiments — 표 + figure 먼저
2. Method — 수식 + Figure 1, 2
3. Introduction — contribution bullet
4. Related Work
5. Abstract
6. Conclusion

## 다음 action

- [ ] D-NeRF 8 씬 + 4D-DRESS 8 takes 정량 표 완성 (exp010 + exp013 결과 합쳐)
- [ ] Figure 2 (alpha-composit averaging) 설명 그림 그리기 (vault/images/)
- [ ] Figure 3 16-frame panel 정제 (vis/SUMMARY_exp010 활용)
- [ ] Method 4.3 의 수식 LaTeX 으로 정리
- [ ] Related work paper 5-8 개 후보 정리 ([[../docs/REPORT.md]] 참고)
