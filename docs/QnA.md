# Q&A — viewer 데모 4문답

## 1. http://localhost:18080 에 보이는 object 는 어떤 dataset?

**D-NeRF `standup` scene** — 안전모 + 형광 vest + 회색 셔츠/바지 + 부츠를 입은 합성 작업자
캐릭터 (Blender 800×800, 시간축 0–1).

- 위치: `1_input/dnerf/standup/`
- 학습 카메라: 150 프레임 (각 프레임마다 다른 pose + time → monocular video 와 동치)
- Wu et al. (2023) "4D Gaussian Splatting for Real-Time Dynamic Scene Rendering" 의 공식 D-NeRF
  검증 셋 8 개 중 하나 (jumpingjacks · standup · hellwarrior · mutant · trex · lego · bouncingballs · hook).

---

## 2. 그 dataset 의 input 값은 무엇으로 주었나?

학습 시 **2 가지 입력**을 사용:

| 입력 | 무엇 | 어디서 |
|---|---|---|
| (1) RGB + pose | `transforms_train.json` (frame 별 R, T, FoV, time) + 800×800 PNG | `1_input/dnerf/standup/` |
| (2) cloth supervision mask | SAM 3 가 prompt `"shirt"` 로 만든 per-frame 2D bool mask | `cache/sam3_dnerf/standup/masks.npz` |

(1) 은 4DGaussians 가 RGB 재구성·deformation 학습용으로,
(2) 는 우리가 추가한 per-Gaussian `cloth_logit` 학습 (BCE + soft-cal) 용으로 들어갑니다.

학습 1줄 (참고):
```bash
./2_process/train.sh standup both
```

---

## 3. 기존 4D-GS 연구 / 4DGS editing 연구 vs 우리 연구의 차별점

| | 기존 4D-GS (Wu et al., 2023 등) | 4DGS editing (GaussianEditor, Edit-DyGS 등) | **우리** |
|---|---|---|---|
| 가우시안 의미 정보 | RGB·shape 만 | 없음 (텍스트 prompt 로 매번 추론) | **per-Gaussian `cloth_logit` channel** (학습된 라벨) |
| 옷 분리 | 없음 | 2D inpainting/SDS optimization 으로 매 편집마다 재학습 | **학습 1번 → 편집 무한** (SH HSV-swap 만) |
| supervision | RGB L1 + ssim | 텍스트-CLIP / SDS | **SAM3 mask BCE + memory-based soft-label calibration** |
| 라벨 모호성 처리 | n/a | 없음 (hard) | **soft-cal**: easy 샘플 메모리 → distance → α(d)=0.5σ((d−d_ref)/τ), C±α 로 target 보정 |
| 편집 비용 | 편집 자체가 없음 | 분당-시간 단위 GPU optimization | **즉시** (GPU 1 회 forward) |
| 일관성 보장 | n/a | 시간 일관성 별도 제약 필요 | **가우시안 단위라 자동 일관성** (시간·시점) |

핵심: "**라벨이 가우시안에 박혀 있으니, 편집은 그냥 색 바꾸기**" — 4D 편집을
*최적화 문제* 가 아닌 *직접 조작* 으로 변환.

---

## 4. 어느 과정에서 옷 변경 트릭을 넣었나?

학습 시점 3 군데 + 편집 시점 1 군데, 총 4 군데:

```
[학습]
(a) 4DGaussians/scene/gaussian_model.py
    └─ _cloth_logit = nn.Parameter(N, 1) 추가, optimizer group 등록
(b) 4DGaussians/gaussian_renderer/__init__.py
    └─ 메인 RGB 패스(diff_gaussian_rasterization) 직후
       gsplat 2nd-pass 로 cloth_logit 채널만 1ch 으로 렌더 (BCE loss 용)
(c) 4DGaussians/train.py + utils/soft_cal.py
    └─ rendered cloth_logit_image vs SAM3 mask 의 per-pixel BCE
       + per-Gaussian soft-cal aux loss (memory bank / distance / α 계산)

[편집]
(d) 2_process/recolor.py
    └─ 학습된 PLY 로드 → sigmoid(cloth_logit) > thresh 인 가우시안만 골라서
       _features_dc (SH degree 0 의 RGB 항) 를 RGB→HSV→hue swap→RGB 로 변환
       → 다시 렌더 (시간 t 마다 deformation MLP 거쳐 자세 유지된 채 색만 변경)
```

요약: **(a)–(c) 에서 옷이 어디인지 가우시안 단위로 가르치고, (d) 에서
"옷이라고 학습된 가우시안의 색만 바꾸는" 1-line trick** 을 적용.
