---
title: SAM 3 mask supervision
layer: 2
date: 2026-05-11
---

# SAM 3 mask supervision

> 옷 segmentation 을 위한 GT 신호로 SAM 3 (Sam3Model, image-mode) 출력을 사용. layer 2.

## 왜 SAM 3?

옷 라벨은 사람이 일일이 그리기 비현실적 (D-NeRF 100~150 frame × 4 view 등). SAM 3 가 *zero-shot text-prompt* 로 segmentation 가능 → 자동화.

## 동작

```python
processor = Sam3Processor.from_pretrained("facebook/sam3")
model = Sam3Model.from_pretrained("facebook/sam3", torch_dtype=torch.bfloat16)
inp = processor(images=img, text="shirt", return_tensors="pt")
out = model(**inp)
mask = post_process(out, threshold=0.3)  # (H, W) bool
```

D-NeRF 는 비디오가 아니라 독립 frame 들 → image-mode (Sam3VideoModel 안 씀).

## 캐싱

매 학습 step 마다 SAM3 forward 는 비효율 → 한 번에 모든 frame 처리해서 `masks.npz` 로 저장:

```bash
python sam3_cache.py --root data/dnerf/data --scenes standup \
  --prompts "shirt" --out cache/sam3_dnerf
```

→ `cache/sam3_dnerf/standup/masks.npz` (uint8 packed bits, T×H×W)

## 학습 시 사용

```python
# train.py: per-pixel BCE between rendered cloth_logit_image vs cached mask
cl_pred = render(viewpoint, gaussians, ...)["cloth_logit"]  # gsplat 2nd-pass
target = mask_cache[viewpoint.uid]  # (H, W) bool
loss_cloth = BCE_with_logits(cl_pred, target.float())
total_loss = L_rgb + λ_cloth * loss_cloth
```

기본값: `λ_cloth=0.1`, `cloth_warmup=3000` (RGB 가 어느 정도 학습된 뒤 supervision 시작 — NaN 방지).

## prompt 의 중요성

prompt 정확도 = mask 정확도. D-NeRF 씬별 prompt:
- `jumpingjacks`: "shirt" → coverage **1.93%** (캐릭터 작아서 SAM3 가 hood 만 잡음)
- `standup`: "shirt" → coverage **~50%** (vest+셔츠+바지+부츠 모두)
- `hellwarrior`: "armor" → coverage **~60%** (몸 전체 over-cover)
- `mutant` v1 "suit" → **0%** ❌ → v2 "monster body and armor" → 9.9% ✓

prompt 가 모호하면 over-cover (얼굴까지 옷으로 분류) 또는 under-cover (옷의 일부만). [[soft_label_calibration]] 으로 mask 모호성 완화.

## 깰 수 있는 부분

- SAM 3 hard mask 만 사용 — confidence map (`mask_threshold` 이전 logit) 을 soft target 으로 직접 쓰면 calibration 일부 대체 가능
- 단일 prompt — 여러 prompt 결과 union/intersect 으로 robust 화 가능
- 한 번 캐싱 후 고정 — online refinement (학습된 cloth_logit 을 SAM3 prompt 에 feedback) 가능
