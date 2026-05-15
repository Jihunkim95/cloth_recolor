---
title: exp007 — Multi-class cloth_logit (K=2/3) on 8 D-NeRF scenes
status: done
date: 2026-05-11
code: 4DGaussians/train.py --num-classes K, --sam3-mc-caches; soft_cal disabled
---

# exp007

## 질문

[[exp006_jumpingjacks_8way_grid]] 에서 단일 prompt 의 hard-BCE / soft-cal 이 *cloth_pct 100% collapse* 를 보임. K-class **softmax + cross-entropy** 로 바꾸면 collapse 해소되는가? cloth class 가 hoodie 같은 명확한 옷 영역에 attribute 화 되는가?

## 설정

- 가우시안에 `_cloth_logit (N, 1)` → `_cloth_logit (N, K)` 확장 ([[../notes/cloth_logit_channel]] 갱신 예정)
- K 클래스: 0=cloth, (1=skin if applicable), K-1=other (implicit, supervision 없음)
- supervision: gsplat 2nd-pass 가 K-channel 출력 → pixel 별 argmax(SAM3 masks) → cross-entropy
- K=3 (jumpingjacks, standup — skin prompt 가능): cloth + skin + other
- K=2 (hellwarrior, bouncingballs, hook, lego, mutant, trex): cloth + other
- 학습: 14000 iter, λ_cloth=0.1, cloth_warmup=3000, soft-cal 비활성
- 8 GPU 병렬, 학습 ~5-7분

### SAM3 prompt 별 coverage

| scene | cloth prompt | cov | skin prompt | cov |
|---|---|---|---|---|
| jumpingjacks | "orange hoodie OR shirt" | 3.88% | "person face arms legs" | 0.02% |
| standup | "safety vest shirt pants" | 3.94% | "face hands hardhat" | 0.84% |
| hellwarrior | "armor" | 7.63% | — | — |
| bouncingballs | "ball" | 6.55% | — | — |
| hook | "puppet body" | 11.39% | — | — |
| lego | "lego figure body" | 5.7% | — | — |
| mutant | "monster body and armor" | 9.86% | — | — |
| trex | "dinosaur" | 4.26% | — | — |

skin coverage 가 모두 < 1% → K=3 인 두 씬도 사실상 cloth + other 동작.

## 예측

- collapse 해소: cloth_pct (target=0) 이 100% → SAM3 GT 수준으로 강하게 감소
- target_class=0 recolor 시 옷만 깔끔하게 색 변경
- RGB PSNR 은 ±0.5 dB 안에서 유지

## 결과

### cloth_pct (target_class=0 @ thresh=0.3)

| scene | K | K=1 (hard-BCE) | K=multi target=0 | Δ |
|---|---|---|---|---|
| jumpingjacks | 3 | 100.0% | **17.88%** | -82 %p |
| standup | 3 | 100.0% | 24.37% | -76 %p |
| hellwarrior | 2 | 100.0% | 50.51% | -49 %p |
| bouncingballs | 2 | 100.0% | 42.18% | -58 %p |
| hook | 2 | 100.0% | 42.63% | -57 %p |
| lego | 2 | 100.0% | 38.28% | -62 %p |
| mutant | 2 | 100.0% | 37.96% | -62 %p |
| trex | 2 | 100.0% | 49.89% | -50 %p |

✅ **collapse 100% → 18-50% 로 강하게 감소**

### 시각 (jumpingjacks K=3, target_class=2)

뜻밖에도 **target_class=0 으로 recolor 하면 hoodie 가 거의 안 잡힘**.
대신 **target_class=2 (other)** 가 hoodie 의 일부를 잡고 partial blue recolor 됨.

이유 분석:
- SAM3 cloth coverage 3.88% (매우 좁음) → 학습 시 CE target 분포:
  - hoodie 영역 중 SAM3-marked (3.88%) 픽셀 → class 0
  - hoodie 영역 중 unmarked (96%) 픽셀 → class 2 (other)
  - 배경 픽셀 → class 2
- 결과: class 0 = "SAM3 가 명시적으로 잡은 미세한 영역", class 2 = "그 외 모든 visible 영역 (hoodie 의 대부분 + 배경)"
- → multi-class 자체는 작동하나 **SAM3 coverage 가 너무 낮아 의도된 의미 분리가 안 됨**

## 예측 맞았나?

- collapse 해소: ✅ 100% → 18-50%
- target=0 깔끔한 recolor: ❌ target=0 은 너무 좁음, target=2 가 의도하지 않게 hoodie 의 일부 captured
- RGB PSNR 유지: ⚠️ 일부 씬 변동 큼 (parsing 정확도 한계, but train PSNR 기준 모두 학습 안정)

## 결론

**Multi-class framework 는 collapse 해소에 성공**.
하지만 **class 의미가 SAM3 prompt coverage 에 결정적으로 의존** — coverage < 10% 시 cloth class 가 mask 의 좁은 영역만 잡고, "other" 가 의도치 않게 옷의 대부분 흡수.

## 다음

1. **Multi-prompt union for cloth class** — SAM3 cloth mask 를 여러 prompt 결과 OR 로 확장. 예: jumpingjacks cloth = `"orange hoodie"` ∪ `"sweater"` ∪ `"sweatshirt"` ∪ `"jacket"`.
   목표: SAM3 cloth coverage 3.88% → 30%+
2. **CE class weighting** — 희소한 class 0 의 loss 에 가중치 (예: 10×) → 모델이 class 0 우선 학습
3. **Point-prompt SAM3** — text 대신 model 의 cloth_logit > 0.5 pixel 을 SAM3 point prompt 로 → mask 의 hoodie 외곽 정확성 향상
4. K=3 의 잠재력은 *cloth coverage 가 충분히 클 때* 만 발휘됨 — 위 (1) 우선
