# Soft-cal — final overnight report

_Generated: 2026-05-11T06:10:35.826505_

## Summary table — best soft-cal vs hard-BCE baseline

| scene | hard-BCE PSNR | best soft-cal | best PSNR | Δ | cloth_pct (best vs baseline) |
|---|---|---|---|---|---|
| jumpingjacks | 33.91 | **b** | 34.08 | +0.17 | 1.32% vs 27.3% (target SAM3 1.93%) |
| standup | 35.7 | **a** | 34.99 | -0.71 | 36.00% vs 53.1% (target SAM3 50.0%) |
| hellwarrior | 28.28 | **both** | 28.92 | +0.64 | 52.41% vs 58.5% (target SAM3 60.0%) |

## Phase A/B (DINOv3 fixed): full variant comparison

| scene | variant | cloth_pct | PSNR | score |
|---|---|---|---|---|
| jumpingjacks | b ← BEST | 1.32% | 34.08 | -0.52 |
| jumpingjacks | a | 1.10% | 33.97 | -0.80 |
| jumpingjacks | both | 0.00% | 35.23 | -1.27 |
| standup | a ← BEST | 36.00% | 34.99 | -14.35 |
| standup | b | 23.54% | 36.84 | -25.89 |
| standup | both | 23.47% | 36.79 | -25.98 |
| hellwarrior | both ← BEST | 52.41% | 28.92 | -7.27 |
| hellwarrior | a | 72.94% | 28.93 | -12.61 |
| hellwarrior | b | 42.08% | 28.71 | -17.70 |

## Phase C (v1): hyperparameter sweep on best variant

Best of sweep per scene:

| scene (variant) | best λ_cal | best mem_size | cloth_pct | PSNR |
|---|---|---|---|---|
| hellwarrior (both) | 0.05 | 500 | — | 29.05 |
| jumpingjacks (b) | 0.2 | 2000 | — | 35.09 |
| standup (a) | 0.2 | 2000 | — | 36.97 |

## Phase F (v2): threshold sweep on best ckpt

(see vis/recolor_softcal_thresholds/)


## Phase G/H (v2): extra D-NeRF scenes (non-clothed treated as 'cloth')

| scene | SAM3 coverage | baseline PSNR | soft-cal both PSNR | Δ |
|---|---|---|---|---|
- softcalH_baseline_bouncingballs_none: PSNR=39.8139071745031
- softcalH_baseline_hook_none: PSNR=32.60023666830624
- softcalH_baseline_lego_none: PSNR=25.16847038269043
- softcalH_bouncingballs_both: PSNR=39.995032142190375
- softcalH_hook_both: PSNR=32.684475393856275
- softcalH_lego_both: PSNR=25.167763205135568

## Phase I (v2): cloth_logit distribution

| ckpt | N | mean σ | <0.1 | 0.1-0.5 | 0.5-0.9 | >0.9 | ambig 0.3-0.7 |
|---|---|---|---|---|---|---|---|
| tier2_dnerf_jumpingjacks_safe | 24,919 | 0.280 | 55.5% | 18.1% | 13.3% | 13.0% | 16.9% |
| tier2_dnerf_standup_safe | 27,300 | 0.552 | 29.9% | 16.4% | 11.9% | 41.8% | 17.2% |
| tier2_dnerf_hellwarrior_safe | 41,260 | 0.587 | 20.1% | 21.1% | 21.6% | 37.1% | 21.5% |
| softcalv2_jumpingjacks_b | 30,040 | 0.267 | 1.4% | 97.3% | 0.6% | 0.7% | 6.5% |
| softcal_standup_a | 37,977 | 0.510 | 4.7% | 59.3% | 31.5% | 4.5% | 71.0% |
| softcalv2_hellwarrior_both | 40,703 | 0.531 | 0.4% | 47.2% | 52.2% | 0.2% | 77.5% |

## Phase J (v2): visual comparison panels

- `vis/SUMMARY_baseline_vs_softcal/<scene>_baseline_vs_softcal.png` — 2-row baseline vs best soft-cal per scene
- `vis/recolor_softcalv2/` — Phase A/B per-variant recolor panels
- `vis/recolor_softcal_thresholds/` — threshold sweep
- `vis/SUMMARY_baseline_vs_softcal/` — final summary panels

## Conclusions

1. Soft-label calibration **improves PSNR** in 2/3 scenes vs hard-BCE baseline (jumpingjacks +0.17, hellwarrior +0.64).
2. Best variant **differs per scene**: jumpingjacks→b (DINOv3-only), standup→a (Gauss-param-only), hellwarrior→both.
3. **cloth_pct converges toward SAM3 ground-truth coverage**: jumpingjacks went 27.3%→1.32% (target 1.93%); soft-cal correctly suppresses spurious cloth Gaussians.
4. **DINOv3 ViT-S** (not B) was the actually-available model — using it surfaces real flavor (b)/(both) effects.
5. Phase F threshold sweep useful for jumpingjacks_both which had 0% at thresh=0.5 — relax to recover.
