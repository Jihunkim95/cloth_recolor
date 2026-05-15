# Soft-cal overnight final report

Generated: 2026-05-10T21:53:46.192977

## Phase A/B: variant comparison (DINOv3 fixed)

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

## Phase C: λ_cal × mem_size sweep

| scene | λ_cal | mem_size | cloth_pct | PSNR |
|---|---|---|---|---|
| hellwarrior (both) | 0.05 | 2000 | 48.73% | 28.93385629092946 |
| jumpingjacks (b) | 0.05 | 2000 | 0.02% | 35.039320440853345 |
| standup (a) | 0.05 | 2000 | 22.50% | 36.71124738805434 |
| hellwarrior (both) | 0.05 | 500 | 53.46% | 29.04939404655905 |
| jumpingjacks (b) | 0.05 | 500 | 0.01% | 35.05579320122214 |
| standup (a) | 0.05 | 500 | 24.52% | 36.731161678538605 |
| hellwarrior (both) | 0.2 | 2000 | 44.70% | 28.82679120232077 |
| jumpingjacks (b) | 0.2 | 2000 | 0.05% | 35.08704297682818 |
| standup (a) | 0.2 | 2000 | 23.80% | 36.97299497267779 |
| hellwarrior (both) | 0.2 | 500 | 54.30% | 28.876124999102423 |
| jumpingjacks (b) | 0.2 | 500 | 0.00% | 35.049596674302045 |
| standup (a) | 0.2 | 500 | 24.64% | 36.9513806735768 |

## Phase D: mutant (re-cached SAM3)

```
1 scenes × 1 workers
[cuda:0] mutant: 150 frames, prompt='monster body and armor'
[cuda:0] loading SAM 3...

Loading weights:   0%|          | 0/1468 [00:00<?, ?it/s]
Loading weights:  77%|███████▋  | 1124/1468 [00:00<00:00, 11208.24it/s]
Loading weights: 100%|██████████| 1468/1468 [00:00<00:00, 11287.16it/s]
[cuda:0] mutant: 50/150 (7.0 fps)
[cuda:0] mutant: 100/150 (9.6 fps)
[cuda:0] mutant done: 150 frames in 13.9s, avg coverage 9.9% 

```

- softcalD_v2_mutant_a: PSNR=36.892400180592254
- softcalD_v2_mutant_b: PSNR=36.649142770206225
- softcalD_v2_mutant_both: PSNR=36.71793657190659

## Recolor visuals

- Phase A/B: `vis/recolor_softcalv2/<scene>_<variant>/`
- Phase C/D: `vis/recolor_softcal_overnight/<expname>/`
