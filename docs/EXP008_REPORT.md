# exp008 — multi-prompt union + class weighting + longer iter

Generated: 2026-05-11T23:10:00.800279

## SAM3 union coverage per scene

| scene | union coverage | prompts |
|---|---|---|
| jumpingjacks | 3.9% | orange hoodie, sweater, jacket, hoodie |
| standup | 6.55% | safety vest, shirt, pants, construction worker outfit |
| hellwarrior | 11.09% | armor, spiked armor, iron armor, full body armor |
| bouncingballs | 6.55% | red ball, blue ball, green ball, ball |
| hook | 12.2% | puppet body, wooden puppet, clothing, doll body |
| lego | 12.85% | lego figure, lego body, minifigure body, plastic figure |
| mutant | 10.18% | monster armor, spiked armor, monster body and armor |
| trex | 4.31% | dinosaur, tyrannosaurus, t-rex body, reptile |

## Ablation table (cloth_pct @ thresh 0.2, target_class=0)

| scene | config | iter | cloth_pct |
|---|---|---|---|
| - | u_k2_30k_bouncingballs_t02_c0_it14000 | 14000 | 42.58% |
| - | u_k2_30k_hellwarrior_t02_c0_it14000 | 14000 | 48.88% |
| - | u_k2_30k_hook_t02_c0_it14000 | 14000 | 44.48% |
| - | u_k2_30k_jumpingjacks_t02_c0_it14000 | 14000 | 61.94% |
| - | u_k2_30k_lego_t02_c0_it14000 | 14000 | 39.12% |
| - | u_k2_30k_mutant_t02_c0_it14000 | 14000 | 38.97% |
| - | u_k2_30k_standup_t02_c0_it14000 | 14000 | 39.77% |
| - | u_k2_30k_trex_t02_c0_it14000 | 14000 | 43.79% |
| - | u_k2_bouncingballs_t02_c0_it14000 | 14000 | 43.72% |
| - | u_k2_hellwarrior_t02_c0_it14000 | 14000 | 39.19% |
| - | u_k2_hook_t02_c0_it14000 | 14000 | 43.92% |
| - | u_k2_jumpingjacks_t02_c0_it14000 | 14000 | 59.12% |
| - | u_k2_lego_t02_c0_it14000 | 14000 | 37.31% |
| - | u_k2_mutant_t02_c0_it14000 | 14000 | 38.31% |
| - | u_k2_standup_t02_c0_it14000 | 14000 | 49.99% |
| - | u_k2_trex_t02_c0_it14000 | 14000 | 44.58% |
| - | u_k2_w10_1_bouncingballs_t02_c0_it14000 | 14000 | 45.82% |
| - | u_k2_w10_1_hellwarrior_t02_c0_it14000 | 14000 | 34.76% |
| - | u_k2_w10_1_hook_t02_c0_it14000 | 14000 | 49.92% |
| - | u_k2_w10_1_jumpingjacks_t02_c0_it14000 | 14000 | 61.12% |
| - | u_k2_w10_1_lego_t02_c0_it14000 | 14000 | 42.93% |
| - | u_k2_w10_1_mutant_t02_c0_it14000 | 14000 | 41.01% |
| - | u_k2_w10_1_standup_t02_c0_it14000 | 14000 | 46.55% |
| - | u_k2_w10_1_trex_t02_c0_it14000 | 14000 | 47.46% |
| - | u_k2_w3_1_bouncingballs_t02_c0_it14000 | 14000 | 39.65% |
| - | u_k2_w3_1_hellwarrior_t02_c0_it14000 | 14000 | 33.52% |
| - | u_k2_w3_1_hook_t02_c0_it14000 | 14000 | 43.1% |
| - | u_k2_w3_1_jumpingjacks_t02_c0_it14000 | 14000 | 60.36% |
| - | u_k2_w3_1_lego_t02_c0_it14000 | 14000 | 41.16% |
| - | u_k2_w3_1_mutant_t02_c0_it14000 | 14000 | 35.07% |
| - | u_k2_w3_1_standup_t02_c0_it14000 | 14000 | 38.65% |
| - | u_k2_w3_1_trex_t02_c0_it14000 | 14000 | 43.47% |
| - | u_k3_jumpingjacks_t02_c0_it14000 | 14000 | 16.55% |
| - | u_k3_standup_t02_c0_it14000 | 14000 | 18.67% |
