---
title: Backup manifest — D-NeRF essentials only
date: 2026-05-15
tags: [backup, manifest]
---

# Backup Manifest — `backup_dnerf_essentials.zip`

> 정리: 사용 안 하는 4D-DRESS / Neural3DVideo / archive / log 모두 제외하고, **D-NeRF 데이터셋 + 모든 실험 결과 + 일지/코드** 만 백업.

## 위치

`/NHNHOME/WORKSPACE/0526040060_B/research/backup_dnerf_essentials.zip`

**생성 완료** (2026-05-15 16:21, **재빌드 22:02 — exp028 + lego retune 포함**):
- **4.02 GB** compressed (4.68 GB raw)
- **10,105 files** total (+579 from v1)
  - `cloth_recolor/`: 6,917 files
  - `4DGaussians/`: 3,186 files
  - root: `CLAUDE.md`, `README.md`

**v2 추가분** (vs v1):
- exp028 (8 D-NeRF scene × ckpt_exp028_best + baseline_bce)
- exp028 phase 1 (jumpingjacks 8 config sweep)
- lego prompt retune (4 variants × ckpt + recolor)
- 3 new SAM3 lego caches (`sam3_lego_v2_{vehicle, vehicle_specific, toy}`)
- vault note `exp028_loss_search.md`

전체 파일 리스트: [[010_BACKUP_FILE_LIST]] (`010_BACKUP_FILE_LIST.txt`)

## 포함 항목 (whitelist, 71 paths)

### Top-level

- `CLAUDE.md` — project guidelines
- `README.md` — top docs

### `4DGaussians/` (수정된 4DGS 코드)

| 경로 | 내용 |
|---|---|
| `arguments/` | scene 별 학습 hyperparam (longer.py 등) |
| `scene/` | GaussianModel (cloth_logit 채널 추가본), cameras, dataset_readers |
| `utils/` | soft_cal.py (DINOv3+memory), graphics_utils, sh_utils 등 |
| `gaussian_renderer/` | gsplat cloth_logit 2nd-pass 추가본 |
| `submodules/` | depth-diff rasterizer 등 |
| `data/dnerf/` | **D-NeRF Blender scene 8개** (505M) |
| `train.py`, `render.py`, `colmap.sh`, `convert.py`, `metrics.py`, `longer.py`, `assets/` | 학습/렌더링 파이프라인 |

### `cloth_recolor/` (실험 프로젝트)

| 경로 | 내용 |
|---|---|
| `CLAUDE.md`, `README.md`, `.gitignore`, `.git/` | git history 포함 |
| `2_process/` | 모든 학습/recolor/cache 스크립트 (per_gaussian_supervision*, exp02X*, recolor.py, sam3_cache*, edge_metric.py 등) |
| `4_viewer/` | viser viewer |
| `docs/` | 보고서 (REPORT, QnA 등) |
| `vault/` | **Obsidian 일지 + 실험 노트 (exp001~027)** |
| `vis/` | SUMMARY panel + SUMMARY_meeting/ |
| `cache/sam3_*` | SAM3 mask cache 36개 (D-NeRF 관련만, 4D-DRESS/N3V 제외) |
| `3_output/{8 D-NeRF scenes + exp022/024 dirs}` | 학습된 ckpt + recolor 결과 |

### `3_output/` 포함 directory (11개)

- D-NeRF 8 scenes: `bouncingballs`, `hellwarrior`, `hook`, `jumpingjacks`, `lego`, `mutant`, `standup`, `trex`
- 실험 sweep: `exp022_thresh_sweep` (threshold 6단계 × 2 scene), `exp024_edge`, `exp024_edge_sam3gt`

### `cache/` 포함 (39개 dir, +3 from v1)

- `sam3_lego_v2_vehicle/`, `sam3_lego_v2_vehicle_specific/`, `sam3_lego_v2_toy/` — exp028 lego retune 결과 (vehicle_specific 이 winner)
- (이하 v1 동일)


- `sam3_dnerf/` — single prompt baseline 8 scenes
- `sam3_union_<scene>/` × 8 — multi-prompt union
- `sam3_mc_cloth_*` × 9 — multi-class cloth
- `sam3_mc_skin_*` × 3 — skin negative
- `sam3_exp025_{jumpingjacks,standup}_<class>` × 7 — exp025 multi-instance
- `sam3_jj_*` × 4 — jumpingjacks 변형 prompts
- 기타 ablation cache 4개

## 제외 항목 (백업 안 함)

| 항목 | 크기 | 제외 이유 |
|---|---|---|
| `4D-DRESS/` (raw tar) | 879 GB | 사용 안 함 |
| `4D-DRESS_extracted/` | 1.1 TB | 사용 안 함 |
| `Neural3DVideo/` | 179 GB | 사용 안 함 |
| `4DGaussians/data/hypernerf` | 1.7 GB | 우리 실험 미사용 |
| `4DGaussians/data/multipleview` | 1.1 GB | 4D-DRESS scene 용 |
| `cloth_recolor/3_output/00*_*/` × 8 | 3.4 GB | 4D-DRESS ckpts |
| `cloth_recolor/3_output/n3v_*/` × 6 | 2.7 GB | N3V ckpts |
| `cloth_recolor/cache/sam3_n3v*` | 281 MB | N3V SAM3 |
| `cloth_recolor/cache/sam3_4ddress` | 9.4 MB | 4D-DRESS SAM3 |
| `cloth_recolor/archive/` | 125 GB | 옛 exp001-009 잔재 (vault 노트로 정리됨) |
| `cloth_recolor/overnight_logs/` | 32 MB | 옛 log (vault 노트로 대체) |
| root `download.log` | 1.6 GB | 텍스트 log |
| root `*.zip`, `*.html`, `urls.txt` | < 10 MB | 일회용 metadata |
| `SpacetimeGaussians/` | 148 MB | 별도 백업 결정 |

→ 총 **~2.3 TB → 4 GB** (99.8% 회수)

## 복원 방법

```bash
unzip backup_dnerf_essentials.zip -d <restore_dir>
cd <restore_dir>
# D-NeRF dataset 그대로 사용 가능
# 4DGaussians 학습:
cd 4DGaussians
python train.py --config arguments/dnerf/jumpingjacks.py
# cloth_logit Stage 3:
cd ../cloth_recolor
python 2_process/per_gaussian_supervision.py \
  --base-ckpt 3_output/jumpingjacks/ckpt_baseline_hardBCE --base-iter 20000 \
  --sam3-cache cache/sam3_union_jumpingjacks --scene jumpingjacks \
  --out 3_output/jumpingjacks/ckpt_restored
```

## 관련 노트

- [[008Viewer_사용법]] — 백업 후 결과 시각화
- [[009Meeting_Response_2026-05-15]] — 미팅 자료 (포함됨)
- [[007Experiment_Plan_2026-05-15]] — 실험 sprint
