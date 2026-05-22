---
title: 4_viewer/viewer.py 사용법
date: 2026-05-15
pinned: true
tags: [tool, viewer, viser]
---

# 4_viewer/viewer.py 사용법

viser 기반 interactive 3D viewer. 학습된 ckpt 의 cloth_logit 시각화 + HSV recolor 실시간 미리보기.

## 기본 실행

```bash
cd /NHNHOME/WORKSPACE/0526040060_B/research/cloth_recolor
source /home/bjh0309/miniconda3/etc/profile.d/conda.sh && conda activate zcar

# gsplat 의 cloth_logit 2nd-pass 가 동작하려면 CUDA 환경변수 필수
export CUDA_HOME=/home/bjh0309/miniconda3/envs/zcar
export CPATH=/home/bjh0309/miniconda3/envs/zcar/targets/x86_64-linux/include:$CPATH
export LD_LIBRARY_PATH=/home/bjh0309/miniconda3/envs/zcar/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

nohup env CUDA_VISIBLE_DEVICES=0 python 4_viewer/viewer.py \
  --ckpt-dir <CKPT_PATH> \
  --iter <ITER> \
  --port 8080 \
  > /tmp/viewer.log 2>&1 &

# 확인
sleep 10 && tail -5 /tmp/viewer.log    # "viewer at http://0.0.0.0:8080/" 출력 기대
ss -tln | grep 8080
```

브라우저: `http://localhost:8080` (SSH tunnel `-L 8080:localhost:8080` 가정)

## 데이터셋별 `--iter` 값

| 데이터셋 | iter | 비고 |
|---|---|---|
| D-NeRF (jumpingjacks, standup, …) | **20000** | exp010, exp025 모두 |
| 4D-DRESS | **14000** | exp013 |
| N3V (sear_steak, cook_spinach, …) | **14000** | exp014/015/026 |

데이터셋 분기는 viewer.py 가 ckpt 의 `source_path` 기반으로 *자동 감지*:
- `transforms_train.json` 존재 → **D-NeRF** Blender layout
- `cam00/images/` 존재 → **N3V** Neural3D_NDC_Dataset
- 그 외 → **4D-DRESS** multipleview

## 추천 ckpt 예시

### (1) D-NeRF main result — single-class per-Gaussian projection ([[results/exp010_per_gaussian_projection|exp010]])

```bash
--ckpt-dir 3_output/jumpingjacks/ckpt_exp010_per_gaussian --iter 20000
```

→ hoodie 하나만 학습됨. threshold slider 0.7 권장.

### (2) D-NeRF multi-instance ([[results/exp025_multiclass|exp025]])

```bash
--ckpt-dir 3_output/jumpingjacks/ckpt_exp025_union_sf --iter 20000   # v2 (권장)
```

→ K=3 (hoodie / shorts / shoes). **target class slider 0=hoodie, 1=shorts, 2=shoes** 로 전환. hue 도 같이 바꿔서 (hoodie=red 0°, shorts=green 120°, shoes=blue 220°) 시연 가능.

```bash
--ckpt-dir 3_output/standup/ckpt_exp025_multi --iter 20000
# K=3: vest / pants / shoes
```

### (3) N3V — spatial filter ([[results/exp015_n3v_train_time_spatial_filter|exp015]])

```bash
--ckpt-dir 3_output/n3v_sear_steak/ckpt_exp015_spatial --iter 14000
```

→ chef shirt 학습. threshold slider **0.3** 권장 (spatial filter 로 logit 분포 압축됨, [[results/exp022_threshold_sweep|exp022]] 참고).

### (4) N3V — 4D timeframe grid ([[results/exp026_4d_label_mapping|exp026]])

```bash
--ckpt-dir 3_output/n3v_sear_steak/ckpt_exp026_4d --iter 14000
```

→ K=10 (time bucket). target class slider 0-9 가 *시간 bucket* (frame 0-30, 30-60, …). chef 가 옷 안 바뀌면 bucket 별 차이 거의 없음 ([[results/exp026_4d_label_mapping|exp026]] 결론).

### (5) 4D-DRESS GT vertex ([[results/exp013_4ddress_retrain|exp013]])

```bash
--ckpt-dir "3_output/00190_Outer_Take10/ckpt_exp013_gt" --iter 14000
```

→ jacket 학습. SAM3 아닌 GT vertex 사용.

## GUI 컨트롤

| 컨트롤 | 설명 |
|---|---|
| **frame slider** | 시간 (0 ~ n_frames−1) — N3V 300, D-NeRF 200, 4D-DRESS scene dependent |
| **mode dropdown** | `original` / `recolored` / `mask overlay` |
| **target hue (deg)** | 0=red, 60=yellow, 120=green, 180=cyan, 220=blue, 300=magenta |
| **cloth threshold** | sigmoid(cloth_logit) 임계값. D-NeRF=0.7, N3V=0.3 권장 |
| **target class** (K>1 만 보임) | exp025 의 0/1/2 (hoodie/shorts/shoes) 또는 exp026 의 0-9 (bucket) |
| **camera preset buttons** | 학습 카메라 시점으로 즉시 점프 |
| viewport 드래그 | free 6DoF 카메라 이동 (orbit + dolly) |
| WASD + 마우스 | first-person 이동 (viser 표준) |

## 디버깅

### viewer 안 뜸 / Connection refused

- 다른 viewer 가 port 점유 중인지: `ss -tln | grep 8080`
- 죽이고 재시작: `pkill -9 -f "viewer.py" && sleep 2 && <위 명령>`

### cloth_logit 채널이 안 그려짐 (mask overlay 가 비어있음)

- gsplat 의 cloth_logit 2nd-pass 가 silent fail — CUDA 환경변수 (`CUDA_HOME`, `CPATH`, `LD_LIBRARY_PATH`) 누락 가능
- 위 export 명령 다시 확인

### 화면 검정 / SAM3 fail

- ckpt 의 `cfg_args.source_path` 가 *상대경로* 일 때 viewer 가 못 찾는 경우 있음
- viewer 가 자동으로 `4DGaussians/` prefix 붙임 — 이상하면 `source_path` 절대경로로 ckpt 수정

### 4D-DRESS rendering 너무 작음

- multipleview 의 카메라 4개 가까이 있음 — `jump to cam` 버튼으로 preset 가서 확인 후 viewport 드래그

## 관련 노트

- [[results/exp010_per_gaussian_projection]] — single-class baseline
- [[results/exp015_n3v_train_time_spatial_filter]] — N3V calibration
- [[results/exp025_multiclass]] — multi-instance K=3
- [[results/exp026_4d_label_mapping]] — 4D bucket label
- [[007Experiment_Plan_2026-05-15]] — 실험 sprint
