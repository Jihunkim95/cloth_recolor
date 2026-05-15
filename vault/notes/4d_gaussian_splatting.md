---
title: 4D Gaussian Splatting
layer: 2
date: 2026-05-11
---

# 4D Gaussian Splatting (4DGS)

> "시간 t 마다 다른 자세를 가진 가우시안" 을 효율적으로 표현하기 위해 누군가 정한 방식. layer 2 — 깰 수 있다.

## 핵심 아이디어

3D Gaussian Splatting (3DGS) 은 정적인 가우시안 N 개로 한 시점 렌더링.
**4DGS (Wu et al., 2023)** 는 그걸 시간으로 확장 — 매 시점 가우시안 위치/크기/회전이 변함.

### 가우시안 1 개의 attribute

| 그룹 | 항목 | 차원 | 역할 |
|---|---|---|---|
| 위치 | `xyz` | 3 | 어디에 있는가 |
| **shape** | `scale` | 3 (log) | 3축 반지름 — 길쭉/납작/구형 |
| **shape** | `rotation` | 4 (quaternion) | 3D 회전 — 어느 방향으로 누웠는가 |
| 밀도 | `opacity` | 1 (logit) | 얼마나 진하게 보이는가 |
| 색 | `SH_DC` | 3 | view-independent 베이스 RGB |
| 색 | `SH_rest` | 3·(L²+2L) | view-dependent residual (specular 등) |

- **shape = (scale, rotation)** — 가우시안이 차지하는 anisotropic 타원체의 크기·방향
  - scale (5, 0.1, 0.1) → 길쭉한 침바늘
  - scale (3, 3, 0.01) → 얇은 디스크 (천 표면 표현에 유리)
- *의미 정보 (옷/피부/배경) 는 없음* → [[cloth_logit_channel]] 로 추가

**구현 방식 두 갈래**:
1. **Per-time Gaussian set** — 시간마다 가우시안 따로. 메모리 N×T. 🚫 비효율.
2. **Canonical + deformation field** — 가우시안 N 개는 t=0 기준, 시간은 deformation MLP 가 (x,y,z,t) → (Δx, Δscale, Δrot) 예측. ✅ 우리가 쓰는 방식.

## hustvl/4DGaussians 의 deformation field

HexPlane (k-planes) 으로 4D 공간을 6 개 2D plane 으로 factor:
- (x,y), (x,z), (y,z) — 정적 spatial
- (x,t), (y,t), (z,t) — 동적 spatio-temporal

각 가우시안은 자기 위치를 6 plane 에서 lookup → concat → 작은 MLP → Δ 예측.

## 학습 산출물

```
output/<expname>/point_cloud/iteration_14000/
├── point_cloud.ply         ← canonical 가우시안 N 개 (xyz, scale, rot, opacity, SH)
├── deformation.pth         ← HexPlane + MLP weights
└── deformation_table.pth   ← 가우시안별 deform 적용 여부 bool
```

## 왜 이게 우리 연구에 중요한가

- 한 번 학습하면 `deformation(canonical_xyz, t)` 호출만으로 임의 t·시점 렌더 가능
- 가우시안 단위 attribute (예: [[cloth_logit_channel]]) 를 추가하면 *시간 일관성 자동 보존* — 가우시안이 그대로 deform 만 되니까

## 한계 (깰 수 있는 부분)

- canonical 자세 + deformation 가정 — **topology 변경** (옷이 벗겨짐, 천이 찢어짐) 은 표현 불가
- HexPlane 해상도 (기본 64×64×64×100) 가 디테일·시간 모두 결정 → 빠른 모션 / 작은 디테일 손실
- 단안 비디오에서는 모호성 폭발 — 우리는 sparse multi-view 또는 D-NeRF 처럼 카메라 trajectory 다양해야 함

## 우리가 쓰는 코드

`research/4DGaussians/` (hustvl/4DGaussians + 우리 fork mod):
- `scene/gaussian_model.py` — `_cloth_logit` parameter 추가
- `gaussian_renderer/__init__.py` — gsplat 2nd-pass for cloth_logit channel
- `utils/soft_cal.py` — [[soft_label_calibration]] 구현
