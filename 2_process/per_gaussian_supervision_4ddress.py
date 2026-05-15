"""Per-Gaussian SAM3 projection supervision adapted for 4D-DRESS (multipleview).

Differences from D-NeRF version:
  - Loads cameras via readMultipleViewinfos (poses_bounds_multipleview.npy)
  - 4 cameras × T frames = 4T view/frame pairs (D-NeRF has only T monocular views)
  - SAM3 masks loaded per camera ID (0004, 0028, 0052, 0076) — 4 separate .npz

Usage:
    python per_gaussian_supervision_4ddress.py \\
        --base-ckpt 3_output/00169_Outer_Take12/ckpt_baseline_4ddress \\
        --base-iter 14000 \\
        --sam3-take cache/sam3_4ddress/00169/Outer/Take12 \\
        --out 3_output/00169_Outer_Take12/ckpt_exp011_per_gaussian \\
        --iters 2000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent.resolve()
FOURDGS_DIR = (HERE / ".." / ".." / "4DGaussians").resolve()
sys.path.insert(0, str(FOURDGS_DIR))


def _load_mask_cache(sam3_take_dir: Path, device: str):
    """Load per-camera SAM3 masks. Returns: {cam_idx: (T, H, W) tensor}"""
    cams = sorted([d.name for d in sam3_take_dir.iterdir() if d.is_dir() and d.name.isdigit()])
    out = {}
    for ci, cam_id in enumerate(cams):
        z = np.load(sam3_take_dir / cam_id / "masks.npz")
        packed = z["masks_packed"]
        T, H, W = z["shape"].tolist()
        m = np.unpackbits(packed, axis=1)[:, : H * W].reshape(T, H, W).astype(np.float32)
        out[ci] = torch.from_numpy(m).to(device)
        print(f"  cam {cam_id} (idx={ci}): {T}×{H}×{W}, avg cov={m.mean()*100:.2f}%", flush=True)
    return out, cams


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-ckpt", required=True)
    ap.add_argument("--base-iter", type=int, default=14000)
    ap.add_argument("--sam3-take", required=True,
                    help="dir containing {0004,0028,0052,0076}/masks.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    # ============ Load base ckpt ============
    from argparse import Namespace
    base = Path(args.base_ckpt)
    cfg_args = eval((base / "cfg_args").read_text())
    cfg_args.num_classes = 1
    from scene.gaussian_model import GaussianModel
    pc = GaussianModel(cfg_args.sh_degree, cfg_args)
    ply_path = base / "point_cloud" / f"iteration_{args.base_iter}" / "point_cloud.ply"
    pc.load_ply(str(ply_path))
    pc.load_model(str(ply_path.parent))
    N = pc._xyz.shape[0]
    print(f"loaded {base.name}: N={N:,} Gaussians", flush=True)

    if pc._cloth_logit.dim() > 1 and pc._cloth_logit.shape[1] > 1:
        old = pc._cloth_logit.detach()[:, 0:1]
        pc._cloth_logit = torch.nn.Parameter(old.clone().requires_grad_(True))

    # ============ Load SAM3 masks (4 cameras) ============
    print(f"loading SAM3 masks from {args.sam3_take}", flush=True)
    masks_per_cam, cam_ids = _load_mask_cache(Path(args.sam3_take), args.device)
    n_cams = len(masks_per_cam)
    n_frames = masks_per_cam[0].shape[0]
    H_m = masks_per_cam[0].shape[1]
    W_m = masks_per_cam[0].shape[2]
    print(f"total {n_cams} cams × {n_frames} frames = {n_cams * n_frames} view/frame pairs",
          flush=True)

    # ============ Load multipleview cameras ============
    import os
    src_path = cfg_args.source_path
    if not os.path.isabs(src_path):
        src_path = str((FOURDGS_DIR / src_path).resolve())
    print(f"source_path = {src_path}", flush=True)
    from scene.dataset_readers import readMultipleViewinfos
    from scene.cameras import Camera
    from utils.graphics_utils import focal2fov
    scene_info = readMultipleViewinfos(src_path)
    raw_train = scene_info.train_cameras
    total_views = len(raw_train)
    expected_total = n_cams * n_frames
    print(f"raw_train has {total_views} entries (expected {expected_total})", flush=True)
    # multipleview_dataset orders: outer loop cams, inner loop frames
    # i.e. raw_train[c * n_frames + f] = cam c, frame f
    focal0 = raw_train.focal[0]

    def make_camera(idx):
        img, (R, T), t = raw_train[idx]
        return Camera(colmap_id=idx, R=R, T=T,
                      FoVx=focal2fov(focal0, img.shape[2]),
                      FoVy=focal2fov(focal0, img.shape[1]),
                      image=img, gt_alpha_mask=None,
                      image_name=str(idx), uid=idx, time=t)

    # ============ Compute per-Gaussian soft target ============
    print("computing per-Gaussian soft targets via projection over ALL 4-view × frame pairs...",
          flush=True)
    accum = torch.zeros(N, device=args.device)
    count = torch.zeros(N, device=args.device)
    with torch.no_grad():
        canonical = pc._xyz.detach()
        for view_idx in range(total_views):
            cam = make_camera(view_idx)
            cam_idx = view_idx // n_frames    # 0..3
            frame_idx = view_idx % n_frames   # 0..159
            t = torch.full((N, 1), float(cam.time), device=args.device)
            # deformation forward
            try:
                means_t, _, _, _, _ = pc._deformation(canonical, pc._scaling.detach(),
                                                       pc._rotation.detach(), pc._opacity.detach(),
                                                       pc._features_dc.detach(),
                                                       times_sel=t)
            except Exception:
                means_t = canonical + pc._deformation.deformation_net(canonical, t)[:, :3]
            ones = torch.ones((N, 1), device=args.device)
            fpm = cam.full_proj_transform.to(args.device)
            clip = torch.cat([means_t, ones], dim=-1) @ fpm
            w = clip[:, 3:].clamp_min(1e-6)
            ndc = clip[:, :3] / w
            in_front = clip[:, 3] > 0
            in_view = in_front & (ndc[:, 0].abs() < 1) & (ndc[:, 1].abs() < 1)
            px = ((ndc[:, 0] + 1) * 0.5 * W_m).long().clamp(0, W_m - 1)
            py = ((ndc[:, 1] + 1) * 0.5 * H_m).long().clamp(0, H_m - 1)
            mask = masks_per_cam[cam_idx][frame_idx]
            v = mask[py, px]
            accum = accum + torch.where(in_view, v, torch.zeros_like(v))
            count = count + in_view.float()
            if view_idx % 50 == 0:
                print(f"  view {view_idx+1}/{total_views} (cam {cam_ids[cam_idx]}, frame {frame_idx}): "
                      f"in_view avg={float(in_view.float().mean()):.3f}", flush=True)
    soft_target = accum / count.clamp_min(1)
    print(f"per-Gaussian soft_target: mean={float(soft_target.mean()):.3f}, "
          f">0.5={int((soft_target>0.5).sum())} ({float((soft_target>0.5).float().mean())*100:.1f}%), "
          f"ambig [0.3,0.7]={float(((soft_target>0.3)&(soft_target<0.7)).float().mean())*100:.1f}%",
          flush=True)

    # ============ Train cloth_logit only ============
    pc._cloth_logit.requires_grad_(True)
    for p_group in (pc._xyz, pc._features_dc, pc._features_rest,
                    pc._scaling, pc._rotation, pc._opacity):
        p_group.requires_grad_(False)
    for p in pc._deformation.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam([pc._cloth_logit], lr=args.lr)
    print(f"training cloth_logit ({args.iters} steps, lr={args.lr})...", flush=True)
    for step in range(args.iters):
        opt.zero_grad()
        pred = torch.sigmoid(pc._cloth_logit.squeeze(-1))
        loss = torch.nn.functional.binary_cross_entropy(pred, soft_target.detach())
        loss.backward()
        opt.step()
        if (step + 1) % 200 == 0:
            with torch.no_grad():
                cp = (pred > 0.5).float().mean() * 100
            print(f"  step {step+1}: BCE={loss.item():.4f}  cloth_pct@0.5={cp.item():.1f}%",
                  flush=True)

    # ============ Save ============
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cfg_args").write_text((base / "cfg_args").read_text())
    new_pc_dir = out / "point_cloud" / f"iteration_{args.base_iter}"
    new_pc_dir.mkdir(parents=True, exist_ok=True)
    pc.save_ply(str(new_pc_dir / "point_cloud.ply"))
    import shutil
    for f in ["deformation.pth", "deformation_table.pth", "deformation_accum.pth"]:
        src = ply_path.parent / f
        if src.exists():
            shutil.copy2(src, new_pc_dir / f)
    np.save(out / "per_gaussian_soft_target.npy", soft_target.cpu().numpy())
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
