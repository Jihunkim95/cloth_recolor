"""Per-Gaussian SAM3 projection supervision for Neural 3D Video (N3V) scenes.

Layout:
  Neural3DVideo/<scene>/cam<NN>/images/0001.png ... 0300.png
  cache/sam3_n3v/<scene>/<cam>/masks.npz
  3_output/n3v_<scene>/ckpt_baseline_4dgs/ (trained 4DGS, iter 14000)

Approach:
  - Load trained 4DGS ckpt (Gaussian xyz + deformation MLP)
  - For each (cam, frame): project deformed Gaussian xyz → look up SAM3 mask
  - Average → per-Gaussian soft_target
  - Train cloth_logit only with BCE, freeze else
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).parent.resolve()
FOURDGS_DIR = (HERE / ".." / ".." / "4DGaussians").resolve()
sys.path.insert(0, str(FOURDGS_DIR))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-ckpt", required=True)
    ap.add_argument("--base-iter", type=int, default=14000)
    ap.add_argument("--sam3-scene-dir", required=True,
                    help="cache/sam3_n3v/<scene> — contains <cam>/masks.npz per cam")
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--spatial-filter", type=float, default=0.0,
                    help="MAD multiplier on soft_target seed cluster; 0=off, 3=typical")
    ap.add_argument("--spatial-seed-thresh", type=float, default=0.2,
                    help="soft_target threshold to seed the spatial cluster")
    ap.add_argument("--depth-aware", action="store_true",
                    help="only count Gaussian as cloth if it's front-most at its projected pixel")
    ap.add_argument("--depth-tol", type=float, default=1.10,
                    help="relative depth tolerance for visibility (1.10 = 10% behind front)")
    ap.add_argument("--temporal-coherent", action="store_true",
                    help="per-frame 3D cluster: only count if Gaussian deformed position is near in-mask cluster centroid")
    ap.add_argument("--temporal-k", type=float, default=3.0,
                    help="MAD multiplier for per-frame cluster radius")
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

    # ============ Load SAM3 masks per cam ============
    sam_dir = Path(args.sam3_scene_dir)
    cam_dirs = sorted([d for d in sam_dir.iterdir() if d.is_dir() and d.name.startswith("cam")])
    masks_per_cam = {}   # cam_idx -> (T, H, W) tensor
    cam_names = []
    for cam_idx, cd in enumerate(cam_dirs):
        z = np.load(cd / "masks.npz")
        T, H, W = z["shape"].tolist()
        m = np.unpackbits(z["masks_packed"], axis=1)[:, :H*W].reshape(T, H, W).astype(np.float32)
        masks_per_cam[cam_idx] = torch.from_numpy(m).to(args.device)
        cam_names.append(cd.name)
        print(f"  cam {cd.name}: {T}×{H}×{W}, cov={m.mean()*100:.2f}%", flush=True)
    n_cams = len(masks_per_cam)
    n_frames = masks_per_cam[0].shape[0]
    H_m, W_m = masks_per_cam[0].shape[1], masks_per_cam[0].shape[2]
    print(f"total {n_cams} cams × {n_frames} frames = {n_cams * n_frames} view/frame pairs", flush=True)

    # ============ Load N3V cameras ============
    import os
    src_path = cfg_args.source_path
    if not os.path.isabs(src_path):
        src_path = str((FOURDGS_DIR / src_path).resolve())

    from scene.neural_3D_dataset_NDC import Neural3D_NDC_Dataset
    train_ds = Neural3D_NDC_Dataset(
        src_path, "train", 1.0, time_scale=1,
        scene_bbox_min=[-2.5, -2.0, -1.0], scene_bbox_max=[2.5, 2.0, 1.0],
        eval_index=0,
    )
    print(f"train_ds has {len(train_ds)} samples (= n_cam × n_frame)", flush=True)

    # Build a flat list of (cam_idx, frame_idx, R, T, fov_w, fov_h, time)
    # n3d_ndc_dataset stacks images per cam, so index = cam_idx * n_frames + frame_idx
    from scene.cameras import Camera
    from utils.graphics_utils import focal2fov

    # The dataset has image_poses, image_times — let's expose them
    image_poses = train_ds.image_poses     # list of (R, T) per index
    image_times = train_ds.image_times     # list of float per index
    focal = train_ds.focal[0]               # single focal length
    img_shape = (H_m, W_m)                  # use mask shape

    n_total = len(train_ds)
    expected = n_cams * n_frames
    print(f"image_poses entries: {n_total} (expected {expected})", flush=True)

    # ============ Compute per-Gaussian soft target ============
    print("computing per-Gaussian soft targets via projection...", flush=True)
    accum = torch.zeros(N, device=args.device)
    count = torch.zeros(N, device=args.device)
    canonical = pc._xyz.detach()

    def make_cam_obj(idx):
        R, T = image_poses[idx]
        t = image_times[idx]
        # Create dummy image to satisfy Camera init
        dummy = torch.zeros(3, img_shape[0], img_shape[1])
        return Camera(colmap_id=idx, R=R, T=T,
                      FoVx=focal2fov(focal, img_shape[1]),
                      FoVy=focal2fov(focal, img_shape[0]),
                      image=dummy, gt_alpha_mask=None,
                      image_name=str(idx), uid=idx, time=t)

    with torch.no_grad():
        for view_idx in range(n_total):
            cam_idx = view_idx // n_frames
            frame_idx = view_idx % n_frames
            cam = make_cam_obj(view_idx)
            t = torch.full((N, 1), float(cam.time), device=args.device)
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
            ndc = torch.nan_to_num(ndc, nan=0.0, posinf=0.0, neginf=0.0)
            in_front = clip[:, 3] > 0
            in_view = in_front & (ndc[:, 0].abs() < 1) & (ndc[:, 1].abs() < 1)
            in_view = in_view & torch.isfinite(ndc[:, 0]) & torch.isfinite(ndc[:, 1])
            # per-cam mask resolution (flame_steak has mixed: cam00/20 full, cam01-19 half)
            mask = masks_per_cam[cam_idx][frame_idx]
            mH, mW = mask.shape
            px = ((ndc[:, 0] + 1) * 0.5 * mW).long().clamp(0, mW - 1)
            py = ((ndc[:, 1] + 1) * 0.5 * mH).long().clamp(0, mH - 1)
            v = mask[py, px]

            # temporal coherent: per-frame 3D cluster of in-mask Gaussian, check if my deformed pos is near
            if args.temporal_coherent:
                in_mask_now = in_view & (v > 0.5)
                if int(in_mask_now.sum()) > 50:
                    cluster_pts = means_t[in_mask_now]
                    center = cluster_pts.median(dim=0).values
                    d_cluster = (cluster_pts - center).norm(dim=1)
                    cluster_mad = (d_cluster - d_cluster.median()).abs().median().clamp_min(1e-3)
                    cutoff = d_cluster.median() + args.temporal_k * cluster_mad
                    d_all = (means_t - center).norm(dim=1)
                    in_cluster = d_all <= cutoff
                else:
                    in_cluster = torch.zeros_like(in_view)
            else:
                in_cluster = torch.ones_like(in_view)

            # depth-aware visibility: front-most Gaussian per pixel via scatter_min
            if args.depth_aware:
                depth_z = clip[:, 3]  # (N,) +z forward in camera frame
                # Build per-pixel min-depth map (only over in_view Gaussians)
                flat_idx = (py * mW + px).long()
                # use a large value for out-of-view to not affect scatter_min
                src = torch.where(in_view, depth_z, torch.full_like(depth_z, 1e9))
                depth_flat = torch.full((mH * mW,), 1e9, device=args.device)
                depth_flat.scatter_reduce_(0, flat_idx, src, reduce="amin", include_self=True)
                front_depth = depth_flat[flat_idx]
                visible = depth_z <= front_depth * args.depth_tol
                hit_mask = in_view & visible
            else:
                hit_mask = in_view
            # combine with temporal coherence: a hit only counts if Gaussian is in the per-frame cluster
            # (or we're not using temporal-coherent in which case in_cluster is all True)
            #
            # For accum (mask-1 hits) we require in_cluster too, so trail Gaussians don't accumulate
            # cloth-evidence even if their projection lands in the SAM3 mask.
            # For count (visit count) we keep in_view unchanged so Gaussians with no cluster membership
            # at any frame still get count (gives them low soft_target naturally).
            accum_hit = hit_mask & in_cluster
            accum = accum + torch.where(accum_hit, v, torch.zeros_like(v))
            count = count + hit_mask.float()
            if view_idx % 500 == 0:
                print(f"  view {view_idx+1}/{n_total}: in_view avg={float(in_view.float().mean()):.3f}",
                      flush=True)
    soft_target = accum / count.clamp_min(1)
    print(f"per-Gaussian soft_target: mean={float(soft_target.mean()):.3f}, "
          f">0.5={int((soft_target > 0.5).sum())} ({float((soft_target > 0.5).float().mean())*100:.1f}%), "
          f"ambig [0.3,0.7]={float(((soft_target > 0.3) & (soft_target < 0.7)).float().mean())*100:.1f}%",
          flush=True)

    if args.spatial_filter > 0:
        seed_mask = soft_target > args.spatial_seed_thresh
        n_seed = int(seed_mask.sum())
        if n_seed > 50:
            xyz_seed = canonical[seed_mask]
            med = xyz_seed.median(dim=0).values
            dist_seed = (xyz_seed - med).norm(dim=1)
            mad = (dist_seed - dist_seed.median()).abs().median().clamp_min(1e-6)
            cutoff = dist_seed.median() + args.spatial_filter * mad
            dist_all = (canonical - med).norm(dim=1)
            outlier = (dist_all > cutoff) & seed_mask
            soft_target = torch.where(outlier, torch.zeros_like(soft_target), soft_target)
            print(f"spatial filter k={args.spatial_filter} on seed (>{args.spatial_seed_thresh}, n={n_seed}): "
                  f"cutoff={float(cutoff):.3f}, zeroed={int(outlier.sum())} outliers", flush=True)

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
