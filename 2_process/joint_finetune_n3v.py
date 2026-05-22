"""Joint cloth-aware fine-tune for N3V (Plan B3, exp019).

Loads baseline 4DGS ckpt, unfreezes deformation MLP + cloth_logit, then trains
both jointly with L_rgb + lambda_cloth * L_cloth_BCE for K iters. The cloth BCE
uses alpha-composited rendered cloth_logit (gradient flows back to deformation
MLP), so cloth Gaussians naturally cluster together over time and trail/leak
Gaussians get pushed out by RGB consistency.

Starts from cloth_logit initialized via per-Gaussian projection (warm start),
so we avoid the cold-start over-cover that exp001-009 had with tier2.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).parent.resolve()
FOURDGS_DIR = (HERE / ".." / ".." / "4DGaussians").resolve()
sys.path.insert(0, str(FOURDGS_DIR))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-ckpt", required=True)
    ap.add_argument("--base-iter", type=int, default=14000)
    ap.add_argument("--sam3-scene-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--lr-deform", type=float, default=1e-5)
    ap.add_argument("--lr-cloth", type=float, default=0.01)
    ap.add_argument("--lambda-cloth", type=float, default=0.05)
    ap.add_argument("--freeze-cloth", action="store_true",
                    help="freeze cloth_logit (B3a): only train deformation MLP via cloth BCE gradient")
    ap.add_argument("--warmup", type=int, default=200,
                    help="iters before cloth BCE turns on (let RGB stabilize)")
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
    masks_per_cam = {}
    for cam_idx, cd in enumerate(cam_dirs):
        z = np.load(cd / "masks.npz")
        T, H, W = z["shape"].tolist()
        m = np.unpackbits(z["masks_packed"], axis=1)[:, :H*W].reshape(T, H, W).astype(np.float32)
        masks_per_cam[cam_idx] = torch.from_numpy(m).to(args.device)
    n_cams = len(masks_per_cam)
    n_frames = masks_per_cam[0].shape[0]
    print(f"  {n_cams} cams x {n_frames} frames, mask shape={masks_per_cam[0].shape[1:]}", flush=True)

    # ============ Load N3V dataset (for GT images) ============
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
    print(f"  N3V dataset: {len(train_ds)} samples", flush=True)

    # ============ Warm start: per-Gaussian soft_target → cloth_logit ============
    print("warm-starting cloth_logit from per-Gaussian projection...", flush=True)
    accum = torch.zeros(N, device=args.device)
    count = torch.zeros(N, device=args.device)
    canonical = pc._xyz.detach()
    from scene.cameras import Camera
    from utils.graphics_utils import focal2fov
    image_poses = train_ds.image_poses
    image_times = train_ds.image_times
    focal = train_ds.focal[0]
    img_shape = (masks_per_cam[0].shape[1], masks_per_cam[0].shape[2])

    def make_cam_obj(idx):
        R, T = image_poses[idx]
        t = image_times[idx]
        dummy = torch.zeros(3, img_shape[0], img_shape[1])
        return Camera(colmap_id=idx, R=R, T=T,
                      FoVx=focal2fov(focal, img_shape[1]),
                      FoVy=focal2fov(focal, img_shape[0]),
                      image=dummy, gt_alpha_mask=None,
                      image_name=str(idx), uid=idx, time=t)

    n_total = len(train_ds)
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
            ndc = torch.nan_to_num(clip[:, :3] / w, nan=0.0, posinf=0.0, neginf=0.0)
            in_view = (clip[:, 3] > 0) & (ndc[:, 0].abs() < 1) & (ndc[:, 1].abs() < 1)
            mask = masks_per_cam[cam_idx][frame_idx]
            mH, mW = mask.shape
            px = ((ndc[:, 0] + 1) * 0.5 * mW).long().clamp(0, mW - 1)
            py = ((ndc[:, 1] + 1) * 0.5 * mH).long().clamp(0, mH - 1)
            v = mask[py, px]
            accum = accum + torch.where(in_view, v, torch.zeros_like(v))
            count = count + in_view.float()
    soft_target = accum / count.clamp_min(1)
    init_logit = torch.logit(soft_target.clamp(0.05, 0.95))
    pc._cloth_logit.data = init_logit.unsqueeze(-1).clone()
    print(f"  warm cloth_logit: mean prob={float(soft_target.mean()):.3f}, "
          f">0.5={int((soft_target > 0.5).sum())}", flush=True)

    # ============ Setup joint training ============
    # Unfreeze deformation MLP + (optionally) cloth_logit
    pc._cloth_logit.requires_grad_(not args.freeze_cloth)
    for p in pc._deformation.parameters():
        p.requires_grad_(True)
    for p_group in (pc._xyz, pc._features_dc, pc._features_rest,
                    pc._scaling, pc._rotation, pc._opacity):
        p_group.requires_grad_(False)

    deform_params = list(pc._deformation.parameters())
    opt_groups = [{"params": deform_params, "lr": args.lr_deform}]
    if not args.freeze_cloth:
        opt_groups.append({"params": [pc._cloth_logit], "lr": args.lr_cloth})
    opt = torch.optim.Adam(opt_groups)
    msg = f"  optimizer: deform({sum(p.numel() for p in deform_params):,} params @ lr={args.lr_deform})"
    if not args.freeze_cloth:
        msg += f" + cloth_logit({pc._cloth_logit.numel():,} @ lr={args.lr_cloth})"
    else:
        msg += " [cloth_logit FROZEN]"
    print(msg, flush=True)

    # ============ Joint fine-tune ============
    from gaussian_renderer import render
    pipe_args = type("P", (), {"convert_SHs_python": False, "compute_cov3D_python": False, "debug": False})
    bg = torch.zeros(3, device=args.device)
    print(f"joint training {args.iters} iters (lambda_cloth={args.lambda_cloth} after warmup={args.warmup})...",
          flush=True)
    t0 = time.time()
    for step in range(args.iters):
        # Sample one (cam, frame) view
        view_idx = int(torch.randint(0, n_total, (1,)).item())
        cam_idx = view_idx // n_frames
        frame_idx = view_idx % n_frames
        cam = make_cam_obj(view_idx)
        # Load GT image (resize from raw to mask resolution to match)
        gt_img, _, _ = train_ds[view_idx]   # (3, H_orig, W_orig)
        gt_img = gt_img.to(args.device)
        # Resize GT to mask resolution (same as render)
        if gt_img.shape[1:] != img_shape:
            gt_img = F.interpolate(gt_img.unsqueeze(0), size=img_shape, mode="bilinear", align_corners=False).squeeze(0)

        out = render(cam, pc, pipe_args, bg, stage="fine")
        rgb = out["render"]
        cl_img = out.get("cloth_logit")  # (H, W) for K=1

        L_rgb = F.l1_loss(rgb, gt_img)
        if cl_img is not None and step >= args.warmup and args.lambda_cloth > 0:
            mask = masks_per_cam[cam_idx][frame_idx]
            # Ensure same shape (mH x mW)
            if cl_img.shape != mask.shape:
                cl_img = F.interpolate(cl_img.unsqueeze(0).unsqueeze(0), size=mask.shape,
                                        mode="bilinear", align_corners=False).squeeze(0).squeeze(0)
            L_cloth = F.binary_cross_entropy_with_logits(cl_img, mask)
        else:
            L_cloth = torch.tensor(0.0, device=args.device)

        loss = L_rgb + args.lambda_cloth * L_cloth
        opt.zero_grad()
        loss.backward()
        opt.step()

        if (step + 1) % 200 == 0:
            with torch.no_grad():
                cp = (torch.sigmoid(pc._cloth_logit) > 0.5).float().mean() * 100
            elapsed = time.time() - t0
            print(f"  step {step+1}: L_rgb={L_rgb.item():.4f} L_cloth={L_cloth.item():.4f} "
                  f"cloth_pct@0.5={cp.item():.1f}% [{elapsed:.0f}s]", flush=True)

    # ============ Save ============
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cfg_args").write_text((base / "cfg_args").read_text())
    new_pc_dir = out_dir / "point_cloud" / f"iteration_{args.base_iter}"
    new_pc_dir.mkdir(parents=True, exist_ok=True)
    pc.save_ply(str(new_pc_dir / "point_cloud.ply"))
    # save updated deformation
    torch.save(pc._deformation.state_dict(), new_pc_dir / "deformation.pth")
    import shutil
    for f in ["deformation_table.pth", "deformation_accum.pth"]:
        src = ply_path.parent / f
        if src.exists():
            shutil.copy2(src, new_pc_dir / f)
    print(f"saved {out_dir}", flush=True)


if __name__ == "__main__":
    main()
