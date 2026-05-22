"""exp026: 4D timeframe grid soft_target (per-Gaussian, per-bucket).

Extends per_gaussian_supervision_n3v.py to compute soft_target per time bucket:
  soft_target[i, b] = mean mask hit for Gaussian i in time bucket b
where bucket b covers frames [b*B, (b+1)*B) with B = n_frames / n_buckets.

Then per-bucket cloth_logit (N, n_buckets) is trained with BCE. At recolor
time, the Gaussian's cloth probability at time t uses the corresponding bucket.

Compared to exp015 (per-Gaussian, time-invariant), this allows the cloth label
to change over time — useful when a Gaussian moves between cloth/background
across the dynamic sequence.
"""
from __future__ import annotations
import argparse, json, sys, os
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
    ap.add_argument("--n-buckets", type=int, default=10,
                    help="number of time buckets (e.g., 10 for 300 frame → 30 frame per bucket)")
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    B = args.n_buckets
    print(f"4D label mapping: n_buckets = {B}", flush=True)

    # ============ Load base ckpt ============
    from argparse import Namespace
    base = Path(args.base_ckpt)
    cfg_args = eval((base / "cfg_args").read_text())
    cfg_args.num_classes = B   # K = B (each bucket as a class)
    from scene.gaussian_model import GaussianModel
    pc = GaussianModel(cfg_args.sh_degree, cfg_args)
    ply_path = base / "point_cloud" / f"iteration_{args.base_iter}" / "point_cloud.ply"
    pc.load_ply(str(ply_path))
    pc.load_model(str(ply_path.parent))
    N = pc._xyz.shape[0]
    pc._cloth_logit = torch.nn.Parameter(
        torch.zeros(N, B, device=args.device, requires_grad=True))
    print(f"loaded: N={N:,}, cloth_logit shape={pc._cloth_logit.shape}", flush=True)

    # ============ Load SAM3 masks per cam ============
    sam_dir = Path(args.sam3_scene_dir)
    cam_dirs = sorted([d for d in sam_dir.iterdir() if d.is_dir() and d.name.startswith("cam")])
    masks_per_cam = {}
    for ci, cd in enumerate(cam_dirs):
        z = np.load(cd / "masks.npz")
        T, H, W = z["shape"].tolist()
        m = np.unpackbits(z["masks_packed"], axis=1)[:, :H*W].reshape(T, H, W).astype(np.float32)
        masks_per_cam[ci] = torch.from_numpy(m).to(args.device)
    n_cams = len(masks_per_cam)
    n_frames = masks_per_cam[0].shape[0]
    H_m, W_m = masks_per_cam[0].shape[1], masks_per_cam[0].shape[2]
    print(f"  {n_cams} cams × {n_frames} frames, n_buckets={B} → frames/bucket={n_frames//B}", flush=True)

    # ============ N3V camera loader ============
    src_path = cfg_args.source_path
    if not os.path.isabs(src_path):
        src_path = str((FOURDGS_DIR / src_path).resolve())
    from scene.neural_3D_dataset_NDC import Neural3D_NDC_Dataset
    train_ds = Neural3D_NDC_Dataset(src_path, "train", 1.0, time_scale=1,
                                     scene_bbox_min=[-2.5,-2,-1], scene_bbox_max=[2.5,2,1], eval_index=0)
    image_poses = train_ds.image_poses
    image_times = train_ds.image_times
    focal = train_ds.focal[0]
    img_shape = (H_m, W_m)
    from scene.cameras import Camera
    from utils.graphics_utils import focal2fov

    def make_cam(idx):
        R, T = image_poses[idx]
        return Camera(colmap_id=idx, R=R, T=T,
                      FoVx=focal2fov(focal, img_shape[1]),
                      FoVy=focal2fov(focal, img_shape[0]),
                      image=torch.zeros(3, img_shape[0], img_shape[1]),
                      gt_alpha_mask=None, image_name=str(idx), uid=idx,
                      time=image_times[idx])

    # ============ Per-bucket soft_target ============
    print(f"computing per-Gaussian per-bucket soft_target...", flush=True)
    accum = torch.zeros(N, B, device=args.device)
    count = torch.zeros(N, B, device=args.device)
    canonical = pc._xyz.detach()
    n_total = len(train_ds)
    with torch.no_grad():
        for view_idx in range(n_total):
            cam_idx = view_idx // n_frames
            frame_idx = view_idx % n_frames
            bucket_idx = min(frame_idx * B // n_frames, B - 1)
            cam = make_cam(view_idx)
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
            accum[:, bucket_idx] += torch.where(in_view, v, torch.zeros_like(v))
            count[:, bucket_idx] += in_view.float()
            if view_idx % 500 == 0:
                print(f"  view {view_idx+1}/{n_total} bucket={bucket_idx}", flush=True)

    soft_target = accum / count.clamp_min(1)
    for b in range(B):
        st = soft_target[:, b]
        print(f"  bucket[{b}] soft_target: mean={float(st.mean()):.3f} >0.5={int((st > 0.5).sum())}",
              flush=True)

    # ============ Train cloth_logit (N, B) per-bucket BCE ============
    pc._cloth_logit.requires_grad_(True)
    for p_group in (pc._xyz, pc._features_dc, pc._features_rest,
                    pc._scaling, pc._rotation, pc._opacity):
        p_group.requires_grad_(False)
    for p in pc._deformation.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam([pc._cloth_logit], lr=args.lr)
    print(f"training cloth_logit ({args.iters} steps)...", flush=True)
    for step in range(args.iters):
        opt.zero_grad()
        pred = torch.sigmoid(pc._cloth_logit)
        loss = F.binary_cross_entropy(pred, soft_target.detach())
        loss.backward()
        opt.step()
        if (step + 1) % 400 == 0:
            with torch.no_grad():
                cps = [(pred[:, b] > 0.5).float().mean() * 100 for b in range(B)]
            print(f"  step {step+1}: BCE={loss.item():.4f}", flush=True)
            for b in range(B):
                print(f"    bucket[{b}] cloth_pct@0.5={cps[b]:.1f}%", flush=True)

    # ============ Save ============
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cfg_args").write_text((base / "cfg_args").read_text())
    new_pc_dir = out_dir / "point_cloud" / f"iteration_{args.base_iter}"
    new_pc_dir.mkdir(parents=True, exist_ok=True)
    pc.save_ply(str(new_pc_dir / "point_cloud.ply"))
    import shutil
    for f in ["deformation.pth", "deformation_table.pth", "deformation_accum.pth"]:
        src = ply_path.parent / f
        if src.exists():
            shutil.copy2(src, new_pc_dir / f)
    np.save(out_dir / "per_gaussian_bucket_soft_target.npy", soft_target.cpu().numpy())
    (out_dir / "buckets.json").write_text(json.dumps({"n_buckets": B, "n_frames": n_frames}))
    print(f"saved {out_dir}", flush=True)


if __name__ == "__main__":
    main()
