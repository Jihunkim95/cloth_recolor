"""exp025: Multi-class per-Gaussian projection supervision (K instances).

Extends per_gaussian_supervision.py to support K independent instance classes
(e.g., hoodie / shorts / shoes for jumpingjacks). Each class has its own SAM3
cache; the Gaussian gets a K-dim cloth_logit, trained with K independent BCE
losses (one per class).

Usage:
  python per_gaussian_supervision_multiclass.py \
    --base-ckpt 3_output/jumpingjacks/ckpt_baseline_hardBCE --base-iter 20000 \
    --scene jumpingjacks \
    --sam3-caches cache/sam3_exp025_jumpingjacks_hoodie,cache/sam3_exp025_jumpingjacks_shorts,cache/sam3_exp025_jumpingjacks_shoes \
    --class-names hoodie,shorts,shoes \
    --out 3_output/jumpingjacks/ckpt_exp025_multi
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
    ap.add_argument("--base-iter", type=int, default=20000)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--sam3-caches", required=True,
                    help="comma-separated cache dirs (one per class)")
    ap.add_argument("--class-names", required=True,
                    help="comma-separated class names (must match cache count)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--spatial-filter", type=float, default=0.0,
                    help="MAD multiplier per-class on soft_target seed cluster; 0=off, 3=typical")
    ap.add_argument("--spatial-seed-thresh", type=float, default=0.2,
                    help="soft_target threshold to seed per-class spatial cluster")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    cache_dirs = [Path(c) for c in args.sam3_caches.split(",")]
    class_names = [c.strip() for c in args.class_names.split(",")]
    K = len(class_names)
    assert len(cache_dirs) == K, "sam3-caches count must match class-names count"
    print(f"K = {K} classes: {class_names}", flush=True)

    # Load each SAM3 cache
    masks_per_class = []
    for c, name in zip(cache_dirs, class_names):
        z = np.load(c / args.scene / "masks.npz")
        T, H, W = z["shape"].tolist()
        m = np.unpackbits(z["masks_packed"], axis=1)[:, :H*W].reshape(T, H, W).astype(np.float32)
        masks_per_class.append(torch.from_numpy(m).to(args.device))
        cov = float(m.mean()) * 100
        print(f"  class[{name}]: {T}x{H}x{W}, cov={cov:.2f}%", flush=True)
    T = masks_per_class[0].shape[0]
    H_m, W_m = masks_per_class[0].shape[1], masks_per_class[0].shape[2]

    # Load base ckpt with K classes
    from argparse import Namespace
    base = Path(args.base_ckpt)
    cfg_args = eval((base / "cfg_args").read_text())
    cfg_args.num_classes = K
    from scene.gaussian_model import GaussianModel
    pc = GaussianModel(cfg_args.sh_degree, cfg_args)
    ply_path = base / "point_cloud" / f"iteration_{args.base_iter}" / "point_cloud.ply"
    pc.load_ply(str(ply_path))
    pc.load_model(str(ply_path.parent))
    N = pc._xyz.shape[0]
    # Reset cloth_logit to K-dim zeros (in case loaded was K=1)
    pc._cloth_logit = torch.nn.Parameter(
        torch.zeros(N, K, device=args.device, requires_grad=True))
    print(f"loaded {base.name}: N={N:,}, cloth_logit shape={pc._cloth_logit.shape}", flush=True)

    # Build cameras (D-NeRF)
    src_path = cfg_args.source_path
    if not os.path.isabs(src_path):
        src_path = str((FOURDGS_DIR / src_path).resolve())
    from scene.dataset_readers import readCamerasFromTransforms
    from scene.cameras import Camera
    ts = json.loads((Path(src_path) / "transforms_train.json").read_text())
    max_time = max(f.get("time", 0.0) for f in ts["frames"])
    mapper = {f["time"]: (f["time"]/max_time if max_time > 0 else 0.0) for f in ts["frames"]}
    cam_infos = readCamerasFromTransforms(src_path, "transforms_train.json",
                                           white_background=getattr(cfg_args, "white_background", True),
                                           extension=".png", mapper=mapper)
    assert len(cam_infos) == T, f"cam count {len(cam_infos)} != mask T {T}"

    # ============ Per-class soft_target via projection ============
    print(f"computing per-Gaussian per-class soft_target (K={K})...", flush=True)
    accum = torch.zeros(N, K, device=args.device)
    count = torch.zeros(N, device=args.device)
    canonical = pc._xyz.detach()
    with torch.no_grad():
        for i, ci in enumerate(cam_infos):
            cam = Camera(colmap_id=i, R=ci.R, T=ci.T, FoVx=ci.FovX, FoVy=ci.FovY,
                         image=ci.image, gt_alpha_mask=None,
                         image_name=str(i), uid=i, time=ci.time)
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
            px = ((ndc[:, 0] + 1) * 0.5 * W_m).long().clamp(0, W_m - 1)
            py = ((ndc[:, 1] + 1) * 0.5 * H_m).long().clamp(0, H_m - 1)
            for k in range(K):
                v = masks_per_class[k][i, py, px]
                accum[:, k] = accum[:, k] + torch.where(in_view, v, torch.zeros_like(v))
            count = count + in_view.float()
            if i % 50 == 0:
                print(f"  frame {i+1}/{T}: in_view avg={float(in_view.float().mean()):.3f}", flush=True)
    soft_target = accum / count.clamp_min(1).unsqueeze(-1)  # (N, K)
    for k, name in enumerate(class_names):
        st = soft_target[:, k]
        print(f"  class[{name}] soft_target: mean={float(st.mean()):.3f}, "
              f">0.5={int((st > 0.5).sum())} ({float((st > 0.5).float().mean())*100:.1f}%)",
              flush=True)

    # Per-class spatial filter (apply [[exp015]] technique per class)
    if args.spatial_filter > 0:
        for k, name in enumerate(class_names):
            seed_mask = soft_target[:, k] > args.spatial_seed_thresh
            n_seed = int(seed_mask.sum())
            if n_seed < 50:
                print(f"  spatial filter {name}: skip (n_seed={n_seed} < 50)", flush=True)
                continue
            xyz_seed = canonical[seed_mask]
            med = xyz_seed.median(dim=0).values
            dist_seed = (xyz_seed - med).norm(dim=1)
            mad = (dist_seed - dist_seed.median()).abs().median().clamp_min(1e-6)
            cutoff = dist_seed.median() + args.spatial_filter * mad
            dist_all = (canonical - med).norm(dim=1)
            outlier = (dist_all > cutoff) & seed_mask
            soft_target[:, k] = torch.where(outlier, torch.zeros_like(soft_target[:, k]),
                                              soft_target[:, k])
            print(f"  spatial filter {name} k={args.spatial_filter}: seed={n_seed}, "
                  f"cutoff={float(cutoff):.3f}, zeroed={int(outlier.sum())} outliers", flush=True)

    # ============ Train cloth_logit (N, K) with K independent BCE ============
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
        pred = torch.sigmoid(pc._cloth_logit)
        loss = F.binary_cross_entropy(pred, soft_target.detach())
        loss.backward()
        opt.step()
        if (step + 1) % 200 == 0:
            with torch.no_grad():
                cps = [(pred[:, k] > 0.5).float().mean() * 100 for k in range(K)]
            cp_str = " ".join(f"{class_names[k]}={cps[k]:.1f}%" for k in range(K))
            print(f"  step {step+1}: BCE={loss.item():.4f}  cloth_pct@0.5: {cp_str}", flush=True)

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
    np.save(out_dir / "per_gaussian_soft_target.npy", soft_target.cpu().numpy())
    (out_dir / "class_names.json").write_text(json.dumps(class_names))
    print(f"saved {out_dir}", flush=True)


if __name__ == "__main__":
    main()
