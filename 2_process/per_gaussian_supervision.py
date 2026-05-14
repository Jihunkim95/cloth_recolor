"""Per-Gaussian cloth supervision via projection (bypasses per-pixel CE).

For each Gaussian g and each train frame i:
  - deform g.xyz with deformation MLP at frame i's time t_i
  - project to frame i's camera → (px, py)
  - look up SAM3 mask[i, py, px] → 0 or 1
Average target across all frames where g is visible (in-frustum + in-front-of-camera)
→ per-Gaussian soft target in [0, 1].

Then train ONLY cloth_logit with BCE against the soft target (other params frozen).
Saves a new ckpt with refined cloth_logit.

Usage:
    python per_gaussian_supervision.py \\
        --base-ckpt 3_output/jumpingjacks/ckpt_baseline_hardBCE \\
        --base-iter 14000 \\
        --sam3-cache cache/sam3_union_jumpingjacks \\
        --scene jumpingjacks \\
        --out 3_output/jumpingjacks/ckpt_exp010_per_gaussian \\
        --iters 2000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).parent.resolve()
FOURDGS_DIR = (HERE / ".." / ".." / "4DGaussians").resolve()
sys.path.insert(0, str(FOURDGS_DIR))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-ckpt", required=True, help="trained 4DGS ckpt dir (has PLY + deformation.pth)")
    ap.add_argument("--base-iter", type=int, default=14000)
    ap.add_argument("--sam3-cache", required=True, help="dir containing <scene>/masks.npz")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", required=True, help="output ckpt dir (new cloth_logit, copy of other params)")
    ap.add_argument("--iters", type=int, default=2000, help="Adam steps on cloth_logit only")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    # ============ Load base ckpt ============
    from argparse import Namespace
    base = Path(args.base_ckpt)
    cfg_args = eval((base / "cfg_args").read_text())
    cfg_args.num_classes = 1   # force K=1 for binary cloth_logit
    from scene.gaussian_model import GaussianModel
    pc = GaussianModel(cfg_args.sh_degree, cfg_args)
    ply_path = base / "point_cloud" / f"iteration_{args.base_iter}" / "point_cloud.ply"
    pc.load_ply(str(ply_path))
    pc.load_model(str(ply_path.parent))
    N = pc._xyz.shape[0]
    print(f"loaded {base.name}: N={N:,} Gaussians", flush=True)

    # If loaded ckpt had K>1 cloth_logit (multi-class), collapse to K=1
    if pc._cloth_logit.dim() > 1 and pc._cloth_logit.shape[1] > 1:
        # take class-0 logit
        old = pc._cloth_logit.detach()[:, 0:1]
        pc._cloth_logit = torch.nn.Parameter(old.clone().requires_grad_(True))
        print(f"  collapsed K={old.shape[1]+(pc._cloth_logit.shape[1]-1)} → K=1 for binary supervision")

    # ============ Load SAM3 masks ============
    cache_npz = Path(args.sam3_cache) / args.scene / "masks.npz"
    z = np.load(cache_npz)
    T, H_m, W_m = z["shape"].tolist()
    masks = np.unpackbits(z["masks_packed"], axis=1)[:, : H_m * W_m].reshape(T, H_m, W_m).astype(np.float32)
    masks_t = torch.from_numpy(masks).to(args.device)
    print(f"SAM3 masks: T={T}, HxW={H_m}x{W_m}, avg coverage={masks.mean()*100:.2f}%", flush=True)

    # ============ Load train cameras (D-NeRF) ============
    import os
    src_path = cfg_args.source_path
    if not os.path.isabs(src_path):
        src_path = str((FOURDGS_DIR / src_path).resolve())
    from scene.dataset_readers import readCamerasFromTransforms
    from scene.cameras import Camera
    ts = json.loads((Path(src_path) / "transforms_train.json").read_text())
    max_time = max(f.get("time", 0.0) for f in ts["frames"])
    mapper = {f["time"]: f["time"]/max_time if max_time > 0 else 0.0 for f in ts["frames"]}
    cam_infos = readCamerasFromTransforms(
        src_path, "transforms_train.json",
        white_background=getattr(cfg_args, "white_background", True),
        extension=".png", mapper=mapper,
    )
    cams = [Camera(colmap_id=ci.uid, R=ci.R, T=ci.T, FoVx=ci.FovX, FoVy=ci.FovY,
                   image=ci.image, gt_alpha_mask=None, image_name=ci.image_name,
                   uid=ci.uid, time=ci.time)
            for ci in cam_infos]
    print(f"loaded {len(cams)} train cameras", flush=True)
    assert len(cams) == T, f"mask T={T} != #cameras {len(cams)}"

    # ============ Compute per-Gaussian soft target via projection ============
    print("computing per-Gaussian soft targets via projection across all frames...", flush=True)
    accum = torch.zeros(N, device=args.device)
    count = torch.zeros(N, device=args.device)
    with torch.no_grad():
        # We need deformed xyz at each frame's time. Use the deformation MLP.
        canonical = pc._xyz.detach()   # (N, 3)
        # deformation forward signature varies; use the simple approach: query MLP at each time
        # 4DGaussians deformation expects (xyz, time, scaling, rotation, opacity) but we only need delta_xyz
        for i, cam in enumerate(cams):
            t = torch.full((N, 1), float(cam.time), device=args.device)
            # Use pc._deformation to compute deformed xyz
            try:
                # 4DGaussians deformation_net.forward signature
                means_t, _, _, _, _ = pc._deformation(canonical, pc._scaling.detach(),
                                                       pc._rotation.detach(), pc._opacity.detach(),
                                                       pc._features_dc.detach(),
                                                       times_sel=t)
            except Exception:
                # Fallback: deformation returns just delta_xyz when called differently
                means_t = canonical + pc._deformation.deformation_net(canonical, t)[:, :3]
            # Project to camera
            ones = torch.ones((N, 1), device=args.device)
            fpm = cam.full_proj_transform.to(args.device)
            clip = torch.cat([means_t, ones], dim=-1) @ fpm
            w = clip[:, 3:].clamp_min(1e-6)
            ndc = clip[:, :3] / w
            in_front = clip[:, 3] > 0
            in_view = in_front & (ndc[:, 0].abs() < 1) & (ndc[:, 1].abs() < 1)
            px = ((ndc[:, 0] + 1) * 0.5 * W_m).long().clamp(0, W_m - 1)
            py = ((ndc[:, 1] + 1) * 0.5 * H_m).long().clamp(0, H_m - 1)
            v = masks_t[i, py, px]              # (N,)
            accum = accum + torch.where(in_view, v, torch.zeros_like(v))
            count = count + in_view.float()
            if i % 50 == 0:
                print(f"  frame {i+1}/{T}: in_view avg={float(in_view.float().mean()):.3f}", flush=True)
    soft_target = accum / count.clamp_min(1)
    print(f"  per-Gaussian soft_target stats: "
          f"mean={float(soft_target.mean()):.3f}, "
          f">0.5={int((soft_target>0.5).sum())} ({float((soft_target>0.5).float().mean())*100:.1f}%), "
          f"in [0.3,0.7]={float(((soft_target>0.3)&(soft_target<0.7)).float().mean())*100:.1f}%",
          flush=True)

    # ============================================================================
    # Stage 3: Train cloth_logit ONLY with BCE against pre-computed soft_target.
    # All other params (xyz, scale, rot, opacity, SH, deformation MLP) stay frozen
    # so the base 4DGS geometry / RGB is preserved.
    # ============================================================================

    # (1) Mark cloth_logit as the only learnable variable
    pc._cloth_logit.requires_grad_(True)

    # (2) Freeze everything else — gradient won't flow into these tensors
    for p_group in (pc._xyz, pc._features_dc, pc._features_rest,
                    pc._scaling, pc._rotation, pc._opacity):
        p_group.requires_grad_(False)
    for p in pc._deformation.parameters():
        p.requires_grad_(False)

    # (3) Adam optimizer is given ONLY cloth_logit — physically cannot update others
    #     even if their .grad were non-None (which they won't be due to freeze above).
    opt = torch.optim.Adam([pc._cloth_logit], lr=args.lr)

    print(f"training cloth_logit for {args.iters} steps (lr={args.lr})...", flush=True)

    # (4) Training loop: ~2000 Adam steps. Each step uses the SAME soft_target
    #     (computed once in Stage 2, frozen). Same target every iteration — model
    #     just learns to match it. Typically converges in 200-500 steps.
    for step in range(args.iters):
        # ─ Clear previous gradients (PyTorch accumulates by default)
        opt.zero_grad()

        # ─ Forward: logit ℓ ∈ (-∞,+∞) → probability σ(ℓ) ∈ (0,1).
        #   squeeze(-1) converts (N, 1) → (N,) to match soft_target shape.
        pred = torch.sigmoid(pc._cloth_logit.squeeze(-1))

        # ─ BCE loss (forward):  L = -mean_i[ t_i·log p_i + (1-t_i)·log(1-p_i) ]
        #   .detach() on soft_target: explicitly mark as constant — no gradient
        #   flows back through the projection chain in Stage 2.
        loss = torch.nn.functional.binary_cross_entropy(pred, soft_target.detach())

        # ─ Backprop: autograd computes  ∂L/∂ℓ_i = (σ(ℓ_i) - t_i) / N
        #   Stored in pc._cloth_logit.grad. All other tensors stay grad=None.
        loss.backward()

        # ─ Adam update: ℓ_i ← ℓ_i - lr · m̂_i / √v̂_i (per-parameter adaptive)
        opt.step()

        # ─ Log every 200 steps: BCE value + how many Gaussians are predicted cloth
        if (step + 1) % 200 == 0:
            with torch.no_grad():
                cp = (pred > 0.5).float().mean() * 100   # cloth_pct @ threshold 0.5
            print(f"  step {step+1}/{args.iters}: BCE={loss.item():.4f}  cloth_pct@0.5={cp.item():.1f}%",
                  flush=True)

    # ============ Save new ckpt ============
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # cfg_args copy
    (out / "cfg_args").write_text((base / "cfg_args").read_text())
    # point_cloud dir
    new_iter = args.base_iter
    new_pc_dir = out / "point_cloud" / f"iteration_{new_iter}"
    new_pc_dir.mkdir(parents=True, exist_ok=True)
    # Save updated PLY (with new cloth_logit, K=1)
    pc.save_ply(str(new_pc_dir / "point_cloud.ply"))
    # Copy deformation files (others unchanged)
    import shutil
    for f in ["deformation.pth", "deformation_table.pth", "deformation_accum.pth"]:
        src = ply_path.parent / f
        if src.exists():
            shutil.copy2(src, new_pc_dir / f)
    print(f"saved {out}", flush=True)

    # Also save soft_target as .npy for debug
    np.save(out / "per_gaussian_soft_target.npy", soft_target.cpu().numpy())


if __name__ == "__main__":
    main()
