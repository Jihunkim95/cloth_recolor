"""exp027: Memory-based negative mining + soft-label calibration (9페이지 core).

D-NeRF only (per user instruction).

Pipeline:
1. Compute per-Gaussian soft_target via projection (exp010 style).
2. Build easy-sample memory:
   - easy_cloth = top-K Gaussians with highest soft_target  (target≈1)
   - easy_noncloth = top-K Gaussians with lowest soft_target (target≈0)
3. For each Gaussian, compute feature distance to its likely-class memory:
   - feat (a): Gaussian-param L2 (14-d concat)
   - feat (b): DINOv3 patch token cosine (768-d, optional)
4. Soft-label calibration:
   - If soft_target > 0.5:  target_cal = 1 - β·sigmoid((d - d_ref)/τ)   ← pull cloth-side toward 0.5 if outlier
   - Else:                  target_cal = 0 + α·sigmoid((d - d_ref)/τ)   ← pull noncloth-side toward 0.5 if outlier
5. BCE on calibrated target.

Ablation flavors: --flavor a / b / both / none (none = exp010 baseline).
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
    ap.add_argument("--sam3-cache", required=True, help="dir containing <scene>/masks.npz")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--flavor", choices=["none", "a", "b", "both"], default="a",
                    help="memory feature: a=Gaussian param, b=DINOv3 patch, both=avg")
    ap.add_argument("--mem-k", type=int, default=1000, help="top-K easy samples per class")
    ap.add_argument("--cal-strength", type=float, default=0.45,
                    help="max calibration magnitude in [0, 0.5]; e.g. 0.45 caps target at [0.05, 0.95]")
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
    if pc._cloth_logit.dim() > 1 and pc._cloth_logit.shape[1] > 1:
        old = pc._cloth_logit.detach()[:, 0:1]
        pc._cloth_logit = torch.nn.Parameter(old.clone().requires_grad_(True))
    elif pc._cloth_logit.numel() == 0:
        pc._cloth_logit = torch.nn.Parameter(torch.zeros(N, 1, device=args.device, requires_grad=True))
    print(f"loaded: N={N:,}, flavor={args.flavor}", flush=True)

    # ============ Load SAM3 masks ============
    z = np.load(Path(args.sam3_cache) / args.scene / "masks.npz")
    T, H_m, W_m = z["shape"].tolist()
    masks_t = torch.from_numpy(
        np.unpackbits(z["masks_packed"], axis=1)[:, :H_m*W_m].reshape(T, H_m, W_m).astype(np.float32)
    ).to(args.device)
    print(f"SAM3: {T}x{H_m}x{W_m}, cov={float(masks_t.mean())*100:.2f}%", flush=True)

    # ============ Build cameras (D-NeRF) ============
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

    # ============ Per-Gaussian soft_target via projection ============
    print("computing per-Gaussian soft_target...", flush=True)
    accum = torch.zeros(N, device=args.device)
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
                                                       pc._features_dc.detach(), times_sel=t)
            except Exception:
                means_t = canonical + pc._deformation.deformation_net(canonical, t)[:, :3]
            ones = torch.ones((N, 1), device=args.device)
            clip = torch.cat([means_t, ones], dim=-1) @ cam.full_proj_transform.to(args.device)
            z = clip[:, 3:].clamp_min(1e-6)
            ndc = torch.nan_to_num(clip[:, :3] / z, nan=0.0, posinf=0.0, neginf=0.0)
            in_view = (clip[:, 3] > 0) & (ndc[:, 0].abs() < 1) & (ndc[:, 1].abs() < 1)
            px = ((ndc[:, 0] + 1) * 0.5 * W_m).long().clamp(0, W_m - 1)
            py = ((ndc[:, 1] + 1) * 0.5 * H_m).long().clamp(0, H_m - 1)
            v = masks_t[i, py, px]
            accum = accum + torch.where(in_view, v, torch.zeros_like(v))
            count = count + in_view.float()
    soft_target = accum / count.clamp_min(1)
    print(f"  soft_target: mean={float(soft_target.mean()):.3f}, "
          f">0.5={int((soft_target > 0.5).sum())} ({float((soft_target > 0.5).float().mean())*100:.1f}%)",
          flush=True)

    # ============ Easy-sample memory + calibration ============
    if args.flavor == "none":
        target_cal = soft_target.clone()
        print("[flavor=none] using raw soft_target (no calibration)", flush=True)
    else:
        # Build features
        from utils.soft_cal import feat_a, DinoV3PatchExtractor
        feats = []
        feat_names = []
        if args.flavor in ("a", "both"):
            fa = feat_a(pc)  # (N, 14)
            feats.append(fa)
            feat_names.append("a")
            print(f"  feat_a: shape={fa.shape}", flush=True)
        if args.flavor in ("b", "both"):
            # Render middle frame for DINOv3 patch extraction
            from gaussian_renderer import render
            pipe_args = type("P", (), {"convert_SHs_python": False, "compute_cov3D_python": False, "debug": False})
            bg = torch.tensor([1, 1, 1] if getattr(cfg_args, "white_background", True) else [0, 0, 0],
                              dtype=torch.float32, device=args.device)
            mid = len(cam_infos) // 2
            ci = cam_infos[mid]
            cam = Camera(colmap_id=mid, R=ci.R, T=ci.T, FoVx=ci.FovX, FoVy=ci.FovY,
                         image=ci.image, gt_alpha_mask=None,
                         image_name=str(mid), uid=mid, time=ci.time)
            with torch.no_grad():
                out = render(cam, pc, pipe_args, bg, stage="fine")
                rgb = out["render"].clamp(0, 1)
            dino = DinoV3PatchExtractor(device=args.device)
            ok = dino.lazy_load()
            if not ok:
                print("  [warn] DINOv3 unavailable, skipping flavor (b)", flush=True)
            else:
                ext = dino.encode_image(rgb)
                if ext is None:
                    print("  [warn] DINOv3 encode failed, skipping flavor (b)", flush=True)
                else:
                    patch_grid, (Hp, Wp), (H_in, W_in) = ext
                    xyz = pc._xyz.detach()
                    ones = torch.ones((N, 1), device=args.device)
                    clip = torch.cat([xyz, ones], dim=-1) @ cam.full_proj_transform.to(args.device)
                    ndc = clip[:, :3] / clip[:, 3:].clamp_min(1e-6)
                    px = ((ndc[:, 0] + 1) * 0.5 * W_in).long().clamp(0, W_in - 1)
                    py = ((ndc[:, 1] + 1) * 0.5 * H_in).long().clamp(0, H_in - 1)
                    ix = (px // dino.patch_size).clamp(0, Wp - 1)
                    iy = (py // dino.patch_size).clamp(0, Hp - 1)
                    fb = patch_grid[iy, ix, :]  # (N, dim)
                    feats.append(fb)
                    feat_names.append("b")
                    print(f"  feat_b: shape={fb.shape}", flush=True)

        if not feats:
            print("  no features available, falling back to flavor=none", flush=True)
            target_cal = soft_target.clone()
        else:
            # easy sample indices
            K = min(args.mem_k, N // 2)
            ec_idx = torch.topk(soft_target, K, largest=True).indices
            en_idx = torch.topk(soft_target, K, largest=False).indices
            print(f"  memory: K={K} cloth (highest soft_target), K={K} noncloth (lowest)", flush=True)

            # Per-Gaussian distance to own-class memory
            alpha = torch.zeros(N, device=args.device)
            cloth_side = soft_target > 0.5
            for fa, name in zip(feats, feat_names):
                ec_f = fa[ec_idx]  # (K, F)
                en_f = fa[en_idx]
                # distance from each Gaussian to nearest in its likely-class memory
                # cloth-side: dist to ec; noncloth-side: dist to en
                d_c = torch.cdist(fa, ec_f).min(dim=1).values  # (N,)
                d_n = torch.cdist(fa, en_f).min(dim=1).values
                d = torch.where(cloth_side, d_c, d_n)
                # calibration scale: dref=median(d on hard samples), tau=std
                hard_mask = (soft_target > 0.3) & (soft_target < 0.7)
                if int(hard_mask.sum()) > 32:
                    d_hard = torch.where(cloth_side[hard_mask],
                                         d_c[hard_mask], d_n[hard_mask])
                    dref = float(d_hard.median())
                    tau = max(float(d_hard.std()), 1e-3)
                else:
                    dref = float(d.median())
                    tau = max(float(d.std()), 1e-3)
                alpha_f = args.cal_strength * torch.sigmoid((d - dref) / tau)  # ∈ [0, cal_strength]
                alpha = alpha + alpha_f
                print(f"  flavor[{name}]: d_ref={dref:.3f}, τ={tau:.3f}, "
                      f"alpha mean={float(alpha_f.mean()):.3f}", flush=True)
            alpha = alpha / len(feats)  # average across enabled flavors
            # Apply calibration: cloth-side target = 1 - α; noncloth-side = 0 + α
            target_cal = torch.where(cloth_side, 1.0 - alpha, alpha)
            print(f"  target_cal: mean={float(target_cal.mean()):.3f}, "
                  f">0.5={int((target_cal > 0.5).sum())}", flush=True)

    # ============ Train cloth_logit ============
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
        loss = F.binary_cross_entropy(pred, target_cal.detach())
        loss.backward()
        opt.step()
        if (step + 1) % 200 == 0:
            with torch.no_grad():
                cp = (pred > 0.5).float().mean() * 100
            print(f"  step {step+1}: BCE={loss.item():.4f}  cloth_pct@0.5={cp.item():.1f}%",
                  flush=True)

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
    np.save(out_dir / "soft_target_raw.npy", soft_target.cpu().numpy())
    np.save(out_dir / "soft_target_cal.npy", target_cal.cpu().numpy())
    (out_dir / "calib_meta.json").write_text(json.dumps({
        "flavor": args.flavor, "mem_k": args.mem_k, "cal_strength": args.cal_strength}))
    print(f"saved {out_dir}", flush=True)


if __name__ == "__main__":
    main()
