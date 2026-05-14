"""Tier 2 HSV recolor for trained 4DGaussians+cloth_logit checkpoint.

Loads PLY (now contains cloth_logit attribute) + deformation network + scene cameras.
For each Gaussian where sigmoid(cloth_logit) > threshold, swap SH_DC Hue, then re-render
sample (cam, frame) views.

Usage:
    python tier2_4dgs_recolor.py \\
        --ckpt-dir ../4DGaussians/output/00169_tier2 \\
        --hue 220 --threshold 0.5 \\
        --out vis/recolor_4dgs_tier2_blue --num-frames 4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData

HERE = Path(__file__).parent.resolve()
FOURDGS_DIR = (HERE / ".." / ".." / "4DGaussians").resolve()
sys.path.insert(0, str(FOURDGS_DIR))

C0 = 0.28209479177387814


def rgb_to_hsv_np(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    cmax = rgb.max(axis=-1); cmin = rgb.min(axis=-1)
    delta = cmax - cmin
    h = np.zeros_like(cmax)
    nz = delta > 1e-8
    rm = nz & (cmax == r); gm = nz & (cmax == g) & ~rm; bm = nz & (cmax == b) & ~rm & ~gm
    h[rm] = ((g[rm] - b[rm]) / delta[rm]) % 6
    h[gm] = (b[gm] - r[gm]) / delta[gm] + 2
    h[bm] = (r[bm] - g[bm]) / delta[bm] + 4
    h /= 6
    s = np.where(cmax > 1e-8, delta / np.maximum(cmax, 1e-8), 0.0)
    return np.stack([h, s, cmax], axis=-1)


def hsv_to_rgb_np(hsv):
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    h6 = (h % 1.0) * 6.0
    i = np.floor(h6).astype(np.int32) % 6
    f = h6 - np.floor(h6)
    p = v * (1 - s); q = v * (1 - s * f); t = v * (1 - s * (1 - f))
    out = np.zeros_like(hsv)
    for ci, stk in [(0, (v, t, p)), (1, (q, v, p)), (2, (p, v, t)),
                     (3, (p, q, v)), (4, (t, p, v)), (5, (v, p, q))]:
        m = i == ci
        if m.any():
            out[m] = np.stack([stk[0][m], stk[1][m], stk[2][m]], axis=-1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--iter", type=int, default=14000)
    ap.add_argument("--hue", type=float, default=220.0)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--target-class", type=int, default=0,
                    help="for K>1 cloth_logit: which class to recolor (0=cloth/first explicit class)")
    ap.add_argument("--num-frames", type=int, default=4, help="frames per cam to render")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--save-ply", action="store_true",
                    help="also write recolored point_cloud.ply (cloth Gaussians' SH baked to new hue)")
    ap.add_argument("--spatial-filter", type=float, default=0.0,
                    help="MAD multiplier for spatial outlier filter; 0=off, 3=typical")
    ap.add_argument("--min-sat", type=float, default=0.0,
                    help="force saturation to at least this (0=disabled, 0.5+ makes black clothes visible)")
    ap.add_argument("--min-val", type=float, default=0.0,
                    help="force V (brightness) to at least this (0=disabled). For black cloth recolor, set 0.5+")
    args = ap.parse_args()

    ckpt = Path(args.ckpt_dir)
    from argparse import Namespace  # used by cfg_args.eval
    cfg_args = eval((ckpt / "cfg_args").read_text())

    # Load model
    from scene.gaussian_model import GaussianModel
    pc = GaussianModel(cfg_args.sh_degree, cfg_args)
    ply_path = ckpt / "point_cloud" / f"iteration_{args.iter}" / "point_cloud.ply"
    pc.load_ply(str(ply_path))
    pc.load_model(str(ply_path.parent))
    N = pc._xyz.shape[0]

    # Threshold cloth Gaussians
    cl = pc._cloth_logit.detach()  # (N, K)
    K = cl.shape[1] if cl.dim() > 1 else 1
    if K == 1:
        cloth_prob = torch.sigmoid(cl).squeeze(-1)
    else:
        # multi-class: softmax → select target class probability, mask where class wins
        sm = torch.softmax(cl, dim=1)
        argmax_class = sm.argmax(dim=1)
        cloth_prob = torch.where(argmax_class == args.target_class,
                                 sm[:, args.target_class],
                                 torch.zeros_like(sm[:, 0]))
    cloth_mask = cloth_prob > args.threshold
    n_cloth = int(cloth_mask.sum())
    print(f"N={N:,} cloth_gaussians={n_cloth:,} ({n_cloth/N*100:.1f}%) hue={args.hue}°")

    if args.spatial_filter > 0 and n_cloth > 10:
        xyz = pc._xyz.detach()
        xyz_c = xyz[cloth_mask]
        med = xyz_c.median(dim=0).values
        dist = (xyz_c - med).norm(dim=1)
        mad = (dist - dist.median()).abs().median()
        max_dist = dist.median() + args.spatial_filter * mad.clamp_min(1e-6)
        keep = dist <= max_dist
        idx = cloth_mask.nonzero(as_tuple=False).squeeze(-1)
        new_mask = torch.zeros_like(cloth_mask)
        new_mask[idx[keep]] = True
        n_kept = int(new_mask.sum())
        print(f"  spatial filter k={args.spatial_filter}: {n_cloth} → {n_kept} "
              f"(removed {n_cloth-n_kept}, median_dist={float(dist.median()):.3f}, mad={float(mad):.3f})")
        cloth_mask = new_mask

    # Recolor SH_DC
    sh_dc_orig = pc._features_dc.detach().clone()  # (N, 1, 3)
    rgb = (sh_dc_orig[:, 0] * C0 + 0.5).clamp(0, 1).cpu().numpy()  # (N, 3)
    hsv = rgb_to_hsv_np(rgb)
    new_hsv = hsv.copy()
    new_hsv[..., 0] = args.hue / 360.0
    if args.min_sat > 0:
        new_hsv[..., 1] = np.maximum(new_hsv[..., 1], args.min_sat)
    if args.min_val > 0:
        new_hsv[..., 2] = np.maximum(new_hsv[..., 2], args.min_val)
    new_rgb = hsv_to_rgb_np(new_hsv).clip(0, 1)
    new_sh_dc = (new_rgb - 0.5) / C0
    new_sh_dc_t = torch.from_numpy(new_sh_dc.astype(np.float32)).to(args.device)
    sh_dc_recolored = sh_dc_orig.clone()
    sh_dc_recolored[cloth_mask, 0] = new_sh_dc_t[cloth_mask]

    # Build cameras — handle multipleview vs Blender (D-NeRF)
    import os
    src_path = cfg_args.source_path
    if not os.path.isabs(src_path):
        src_path = str((FOURDGS_DIR / src_path).resolve())
    from scene.cameras import Camera
    from utils.graphics_utils import focal2fov

    is_dnerf = (Path(src_path) / "transforms_train.json").exists()
    # N3V detection: source dir contains cam*.mp4 files
    is_n3v = bool(list(Path(src_path).glob("cam*.mp4"))) and not is_dnerf
    if is_n3v:
        # Neural 3D Video — use Neural3D_NDC_Dataset to get poses + times
        from scene.neural_3D_dataset_NDC import Neural3D_NDC_Dataset
        train_ds = Neural3D_NDC_Dataset(
            src_path, "train", 1.0, time_scale=1,
            scene_bbox_min=[-2.5, -2.0, -1.0], scene_bbox_max=[2.5, 2.0, 1.0],
            eval_index=0,
        )
        image_poses = train_ds.image_poses
        image_times = train_ds.image_times
        focal = train_ds.focal[0]
        # use a typical N3V resolution
        H_img, W_img = 1014, 1352   # downsampled N3V; will be overridden by dummy image size

        def make_camera(idx):
            R, T = image_poses[idx]
            t = image_times[idx]
            dummy = torch.zeros(3, H_img, W_img)
            return Camera(colmap_id=idx, R=R, T=T,
                          FoVx=focal2fov(focal, W_img),
                          FoVy=focal2fov(focal, H_img),
                          image=dummy, gt_alpha_mask=None,
                          image_name=str(idx), uid=idx, time=t)

        n_total = len(train_ds)
        panel_idx = np.linspace(0, n_total - 1, args.num_frames * 4).astype(int).tolist()[: args.num_frames * 4]
    elif is_dnerf:
        # Blender / D-NeRF: load transforms_train.json directly, build Camera per frame
        from scene.dataset_readers import readCamerasFromTransforms
        ts = json.loads((Path(src_path) / "transforms_train.json").read_text())
        # 4DGaussians' loader assigns timestamps from a mapper
        max_time = max(f.get("time", 0.0) for f in ts["frames"])
        # mapper keys are the time VALUE, not file_path (per readCamerasFromTransforms)
        timestamp_mapper = {f["time"]: (f["time"] / max_time if max_time > 0 else 0.0)
                            for f in ts["frames"]}
        cam_infos = readCamerasFromTransforms(
            src_path, "transforms_train.json",
            white_background=getattr(cfg_args, "white_background", True),
            extension=".png", mapper=timestamp_mapper,
        )

        def make_camera(idx):
            ci = cam_infos[idx]
            return Camera(colmap_id=ci.uid, R=ci.R, T=ci.T,
                          FoVx=ci.FovX, FoVy=ci.FovY,
                          image=ci.image, gt_alpha_mask=None,
                          image_name=ci.image_name, uid=ci.uid, time=ci.time)

        n_total = len(cam_infos)
        # pick num_frames evenly spaced
        panel_idx = np.linspace(0, n_total - 1, args.num_frames * 4).astype(int).tolist()[: args.num_frames * 4]
    else:
        from scene.dataset_readers import readMultipleViewinfos
        scene_info = readMultipleViewinfos(src_path)
        raw_train = scene_info.train_cameras
        focal0 = raw_train.focal[0]

        def make_camera(idx):
            img, (R, T), t = raw_train[idx]
            return Camera(colmap_id=idx, R=R, T=T,
                          FoVx=focal2fov(focal0, img.shape[2]),
                          FoVy=focal2fov(focal0, img.shape[1]),
                          image=img, gt_alpha_mask=None,
                          image_name=str(idx), uid=idx, time=t)

        n_total = len(raw_train)
        n_cams = 4
        n_per_cam = n_total // n_cams
        panel_idx = []
        for ci in range(n_cams):
            for fi in np.linspace(0, n_per_cam - 1, args.num_frames).astype(int):
                panel_idx.append(ci * n_per_cam + int(fi))

    # Render
    from gaussian_renderer import render
    pipe_args = type("P", (), {"convert_SHs_python": False, "compute_cov3D_python": False, "debug": False})
    bg = torch.zeros(3, device=args.device)

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    n_panels = 0
    with torch.no_grad():
        for k in panel_idx:
            try:
                vp = make_camera(k)
                pc._features_dc.data = sh_dc_orig
                r1 = render(vp, pc, pipe_args, bg, stage="fine")
                rgb_o = (r1["render"].detach().clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                # Mask overlay (cloth_logit channel rendered if available)
                cl_img = r1.get("cloth_logit")
                if cl_img is not None:
                    mask_render = (torch.sigmoid(cl_img.detach()) > 0.5).cpu().numpy()
                    overlay = rgb_o.astype(np.float32).copy()
                    overlay[mask_render] = (overlay[mask_render] * 0.55
                                            + np.array([0, 255, 0], dtype=np.float32) * 0.45)
                    overlay = overlay.clip(0, 255).astype(np.uint8)
                else:
                    overlay = rgb_o
                pc._features_dc.data = sh_dc_recolored
                r2 = render(vp, pc, pipe_args, bg, stage="fine")
                rgb_r = (r2["render"].detach().clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                pc._features_dc.data = sh_dc_orig
                panel = np.concatenate([rgb_o, rgb_r, overlay], axis=1)
                Image.fromarray(panel).save(out_dir / f"panel_{k:04d}.png")
                n_panels += 1
            except Exception as e:
                print(f"  fail @ idx={k}: {type(e).__name__}: {e}")

    # optional: bake recolor into a new PLY file
    if args.save_ply:
        pc._features_dc.data = sh_dc_recolored
        ply_out = out_dir / f"point_cloud_recolored_hue{int(args.hue)}_th{args.threshold}.ply"
        pc.save_ply(str(ply_out))
        pc._features_dc.data = sh_dc_orig
        print(f"  recolored PLY saved: {ply_out}")

    summary = {
        "ckpt_dir": str(ckpt), "n_gaussians": N, "n_cloth": n_cloth,
        "cloth_pct": round(n_cloth / N * 100, 2),
        "hue": args.hue, "threshold": args.threshold, "n_panels": n_panels,
        "iter": args.iter,
        "ply_saved": bool(args.save_ply),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n{n_panels} panels saved → {out_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
