"""COLMAP sparse-point classification (Plan #3 Phase 1).

For each COLMAP sparse point, compute "cloth probability" by projecting it to
each training (cam, frame) and looking up the SAM3 union mask. Then propagate
the label to baseline 4DGS Gaussians via nearest-neighbor in canonical xyz.

Output: augmented PLY with extra `cloth_logit_0` attribute set to ±10 logit
(hard label) so existing recolor.py works without modification.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).parent.resolve()
FOURDGS_DIR = (HERE / ".." / ".." / "4DGaussians").resolve()
sys.path.insert(0, str(FOURDGS_DIR))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-ckpt", required=True)
    ap.add_argument("--base-iter", type=int, default=14000)
    ap.add_argument("--colmap-ply", required=True,
                    help="COLMAP sparse points (.ply with x,y,z)")
    ap.add_argument("--sam3-scene-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cloth-thresh", type=float, default=0.15,
                    help="cloth probability threshold for COLMAP point classification")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    # ============ Load COLMAP sparse points ============
    from plyfile import PlyData
    p = PlyData.read(args.colmap_ply)["vertex"]
    sparse_xyz = np.stack([p["x"], p["y"], p["z"]], axis=1).astype(np.float32)
    M = len(sparse_xyz)
    print(f"COLMAP: {M:,} sparse points", flush=True)
    sparse_xyz_t = torch.from_numpy(sparse_xyz).to(args.device)

    # ============ Load base ckpt for cameras ============
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
    print(f"baseline 4DGS: N={N:,} Gaussians", flush=True)

    # ============ Load SAM3 masks ============
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
    H_m, W_m = masks_per_cam[0].shape[1], masks_per_cam[0].shape[2]
    print(f"  {n_cams} cams x {n_frames} frames, mask {H_m}x{W_m}", flush=True)

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
    image_poses = train_ds.image_poses
    image_times = train_ds.image_times
    focal = train_ds.focal[0]
    img_shape = (H_m, W_m)
    n_total = len(train_ds)
    from scene.cameras import Camera
    from utils.graphics_utils import focal2fov

    def make_cam_obj(idx):
        R, T = image_poses[idx]
        t = image_times[idx]
        dummy = torch.zeros(3, img_shape[0], img_shape[1])
        return Camera(colmap_id=idx, R=R, T=T,
                      FoVx=focal2fov(focal, img_shape[1]),
                      FoVy=focal2fov(focal, img_shape[0]),
                      image=dummy, gt_alpha_mask=None,
                      image_name=str(idx), uid=idx, time=t)

    # ============ Project each sparse point (static, no deformation) ============
    print("classifying sparse points via projection...", flush=True)
    accum = torch.zeros(M, device=args.device)
    count = torch.zeros(M, device=args.device)
    ones = torch.ones((M, 1), device=args.device)
    t0 = time.time()
    with torch.no_grad():
        for view_idx in range(n_total):
            cam_idx = view_idx // n_frames
            frame_idx = view_idx % n_frames
            cam = make_cam_obj(view_idx)
            fpm = cam.full_proj_transform.to(args.device)
            clip = torch.cat([sparse_xyz_t, ones], dim=-1) @ fpm
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
            if view_idx % 1000 == 0:
                print(f"  view {view_idx+1}/{n_total} ({time.time()-t0:.0f}s)", flush=True)
    sparse_cloth_prob = accum / count.clamp_min(1)
    sparse_is_cloth = sparse_cloth_prob > args.cloth_thresh
    n_cloth_sparse = int(sparse_is_cloth.sum())
    print(f"sparse classification: {n_cloth_sparse}/{M} cloth ({n_cloth_sparse/M*100:.1f}%) "
          f"thresh={args.cloth_thresh}", flush=True)
    print(f"  prob distribution: mean={float(sparse_cloth_prob.mean()):.3f}, "
          f">0.1={int((sparse_cloth_prob > 0.1).sum())}, "
          f">0.2={int((sparse_cloth_prob > 0.2).sum())}, "
          f">0.3={int((sparse_cloth_prob > 0.3).sum())}", flush=True)

    # ============ Propagate to 4DGS Gaussians via nearest-neighbor ============
    print("propagating labels to 4DGS Gaussians (NN in canonical xyz)...", flush=True)
    g_xyz = pc._xyz.detach()  # (N, 3)
    # Compute pairwise distances in chunks
    chunk = 5000
    is_cloth_g = torch.zeros(N, dtype=torch.bool, device=args.device)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        # (chunk, M) distances
        d = torch.cdist(g_xyz[s:e].unsqueeze(0), sparse_xyz_t.unsqueeze(0)).squeeze(0)
        nn_idx = d.argmin(dim=1)
        is_cloth_g[s:e] = sparse_is_cloth[nn_idx]
    n_cloth_g = int(is_cloth_g.sum())
    print(f"  4DGS: {n_cloth_g:,}/{N:,} cloth Gaussians ({n_cloth_g/N*100:.1f}%)", flush=True)

    # ============ Set cloth_logit hard label and save ============
    # +10 logit for cloth, -10 for non-cloth → sigmoid >> 0.99 / << 0.01
    pc._cloth_logit.data = torch.where(is_cloth_g.unsqueeze(-1),
                                        torch.full_like(pc._cloth_logit, 10.0),
                                        torch.full_like(pc._cloth_logit, -10.0))

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
    np.save(out_dir / "sparse_cloth_prob.npy", sparse_cloth_prob.cpu().numpy())
    np.save(out_dir / "sparse_xyz.npy", sparse_xyz)
    np.save(out_dir / "is_cloth_g.npy", is_cloth_g.cpu().numpy())
    print(f"saved {out_dir}", flush=True)


if __name__ == "__main__":
    main()
