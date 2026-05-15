"""Per-Gaussian supervision using 4D-DRESS GT vertex labels (no projection, no SAM3).

For each Gaussian g and each frame t:
  - Load mesh-fXXXXX.pkl (V_t shape (n_vt, 3)) and label-fXXXXX.pkl (L_t shape (n_vt,))
  - Deform g.xyz to time t via deformation MLP → g_t
  - Find nearest vertex idx i* = argmin ||V_t - g_t||
  - Inherit label L_t[i*] ∈ {0,1,2,3,4,5}
Aggregate label-frequency per Gaussian → 6-d soft target. Then train cloth_logit (N, 1)
with BCE against (label == target_label) for a specific cloth class.

Usage:
    python per_gaussian_supervision_4ddress_gt.py \\
        --base-ckpt 3_output/00169_Outer_Take12/ckpt_baseline_4ddress \\
        --base-iter 14000 \\
        --semantic-root /NHNHOME/.../4D-DRESS_extracted/00169/Outer/Take12 \\
        --target-label 5 \\
        --out 3_output/00169_Outer_Take12/ckpt_exp012_gt_label5 \\
        --iters 2000
"""
from __future__ import annotations

import argparse
import pickle
import sys
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
    ap.add_argument("--semantic-root", required=True,
                    help="dir containing Meshes_pkl/ + Semantic/labels/")
    ap.add_argument("--target-label", type=int, default=5,
                    help="which vertex label to treat as cloth (default 5 = outer garment)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--gauss-chunk", type=int, default=8192,
                    help="N Gaussians processed per cdist batch")
    ap.add_argument("--max-dist", type=float, default=0.05,
                    help="max distance (scene units) from Gaussian to mesh to assign label; "
                         "Gaussians farther than this counted as background (label=0)")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    # ============ Load ckpt ============
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
    print(f"loaded ckpt: N={N:,} Gaussians", flush=True)

    if pc._cloth_logit.dim() > 1 and pc._cloth_logit.shape[1] > 1:
        old = pc._cloth_logit.detach()[:, 0:1]
        pc._cloth_logit = torch.nn.Parameter(old.clone().requires_grad_(True))

    # ============ Enumerate per-frame mesh + label files ============
    sem_root = Path(args.semantic_root)
    mesh_files = sorted((sem_root / "Meshes_pkl").glob("mesh-f*.pkl"))
    label_files = sorted((sem_root / "Semantic" / "labels").glob("label-f*.pkl"))
    assert len(mesh_files) > 0 and len(label_files) == len(mesh_files), \
        f"#meshes={len(mesh_files)} #labels={len(label_files)}"
    print(f"frames: {len(mesh_files)} mesh+label pairs", flush=True)

    # ============ For each frame, find each Gaussian's nearest vertex & inherit label ============
    print("computing per-Gaussian label aggregation...", flush=True)
    canonical = pc._xyz.detach().to(args.device)
    label_counts = torch.zeros((N, 6), device=args.device)   # 6 classes
    n_frames = len(mesh_files)
    # Read transforms_train-like time mapping: 4D-DRESS uses frame_id range, time = (frame_idx - first) / (last - first)
    frame_indices = [int(f.stem.split("-f")[1]) for f in mesh_files]
    min_f, max_f = min(frame_indices), max(frame_indices)
    times_norm = [(fi - min_f) / max(1, max_f - min_f) for fi in frame_indices]

    with torch.no_grad():
        for fi, (mf, lf) in enumerate(zip(mesh_files, label_files)):
            t_norm = times_norm[fi]
            t_tensor = torch.full((N, 1), float(t_norm), device=args.device)
            # Deform Gaussians
            try:
                g_t, _, _, _, _ = pc._deformation(canonical, pc._scaling.detach(),
                                                  pc._rotation.detach(), pc._opacity.detach(),
                                                  pc._features_dc.detach(),
                                                  times_sel=t_tensor)
            except Exception:
                g_t = canonical + pc._deformation.deformation_net(canonical, t_tensor)[:, :3]
            # Load mesh + labels
            with open(mf, 'rb') as f: mesh = pickle.load(f)
            with open(lf, 'rb') as f: lab = pickle.load(f)
            V = torch.from_numpy(mesh['vertices'].astype(np.float32)).to(args.device)
            L = torch.from_numpy(lab['scan_labels'].astype(np.int64)).to(args.device)
            # Batched NN with distance threshold
            for s in range(0, N, args.gauss_chunk):
                e = min(s + args.gauss_chunk, N)
                d = torch.cdist(g_t[s:e], V)              # (chunk, n_vt)
                d_min, nn = d.min(dim=1)                   # (chunk,) each
                lab_per = L[nn]                            # (chunk,)
                # Gaussians too far → treat as background (label 0)
                lab_per = torch.where(d_min < args.max_dist, lab_per, torch.zeros_like(lab_per))
                # accumulate one-hot count
                label_counts[s:e].scatter_add_(1, lab_per.unsqueeze(1),
                                                torch.ones_like(lab_per, dtype=torch.float32).unsqueeze(1))
            if fi % 20 == 0:
                print(f"  frame {fi+1}/{n_frames}", flush=True)

    # Normalize per Gaussian
    label_probs = label_counts / label_counts.sum(dim=1, keepdim=True).clamp_min(1)
    target_label = args.target_label
    soft_target = label_probs[:, target_label]
    print(f"per-Gaussian label distribution (averaged):", flush=True)
    print(f"  class 0..5 prevalence: {label_probs.mean(0).cpu().numpy()}", flush=True)
    print(f"  target_label={target_label}: soft_target mean={float(soft_target.mean()):.3f}, "
          f">0.5={int((soft_target > 0.5).sum())} ({float((soft_target > 0.5).float().mean())*100:.1f}%)",
          flush=True)

    # ============ Train cloth_logit ============
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
    np.save(out / "per_gaussian_label_probs.npy", label_probs.cpu().numpy())
    np.save(out / "per_gaussian_soft_target.npy", soft_target.cpu().numpy())
    # Meta
    (out / "exp_meta.json").write_text(__import__("json").dumps({
        "target_label": target_label,
        "n_frames": n_frames,
        "n_gaussians": N,
        "label_distribution_mean": label_probs.mean(0).cpu().numpy().tolist(),
        "cloth_pct_at_0.5": float((soft_target > 0.5).float().mean()) * 100,
    }, indent=2))
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
