"""exp028 v2: Loss search with **multi-camera SAM3 mapping validation in the loop**.

Core idea: while training cloth_logit, periodically render the prediction from
multiple training cameras and compare against the ground SAM3 mask at each
(uid → mask). The mapping IoU across K validation cameras tells us *did the
3D label correctly map back to all 2D views*. Best ckpt selected by val IoU.

Loss menu:
  L_BCE     : per-Gaussian projection BCE (exp010 baseline). Always on.
  L_smooth  : k-NN cloth_logit coherence — penalizes isolated misclassifications.
  L_mapping : soft Dice loss between rendered sigmoid(cloth_logit) and SAM3 mask
              at K randomly sampled training cameras per step. Differentiable
              through gsplat 2nd-pass. Less over-cover than alpha-composit BCE
              because Dice balances precision/recall.

Validation (every --val-every steps):
  val_iou = mean IoU(rendered_pred > 0.5, SAM3_mask) across all training cams
  best ckpt = ckpt with highest val_iou

Output: includes per-step training log + val_iou trajectory + best ckpt path.
"""
from __future__ import annotations
import argparse, json, sys, os, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).parent.resolve()
FOURDGS_DIR = (HERE / ".." / ".." / "4DGaussians").resolve()
sys.path.insert(0, str(FOURDGS_DIR))


def build_knn_indices(xyz: torch.Tensor, k: int) -> torch.Tensor:
    N = xyz.shape[0]
    nn = torch.zeros((N, k), dtype=torch.long, device=xyz.device)
    chunk = 4096
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        d = torch.cdist(xyz[s:e].unsqueeze(0), xyz.unsqueeze(0)).squeeze(0)
        for i in range(e - s):
            d[i, s + i] = float("inf")
        nn[s:e] = d.topk(k, largest=False).indices
    return nn


def soft_dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft Dice loss = 1 − 2 * Σ(p·t) / (Σp + Σt). pred ∈ [0,1], target ∈ {0,1}."""
    inter = (pred * target).sum()
    return 1.0 - (2.0 * inter + eps) / (pred.sum() + target.sum() + eps)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-ckpt", required=True)
    ap.add_argument("--base-iter", type=int, default=20000)
    ap.add_argument("--sam3-cache", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--w-smooth", type=float, default=0.0)
    ap.add_argument("--w-mapping", type=float, default=0.0,
                    help="weight on multi-camera rendered Dice loss")
    ap.add_argument("--knn-k", type=int, default=16)
    ap.add_argument("--mapping-batch", type=int, default=2,
                    help="cameras sampled per step for L_mapping")
    ap.add_argument("--val-every", type=int, default=100,
                    help="run val IoU eval every N steps (and save best ckpt)")
    ap.add_argument("--val-cams", type=int, default=16,
                    help="cameras used for validation eval")
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
        pc._cloth_logit = torch.nn.Parameter(pc._cloth_logit.detach()[:, 0:1].clone().requires_grad_(True))
    elif pc._cloth_logit.numel() == 0:
        pc._cloth_logit = torch.nn.Parameter(torch.zeros(N, 1, device=args.device, requires_grad=True))
    print(f"loaded: N={N:,}  loss: BCE smooth={args.w_smooth} mapping={args.w_mapping}",
          flush=True)

    # ============ Load SAM3 masks ============
    z = np.load(Path(args.sam3_cache) / args.scene / "masks.npz")
    T, H_m, W_m = z["shape"].tolist()
    masks_t = torch.from_numpy(
        np.unpackbits(z["masks_packed"], axis=1)[:, :H_m*W_m].reshape(T, H_m, W_m).astype(np.float32)
    ).to(args.device)
    print(f"SAM3: {T}×{H_m}×{W_m}, cov={float(masks_t.mean())*100:.2f}%", flush=True)

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

    def make_cam(idx):
        ci = cam_infos[idx]
        return Camera(colmap_id=idx, R=ci.R, T=ci.T, FoVx=ci.FovX, FoVy=ci.FovY,
                      image=ci.image, gt_alpha_mask=None,
                      image_name=str(idx), uid=idx, time=ci.time)

    # ============ Per-Gaussian soft_target via projection ============
    print("computing per-Gaussian soft_target...", flush=True)
    accum = torch.zeros(N, device=args.device)
    count = torch.zeros(N, device=args.device)
    canonical = pc._xyz.detach()
    with torch.no_grad():
        for i in range(T):
            cam = make_cam(i)
            t = torch.full((N, 1), float(cam.time), device=args.device)
            try:
                means_t, _, _, _, _ = pc._deformation(canonical, pc._scaling.detach(),
                                                       pc._rotation.detach(), pc._opacity.detach(),
                                                       pc._features_dc.detach(), times_sel=t)
            except Exception:
                means_t = canonical + pc._deformation.deformation_net(canonical, t)[:, :3]
            ones = torch.ones((N, 1), device=args.device)
            clip = torch.cat([means_t, ones], dim=-1) @ cam.full_proj_transform.to(args.device)
            zw = clip[:, 3:].clamp_min(1e-6)
            ndc = torch.nan_to_num(clip[:, :3] / zw, nan=0.0, posinf=0.0, neginf=0.0)
            in_view = (clip[:, 3] > 0) & (ndc[:, 0].abs() < 1) & (ndc[:, 1].abs() < 1)
            px = ((ndc[:, 0] + 1) * 0.5 * W_m).long().clamp(0, W_m - 1)
            py = ((ndc[:, 1] + 1) * 0.5 * H_m).long().clamp(0, H_m - 1)
            v = masks_t[i, py, px]
            accum = accum + torch.where(in_view, v, torch.zeros_like(v))
            count = count + in_view.float()
    soft_target = accum / count.clamp_min(1)
    print(f"  soft_target mean={float(soft_target.mean()):.3f} >0.5={int((soft_target>0.5).sum())}",
          flush=True)

    # ============ k-NN for L_smooth ============
    knn_idx = None
    if args.w_smooth > 0:
        print(f"building k-NN (k={args.knn_k})...", flush=True)
        knn_idx = build_knn_indices(canonical, args.knn_k)

    # ============ Renderer setup ============
    from gaussian_renderer import render
    pipe_args = type("P", (), {"convert_SHs_python": False, "compute_cov3D_python": False, "debug": False})
    bg = torch.tensor([1, 1, 1] if getattr(cfg_args, "white_background", True) else [0, 0, 0],
                      dtype=torch.float32, device=args.device)

    def render_cloth_prob_at(idx, requires_grad=False):
        """Render the cloth_logit channel at training cam `idx`, return (H_pred, W_pred) prob map."""
        cam = make_cam(idx)
        if requires_grad:
            out = render(cam, pc, pipe_args, bg, stage="fine")
        else:
            with torch.no_grad():
                out = render(cam, pc, pipe_args, bg, stage="fine")
        cl = out.get("cloth_logit")
        if cl is None:
            return None
        if cl.dim() == 3:
            cl = cl[..., 0]
        return torch.sigmoid(cl)

    def val_iou(sample_cams=None) -> float:
        """Compute mean IoU(rendered_pred > 0.5, SAM3_mask) across cams."""
        if sample_cams is None:
            sample_cams = list(range(T))
        ious = []
        for idx in sample_cams:
            prob = render_cloth_prob_at(idx, requires_grad=False)
            if prob is None:
                continue
            pred_mask = (prob > 0.5).float()
            tgt = masks_t[idx]
            if pred_mask.shape != tgt.shape:
                pred_mask = F.interpolate(pred_mask.unsqueeze(0).unsqueeze(0),
                                          size=tgt.shape, mode="nearest").squeeze()
            inter = (pred_mask * tgt).sum().item()
            union = (pred_mask + tgt - pred_mask * tgt).sum().item()
            ious.append(inter / (union + 1e-6))
        return float(np.mean(ious)) if ious else 0.0

    val_cam_idxs = np.linspace(0, T - 1, min(args.val_cams, T)).astype(int).tolist()
    print(f"val cams: {len(val_cam_idxs)} (idx={val_cam_idxs[:3]}...{val_cam_idxs[-2:]})", flush=True)

    # ============ Train ============
    pc._cloth_logit.requires_grad_(True)
    for p_group in (pc._xyz, pc._features_dc, pc._features_rest,
                    pc._scaling, pc._rotation, pc._opacity):
        p_group.requires_grad_(False)
    for p in pc._deformation.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam([pc._cloth_logit], lr=args.lr)

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    best_val_iou = -1.0
    best_logit = pc._cloth_logit.detach().clone()
    val_log = []  # list of (step, val_iou)

    print(f"training {args.iters} steps...", flush=True)
    t0 = time.time()
    for step in range(args.iters):
        opt.zero_grad()
        pred_logit = pc._cloth_logit.squeeze(-1)
        pred = torch.sigmoid(pred_logit)

        l_bce = F.binary_cross_entropy(pred, soft_target.detach())

        l_smooth = torch.tensor(0.0, device=args.device)
        if args.w_smooth > 0 and knn_idx is not None:
            neigh_mean = pred_logit[knn_idx].mean(dim=1)
            l_smooth = (pred_logit - neigh_mean).pow(2).mean()

        l_mapping = torch.tensor(0.0, device=args.device)
        if args.w_mapping > 0:
            dice_losses = []
            sampled = np.random.choice(T, args.mapping_batch, replace=False)
            for idx in sampled:
                prob = render_cloth_prob_at(int(idx), requires_grad=True)
                if prob is None:
                    continue
                tgt = masks_t[int(idx)]
                if prob.shape != tgt.shape:
                    prob = F.interpolate(prob.unsqueeze(0).unsqueeze(0),
                                         size=tgt.shape, mode="bilinear", align_corners=False).squeeze()
                dice_losses.append(soft_dice_loss(prob, tgt))
            if dice_losses:
                l_mapping = torch.stack(dice_losses).mean()

        loss = l_bce + args.w_smooth * l_smooth + args.w_mapping * l_mapping
        loss.backward()
        opt.step()

        # Validation
        if (step + 1) % args.val_every == 0 or step == args.iters - 1:
            v = val_iou(val_cam_idxs)
            val_log.append((step + 1, v))
            cp = (pred.detach() > 0.5).float().mean() * 100
            marker = ""
            if v > best_val_iou:
                best_val_iou = v
                best_logit = pc._cloth_logit.detach().clone()
                marker = "  ★ NEW BEST"
            print(f"  step {step+1}: loss={loss.item():.4f}  bce={l_bce.item():.4f}  "
                  f"smooth={float(l_smooth):.4f}  map={float(l_mapping):.4f}  "
                  f"cloth@0.5={cp.item():.1f}%  val_iou={v:.4f}{marker}  [{time.time()-t0:.0f}s]",
                  flush=True)

    # ============ Save best ============
    pc._cloth_logit.data = best_logit
    (out_dir / "cfg_args").write_text((base / "cfg_args").read_text())
    new_pc_dir = out_dir / "point_cloud" / f"iteration_{args.base_iter}"
    new_pc_dir.mkdir(parents=True, exist_ok=True)
    pc.save_ply(str(new_pc_dir / "point_cloud.ply"))
    import shutil
    for f in ["deformation.pth", "deformation_table.pth", "deformation_accum.pth"]:
        src = ply_path.parent / f
        if src.exists():
            shutil.copy2(src, new_pc_dir / f)
    np.save(out_dir / "soft_target.npy", soft_target.cpu().numpy())
    (out_dir / "loss_config.json").write_text(json.dumps({
        "w_smooth": args.w_smooth, "w_mapping": args.w_mapping, "knn_k": args.knn_k,
        "iters": args.iters, "lr": args.lr,
        "best_val_iou": best_val_iou,
        "val_log": val_log,
    }, indent=2))
    print(f"saved {out_dir}  (best val_iou={best_val_iou:.4f})", flush=True)


if __name__ == "__main__":
    main()
