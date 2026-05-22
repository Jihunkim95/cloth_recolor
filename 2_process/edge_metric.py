"""exp024: Edge metric for cloth/non-cloth boundary alignment (GT-free).

Idea: a "good" cloth_logit should have its spatial gradient peak ALIGNED with
the original RGB image edge at the cloth/non-cloth boundary. We:

1. Render 4DGS RGB image + cloth_logit channel for several views/frames.
2. Compute Sobel edge on RGB → E_rgb (binary at tau_rgb).
3. Compute Sobel edge on sigmoid(cloth_logit) → E_pred (binary at tau_pred).
4. Metrics:
   - **Edge IoU @ tau**: |E_rgb ∩ E_pred| / |E_rgb ∪ E_pred| within mask dilation
   - **Chamfer distance**: mean distance from each E_pred pixel to nearest E_rgb pixel,
     restricted to pixels within k-pixel band around SAM3 mask boundary
5. Qualitative: zoom-in overlays of (RGB, cloth_logit heatmap, edge overlay) per scene.
"""
from __future__ import annotations
import argparse, sys, json, os, math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).parent.resolve()
FOURDGS_DIR = (HERE / ".." / ".." / "4DGaussians").resolve()
sys.path.insert(0, str(FOURDGS_DIR))


def sobel_edge(img: np.ndarray, tau: float = 0.1) -> np.ndarray:
    """Sobel magnitude → binary edge. img: (H, W, 3) float [0,1] or (H, W) float."""
    import cv2
    if img.ndim == 3:
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    else:
        gray = img.astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    return (mag > tau).astype(np.uint8)


def chamfer_distance(edge_a: np.ndarray, edge_b: np.ndarray) -> float:
    """Mean distance from edge_a pixels to nearest edge_b pixel (cv2 distance transform)."""
    import cv2
    if edge_a.sum() == 0 or edge_b.sum() == 0:
        return float("nan")
    # distance transform of !edge_b → gives nearest distance from each pixel to edge_b
    dist = cv2.distanceTransform((1 - edge_b).astype(np.uint8), cv2.DIST_L2, 3)
    return float(dist[edge_a.astype(bool)].mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--iter", type=int, default=14000)
    ap.add_argument("--sam3-cache", help="optional SAM3 mask npz for boundary band restriction")
    ap.add_argument("--use-sam3-as-gt", action="store_true",
                    help="use SAM3 mask boundary as pseudo-GT (more accurate proxy than RGB edge)")
    ap.add_argument("--n-frames", type=int, default=8, help="number of views to evaluate")
    ap.add_argument("--tau-rgb", type=float, default=0.15)
    ap.add_argument("--tau-pred", type=float, default=0.1)
    ap.add_argument("--band-px", type=int, default=10, help="restrict metric to ±band_px around SAM3 boundary")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from argparse import Namespace
    base = Path(args.ckpt_dir)
    cfg_args = eval((base / "cfg_args").read_text())
    cfg_args.num_classes = 1
    from scene.gaussian_model import GaussianModel
    pc = GaussianModel(cfg_args.sh_degree, cfg_args)
    ply_path = base / "point_cloud" / f"iteration_{args.iter}" / "point_cloud.ply"
    pc.load_ply(str(ply_path))
    pc.load_model(str(ply_path.parent))
    print(f"loaded {base.name}: N={pc._xyz.shape[0]:,}", flush=True)

    # Build cameras — handle D-NeRF vs N3V
    src_path = cfg_args.source_path
    if not os.path.isabs(src_path):
        src_path = str((FOURDGS_DIR / src_path).resolve())

    is_dnerf = (Path(src_path) / "transforms_train.json").exists()
    is_n3v = (Path(src_path) / "cam00" / "images").exists()
    from scene.cameras import Camera
    from utils.graphics_utils import focal2fov

    cams_list = []
    if is_dnerf:
        from scene.dataset_readers import readCamerasFromTransforms
        ts = json.loads((Path(src_path) / "transforms_train.json").read_text())
        max_time = max(f.get("time", 0.0) for f in ts["frames"])
        mapper = {f["time"]: (f["time"]/max_time if max_time > 0 else 0.0) for f in ts["frames"]}
        cis = readCamerasFromTransforms(src_path, "transforms_train.json",
                                         white_background=getattr(cfg_args, "white_background", True),
                                         extension=".png", mapper=mapper)
        idxs = np.linspace(0, len(cis) - 1, args.n_frames).astype(int)
        for i in idxs:
            ci = cis[int(i)]
            cams_list.append((Camera(colmap_id=int(i), R=ci.R, T=ci.T, FoVx=ci.FovX, FoVy=ci.FovY,
                                      image=ci.image, gt_alpha_mask=None, image_name=str(i), uid=int(i),
                                      time=ci.time), int(i)))
    elif is_n3v:
        from scene.neural_3D_dataset_NDC import Neural3D_NDC_Dataset
        ds = Neural3D_NDC_Dataset(src_path, "train", 1.0, time_scale=1,
                                   scene_bbox_min=[-2.5,-2,-1], scene_bbox_max=[2.5,2,1], eval_index=0)
        idxs = np.linspace(0, len(ds) - 1, args.n_frames).astype(int)
        focal = ds.focal[0]
        H, W = 2028, 2704  # full N3V res
        for i in idxs:
            R, T = ds.image_poses[int(i)]
            cams_list.append((Camera(colmap_id=int(i), R=R, T=T,
                                      FoVx=focal2fov(focal, W), FoVy=focal2fov(focal, H),
                                      image=torch.zeros(3, H, W), gt_alpha_mask=None,
                                      image_name=str(i), uid=int(i), time=ds.image_times[int(i)]), int(i)))
    else:
        raise SystemExit("unsupported scene type")
    print(f"evaluating {len(cams_list)} views", flush=True)

    from gaussian_renderer import render
    pipe_args = type("P", (), {"convert_SHs_python": False, "compute_cov3D_python": False, "debug": False})
    bg = torch.zeros(3, device=args.device)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Optional: SAM3 mask cache for boundary GT
    sam3_masks = None
    if args.sam3_cache:
        z = np.load(args.sam3_cache)
        T, H_s, W_s = z["shape"].tolist()
        sam3_masks = np.unpackbits(z["masks_packed"], axis=1)[:, :H_s*W_s].reshape(T, H_s, W_s).astype(bool)
        print(f"  SAM3 cache: {T}x{H_s}x{W_s}", flush=True)

    ious = []
    chamfers = []
    per_view = []
    with torch.no_grad():
        for vi, (cam, uid) in enumerate(cams_list):
            out = render(cam, pc, pipe_args, bg, stage="fine")
            rgb = out["render"].clamp(0, 1).cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
            cl = out.get("cloth_logit")
            if cl is None:
                print(f"  view {vi}: no cloth_logit channel (skipping)", flush=True)
                continue
            pred = torch.sigmoid(cl.detach()).cpu().numpy()
            if pred.ndim == 3:
                pred = pred[..., 0]

            import cv2
            if args.use_sam3_as_gt and sam3_masks is not None and uid < len(sam3_masks):
                # SAM3 boundary as pseudo-GT
                m = sam3_masks[uid].astype(np.uint8)
                # Resize mask to match render if needed
                if m.shape != pred.shape:
                    m = cv2.resize(m, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
                e_rgb = sobel_edge(m.astype(np.float32), 0.5)
                e_pred = sobel_edge(pred, args.tau_pred)
                band = cv2.dilate(m, np.ones((2*args.band_px+1, 2*args.band_px+1), np.uint8))
            else:
                e_rgb = sobel_edge(rgb, args.tau_rgb)
                e_pred = sobel_edge(pred, args.tau_pred)
                band = cv2.dilate((pred > 0.5).astype(np.uint8), np.ones((2*args.band_px+1, 2*args.band_px+1), np.uint8))
            e_rgb_b = e_rgb & band
            e_pred_b = e_pred & band
            inter = int((e_rgb_b & e_pred_b).sum())
            union = int((e_rgb_b | e_pred_b).sum())
            iou = inter / union if union > 0 else 0.0
            cd = chamfer_distance(e_pred_b, e_rgb_b)
            ious.append(iou); chamfers.append(cd)
            per_view.append({"view": uid, "iou": iou, "chamfer": cd,
                             "n_e_rgb": int(e_rgb_b.sum()), "n_e_pred": int(e_pred_b.sum())})
            # Save qualitative overlay
            overlay = (rgb * 0.6).copy()
            overlay[e_rgb_b > 0] = [1.0, 0.0, 0.0]   # rgb edge red
            overlay[e_pred_b > 0] = [0.0, 1.0, 0.0]   # pred edge green
            overlay[(e_rgb_b > 0) & (e_pred_b > 0)] = [1.0, 1.0, 0.0]  # both yellow
            heat = np.stack([pred, np.zeros_like(pred), 1 - pred], axis=-1)  # blue→red heatmap
            panel = np.concatenate([
                (rgb * 255).astype(np.uint8),
                (heat * 255).astype(np.uint8),
                (overlay * 255).clip(0, 255).astype(np.uint8),
            ], axis=1)
            Image.fromarray(panel).save(out_dir / f"view_{uid:04d}.png")
            print(f"  view {uid}: iou={iou:.3f} chamfer={cd:.2f} (e_rgb={int(e_rgb_b.sum())}, e_pred={int(e_pred_b.sum())})",
                  flush=True)

    mean_iou = float(np.nanmean(ious)) if ious else 0.0
    mean_cd = float(np.nanmean(chamfers)) if chamfers else float("nan")
    result = {"ckpt_dir": str(base), "n_views": len(ious),
              "edge_iou_mean": mean_iou, "chamfer_mean": mean_cd,
              "per_view": per_view}
    (out_dir / "metric.json").write_text(json.dumps(result, indent=2))
    print(f"\n=== RESULT ===", flush=True)
    print(f"  edge IoU mean = {mean_iou:.3f}")
    print(f"  Chamfer mean  = {mean_cd:.2f} px")
    print(f"  saved {out_dir / 'metric.json'}", flush=True)


if __name__ == "__main__":
    main()
