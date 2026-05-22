"""exp021: Pipeline efficiency comparison (Meeting 0.2).

baseline: each frame = 4DGS render + SAM3 detect + HSV swap masked pixels
ours:     each frame = 4DGS render with cloth_logit + HSV swap > threshold

Measures per-frame wall-time across 200 frames (jumpingjacks).
"""
from __future__ import annotations
import argparse, sys, time, math, os
from pathlib import Path
import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).parent.resolve()
FOURDGS_DIR = (HERE / ".." / ".." / "4DGaussians").resolve()
sys.path.insert(0, str(FOURDGS_DIR))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--iter", type=int, default=20000)
    ap.add_argument("--n-frames", type=int, default=200)
    ap.add_argument("--hue", type=float, default=220.0)
    ap.add_argument("--threshold", type=float, default=0.5)
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
    print(f"loaded: N={pc._xyz.shape[0]:,}", flush=True)

    # Build cameras (D-NeRF transforms_train)
    import json
    src_path = cfg_args.source_path
    if not os.path.isabs(src_path):
        src_path = str((FOURDGS_DIR / src_path).resolve())
    from scene.dataset_readers import readCamerasFromTransforms
    ts = json.loads((Path(src_path) / "transforms_train.json").read_text())
    max_time = max(f.get("time", 0.0) for f in ts["frames"])
    mapper = {f["time"]: (f["time"]/max_time if max_time > 0 else 0.0) for f in ts["frames"]}
    cam_infos = readCamerasFromTransforms(
        src_path, "transforms_train.json",
        white_background=getattr(cfg_args, "white_background", True),
        extension=".png", mapper=mapper)
    n = min(args.n_frames, len(cam_infos))
    print(f"using {n} frames", flush=True)

    from scene.cameras import Camera
    from gaussian_renderer import render
    pipe_args = type("P", (), {"convert_SHs_python": False, "compute_cov3D_python": False, "debug": False})
    bg = torch.tensor([1, 1, 1] if getattr(cfg_args, "white_background", True) else [0, 0, 0],
                      dtype=torch.float32, device=args.device)

    # SH_DC HSV swap for cloth Gaussians (ours)
    from utils.sh_utils import RGB2SH, SH2RGB
    import colorsys
    C0 = 0.28209479177387814
    sh_dc_orig = pc._features_dc.detach().clone()  # (N,1,3)
    cloth_prob = torch.sigmoid(pc._cloth_logit.detach()).squeeze(-1)
    cloth_mask = cloth_prob > args.threshold
    n_cloth = int(cloth_mask.sum())
    print(f"cloth_pct@{args.threshold}: {n_cloth} ({n_cloth/pc._xyz.shape[0]*100:.2f}%)", flush=True)

    # Precompute recolored SH_DC for ours method
    rgb = (sh_dc_orig[:, 0] * C0 + 0.5).clamp(0, 1).cpu().numpy()
    import matplotlib.colors as mcolors
    hsv = np.array([mcolors.rgb_to_hsv(c) for c in rgb])
    hsv[..., 0] = args.hue / 360.0
    new_rgb = np.array([mcolors.hsv_to_rgb(h) for h in hsv]).clip(0, 1).astype(np.float32)
    new_sh_dc = (new_rgb - 0.5) / C0
    new_sh_dc_t = torch.from_numpy(new_sh_dc).to(args.device)
    sh_dc_recolored = sh_dc_orig.clone()
    sh_dc_recolored[cloth_mask, 0] = new_sh_dc_t[cloth_mask]

    # ============ Time ours method ============
    print("=== timing ours (4DGS render + precomputed cloth → swap → render) ===", flush=True)
    times_ours = []
    with torch.no_grad():
        for i in range(n):
            vp = cam_infos[i]
            cam = Camera(colmap_id=i, R=vp.R, T=vp.T, FoVx=vp.FovX, FoVy=vp.FovY,
                         image=vp.image, gt_alpha_mask=None, image_name=str(i), uid=i, time=vp.time)
            torch.cuda.synchronize()
            t0 = time.time()
            # 1) render with recolored SH_DC
            pc._features_dc.data = sh_dc_recolored
            out = render(cam, pc, pipe_args, bg, stage="fine")
            _ = out["render"]
            torch.cuda.synchronize()
            t1 = time.time()
            times_ours.append(t1 - t0)
    pc._features_dc.data = sh_dc_orig
    avg_ours = np.mean(times_ours) * 1000
    print(f"  ours avg: {avg_ours:.1f} ms/frame", flush=True)

    # ============ Time baseline (with SAM3 per frame) ============
    print("=== timing baseline (4DGS render + SAM3 per-frame + 2D mask swap) ===", flush=True)
    from transformers import Sam3Model, Sam3Processor
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    model = (Sam3Model.from_pretrained("facebook/sam3", torch_dtype=torch.bfloat16)
             .to(args.device).eval())
    prompt = "hoodie"

    times_baseline = []
    times_render_only = []
    times_sam3_only = []
    times_swap_only = []
    with torch.no_grad():
        for i in range(n):
            vp = cam_infos[i]
            cam = Camera(colmap_id=i, R=vp.R, T=vp.T, FoVx=vp.FovX, FoVy=vp.FovY,
                         image=vp.image, gt_alpha_mask=None, image_name=str(i), uid=i, time=vp.time)
            torch.cuda.synchronize()
            t0 = time.time()
            # 1) Render 4DGS with original SH
            out = render(cam, pc, pipe_args, bg, stage="fine")
            rgb = out["render"].clamp(0, 1)
            torch.cuda.synchronize()
            t1 = time.time()
            # 2) Run SAM3 on rendered image
            img_np = (rgb.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)
            inputs = processor(images=img_pil, text=prompt, return_tensors="pt").to(args.device)
            for k in inputs:
                if inputs[k].dtype.is_floating_point:
                    inputs[k] = inputs[k].to(torch.bfloat16)
            outputs = model(**inputs)
            post = processor.post_process_instance_segmentation(
                outputs, threshold=0.3, mask_threshold=0.5, target_sizes=[[img_np.shape[0], img_np.shape[1]]])
            obj_masks = post[0].get("masks")
            if obj_masks is not None and obj_masks.numel() > 0 and obj_masks.shape[0] > 0:
                mask = obj_masks.bool().any(dim=0).cpu().numpy()
            else:
                mask = np.zeros((img_np.shape[0], img_np.shape[1]), dtype=bool)
            torch.cuda.synchronize()
            t2 = time.time()
            # 3) HSV swap on masked pixels
            import matplotlib.colors as mc
            rgb_norm = img_np.astype(np.float32) / 255.0
            hsv_img = mc.rgb_to_hsv(rgb_norm)
            hsv_img[mask, 0] = args.hue / 360.0
            rgb_out = mc.hsv_to_rgb(hsv_img).clip(0, 1)
            torch.cuda.synchronize()
            t3 = time.time()
            times_baseline.append(t3 - t0)
            times_render_only.append(t1 - t0)
            times_sam3_only.append(t2 - t1)
            times_swap_only.append(t3 - t2)
    avg_baseline = np.mean(times_baseline) * 1000
    avg_render = np.mean(times_render_only) * 1000
    avg_sam3 = np.mean(times_sam3_only) * 1000
    avg_swap = np.mean(times_swap_only) * 1000
    print(f"  baseline avg: {avg_baseline:.1f} ms/frame", flush=True)
    print(f"    render: {avg_render:.1f} ms   sam3: {avg_sam3:.1f} ms   swap: {avg_swap:.1f} ms", flush=True)

    # ============ Save ============
    import json
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "n_frames": n,
        "ckpt_dir": str(base),
        "ours_avg_ms": avg_ours,
        "baseline_avg_ms": avg_baseline,
        "baseline_render_ms": avg_render,
        "baseline_sam3_ms": avg_sam3,
        "baseline_swap_ms": avg_swap,
        "speedup": avg_baseline / avg_ours,
        "cloth_pct": n_cloth / pc._xyz.shape[0] * 100,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(f"\n=== RESULT ===", flush=True)
    print(f"baseline: {avg_baseline:.1f} ms/frame (render {avg_render:.0f} + sam3 {avg_sam3:.0f} + swap {avg_swap:.0f})")
    print(f"ours:     {avg_ours:.1f} ms/frame")
    print(f"speedup:  {avg_baseline/avg_ours:.2f}×")
    print(f"saved {out_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
