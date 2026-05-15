"""Interactive viser viewer for trained 4DGaussians + Tier 2 cloth_logit checkpoint.

Usage:
    python tier2_viewer.py --ckpt-dir ../4DGaussians/output/tier2_00169_Outer_Take12 \\
        [--port 8080] [--iter 14000]

Open http://<host>:<port>/ in browser.

Controls:
- frame             : time slider (0..N-1)
- mode              : original / recolored / mask overlay
- target hue (deg)  : 0=red, 120=green, 220=blue
- cloth threshold   : sigmoid(cloth_logit) > t ⇒ recolor
- jump to cam 01-04 : preset camera positions
- viewport drag     : free 6DoF
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import viser
import nerfview

HERE = Path(__file__).parent.resolve()
FOURDGS_DIR = (HERE / ".." / ".." / "4DGaussians").resolve()
sys.path.insert(0, str(FOURDGS_DIR))

C0 = 0.28209479177387814


def rgb_to_hsv_torch(rgb):
    r, g, b = rgb.unbind(-1)
    cmax, _ = rgb.max(-1); cmin, _ = rgb.min(-1)
    delta = cmax - cmin
    h = torch.zeros_like(cmax)
    nz = delta > 1e-8
    rm = nz & (cmax == r); gm = nz & (cmax == g) & ~rm; bm = nz & (cmax == b) & ~rm & ~gm
    h[rm] = ((g[rm] - b[rm]) / delta[rm]) % 6
    h[gm] = (b[gm] - r[gm]) / delta[gm] + 2
    h[bm] = (r[bm] - g[bm]) / delta[bm] + 4
    h = h / 6
    s = torch.where(cmax > 1e-8, delta / cmax.clamp_min(1e-8), torch.zeros_like(cmax))
    return torch.stack([h, s, cmax], dim=-1)


def hsv_to_rgb_torch(hsv):
    h, s, v = hsv.unbind(-1)
    h6 = (h % 1.0) * 6.0
    i = h6.floor().long() % 6
    f = h6 - h6.floor()
    p = v * (1 - s); q = v * (1 - s * f); t = v * (1 - s * (1 - f))
    out = torch.zeros_like(hsv)
    for ci, stk in [(0, (v, t, p)), (1, (q, v, p)), (2, (p, v, t)),
                     (3, (p, q, v)), (4, (t, p, v)), (5, (v, p, q))]:
        m = i == ci
        if m.any():
            out[m] = torch.stack([stk[0][m], stk[1][m], stk[2][m]], dim=-1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--iter", type=int, default=14000)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = args.device
    ckpt = Path(args.ckpt_dir)

    # ---- Load 4DGaussians + deformation + cloth_logit ----
    from argparse import Namespace
    cfg_args = eval((ckpt / "cfg_args").read_text())
    print(f"loading {args.ckpt_dir} (iter {args.iter})...", flush=True)
    from scene.gaussian_model import GaussianModel
    pc = GaussianModel(cfg_args.sh_degree, cfg_args)
    ply_path = ckpt / "point_cloud" / f"iteration_{args.iter}" / "point_cloud.ply"
    pc.load_ply(str(ply_path))
    pc.load_model(str(ply_path.parent))
    N = pc._xyz.shape[0]
    # K=1 legacy: sigmoid; K>1 multi-class: softmax over K
    _cl = pc._cloth_logit.detach()
    K_classes = _cl.shape[1] if _cl.dim() > 1 else 1
    if K_classes == 1:
        cloth_prob = torch.sigmoid(_cl).squeeze(-1)
        n_pos = int((cloth_prob > 0.5).sum())
    else:
        sm = torch.softmax(_cl, dim=1)
        argmax_class = sm.argmax(dim=1)
        cloth_prob = sm[:, 0]    # class 0 default for stats
        n_pos = int((argmax_class == 0).sum())
    print(f"  N={N:,} K_classes={K_classes} class0_gaussians={n_pos:,} ({n_pos/N*100:.1f}%)", flush=True)

    # Save original SH_DC for switching
    sh_dc_orig = pc._features_dc.detach().clone()

    # ---- Source data path for cameras ----
    import os, json
    src_path = cfg_args.source_path
    if not os.path.isabs(src_path):
        src_path = str((FOURDGS_DIR / src_path).resolve())
    from scene.cameras import Camera
    from utils.graphics_utils import focal2fov

    is_dnerf = (Path(src_path) / "transforms_train.json").exists()
    is_n3v = (Path(src_path) / "cam00" / "images").exists()

    if is_n3v:
        from scene.neural_3D_dataset_NDC import Neural3D_NDC_Dataset
        train_ds = Neural3D_NDC_Dataset(
            src_path, "train", 1.0, time_scale=1,
            scene_bbox_min=[-2.5, -2.0, -1.0], scene_bbox_max=[2.5, 2.0, 1.0],
            eval_index=0,
        )
        image_poses = train_ds.image_poses
        image_times = train_ds.image_times
        focal0 = train_ds.focal[0]
        cams = sorted([d.name for d in Path(src_path).iterdir() if d.is_dir() and d.name.startswith("cam")])
        n_cams = len(cams)
        n_total = len(image_poses)
        n_frames = n_total // n_cams
        from PIL import Image as _PILImage
        first_img = next((Path(src_path) / cams[0] / "images").glob("*.png"))
        with _PILImage.open(first_img) as im:
            W0, H0 = im.size
        cam_presets = []
        for ci in range(min(n_cams, 8)):
            idx = ci * n_frames
            R, T = image_poses[idx]
            cam_presets.append({
                "R": R, "T": T,
                "FoVx": focal2fov(focal0, W0),
                "FoVy": focal2fov(focal0, H0),
                "img_shape": (H0, W0),
            })
        print(f"  N3V: {n_cams} cams × {n_frames} frames", flush=True)
    elif is_dnerf:
        # D-NeRF (Blender) layout — read transforms_train, build per-frame Camera.
        from scene.dataset_readers import readCamerasFromTransforms
        ts = json.loads((Path(src_path) / "transforms_train.json").read_text())
        max_time = max(f.get("time", 0.0) for f in ts["frames"])
        timestamp_mapper = {f["time"]: (f["time"]/max_time if max_time > 0 else 0.0)
                            for f in ts["frames"]}
        cam_infos = readCamerasFromTransforms(
            src_path, "transforms_train.json",
            white_background=getattr(cfg_args, "white_background", True),
            extension=".png", mapper=timestamp_mapper,
        )
        n_total = len(cam_infos)
        n_frames = n_total
        n_cams = 1   # D-NeRF train cams are independent timestamped poses, treat as one cam set
        def _hw(im):
            if hasattr(im, "shape"):  # torch / numpy
                s = im.shape
                return (s[-2], s[-1])
            return (im.size[1], im.size[0])  # PIL: (W, H) → (H, W)
        h0, w0 = _hw(cam_infos[0].image)
        cam_presets = [{
            "R": cam_infos[0].R, "T": cam_infos[0].T,
            "FoVx": cam_infos[0].FovX, "FoVy": cam_infos[0].FovY,
            "img_shape": (h0, w0),
        }, {
            "R": cam_infos[n_total // 2].R, "T": cam_infos[n_total // 2].T,
            "FoVx": cam_infos[n_total // 2].FovX, "FoVy": cam_infos[n_total // 2].FovY,
            "img_shape": (h0, w0),
        }]
        print(f"  D-NeRF: {n_total} timestamped train cams", flush=True)
    else:
        from scene.dataset_readers import readMultipleViewinfos
        scene_info = readMultipleViewinfos(src_path)
        raw_train = scene_info.train_cameras
        n_total = len(raw_train)
        n_cams = len([d for d in (Path(src_path)).iterdir() if d.is_dir() and d.name.startswith("cam")])
        n_frames = n_total // n_cams
        print(f"  multipleview: {n_cams} cams × {n_frames} frames", flush=True)
        cam_presets = []
        for ci in range(n_cams):
            idx = ci * n_frames
            img, (R, T), _ = raw_train[idx]
            cam_presets.append({
                "R": R, "T": T,
                "FoVx": focal2fov(raw_train.focal[0], img.shape[2]),
                "FoVy": focal2fov(raw_train.focal[0], img.shape[1]),
                "img_shape": (img.shape[1], img.shape[2]),
            })

    # ---- viser ----
    server = viser.ViserServer(host=args.host, port=args.port)
    print(f"viewer at http://{args.host}:{args.port}/  (Ctrl+C to quit)", flush=True)

    time_slider = server.gui.add_slider("frame", min=0, max=n_frames - 1, step=1, initial_value=n_frames // 2)
    mode = server.gui.add_dropdown("mode",
                                    options=("original", "recolored", "mask overlay"),
                                    initial_value="original")
    hue_slider = server.gui.add_slider("target hue (deg)", min=0, max=360, step=5, initial_value=220)
    thresh_slider = server.gui.add_slider("cloth threshold", min=0.0, max=1.0, step=0.05, initial_value=0.5)
    # K>1: which class to recolor (0..K-1)
    target_class_slider = (server.gui.add_slider("target class", min=0, max=K_classes - 1,
                                                  step=1, initial_value=0)
                           if K_classes > 1 else None)
    info_md = server.gui.add_markdown(
        f"**4DGS Tier 2 ckpt:** `{ckpt.name}`  \n"
        f"N = **{N:,}** Gaussians, cloth (>0.5) = **{int((cloth_prob>0.5).sum()):,}** ({(cloth_prob>0.5).float().mean()*100:.1f}%)  \n"
        f"Drag viewport for 6DoF camera. Hue/threshold update live."
    )

    # Camera preset buttons
    for ci, preset in enumerate(cam_presets):
        btn = server.gui.add_button(f"jump to cam {ci+1:02d}")
        def _make_handler(p=preset):
            def handler(_):
                from scipy.spatial.transform import Rotation as Rot
                # build c2w in OpenCV convention from R (c2w rotation), T (w2c translation)
                R_c2w = np.asarray(p["R"], dtype=np.float64)
                T_w2c = np.asarray(p["T"], dtype=np.float64)
                # cam center = -R_c2w @ T_w2c (since R^T_w2c @ T_w2c = -cam_center)
                # actually: w2c → cam_center = -inv(R_w2c) @ T_w2c = -R_c2w @ T_w2c
                cam_center = -R_c2w @ T_w2c
                # viser camera_state.c2w is OpenCV c2w (rotation + position)
                quat = Rot.from_matrix(R_c2w).as_quat(scalar_first=True)  # wxyz
                for client in server.get_clients().values():
                    client.camera.position = cam_center
                    client.camera.wxyz = quat
            return handler
        btn.on_click(_make_handler())

    # ---- Render callback ----
    from gaussian_renderer import render
    pipe_args = type("P", (), {"convert_SHs_python": False, "compute_cov3D_python": False, "debug": False})
    bg_color = torch.zeros(3, device=device)

    @torch.no_grad()
    def render_fn(camera_state: nerfview.CameraState,
                  render_tab_state: nerfview.RenderTabState) -> np.ndarray:
        W = int(render_tab_state.viewer_width)
        H = max(1, int(W / camera_state.aspect))
        # viser c2w (OpenCV) → R_c2w, T_w2c for 4DGaussians Camera
        c2w = np.asarray(camera_state.c2w, dtype=np.float64)
        R_c2w = c2w[:3, :3]
        cam_center = c2w[:3, 3]
        # T_w2c = -R_w2c @ cam_center = -R_c2w.T @ cam_center
        T_w2c = -R_c2w.T @ cam_center
        # FoVy from camera_state.fov (vertical), FoVx via aspect
        FoVy = float(camera_state.fov)
        FoVx = 2.0 * np.arctan(np.tan(FoVy * 0.5) * camera_state.aspect)

        # Construct Camera with dummy image of right size
        dummy = torch.zeros(3, H, W)
        time_norm = float(time_slider.value) / max(1, n_frames - 1)  # [0, 1]
        vp = Camera(colmap_id=0, R=R_c2w, T=T_w2c,
                    FoVx=FoVx, FoVy=FoVy, image=dummy, gt_alpha_mask=None,
                    image_name="viser", uid=0, time=time_norm)

        m = mode.value; thr = float(thresh_slider.value); hue = float(hue_slider.value)
        tgt_class = int(target_class_slider.value) if target_class_slider is not None else 0

        if m in ("original", "mask overlay"):
            pc._features_dc.data = sh_dc_orig
            r = render(vp, pc, pipe_args, bg_color, stage="fine")
            rgb_t = r["render"].clamp(0, 1)
            img = (rgb_t.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            cl_img = r.get("cloth_logit")
            if m == "mask overlay" and cl_img is not None:
                if cl_img.dim() == 3:    # (H, W, K) multi-class
                    sm = torch.softmax(cl_img, dim=-1)
                    mask = (sm[..., tgt_class] > thr).cpu().numpy()
                else:                    # (H, W) legacy K=1
                    mask = (torch.sigmoid(cl_img) > thr).cpu().numpy()
                green = np.array([0, 255, 0], dtype=np.float32)
                out = img.astype(np.float32)
                out[mask] = (out[mask] * 0.55 + green * 0.45)
                img = out.clip(0, 255).astype(np.uint8)
            return img
        else:  # recolored
            # Modify SH_DC for cloth Gaussians (per-Gaussian selection)
            cl = pc._cloth_logit.detach()
            if cl.dim() > 1 and cl.shape[1] > 1:   # multi-class
                sm = torch.softmax(cl, dim=1)
                argmax_c = sm.argmax(dim=1)
                cloth_mask_g = (argmax_c == tgt_class) & (sm[:, tgt_class] > thr)
            else:
                cloth_mask_g = torch.sigmoid(cl).squeeze(-1) > thr
            rgb_per = (sh_dc_orig[:, 0] * C0 + 0.5).clamp(0, 1)  # (N, 3)
            hsv = rgb_to_hsv_torch(rgb_per)
            new_hsv = hsv.clone()
            new_hsv[..., 0] = hue / 360.0
            new_rgb = hsv_to_rgb_torch(new_hsv).clamp(0, 1)
            new_sh_dc_term = (new_rgb - 0.5) / C0
            sh_dc_mod = sh_dc_orig.clone()
            sh_dc_mod[cloth_mask_g, 0] = new_sh_dc_term[cloth_mask_g]
            pc._features_dc.data = sh_dc_mod
            r = render(vp, pc, pipe_args, bg_color, stage="fine")
            pc._features_dc.data = sh_dc_orig  # restore for next call
            rgb_t = r["render"].clamp(0, 1)
            return (rgb_t.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

    viewer = nerfview.Viewer(server, render_fn, mode="rendering")

    # GUI sliders / dropdown don't move the camera, so nerfview's camera-state-cache
    # would skip re-renders. Force a re-render whenever any control changes.
    def _force_rerender(_=None):
        try:
            viewer.rerender(_)
        except Exception:
            try:
                viewer.state.status = "rendering"
            except Exception:
                pass
    _controls = [time_slider, mode, hue_slider, thresh_slider]
    if target_class_slider is not None:
        _controls.append(target_class_slider)
    for ctrl in _controls:
        ctrl.on_update(_force_rerender)

    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
