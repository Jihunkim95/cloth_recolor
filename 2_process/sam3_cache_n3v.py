"""SAM 3 image-mode mask cache for Neural 3D Video (N3V) scenes.

Layout (after ffmpeg extraction):
  Neural3DVideo/<scene>/cam<NN>/images/0001.png ... 0300.png

Output:
  cache/sam3_n3v/<scene>/<cam>/masks.npz   (300 packed bool masks)
  cache/sam3_n3v/<scene>/<cam>/meta.json

Usage:
    python sam3_cache_n3v.py --scene flame_steak --prompt "apron" --gpu 0
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image


def cache_one_cam(scene_dir: Path, cam: str, prompts, out_dir: Path, device: str):
    if isinstance(prompts, str):
        prompts = [prompts]
    cam_images = sorted((scene_dir / cam / "images").glob("*.png"))
    if not cam_images:
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_dir / "masks.npz"
    if out_npz.exists():
        return -1   # already cached

    from transformers import Sam3Model, Sam3Processor
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    model = (Sam3Model.from_pretrained("facebook/sam3", torch_dtype=torch.bfloat16)
             .to(device).eval())

    # Read first image to get H, W
    with Image.open(cam_images[0]) as img0:
        if img0.mode == "RGBA":
            bg = Image.new("RGB", img0.size, (255, 255, 255))
            bg.paste(img0, mask=img0.split()[-1])
            img0 = bg
        else:
            img0 = img0.convert("RGB")
        W, H = img0.size

    masks = np.zeros((len(cam_images), H, W), dtype=bool)
    obj_total = 0
    t0 = time.time()
    with torch.no_grad():
        for fi, fp in enumerate(cam_images):
            img = Image.open(fp).convert("RGB")
            union = np.zeros((H, W), dtype=bool)
            for prompt in prompts:
                inputs = processor(images=img, text=prompt, return_tensors="pt").to(device)
                for k in inputs:
                    if inputs[k].dtype.is_floating_point:
                        inputs[k] = inputs[k].to(torch.bfloat16)
                outputs = model(**inputs)
                post = processor.post_process_instance_segmentation(
                    outputs, threshold=0.3, mask_threshold=0.5, target_sizes=[[H, W]])
                obj_masks = post[0].get("masks")
                if obj_masks is not None and obj_masks.numel() > 0 and obj_masks.shape[0] > 0:
                    union |= obj_masks.bool().any(dim=0).cpu().numpy()
                    obj_total += int(obj_masks.shape[0])
            masks[fi] = union

    took = time.time() - t0
    packed = np.packbits(masks.reshape(masks.shape[0], -1), axis=1)
    np.savez_compressed(out_npz,
        masks_packed=packed,
        shape=np.array(masks.shape, dtype=np.int64),
        frame_ids=np.arange(len(cam_images), dtype=np.int64))
    meta = {"scene": scene_dir.name, "cam": cam, "prompts": prompts,
            "n_frames": len(cam_images), "image_hw": [H, W],
            "took_sec": round(took, 2), "fps": round(len(cam_images)/took, 2),
            "objects_total": obj_total,
            "coverage_avg_pct": round(float(masks.mean())*100, 2)}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return len(cam_images)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--prompt", default="apron",
                    help="single prompt or comma-separated for union (e.g. 'shirt,apron')")
    ap.add_argument("--root", default="/NHNHOME/WORKSPACE/0526040060_B/research/Neural3DVideo")
    ap.add_argument("--out-root", default="/NHNHOME/WORKSPACE/0526040060_B/research/cloth_recolor/cache/sam3_n3v")
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args()

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda:0"

    prompts = [p.strip() for p in args.prompt.split(",") if p.strip()]
    scene_dir = Path(args.root) / args.scene
    out_scene = Path(args.out_root) / args.scene
    cams = sorted([d.name for d in scene_dir.iterdir() if d.is_dir() and d.name.startswith("cam") and (d / "images").exists()])
    print(f"{args.scene}: {len(cams)} cams, prompts={prompts}", flush=True)

    for cam in cams:
        n = cache_one_cam(scene_dir, cam, prompts, out_scene / cam, device)
        if n == -1:
            print(f"  [skip] {cam} cached", flush=True)
        elif n > 0:
            print(f"  ✓ {cam}: {n} frames", flush=True)
        else:
            print(f"  ✗ {cam}: no images", flush=True)


if __name__ == "__main__":
    main()
