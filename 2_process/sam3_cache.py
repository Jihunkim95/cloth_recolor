"""SAM 3 image-mode segmentation for D-NeRF train frames.

D-NeRF frames are independent pose samples (not video) → use Sam3Model + Sam3Processor
(image-level), text-prompted concept segmentation. Output cached as masks.npz indexed
by linear frame uid (matching the order in transforms_train.json).

Usage:
    python sam3_dnerf_cache.py --root ../4DGaussians/data/dnerf/data \\
        --scenes jumpingjacks,standup,hellwarrior,mutant \\
        --prompts hoodie,shirt,armor,suit \\
        --out cache/sam3_dnerf [--workers 8]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def process_scene(scene: str, prompt: str, root: Path, out_root: Path, device: str,
                  force: bool = False) -> None:
    """Run SAM 3 image-mode on every train frame of a D-NeRF scene; save masks.npz."""
    from transformers import Sam3Model, Sam3Processor

    out_dir = out_root / scene
    out_path = out_dir / "masks.npz"
    if out_path.exists() and not force:
        print(f"[skip] {scene}", flush=True)
        return

    transforms = json.loads((root / scene / "transforms_train.json").read_text())
    frames = transforms["frames"]
    print(f"[{device}] {scene}: {len(frames)} frames, prompt={prompt!r}", flush=True)

    print(f"[{device}] loading SAM 3...", flush=True)
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    model = (Sam3Model.from_pretrained("facebook/sam3", torch_dtype=torch.bfloat16)
             .to(device).eval())

    # Read first image to get H, W
    fp0 = root / scene / (frames[0]["file_path"].lstrip("./") + ".png")
    with Image.open(fp0) as img0:
        # D-NeRF is RGBA — convert to RGB on white bg (matches their training behavior)
        if img0.mode == "RGBA":
            bg = Image.new("RGB", img0.size, (255, 255, 255))
            bg.paste(img0, mask=img0.split()[-1])
            img0 = bg
        else:
            img0 = img0.convert("RGB")
        W, H = img0.size

    masks = np.zeros((len(frames), H, W), dtype=bool)
    n_objects_total = 0
    t0 = time.time()
    with torch.no_grad():
        for fi, frame in enumerate(frames):
            fp = root / scene / (frame["file_path"].lstrip("./") + ".png")
            img = Image.open(fp)
            if img.mode == "RGBA":
                bg = Image.new("RGB", img.size, (255, 255, 255))
                bg.paste(img, mask=img.split()[-1])
                img = bg
            else:
                img = img.convert("RGB")

            inputs = processor(images=img, text=prompt, return_tensors="pt").to(device)
            for k in inputs:
                if inputs[k].dtype.is_floating_point:
                    inputs[k] = inputs[k].to(torch.bfloat16)
            outputs = model(**inputs)
            post = processor.post_process_instance_segmentation(
                outputs, threshold=0.3, mask_threshold=0.5,
                target_sizes=[[H, W]],
            )
            obj_masks = post[0].get("masks")  # (num_objects, H, W) bool
            if obj_masks is not None and obj_masks.numel() > 0 and obj_masks.shape[0] > 0:
                merged = obj_masks.bool().any(dim=0).cpu().numpy()
                n_objects_total += int(obj_masks.shape[0])
            else:
                merged = np.zeros((H, W), dtype=bool)
            masks[fi] = merged
            if fi % 50 == 0 and fi > 0:
                elapsed = time.time() - t0
                fps = fi / elapsed
                print(f"[{device}] {scene}: {fi}/{len(frames)} ({fps:.1f} fps)", flush=True)

    took = time.time() - t0
    out_dir.mkdir(parents=True, exist_ok=True)
    packed = np.packbits(masks.reshape(masks.shape[0], -1), axis=1)
    np.savez_compressed(out_path,
        masks_packed=packed,
        shape=np.array(masks.shape, dtype=np.int64),
        frame_ids=np.arange(len(frames), dtype=np.int64),
    )
    meta = {
        "scene": scene, "prompt": prompt, "n_frames": len(frames),
        "image_hw": [H, W], "took_sec": round(took, 2),
        "fps": round(len(frames) / took, 2),
        "objects_total": n_objects_total,
        "objects_avg_per_frame": round(n_objects_total / len(frames), 3),
        "coverage_avg_pct": round(float(masks.mean()) * 100, 2),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[{device}] {scene} done: {len(frames)} frames in {took:.1f}s, "
          f"avg coverage {meta['coverage_avg_pct']:.1f}% ", flush=True)


def worker(rank: int, jobs: list[tuple], args: argparse.Namespace) -> None:
    device = f"cuda:{rank}"
    torch.cuda.set_device(device)
    for scene, prompt in jobs:
        try:
            process_scene(scene, prompt, Path(args.root), Path(args.out), device, args.force)
        except Exception as e:
            print(f"[worker {rank} FAIL] {scene}: {e}", flush=True)
            traceback.print_exc()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="D-NeRF root (.../dnerf/data)")
    ap.add_argument("--out", required=True, help="cache output root")
    ap.add_argument("--scenes", required=True, help="comma-separated scene names")
    ap.add_argument("--prompts", required=True, help="comma-separated text prompts (one per scene)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    scenes = [s.strip() for s in args.scenes.split(",")]
    prompts = [p.strip() for p in args.prompts.split(",")]
    if len(prompts) != len(scenes):
        print(f"prompts ({len(prompts)}) must match scenes ({len(scenes)})", file=sys.stderr)
        sys.exit(2)
    jobs = list(zip(scenes, prompts))
    print(f"{len(jobs)} scenes × {args.workers} workers", flush=True)

    if args.workers <= 1:
        worker(0, jobs, args)
        return

    mp.set_start_method("spawn", force=True)
    parts = [jobs[i :: args.workers] for i in range(args.workers)]
    procs = []
    for rank, part in enumerate(parts):
        if not part:
            continue
        p = mp.Process(target=worker, args=(rank, part, args), daemon=False)
        p.start()
        procs.append(p)
    for p in procs:
        p.join()


if __name__ == "__main__":
    main()
