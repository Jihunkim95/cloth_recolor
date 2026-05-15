"""SAM 3 multi-prompt union mask cache.

Runs SAM3 image-mode N times (once per prompt), then OR-s the N masks per frame
into one union mask. Saves single masks.npz like sam3_cache.py but coverage is the
union of all prompts. Useful when one prompt under-covers (e.g., "shirt" → 1.93%)
and multiple synonyms together capture more (e.g., shirt ∪ hoodie ∪ jacket).

Usage:
    python sam3_cache_union.py --root .../dnerf/data \\
        --scene jumpingjacks \\
        --prompts "orange hoodie,sweater,jacket,hoodie" \\
        --out cache/sam3_union_jumpingjacks
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="D-NeRF root (.../dnerf/data)")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--prompts", required=True, help="comma-separated prompts")
    ap.add_argument("--out", required=True, help="cache output root")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
    if not prompts:
        raise SystemExit("no prompts")

    out_dir = Path(args.out) / args.scene
    out_path = out_dir / "masks.npz"
    if out_path.exists() and not args.force:
        print(f"[skip] {args.scene} (cached)", flush=True)
        return

    from transformers import Sam3Model, Sam3Processor
    print(f"loading SAM 3...", flush=True)
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    model = (Sam3Model.from_pretrained("facebook/sam3", torch_dtype=torch.bfloat16)
             .to(args.device).eval())

    transforms = json.loads((Path(args.root) / args.scene / "transforms_train.json").read_text())
    frames = transforms["frames"]
    print(f"{args.scene}: {len(frames)} frames × {len(prompts)} prompts = "
          f"{len(frames) * len(prompts)} SAM3 inferences", flush=True)

    fp0 = Path(args.root) / args.scene / (frames[0]["file_path"].lstrip("./") + ".png")
    with Image.open(fp0) as img0:
        if img0.mode == "RGBA":
            bg = Image.new("RGB", img0.size, (255, 255, 255)); bg.paste(img0, mask=img0.split()[-1])
            img0 = bg
        else:
            img0 = img0.convert("RGB")
        W, H = img0.size

    union = np.zeros((len(frames), H, W), dtype=bool)
    per_prompt_obj = {p: 0 for p in prompts}
    t0 = time.time()
    with torch.no_grad():
        for fi, frame in enumerate(frames):
            fp = Path(args.root) / args.scene / (frame["file_path"].lstrip("./") + ".png")
            img = Image.open(fp)
            if img.mode == "RGBA":
                bg = Image.new("RGB", img.size, (255, 255, 255)); bg.paste(img, mask=img.split()[-1])
                img = bg
            else:
                img = img.convert("RGB")
            frame_union = np.zeros((H, W), dtype=bool)
            for prompt in prompts:
                inputs = processor(images=img, text=prompt, return_tensors="pt").to(args.device)
                for k in inputs:
                    if inputs[k].dtype.is_floating_point:
                        inputs[k] = inputs[k].to(torch.bfloat16)
                outputs = model(**inputs)
                post = processor.post_process_instance_segmentation(
                    outputs, threshold=0.3, mask_threshold=0.5, target_sizes=[[H, W]])
                obj_masks = post[0].get("masks")
                if obj_masks is not None and obj_masks.numel() > 0 and obj_masks.shape[0] > 0:
                    merged = obj_masks.bool().any(dim=0).cpu().numpy()
                    per_prompt_obj[prompt] += int(obj_masks.shape[0])
                    frame_union = frame_union | merged
            union[fi] = frame_union
            if fi % 50 == 0 and fi > 0:
                fps = fi / (time.time() - t0)
                print(f"{args.scene}: {fi}/{len(frames)} ({fps:.1f} frames/s)", flush=True)

    took = time.time() - t0
    out_dir.mkdir(parents=True, exist_ok=True)
    packed = np.packbits(union.reshape(union.shape[0], -1), axis=1)
    np.savez_compressed(out_path,
        masks_packed=packed,
        shape=np.array(union.shape, dtype=np.int64),
        frame_ids=np.arange(len(frames), dtype=np.int64))
    meta = {
        "scene": args.scene, "prompts": prompts, "n_frames": len(frames),
        "image_hw": [H, W], "took_sec": round(took, 2),
        "objects_per_prompt": per_prompt_obj,
        "coverage_avg_pct": round(float(union.mean()) * 100, 2),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"{args.scene} union done: coverage {meta['coverage_avg_pct']:.2f}% ({took:.1f}s)",
          flush=True)


if __name__ == "__main__":
    main()
