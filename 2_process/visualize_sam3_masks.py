"""Visualize SAM3 mask cache as overlay grid on original images.

Usage:
    python visualize_sam3_masks.py \\
        --cache-npz cache/sam3_union_jumpingjacks/jumpingjacks/masks.npz \\
        --image-root 4DGaussians/data/dnerf/data/jumpingjacks \\
        --out vis/sam3_overlay_union_jj.png \\
        --n-frames 16
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-npz", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-frames", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=0.45, help="overlay alpha")
    args = ap.parse_args()

    cache_npz = Path(args.cache_npz)
    image_root = Path(args.image_root)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Load masks
    z = np.load(cache_npz)
    T, H, W = z["shape"].tolist()
    masks = np.unpackbits(z["masks_packed"], axis=1)[:, : H * W].reshape(T, H, W).astype(bool)
    print(f"masks: T={T}, H×W={H}×{W}, coverage avg={masks.mean()*100:.2f}%")

    # Meta (prompt info)
    meta_path = cache_npz.parent / "meta.json"
    meta_txt = ""
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        prompt = meta.get("prompt") or meta.get("prompts") or "?"
        meta_txt = f"prompt={prompt}  cov={meta.get('coverage_avg_pct', '?')}%"

    # Image source — match D-NeRF dataset layout
    transforms = json.loads((image_root / "transforms_train.json").read_text())
    frames = transforms["frames"]
    n = min(args.n_frames, T, len(frames))
    idx = np.linspace(0, T - 1, n).astype(int).tolist()

    # Build grid
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    tile_w = 400
    tile_h = int(H * tile_w / W)
    grid = Image.new("RGB", (cols * tile_w, rows * tile_h + 30), (255, 255, 255))
    draw_grid = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    draw_grid.text((10, 6), f"{cache_npz.parent.name}/{cache_npz.name}  |  {meta_txt}",
                   fill=(0, 0, 0), font=font)

    red = np.array([255, 0, 0], dtype=np.float32)
    for i, fi in enumerate(idx):
        # frame i image
        fp = image_root / (frames[fi]["file_path"].lstrip("./") + ".png")
        img = Image.open(fp)
        if img.mode == "RGBA":
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[-1])
            img = bg
        else:
            img = img.convert("RGB")
        img_arr = np.array(img, dtype=np.float32)
        m = masks[fi]
        # red overlay
        img_arr[m] = img_arr[m] * (1 - args.alpha) + red * args.alpha
        overlay = Image.fromarray(img_arr.clip(0, 255).astype(np.uint8))
        overlay = overlay.resize((tile_w, tile_h), Image.LANCZOS)
        r, c = i // cols, i % cols
        grid.paste(overlay, (c * tile_w, 30 + r * tile_h))
        # frame label
        ImageDraw.Draw(grid).text((c * tile_w + 5, 30 + r * tile_h + 5),
                                  f"f{fi}", fill=(255, 255, 0), font=font)
    grid.save(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
