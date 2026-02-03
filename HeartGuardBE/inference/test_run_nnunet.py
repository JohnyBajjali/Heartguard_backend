from __future__ import annotations
from pathlib import Path

import cv2
import numpy as np

from .preprocess_3x4plus1 import preprocess_3x4plus1
from .nnunet_runner import run_nnunet
import os
import shutil
import sys
from pathlib import Path


BASE = Path(__file__).resolve().parents[2] / "media"

# 0..12 (13 classes total)
NUM_CLASSES = 13

# BGR colors (OpenCV uses BGR, not RGB)
# 0 = background -> black
PALETTE_BGR = np.array(
    [
        (0, 0, 0),         # 0 background
        (0, 0, 255),       # 1 red
        (0, 128, 255),     # 2 orange
        (0, 255, 255),     # 3 yellow
        (0, 255, 0),       # 4 green
        (255, 255, 0),     # 5 cyan
        (255, 0, 0),       # 6 blue
        (255, 0, 255),     # 7 magenta
        (128, 0, 255),     # 8 purple-ish
        (255, 128, 0),     # 9 teal-ish
        (128, 255, 0),     # 10 lime-ish
        (0, 255, 128),     # 11 spring-green
        (255, 0, 128),     # 12 pink
    ],
    dtype=np.uint8
)


def pick_first_png(folder: Path) -> Path:
    folder = folder.resolve()
    png = next(folder.glob("*.png"), None)
    if png is None:
        raise FileNotFoundError(f"No PNG found in {folder}")
    return png


def read_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    return img


def read_mask_u8(path: Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise ValueError(f"Failed to read mask: {path}")
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    if m.dtype != np.uint8:
        m = np.clip(np.rint(m), 0, 255).astype(np.uint8)
    return m


def colorize_multiclass_mask(mask_u8: np.ndarray) -> np.ndarray:
    """
    mask_u8: (H,W) uint8 containing values 0..12
    returns: (H,W,3) BGR colored image
    """
    # clamp unexpected labels
    m = mask_u8.copy()
    m[m >= NUM_CLASSES] = 0
    colored = PALETTE_BGR[m]  # fancy indexing -> (H,W,3)
    return colored


def overlay_on_gray(gray_u8: np.ndarray, colored_bgr: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    Overlay colored mask on a grayscale image.
    Background class (0) should not colorize the image: we only overlay where mask != 0.
    """
    base_bgr = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)

    # only overlay where mask has color != background
    mask_nonzero = np.any(colored_bgr != (0, 0, 0), axis=2).astype(np.uint8) * 255
    mask_nonzero_3 = cv2.cvtColor(mask_nonzero, cv2.COLOR_GRAY2BGR)

    blended = cv2.addWeighted(base_bgr, 1.0, colored_bgr, alpha, 0.0)

    # keep original where background
    out = np.where(mask_nonzero_3 > 0, blended, base_bgr)
    return out


def save_side_by_side(pre_gray: np.ndarray, color_bgr: np.ndarray, overlay_bgr: np.ndarray, out_path: Path) -> None:
    # convert pre_gray to BGR
    pre_bgr = cv2.cvtColor(pre_gray, cv2.COLOR_GRAY2BGR)

    # ensure same height
    H = pre_bgr.shape[0]
    imgs = []
    for img in [pre_bgr, color_bgr, overlay_bgr]:
        if img.shape[0] != H:
            img = cv2.resize(img, (int(img.shape[1] * H / img.shape[0]), H), interpolation=cv2.INTER_NEAREST)
        imgs.append(img)

    combined = np.concatenate(imgs, axis=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), combined):
        raise RuntimeError(f"Failed to write preview: {out_path}")


if __name__ == "__main__":
    # ====== 503 (3x4+1) MULTICLASS TEST ======
    src_dir = BASE / "test_in" / "503"
    src = pick_first_png(src_dir)

    out_dir = BASE / "test_out" / "503"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) preprocess
    pre = out_dir / f"pre_{src.stem}.png"
    preprocess_3x4plus1(src, pre)
    print("✅ Preprocessed saved:", pre)

    # 2) run nnU-Net (503 = PNG-trained, multiclass)
    pred = run_nnunet(pre, dataset_id="503", is_png=True)
    if pred.suffix.lower() != ".png":
        raise RuntimeError(f"Expected PNG prediction for Dataset503, got: {pred}")

    # 3) save raw mask (0..12)
    mask_png = out_dir / f"mask_503_{src.stem}.png"
    mask_png.write_bytes(pred.read_bytes())
    print("✅ Raw multiclass MASK saved:", mask_png)

    # 4) load and inspect values
    mask = read_mask_u8(mask_png)
    vals = np.unique(mask)
    print("Mask dtype:", mask.dtype, "shape:", mask.shape)
    print("Unique values:", vals.tolist())
    print("Min/Max:", int(vals.min()), int(vals.max()))

    # 5) colorize classes
    mask_color = colorize_multiclass_mask(mask)
    color_png = out_dir / f"mask_503_{src.stem}_color.png"
    if not cv2.imwrite(str(color_png), mask_color):
        raise RuntimeError(f"Failed to write color mask: {color_png}")
    print("✅ Colorized MASK saved:", color_png)

    # 6) overlay on preprocessed image (recommended for quick visual QA)
    pre_gray = read_gray(pre)
    overlay = overlay_on_gray(pre_gray, mask_color, alpha=0.45)
    overlay_png = out_dir / f"overlay_503_{src.stem}.png"
    if not cv2.imwrite(str(overlay_png), overlay):
        raise RuntimeError(f"Failed to write overlay: {overlay_png}")
    print("✅ Overlay saved:", overlay_png)

    # 7) side-by-side preview: pre | colored mask | overlay
    preview_png = out_dir / f"preview_503_{src.stem}.png"
    save_side_by_side(pre_gray, mask_color, overlay, preview_png)
    print("✅ Preview (pre | mask | overlay) saved:", preview_png)

    print("\nDONE. Open these:")
    print(" -", color_png)
    print(" -", overlay_png)
    print(" -", preview_png)
