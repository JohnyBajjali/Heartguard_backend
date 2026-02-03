from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .vis_504_outputs import plot_like_original_3x4 as plot_like_original_3x4_vis

from .preprocess_3x4 import preprocess_3x4
from .nnunet_runner import run_nnunet
from .postprocess import nii_to_png_mask

from .digitize_binary_3x4 import (
    digitize_from_binary_mask_3x4,
    LEAD_LAYOUT_3x4,
)

from HeartGuardBE.protoecgnet_Model.preprocess import preprocess_single_ecg
from HeartGuardBE.protoecgnet_Model.predictor import FusionPredictor

BASE = Path(__file__).resolve().parents[2] / "media"


# -----------------------------
# basic helpers
# -----------------------------
def pick_first_png(folder: Path) -> Path:
    folder = folder.resolve()
    png = next(folder.glob("*.png"), None)
    if png is None:
        raise FileNotFoundError(f"No PNG found in {folder}")
    return png


def save_leads_json(leads: dict[str, np.ndarray], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: np.asarray(v).astype(float).tolist() for k, v in leads.items()}
    out_path.write_text(json.dumps(payload), encoding="utf-8")


def _print_checkpoint_help():
    print("\n⚠️ ProtoECGNet checkpoints could not be loaded.")
    print("This is almost always because the .ckpt files are Git-LFS pointers (text files).")
    print("\nFix:")
    print("  git lfs install")
    print("  git lfs pull")
    print("\nQuick check:")
    print("  Get-Content path\\to\\last.ckpt -TotalCount 2")
    print("If you see 'version https://git-lfs.github.com/spec/v1' => it's a pointer, not weights.\n")


def read_mask_u8(p: Path) -> np.ndarray:
    m = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(p)
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    if m.dtype != np.uint8:
        m = np.clip(np.rint(m), 0, 255).astype(np.uint8)
    return m


# -----------------------------
# overlays (what you wanted)
# -----------------------------
def overlay_mask_vs_path(mask_u8: np.ndarray, meta: dict, out_png: Path) -> None:
    """
    GREEN = mask
    RED   = extracted y_path_pix (directly from digitizer, ROI coords)
    """
    H, W = mask_u8.shape
    mask01 = (mask_u8 > 0).astype(np.uint8)

    vis = np.zeros((H, W, 3), dtype=np.uint8)
    vis[mask01 > 0] = (0, 255, 0)

    boxes = meta.get("boxes", {})
    y_paths = meta.get("y_path_pix", {})

    for lead, box in boxes.items():
        y_list = y_paths.get(lead, None)
        if not y_list:
            continue

        y1, y2, x1, x2 = box
        roi_w = x2 - x1
        y_path = np.array(y_list, dtype=np.float32)

        if y_path.size != roi_w:
            y_path = np.interp(
                np.linspace(0, 1, roi_w, dtype=np.float32),
                np.linspace(0, 1, y_path.size, dtype=np.float32),
                y_path,
            ).astype(np.float32)

        y_int = np.clip(np.rint(y_path).astype(int), 0, (y2 - y1) - 1)
        xs = np.arange(x1, x2, dtype=np.int32)
        vis[y1 + y_int, xs] = (0, 0, 255)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), vis)


def overlay_mask_vs_rendered_from_mv(mask_u8: np.ndarray, leads_mV: dict, meta: dict, out_png: Path) -> None:
    """
    GREEN = mask
    RED   = reconstructed pixels using digitizer meta:
            y_pix(ROI) = baseline_pix - mv * px_per_mV
    """
    H, W = mask_u8.shape
    mask01 = (mask_u8 > 0).astype(np.uint8)

    px_per_mV = float(meta["px_per_mV"])
    boxes = meta.get("boxes", {})
    baselines = meta.get("baseline_pix", {})

    vis = np.zeros((H, W, 3), dtype=np.uint8)
    vis[mask01 > 0] = (0, 255, 0)

    for lead, box in boxes.items():
        if lead not in leads_mV:
            continue

        y1, y2, x1, x2 = box
        roi_h = y2 - y1
        roi_w = x2 - x1
        if roi_h <= 2 or roi_w <= 2:
            continue

        baseline = float(baselines.get(lead, roi_h * 0.55))  # ROI coords
        mv = np.asarray(leads_mV[lead], dtype=np.float32)

        mv_rs = np.interp(
            np.linspace(0, 1, roi_w, dtype=np.float32),
            np.linspace(0, 1, mv.size, dtype=np.float32),
            mv,
        ).astype(np.float32)

        y_pix = baseline - (mv_rs * px_per_mV)
        y_int = np.clip(np.rint(y_pix).astype(int), 0, roi_h - 1)

        xs = np.arange(x1, x2, dtype=np.int32)
        vis[y1 + y_int, xs] = (0, 0, 255)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), vis)


# -----------------------------
# keep: digitized like original
# -----------------------------
def _draw_ecg_paper(ax, width_mm: float, height_mm: float):
    ax.set_facecolor("white")

    # minor grid (1mm)
    for x in np.arange(0, width_mm + 1, 1.0):
        ax.axvline(x, linewidth=0.3, alpha=0.25)
    for y in np.arange(0, height_mm + 1, 1.0):
        ax.axhline(y, linewidth=0.3, alpha=0.25)

    # major grid (5mm)
    for x in np.arange(0, width_mm + 1, 5.0):
        ax.axvline(x, linewidth=0.8, alpha=0.35)
    for y in np.arange(0, height_mm + 1, 5.0):
        ax.axhline(y, linewidth=0.8, alpha=0.35)


def plot_like_original_3x4(
    leads_mV: dict[str, np.ndarray],
    out_png: Path,
    cell_w_mm: float = 65.0,
    cell_h_mm: float = 30.0,
    mm_per_mV: float = 10.0,
    dpi: int = 200,
):
    """
    Creates: digitized_like_original_3x4.png
    """
    cols, rows = 4, 3
    width_mm = cols * cell_w_mm
    height_mm = rows * cell_h_mm

    fig_w_in = width_mm / 25.4
    fig_h_in = height_mm / 25.4

    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=dpi)
    ax = plt.gca()

    _draw_ecg_paper(ax, width_mm, height_mm)

    # paper coords: (0,0) top-left
    ax.set_xlim(0, width_mm)
    ax.set_ylim(height_mm, 0)

    for idx, lead in enumerate(LEAD_LAYOUT_3x4):
        r = idx // cols
        c = idx % cols

        x0 = c * cell_w_mm
        y0 = r * cell_h_mm

        y = leads_mV.get(lead, None)
        if y is None:
            continue

        y = np.asarray(y, dtype=np.float32)
        x = np.linspace(0, cell_w_mm, num=len(y), dtype=np.float32)
        y_mm = y * float(mm_per_mV)

        baseline = y0 + cell_h_mm * 0.55
        ax.plot(x0 + x, baseline - y_mm, linewidth=1.2)

        ax.text(
            x0 + 3.0, y0 + 6.0, lead,
            fontsize=10, fontweight="bold",
            va="top", ha="left"
        )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_png), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# -----------------------------
# main
# -----------------------------
if __name__ == "__main__":
    src_dir = BASE / "test_in" / "504"
    src = pick_first_png(src_dir)

    out_dir = BASE / "test_out" / "505"
    out_dir.mkdir(parents=True, exist_ok=True)

    px_per_mV = 40.0
    target_len = 1000
    fs_assumed = 100.0

    # 1) preprocess
    pre = out_dir / f"pre_{src.stem}.png"
    preprocess_3x4(src, pre)
    print("✅ Preprocessed:", pre)

    # 2) nnUNet 504 -> NIfTI
    pred_nii = run_nnunet(pre, dataset_id="504", is_png=False)
    print("✅ nnU-Net output:", pred_nii)

    if pred_nii.suffixes[-2:] != [".nii", ".gz"]:
        raise RuntimeError(f"Expected .nii.gz from 504, got: {pred_nii}")

    # 3) NIfTI -> binary mask PNG
    mask_png = out_dir / f"mask_504_{src.stem}.png"
    nii_to_png_mask(pred_nii, mask_png, binary=True)
    print("✅ Binary mask:", mask_png)

    # 4) digitize -> leads + meta
    debug_dir = out_dir / f"debug_digitize_{src.stem}"
    leads_mV, meta = digitize_from_binary_mask_3x4(
        str(mask_png),
        target_len_samples=target_len,
        px_per_mV=px_per_mV,
        debug_dir=str(debug_dir),
        return_meta=True,
    )
    print("✅ Digitized leads. Debug:", debug_dir)

    # Save leads + meta
    leads_json = out_dir / f"leads_504_{src.stem}.json"
    save_leads_json(leads_mV, leads_json)
    print("✅ Leads JSON:", leads_json)

    meta_path = out_dir / "digitize_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("✅ Saved:", meta_path)

    # 5) overlays
    mask_u8 = read_mask_u8(mask_png)

    path_overlay = out_dir / "mask_vs_path_overlay.png"
    overlay_mask_vs_path(mask_u8, meta, path_overlay)
    print("✅ Saved:", path_overlay)

    render_overlay = out_dir / "mask_vs_rendered_overlay.png"
    overlay_mask_vs_rendered_from_mv(mask_u8, leads_mV, meta, render_overlay)
    print("✅ Saved:", render_overlay)

    # 6) digitized like original (kept)
    like_png = out_dir / "digitized_like_original_3x4.png"
    plot_like_original_3x4_vis(
        leads_mV, like_png,
        cell_w_mm=62.0,  # smaller cells -> smaller overall width
        cell_h_mm=30.0,  # smaller cells -> smaller overall height
        row_gap_mm=4.0,  # rows closer together
        col_gap_mm=4.0,  # columns closer together
        auto_gain=True,
        never_clip=True,
        headroom=1.10,  # less extra headroom -> less “empty” feel
        max_gain_mm_per_mV=7.0,
        out_width_px=1400,  # big change: output pixel width for Flutter
    )

    print("✅ Saved:", like_png)

    # 7) print correctness per lead
    print("\n--- Digitization correctness (per lead) ---")
    for lead in LEAD_LAYOUT_3x4:
        hr = meta.get("hit_rate", {}).get(lead, None)
        err = meta.get("mean_abs_px_err", {}).get(lead, None)
        if hr is None or err is None:
            continue
        print(f"{lead:>3s} | hit_rate={100.0*float(hr):6.2f}% | mean_abs_px_err={float(err):7.3f}px")

    # 8) proto preprocess -> (1,1000,12)
    X, fs_out = preprocess_single_ecg(
        leads_mV,
        fs_in=int(fs_assumed),
        fs_target=int(fs_assumed),
        target_len=target_len,
        remove_baseline=True,
    )
    print("\n✅ Proto input X:", X.shape, "fs_out:", fs_out)

    npy_path = out_dir / f"proto_input_{src.stem}.npy"
    np.save(str(npy_path), X.astype(np.float32))
    print("✅ Saved proto input:", npy_path)

    # 9) proto predict
    try:
        probs = FusionPredictor.instance().predict_from_X(X)
        probs_path = out_dir / f"proto_probs_{src.stem}.npy"
        np.save(str(probs_path), probs.astype(np.float32))
        print("✅ Saved probs:", probs_path)

        top = np.argsort(-probs)[:10]
        print("\nTop-10 indices + probs:")
        for i in top:
            print(int(i), float(probs[i]))

    except Exception as e:
        print("\n❌ Proto prediction step failed:")
        print("   ", repr(e))
        _print_checkpoint_help()
        print("Pipeline outputs up to proto_input were saved ✅")

    print("\nDONE ✅")
