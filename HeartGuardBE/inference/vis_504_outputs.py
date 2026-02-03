from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .preprocess_3x4 import preprocess_3x4
from .nnunet_runner import run_nnunet
from .postprocess import nii_to_png_mask
from .digitize_binary_3x4 import (
    digitize_from_binary_mask_3x4,
    split_mask_into_cells_3x4,
)

# Printed order (row-major):
LEAD_LAYOUT_3x4 = [
    "I", "aVR", "V1", "V4",
    "II", "aVL", "V2", "V5",
    "III", "aVF", "V3", "V6",
]

BASE = Path(__file__).resolve().parents[2] / "media"


# -----------------------------
# small IO helpers
# -----------------------------
def pick_first_png(folder: Path) -> Path:
    folder = folder.resolve()
    png = next(folder.glob("*.png"), None)
    if png is None:
        raise FileNotFoundError(f"No PNG found in {folder}")
    return png

def read_mask_u8(p: Path) -> np.ndarray:
    m = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(p)
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    if m.dtype != np.uint8:
        m = np.clip(np.rint(m), 0, 255).astype(np.uint8)
    return m

def save_leads_json(leads: dict[str, np.ndarray], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: np.asarray(v).astype(float).tolist() for k, v in leads.items()}
    out_path.write_text(json.dumps(payload), encoding="utf-8")


# -----------------------------
# ECG paper plot (NO CLIPPING)
# -----------------------------
def _draw_ecg_paper(ax, width_mm: float, height_mm: float):
    ax.set_facecolor("white")
    # minor grid 1mm
    for x in np.arange(0, width_mm + 1, 1.0):
        ax.axvline(x, linewidth=0.3, alpha=0.25)
    for y in np.arange(0, height_mm + 1, 1.0):
        ax.axhline(y, linewidth=0.3, alpha=0.25)
    # major grid 5mm
    for x in np.arange(0, width_mm + 1, 5.0):
        ax.axvline(x, linewidth=0.8, alpha=0.35)
    for y in np.arange(0, height_mm + 1, 5.0):
        ax.axhline(y, linewidth=0.8, alpha=0.35)

def _lead_pos_neg_mV(y: np.ndarray) -> tuple[float, float]:
    """pos=max(y), neg=-min(y) ignoring NaN/Inf"""
    y = np.asarray(y, dtype=np.float32)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return 0.0, 0.0
    pos = float(np.max(y))
    neg = float(-np.min(y))
    return max(pos, 0.0), max(neg, 0.0)

def plot_like_original_3x4(
    leads_mV: dict[str, np.ndarray],
    out_png,
    *,
    # --- cell geometry
    cell_w_mm: float = 75.0,
    cell_h_mm: float = 42.0,
    row_gap_mm: float = 10.0,
    col_gap_mm: float = 8.0,

    # --- page padding (prevents edge cut)
    page_pad_left_mm: float = 6.0,
    page_pad_right_mm: float = 6.0,
    page_pad_top_mm: float = 6.0,
    page_pad_bottom_mm: float = 10.0,

    # --- inner padding inside each cell
    inner_top_mm: float = 5.0,
    inner_bottom_mm: float = 5.0,
    inner_x_mm: float = 2.5,

    # --- scaling
    auto_gain: bool = True,
    mm_per_mV: float = 10.0,          # used only if auto_gain=False
    max_gain_mm_per_mV: float = 10.0,
    min_gain_mm_per_mV: float = 0.2,  # keep small (safe). DO NOT set this to 2.
    headroom: float = 1.12,

    # if True: NEVER force min_gain (guarantees no clipping)
    never_clip: bool = True,

    # baseline preference in usable region
    baseline_frac_target: float = 0.50,

    # output sizing
    dpi: int = 260,
    out_width_px: int | None = 6000,
):
    cols, rows = 4, 3

    # --- sanity clamp paddings
    inner_top_mm = float(np.clip(inner_top_mm, 0.0, cell_h_mm - 1e-3))
    inner_bottom_mm = float(np.clip(inner_bottom_mm, 0.0, cell_h_mm - inner_top_mm - 1e-3))
    usable_h = float(max(1e-3, cell_h_mm - inner_top_mm - inner_bottom_mm))

    # --- page size (mm)
    grid_w_mm = cols * cell_w_mm + (cols - 1) * col_gap_mm
    grid_h_mm = rows * cell_h_mm + (rows - 1) * row_gap_mm
    width_mm  = page_pad_left_mm + grid_w_mm + page_pad_right_mm
    height_mm = page_pad_top_mm + grid_h_mm + page_pad_bottom_mm

    # --- choose ONE global gain that fits ALL leads (peak-to-peak based)
    gain = float(mm_per_mV)
    if auto_gain:
        fit_gains = []
        for lead in LEAD_LAYOUT_3x4:
            y = leads_mV.get(lead, None)
            if y is None:
                continue
            pos, neg = _lead_pos_neg_mV(y)
            p2p = pos + neg
            if p2p <= 1e-6:
                continue
            fit_gains.append(usable_h / (headroom * p2p))

        if fit_gains:
            gain_fit = float(min(fit_gains))
        else:
            gain_fit = float(mm_per_mV)

        # cap max gain always
        gain = min(gain_fit, float(max_gain_mm_per_mV))

        # ONLY apply min_gain if you explicitly allow it (otherwise it can clip V3!)
        if not never_clip:
            gain = max(gain, float(min_gain_mm_per_mV))

    # --- figure size
    fig_w_in = width_mm / 25.4
    fig_h_in = height_mm / 25.4
    if out_width_px is not None:
        dpi = int(np.clip(out_width_px / fig_w_in, 80, 1200))

    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=dpi)
    ax = plt.gca()

    _draw_ecg_paper(ax, width_mm, height_mm)

    # paper coords: (0,0) top-left
    ax.set_xlim(0, width_mm)
    ax.set_ylim(height_mm, 0)

    # fill canvas (NO bbox_inches='tight')
    ax.set_position([0, 0, 1, 1])
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    colors = [
        "tab:blue", "tab:orange", "tab:green", "tab:red",
        "tab:purple", "tab:brown", "tab:pink", "tab:gray",
        "tab:olive", "tab:cyan", "tab:blue", "tab:orange",
    ]

    grid_x0 = page_pad_left_mm
    grid_y0 = page_pad_top_mm

    for idx, lead in enumerate(LEAD_LAYOUT_3x4):
        r = idx // cols
        c = idx % cols

        cell_x0 = grid_x0 + c * (cell_w_mm + col_gap_mm)
        cell_y0 = grid_y0 + r * (cell_h_mm + row_gap_mm)

        y = leads_mV.get(lead, None)
        if y is None:
            continue
        y = np.asarray(y, dtype=np.float32)
        if y.size == 0:
            continue

        # ensure no NaN/Inf breaks baseline math
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        x = np.linspace(inner_x_mm, cell_w_mm - inner_x_mm, num=len(y), dtype=np.float32)

        pos, neg = _lead_pos_neg_mV(y)

        # baseline constraints so both + and - fit inside the cell
        b_min = inner_top_mm + headroom * pos * gain
        b_max = (cell_h_mm - inner_bottom_mm) - headroom * neg * gain

        b_target = inner_top_mm + baseline_frac_target * usable_h

        if b_min > b_max:
            # This only happens if input has crazy amplitudes.
            # Final guarantee: shrink gain locally for this lead so it fits.
            p2p = pos + neg
            if p2p > 1e-6:
                local_gain = usable_h / (headroom * p2p)
                local_gain = min(local_gain, gain)  # only shrink, never enlarge
            else:
                local_gain = gain

            b_min = inner_top_mm + headroom * pos * local_gain
            b_max = (cell_h_mm - inner_bottom_mm) - headroom * neg * local_gain
            b = 0.5 * (b_min + b_max)
            y_mm = y * local_gain
        else:
            b = float(np.clip(b_target, b_min, b_max))
            y_mm = y * gain

        baseline = cell_y0 + b

        ax.plot(cell_x0 + x, baseline - y_mm, linewidth=1.35, color=colors[idx])

        ax.text(
            cell_x0 + inner_x_mm + 1.0,
            cell_y0 + 6.0,
            lead,
            fontsize=11,
            fontweight="bold",
            va="top",
            ha="left",
            color="black",
        )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        str(out_png),
        dpi=dpi,
        facecolor="white",
        edgecolor="none",
        transparent=False,
        pad_inches=0.0,
    )
    plt.close(fig)

    print(f"✅ saved: {out_png}")
    print(f"✅ dpi={dpi}  approx_px={int(fig_w_in*dpi)}x{int(fig_h_in*dpi)}  gain={gain:.4f} mm/mV  never_clip={never_clip}")
# -----------------------------
# Optional: truth-check overlay
# -----------------------------
def overlay_mask_vs_rendered_signal(
    mask_u8: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    leads_mV: dict[str, np.ndarray],
    out_png: Path,
    px_per_mV: float,
    baseline_frac: float = 0.55,
):
    """
    GREEN = mask pixels
    RED   = digitized signals rendered back in pixel space
    """
    H, W = mask_u8.shape
    mask01 = (mask_u8 > 0).astype(np.uint8)

    vis = np.zeros((H, W, 3), dtype=np.uint8)
    vis[mask01 > 0] = (0, 255, 0)

    for i, (y1, y2, x1, x2) in enumerate(boxes):
        lead = LEAD_LAYOUT_3x4[i]
        mv = leads_mV.get(lead, None)
        if mv is None:
            continue

        roi_h = y2 - y1
        roi_w = x2 - x1
        if roi_h <= 2 or roi_w <= 2:
            continue

        mv = np.asarray(mv, dtype=np.float32)
        mv_rs = np.interp(
            np.linspace(0.0, 1.0, roi_w, dtype=np.float32),
            np.linspace(0.0, 1.0, len(mv), dtype=np.float32),
            mv,
        ).astype(np.float32)

        baseline = y1 + int(baseline_frac * roi_h)
        y_pix = baseline - (mv_rs * float(px_per_mV))
        y_int = np.clip(np.rint(y_pix).astype(int), y1, y2 - 1)

        xs = np.arange(x1, x2, dtype=np.int32)
        vis[y_int, xs] = (0, 0, 255)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), vis)
    print("✅ Saved:", out_png)


# -----------------------------
# main
# -----------------------------
if __name__ == "__main__":
    src_dir = BASE / "test_in" / "504"
    src = pick_first_png(src_dir)

    out_dir = BASE / "test_out" / "504"
    out_dir.mkdir(parents=True, exist_ok=True)

    px_per_mV = 40.0
    target_len = 1000

    # 1) preprocess
    pre = out_dir / f"pre_{src.stem}.png"
    preprocess_3x4(src, pre)
    print("✅ Preprocessed:", pre)

    # 2) nnUNet 504 -> NIfTI
    pred_nii = run_nnunet(pre, dataset_id="504", is_png=False)
    print("✅ nnU-Net output:", pred_nii)

    # 3) NIfTI -> binary mask PNG
    mask_png = out_dir / f"mask_504_{src.stem}.png"
    nii_to_png_mask(pred_nii, mask_png, binary=True)
    print("✅ Binary mask:", mask_png)

    # 4) digitize -> leads dict (+meta to reuse boxes)
    debug_dir = out_dir / f"debug_digitize_{src.stem}"
    leads_mV, meta = digitize_from_binary_mask_3x4(
        str(mask_png),
        target_len_samples=target_len,
        px_per_mV=px_per_mV,
        debug_dir=str(debug_dir),
        return_meta=True,
    )
    print("✅ Digitized leads. Debug:", debug_dir)

    # Save leads JSON (optional)
    leads_json = out_dir / f"leads_504_{src.stem}.json"
    save_leads_json(leads_mV, leads_json)
    print("✅ Leads JSON:", leads_json)

    # 5) USER-FACING OUTPUT (bigger + spaced + no clipping)
    out_like = out_dir / "digitized_like_original_3x4.png"
    plot_like_original_3x4(
        leads_mV,
        out_like,
        cell_w_mm=75.0,
        cell_h_mm=42.0,
        row_gap_mm=10.0,
        col_gap_mm=8.0,
        auto_gain=True,          # ✅ guarantees V3 won’t clip
        never_clip=True,
        max_gain_mm_per_mV=10.0, # keep realistic when possible
        min_gain_mm_per_mV=2.0,
        dpi=220,
    )

    # 6) Optional truth-check overlay (mask vs rendered signal)
    try:
        mask_u8 = read_mask_u8(mask_png)
        boxes = [tuple(v) for v in meta["boxes"].values()]  # lead->box in order
        overlay_png = out_dir / "mask_vs_rendered_overlay.png"
        overlay_mask_vs_rendered_signal(
            mask_u8=mask_u8,
            boxes=boxes,
            leads_mV=leads_mV,
            out_png=overlay_png,
            px_per_mV=px_per_mV,
            baseline_frac=0.55,
        )
    except Exception as e:
        print("⚠️ overlay failed:", repr(e))

    print("\nDONE ✅")
