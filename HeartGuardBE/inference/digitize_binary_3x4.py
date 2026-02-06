from __future__ import annotations

from pathlib import Path
import json
from typing import Tuple, Optional

import numpy as np
import cv2


STD_LEADS = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]

LEAD_LAYOUT_3x4 = [
    "I",  "aVR", "V1", "V4",
    "II", "aVL", "V2", "V5",
    "III","aVF", "V3", "V6",
]


def hampel_filter_1d(x: np.ndarray, window: int = 7, n_sigmas: float = 3.0) -> np.ndarray:
    """
    Hampel filter for outlier removal. Keeps morphology better than median-only smoothing.
    window: odd number, typical 7..21
    """
    x = np.asarray(x, dtype=np.float32)
    n = x.size
    if n < 3:
        return x.copy()

    w = int(window) | 1
    k = w // 2

    y = x.copy()
    # scale factor for MAD->sigma for normal dist
    c = 1.4826

    for i in range(n):
        a = max(0, i - k)
        b = min(n, i + k + 1)
        med = np.median(x[a:b])
        mad = c * np.median(np.abs(x[a:b] - med))
        if mad < 1e-12:
            continue
        if np.abs(x[i] - med) > n_sigmas * mad:
            y[i] = med
    return y


def savgol_smooth_1d(x: np.ndarray, window: int = 11, poly: int = 3) -> np.ndarray:
    """
    Savitzky–Golay smoothing via local polynomial least squares (no SciPy).
    window: odd number (9..21 typical). poly: 2 or 3.
    Preserves QRS shape better than moving average.
    """
    x = np.asarray(x, dtype=np.float32)
    n = x.size
    if n < 5:
        return x.copy()

    w = int(window) | 1
    w = min(w, n if n % 2 == 1 else n - 1)
    if w < 5:
        return x.copy()

    poly = int(poly)
    poly = max(1, min(poly, w - 2))
    half = w // 2

    # Build Vandermonde for positions [-half..half]
    t = np.arange(-half, half + 1, dtype=np.float32)
    A = np.vstack([t**p for p in range(poly + 1)]).T  # (w, poly+1)

    # Compute convolution coeffs for smoothing (evaluate at t=0)
    # Coeffs = e0^T (A^+), where e0 selects constant term at t=0
    pinv = np.linalg.pinv(A)  # (poly+1, w)
    coeff = pinv[0].astype(np.float32)  # weights for y(0)

    # Pad edge with reflection to avoid edge shrink
    xpad = np.pad(x, (half, half), mode="reflect")
    y = np.convolve(xpad, coeff[::-1], mode="valid").astype(np.float32)
    return y


def smooth_ecg_lead_mV(
    y: np.ndarray,
    *,
    hampel_window: int = 7,
    hampel_sigmas: float = 3.0,
    sg_window: int = 11,
    sg_poly: int = 3,
    blend: float = 1.0,
) -> np.ndarray:
    """
    Safe smoothing pipeline:
      - Hampel removes tiny digitization glitches
      - SavGol lightly smooths while preserving morphology
    blend: 0..1, how much smoothed replaces original (1=full smoothed)
    """
    y = np.asarray(y, dtype=np.float32)
    if y.size < 5:
        return y.copy()

    y1 = hampel_filter_1d(y, window=hampel_window, n_sigmas=hampel_sigmas)
    y2 = savgol_smooth_1d(y1, window=sg_window, poly=sg_poly)

    blend = float(np.clip(blend, 0.0, 1.0))
    return (1.0 - blend) * y + blend * y2


# -------------------------------
# IO + basic helpers
# -------------------------------
def load_mask_u8(p: str) -> np.ndarray:
    m = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(p)
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    if m.dtype != np.uint8:
        m = np.clip(np.rint(m), 0, 255).astype(np.uint8)
    return m


def clean_bin_mask(
    bin01: np.ndarray,
    ksize: int = 3,
    iters_close: int = 1,
    do_open: bool = False,   # IMPORTANT: keep OFF by default
) -> np.ndarray:
    """
    CLOSE fills tiny gaps.
    OPEN can shave QRS tips -> default OFF.
    """
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    x = (bin01 > 0).astype(np.uint8) * 255
    x = cv2.morphologyEx(x, cv2.MORPH_CLOSE, k, iterations=iters_close)
    if do_open:
        x = cv2.morphologyEx(x, cv2.MORPH_OPEN, k, iterations=1)
    return (x > 0).astype(np.uint8)


def pixel_y_to_mV(y_pix: np.ndarray, px_per_mV: float, baseline_pix: float) -> np.ndarray:
    # ECG convention: up is positive, image y grows downward
    y_pix = np.asarray(y_pix, np.float32)
    return ((baseline_pix - y_pix) / float(px_per_mV)).astype(np.float32)


def _smooth_1d(x: np.ndarray, k: int) -> np.ndarray:
    k = max(3, int(k) | 1)
    w = np.ones((k,), dtype=np.float32) / float(k)
    return np.convolve(x.astype(np.float32), w, mode="same")


def _crop_to_foreground(bin01: np.ndarray, margin_frac: float = 0.01) -> tuple[np.ndarray, tuple[int,int,int,int]]:
    ys, xs = np.where(bin01 > 0)
    H, W = bin01.shape
    if xs.size == 0:
        return bin01, (0, H, 0, W)

    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1

    my = int(margin_frac * H)
    mx = int(margin_frac * W)

    y1 = max(0, y1 - my); y2 = min(H, y2 + my)
    x1 = max(0, x1 - mx); x2 = min(W, x2 + mx)

    return bin01[y1:y2, x1:x2], (y1, y2, x1, x2)


def _search_min(profile: np.ndarray, center: int, halfwin: int) -> int:
    n = len(profile)
    a = max(0, center - halfwin)
    b = min(n, center + halfwin + 1)
    j = int(np.argmin(profile[a:b]))
    return a + j


# -------------------------------
# Stable split: 3 rows x 4 cols
# -------------------------------
def split_mask_into_cells_3x4(
    bin01: np.ndarray,
    pad_x: int = 0,   # default 0: don't cut tips
    pad_y: int = 0,   # default 0: don't cut tips
) -> tuple[list[tuple[int,int,int,int]], list[int], list[int]]:
    cropped, (Y1, Y2, X1, X2) = _crop_to_foreground(bin01, margin_frac=0.01)
    H, W = cropped.shape

    col_prof = cropped.sum(axis=0).astype(np.float32)
    row_prof = cropped.sum(axis=1).astype(np.float32)

    col_s = _smooth_1d(col_prof, k=max(31, W // 70))
    row_s = _smooth_1d(row_prof, k=max(31, H // 70))

    v_targets = [int(W * 0.25), int(W * 0.50), int(W * 0.75)]
    h_targets = [int(H * (1/3)), int(H * (2/3))]

    v_halfwin = max(10, int(W * 0.08))
    h_halfwin = max(10, int(H * 0.10))

    v_seps = sorted({_search_min(col_s, t, v_halfwin) for t in v_targets})
    h_seps = sorted({_search_min(row_s, t, h_halfwin) for t in h_targets})

    if len(v_seps) != 3:
        v_seps = [int(W*0.25), int(W*0.50), int(W*0.75)]
    if len(h_seps) != 2:
        h_seps = [int(H*(1/3)), int(H*(2/3))]

    xs = [0] + v_seps + [W]
    ys = [0] + h_seps + [H]

    boxes: list[tuple[int,int,int,int]] = []
    for r in range(3):
        for c in range(4):
            x1, x2 = xs[c], xs[c+1]
            y1, y2 = ys[r], ys[r+1]

            x1 = min(max(0, x1 + pad_x), W - 2)
            x2 = min(max(x1 + 1, x2 - pad_x), W)
            y1 = min(max(0, y1 + pad_y), H - 2)
            y2 = min(max(y1 + 1, y2 - pad_y), H)

            boxes.append((Y1 + y1, Y1 + y2, X1 + x1, X1 + x2))

    return boxes, h_seps, v_seps


# -------------------------------
# Baseline + trace extraction
# -------------------------------
def baseline_mode_from_rows(roi01: np.ndarray, band=(0.25, 0.75)) -> float:
    H = roi01.shape[0]
    row_counts = roi01.sum(axis=1).astype(np.int32)
    if row_counts.max() <= 0:
        return float(H * 0.55)

    a = max(0, min(H-1, int(band[0] * H)))
    b = max(a+1, min(H, int(band[1] * H)))

    mid = row_counts[a:b]
    if mid.max() > 0:
        return float(a + int(np.argmax(mid)))
    return float(int(np.argmax(row_counts)))


def extract_trace_y_from_mask(
    roi01: np.ndarray,
    *,
    baseline_pix: Optional[float] = None,
    min_pixels_per_col: int = 1,
    spike_span_frac: float = 0.22,
) -> Tuple[np.ndarray, np.ndarray]:

    H, W = roi01.shape
    xs: list[int] = []
    ys: list[float] = []

    if baseline_pix is None:
        row_counts = roi01.sum(axis=1).astype(np.int32)
        if row_counts.max() <= 0:
            baseline_pix = float(H * 0.55)
        else:
            a = max(0, min(H - 1, int(0.25 * H)))
            b = max(a + 1, min(H, int(0.75 * H)))
            mid = row_counts[a:b]
            baseline_pix = float(a + int(np.argmax(mid))) if mid.max() > 0 else float(int(np.argmax(row_counts)))

    prev_y = np.nan
    prev_dy = 0.0
    side = 0  # -1 above baseline, +1 below baseline

    # --- NEW knobs ---
    max_jump = 0.40 * H       # cap vertical jump per column
    alpha_vel = 0.35          # smooth velocity update
    edge_margin = 0.04 * H    # how far to push off border when clipped

    for x in range(W):
        rows = np.flatnonzero(roi01[:, x])
        if rows.size < min_pixels_per_col:
            continue

        y_top = float(rows.min())
        y_bot = float(rows.max())
        y_med = float(np.median(rows))
        span = y_bot - y_top

        top_touch = (y_top <= 1.0)
        bot_touch = (y_bot >= H - 2.0)

        if not np.isfinite(prev_y):
            y = y_med
            side = -1 if y < baseline_pix else +1
        else:
            pred = prev_y + prev_dy

            if span >= spike_span_frac * H:
                desired_side = -1 if pred < baseline_pix else +1
                cand = y_top if desired_side < 0 else y_bot
                cand_side = -1 if cand < baseline_pix else +1

                flip_margin = 0.06 * H
                if cand_side != side and abs(prev_y - baseline_pix) > flip_margin:
                    cand = y_top if side < 0 else y_bot

                # --- NEW: avoid trusting clipped border extremes ---
                if top_touch and cand == y_top:
                    cand = min(y_bot, y_top + edge_margin)
                if bot_touch and cand == y_bot:
                    cand = max(y_top, y_bot - edge_margin)

                y = float(cand)
                side = -1 if y < baseline_pix else +1
            else:
                candidates = np.array([y_top, y_med, y_bot], dtype=np.float32)

                # --- NEW: penalize border-touch candidates slightly (reduces clipping lock) ---
                edge_pen = np.zeros_like(candidates)
                if top_touch:
                    edge_pen[0] += 0.35 * H
                if bot_touch:
                    edge_pen[2] += 0.35 * H

                jump_pen = 0.12 * np.abs(candidates - prev_y)
                slope_pen = np.abs(candidates - pred)

                y = float(candidates[int(np.argmin(slope_pen + jump_pen + edge_pen))])
                side = -1 if y < baseline_pix else +1

        # --- NEW: hard jump cap to prevent teleports ---
        if np.isfinite(prev_y):
            dy = y - prev_y
            if abs(dy) > max_jump:
                y = float(prev_y + np.clip(dy, -max_jump, max_jump))

        xs.append(x)
        ys.append(y)

        # --- NEW: velocity smoothing ---
        if np.isfinite(prev_y):
            inst = float(y - prev_y)
            prev_dy = (1 - alpha_vel) * prev_dy + alpha_vel * inst
        prev_y = float(y)

    if not xs:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float32)

    return np.asarray(xs, np.int32), np.asarray(ys, np.float32)


def resample_y_over_x(xs: np.ndarray, ys: np.ndarray, out_len: int) -> np.ndarray:
    xs = np.asarray(xs, np.float32)
    ys = np.asarray(ys, np.float32)
    if xs.size < 2:
        return np.zeros((out_len,), dtype=np.float32)

    x0, x1 = float(xs.min()), float(xs.max())
    x_new = np.linspace(x0, x1, out_len, dtype=np.float32)
    y_new = np.interp(x_new, xs, ys).astype(np.float32)
    return y_new


# -------------------------------
# CORRECT metrics (measured on ROI grid, not on resampled array)
# -------------------------------
def mean_pixel_error_on_roi(roi01: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> float:
    H, W = roi01.shape
    if xs.size == 0:
        return float("inf")

    errs = []
    for x, y in zip(xs.tolist(), ys.tolist()):
        x = int(x)
        if x < 0 or x >= W:
            continue
        col = np.flatnonzero(roi01[:, x])
        if col.size == 0:
            continue
        errs.append(float(np.min(np.abs(col.astype(np.float32) - float(y)))))
    return float(np.mean(errs)) if errs else float("inf")


def hit_rate_on_roi(roi01: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> float:
    H, W = roi01.shape
    if xs.size == 0:
        return 0.0

    hits = 0
    tot = 0
    for x, y in zip(xs.tolist(), ys.tolist()):
        x = int(x)
        if x < 0 or x >= W:
            continue
        yy = int(np.clip(np.rint(y), 0, H - 1))
        tot += 1
        if roi01[yy, x] > 0:
            hits += 1
    return float(hits / max(1, tot))


def _debug_draw_boxes(mask_u8: np.ndarray, boxes: list[tuple[int,int,int,int]], out_png: Path):
    vis = cv2.cvtColor((mask_u8 > 0).astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
    for i, (y1,y2,x1,x2) in enumerate(boxes):
        cv2.rectangle(vis, (x1,y1), (x2-1,y2-1), (0,255,0), 2)
        label = LEAD_LAYOUT_3x4[i]
        cv2.putText(vis, label, (x1+6, y1+26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), vis)


# -------------------------------
# Main API
# -------------------------------
def digitize_from_binary_mask_3x4(
    mask_png_path: str,
    target_len_samples: int,
    px_per_mV: float = 40.0,
    debug_dir: str | None = None,
    return_meta: bool = False,
    *,
    split_pad_x: int = 0,          # keep 0
    split_pad_y: int = 0,          # keep 0
    clean_do_open: bool = False,   # keep False
    prefer: str = "median",
):
    mask_u8 = load_mask_u8(mask_png_path)
    bin01 = (mask_u8 > 0).astype(np.uint8)
    bin01 = clean_bin_mask(bin01, ksize=3, iters_close=1, do_open=clean_do_open)

    boxes, _, _ = split_mask_into_cells_3x4(bin01, pad_x=split_pad_x, pad_y=split_pad_y)

    dbg = Path(debug_dir) if debug_dir else None
    if dbg:
        dbg.mkdir(parents=True, exist_ok=True)
        _debug_draw_boxes(mask_u8, boxes, dbg / "rois_overlay.png")

    layout_leads_mV: dict[str, np.ndarray] = {}
    meta = {
        "px_per_mV": float(px_per_mV),
        "target_len": int(target_len_samples),
        "boxes": {},          # lead -> [y1,y2,x1,x2]
        "baseline_pix": {},   # lead -> baseline in ROI coords
        "xs_pix": {},         # lead -> sparse x samples (ROI coords)
        "ys_pix": {},         # lead -> sparse y samples (ROI coords)
        "mean_abs_px_err": {},# lead -> float (computed on xs/ys)
        "hit_rate": {},       # lead -> float (computed on xs/ys)
    }

    for i, (y1, y2, x1, x2) in enumerate(boxes):
        lead = LEAD_LAYOUT_3x4[i]
        roi = bin01[y1:y2, x1:x2].astype(np.uint8)
        meta["boxes"][lead] = [int(y1), int(y2), int(x1), int(x2)]

        if int(roi.sum()) < 30:
            layout_leads_mV[lead] = np.zeros((target_len_samples,), dtype=np.float32)
            meta["baseline_pix"][lead] = float(roi.shape[0] * 0.55)
            meta["xs_pix"][lead] = []
            meta["ys_pix"][lead] = []
            meta["mean_abs_px_err"][lead] = None
            meta["hit_rate"][lead] = None
            continue

        baseline = baseline_mode_from_rows(roi)
        xs, ys = extract_trace_y_from_mask(
            roi,
            baseline_pix=baseline,
            min_pixels_per_col=1,
            spike_span_frac=0.22,
        )

        if xs.size < 2:
            layout_leads_mV[lead] = np.zeros((target_len_samples,), dtype=np.float32)
            meta["baseline_pix"][lead] = float(baseline)
            meta["xs_pix"][lead] = []
            meta["ys_pix"][lead] = []
            meta["mean_abs_px_err"][lead] = None
            meta["hit_rate"][lead] = None
            continue

        # resample ONLY for the output signal length
        y_resampled = resample_y_over_x(xs, ys, out_len=int(target_len_samples))
        mv = pixel_y_to_mV(y_resampled, px_per_mV=float(px_per_mV), baseline_pix=float(baseline))
        mv = smooth_ecg_lead_mV(
            mv,
            hampel_window=7,
            hampel_sigmas=3.0,
            sg_window=11,
            sg_poly=3,
            blend=1.0,
        )
        layout_leads_mV[lead] = mv.astype(np.float32)

        # store meta + CORRECT metrics on xs/ys
        meta["baseline_pix"][lead] = float(baseline)
        meta["xs_pix"][lead] = xs.astype(int).tolist()
        meta["ys_pix"][lead] = ys.astype(np.float32).tolist()
        meta["mean_abs_px_err"][lead] = float(mean_pixel_error_on_roi(roi, xs, ys))
        meta["hit_rate"][lead] = float(hit_rate_on_roi(roi, xs, ys))

        if dbg:
            vis = cv2.cvtColor((roi * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            for x, y in zip(xs.tolist(), ys.tolist()):
                yy = int(np.clip(np.rint(y), 0, roi.shape[0]-1))
                vis[yy, int(x)] = (0, 0, 255)
            cv2.imwrite(str(dbg / f"overlay_{lead}.png"), vis)

    leads_mV: dict[str, np.ndarray] = {}
    for k in STD_LEADS:
        leads_mV[k] = layout_leads_mV.get(k, np.zeros((target_len_samples,), dtype=np.float32))

    if dbg:
        (dbg / "digitize_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if return_meta:
        return leads_mV, meta
    return leads_mV