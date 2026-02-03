from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import cv2

STD_LEADS = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]

LEAD_LAYOUT_3x4 = [
    "I",  "aVR", "V1", "V4",
    "II", "aVL", "V2", "V5",
    "III","aVF", "V3", "V6",
]

# -------------------------------
# IO + utilities
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
    do_open: bool = False,   # ✅ IMPORTANT: default OFF to avoid shaving QRS tips
) -> np.ndarray:
    """
    CLOSE fills tiny gaps in the trace.
    OPEN can erase thin spike tips -> keep OFF unless your mask is very noisy.
    """
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    x = (bin01 > 0).astype(np.uint8) * 255
    x = cv2.morphologyEx(x, cv2.MORPH_CLOSE, k, iterations=iters_close)
    if do_open:
        x = cv2.morphologyEx(x, cv2.MORPH_OPEN, k, iterations=1)
    return (x > 0).astype(np.uint8)


def resample_1d(y: np.ndarray, target_len: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if len(y) == target_len:
        return y
    x_old = np.linspace(0.0, 1.0, num=len(y), dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, num=target_len, dtype=np.float32)
    return np.interp(x_new, x_old, y).astype(np.float32)


def fill_nans_linear(y: np.ndarray) -> np.ndarray:
    y = y.astype(np.float32)
    n = len(y)
    x = np.arange(n, dtype=np.float32)
    good = np.isfinite(y)
    if good.sum() == 0:
        return np.zeros((n,), dtype=np.float32)
    if good.sum() == 1:
        y[:] = y[good][0]
        return y
    y[~good] = np.interp(x[~good], x[good], y[good]).astype(np.float32)
    return y


def ypix_to_mV(y_pix: np.ndarray, baseline_pix: float, px_per_mV: float) -> np.ndarray:
    # up = positive (image y grows downward)
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
# Stable split (bbox + constrained minima)
# -------------------------------
def split_mask_into_cells_3x4(
    bin01: np.ndarray,
    pad_x: int = 0,   # ✅ default 0 so we don't cut V3/V2 tips
    pad_y: int = 0,   # ✅ default 0 so we don't cut bottom spikes
) -> tuple[list[tuple[int,int,int,int]], list[int], list[int]]:
    """
    Returns:
      boxes: 12 boxes (y1,y2,x1,x2) row-major in FULL image coords
      h_seps, v_seps in CROPPED coords (debug)
    """
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

    v_seps = [_search_min(col_s, t, v_halfwin) for t in v_targets]
    h_seps = [_search_min(row_s, t, h_halfwin) for t in h_targets]

    v_seps = sorted(list(dict.fromkeys(v_seps)))
    h_seps = sorted(list(dict.fromkeys(h_seps)))

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

            # shrink a little if you want, but default is 0 (no shrink)
            x1 = min(max(0, x1 + pad_x), W - 2)
            x2 = min(max(x1 + 1, x2 - pad_x), W)
            y1 = min(max(0, y1 + pad_y), H - 2)
            y2 = min(max(y1 + 1, y2 - pad_y), H)

            boxes.append((Y1 + y1, Y1 + y2, X1 + x1, X1 + x2))

    return boxes, h_seps, v_seps


# -------------------------------
# Baseline + spike-aware trace extraction
# -------------------------------
def _baseline_mode_from_rows(roi01: np.ndarray, band=(0.25, 0.75)) -> float:
    """
    Baseline from row-histogram peak, but restricted to the middle band so
    huge QRS columns don't hijack the mode.
    """
    H = roi01.shape[0]
    row_counts = roi01.sum(axis=1).astype(np.int32)
    if row_counts.max() <= 0:
        return float(H * 0.55)

    a = int(band[0] * H)
    b = int(band[1] * H)
    a = max(0, min(H-1, a))
    b = max(a+1, min(H, b))

    mid = row_counts[a:b]
    if mid.max() > 0:
        return float(a + int(np.argmax(mid)))

    return float(int(np.argmax(row_counts)))


def extract_trace_spike_aware(roi01: np.ndarray) -> np.ndarray:
    """
    Stronger + more consistent spike handling:
    - keeps a persistent "side" (above/below baseline) through a spike
    - prefers extremes in tall columns, not the median
    """
    H, W = roi01.shape
    y = np.full((W,), np.nan, dtype=np.float32)

    col_counts = roi01.sum(axis=0).astype(np.int32)
    bad_cols = col_counts >= int(0.90 * H)  # stricter; ignore huge vertical junk

    baseline_est = _baseline_mode_from_rows(roi01)

    prev_y = np.nan
    prev_dy = 0.0

    # persistent side state: -1 means above baseline, +1 below baseline
    side = 0

    # centroid for trend hint
    y_cent = np.full((W,), np.nan, dtype=np.float32)
    for x in range(W):
        ys = np.where(roi01[:, x] > 0)[0]
        if ys.size:
            y_cent[x] = float(np.mean(ys))

    for x in range(W):
        if bad_cols[x]:
            continue

        ys = np.where(roi01[:, x] > 0)[0]
        if ys.size == 0:
            continue

        y_top = float(ys.min())
        y_bot = float(ys.max())
        y_med = float(np.median(ys))
        span = y_bot - y_top

        if not np.isfinite(prev_y):
            y[x] = y_med
            prev_y = y[x]
            prev_dy = 0.0
            side = -1 if prev_y < baseline_est else +1
            continue

        dy_hint = None
        if x + 1 < W and np.isfinite(y_cent[x]) and np.isfinite(y_cent[x + 1]):
            dy_hint = float(y_cent[x + 1] - y_cent[x])

        # tall column => pick an EXTREME consistently
        if span >= 0.22 * H:
            going_up = (dy_hint < 0) if dy_hint is not None else (prev_dy < 0)
            going_down = (dy_hint > 0) if dy_hint is not None else (prev_dy > 0)

            if going_up:
                desired = -1
            elif going_down:
                desired = +1
            else:
                desired = side

            # choose extreme matching desired side:
            cand = y_top if desired < 0 else y_bot

            # but if previous y is clearly on the other side, don't flip too easily
            flip_margin = 0.06 * H
            if desired != side and abs(prev_y - baseline_est) > flip_margin:
                cand = y_top if side < 0 else y_bot

            y[x] = cand
            side = -1 if y[x] < baseline_est else +1

        else:
            # normal column: follow slope while staying near the trace
            pred = prev_y + prev_dy
            candidates = np.array([y_top, y_med, y_bot], dtype=np.float32)

            # small penalty for jumping between extremes (helps consistency)
            jump_pen = 0.12 * np.abs(candidates - prev_y)
            slope_pen = np.abs(candidates - pred)
            y[x] = float(candidates[int(np.argmin(slope_pen + jump_pen))])

            side = -1 if y[x] < baseline_est else +1

        prev_dy = float(y[x] - prev_y)
        prev_y = float(y[x])

    y = fill_nans_linear(y)

    # ultra-mild smoothing (keeps spikes)
    k = 3
    if W >= k:
        pad = k // 2
        ypad = np.pad(y, (pad, pad), mode="edge")
        y = np.convolve(ypad, np.ones(k, dtype=np.float32) / k, mode="valid").astype(np.float32)

    return y.astype(np.float32)


def _mean_pixel_error(roi01: np.ndarray, y: np.ndarray) -> float:
    H, W = roi01.shape
    y = np.asarray(y, dtype=np.float32)
    errs = []
    for x in range(min(W, y.size)):
        ys = np.where(roi01[:, x] > 0)[0]
        if ys.size == 0:
            continue
        errs.append(float(np.min(np.abs(ys.astype(np.float32) - y[x]))))
    return float(np.mean(errs)) if errs else float("inf")


def _hit_rate(roi01: np.ndarray, y: np.ndarray) -> float:
    H, W = roi01.shape
    y = np.asarray(y, dtype=np.float32)
    hits = 0
    tot = 0
    for x in range(min(W, y.size)):
        yy = int(np.clip(np.rint(y[x]), 0, H - 1))
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
    split_pad_x: int = 0,          # ✅ default no shrink
    split_pad_y: int = 0,          # ✅ default no shrink
    clean_do_open: bool = False,   # ✅ default OFF
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
        "boxes": {},               # lead -> [y1,y2,x1,x2]
        "baseline_pix": {},        # lead -> baseline in ROI coords
        "y_path_pix": {},          # lead -> list length roi_w (ROI coords)
        "mean_abs_px_err": {},     # lead -> float
        "hit_rate": {},            # lead -> float (0..1)
    }

    for i, (y1, y2, x1, x2) in enumerate(boxes):
        lead = LEAD_LAYOUT_3x4[i]
        roi = bin01[y1:y2, x1:x2].astype(np.uint8)

        meta["boxes"][lead] = [int(y1), int(y2), int(x1), int(x2)]

        if int(roi.sum()) < 30:
            layout_leads_mV[lead] = np.zeros((target_len_samples,), dtype=np.float32)
            meta["baseline_pix"][lead] = float(roi.shape[0] * 0.55)
            meta["y_path_pix"][lead] = []
            meta["mean_abs_px_err"][lead] = None
            meta["hit_rate"][lead] = None
            continue

        y_path = extract_trace_spike_aware(roi)  # ROI coords
        baseline = _baseline_mode_from_rows(roi) # ROI coords

        mv = ypix_to_mV(y_path, baseline_pix=baseline, px_per_mV=px_per_mV)
        mv = resample_1d(mv, target_len_samples).astype(np.float32)
        layout_leads_mV[lead] = mv

        meta["baseline_pix"][lead] = float(baseline)
        meta["y_path_pix"][lead] = y_path.astype(np.float32).tolist()
        meta["mean_abs_px_err"][lead] = float(_mean_pixel_error(roi, y_path))
        meta["hit_rate"][lead] = float(_hit_rate(roi, y_path))

        if dbg:
            vis = cv2.cvtColor((roi * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            y_int = np.clip(np.rint(y_path).astype(int), 0, roi.shape[0] - 1)
            for xx in range(min(roi.shape[1], y_int.size)):
                vis[y_int[xx], xx] = (0, 0, 255)
            cv2.imwrite(str(dbg / f"overlay_{lead}.png"), vis)

    leads_mV: dict[str, np.ndarray] = {}
    for k in STD_LEADS:
        leads_mV[k] = layout_leads_mV.get(k, np.zeros((target_len_samples,), dtype=np.float32))

    if dbg:
        (dbg / "digitize_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if return_meta:
        return leads_mV, meta
    return leads_mV
