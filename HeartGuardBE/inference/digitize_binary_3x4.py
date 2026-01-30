from __future__ import annotations
import numpy as np

from .digitize_common import (
    load_mask_u8,
    skeletonize_bin,
    extract_y_per_x_median,
    fill_nans_linear,
    estimate_px_per_mV_default,
    ypix_to_mV,
    resample_1d,
)

STD_LEADS = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]

def _split_rois_3x4(H: int, W: int) -> list[tuple[int,int,int,int]]:
    """
    Layout assumption (common 3x4 print):
      Row1: I,   aVR, V1, V4
      Row2: II,  aVL, V2, V5
      Row3: III, aVF, V3, V6
    """
    top = int(0.07 * H)
    bot = int(0.93 * H)
    left = int(0.05 * W)
    right = int(0.95 * W)

    H2 = bot - top
    W2 = right - left
    rh = H2 // 3
    cw = W2 // 4

    rois = []
    for r in range(3):
        for c in range(4):
            y1 = top + r * rh
            y2 = top + (r + 1) * rh
            x1 = left + c * cw
            x2 = left + (c + 1) * cw
            rois.append((y1, y2, x1, x2))

    mapping = {
        "I":   0, "aVR": 1, "V1":  2, "V4":  3,
        "II":  4, "aVL": 5, "V2":  6, "V5":  7,
        "III": 8, "aVF": 9, "V3": 10, "V6": 11,
    }
    return [rois[mapping[n]] for n in STD_LEADS]

def digitize_from_binary_mask_3x4(
    mask_png_path: str,
    target_len_samples: int,
    px_per_mV: float | None = None,
) -> dict[str, np.ndarray]:
    """
    mask: binary (0 bg, >0 trace)
    Returns leads_mV dict.
    """
    mask_u8 = load_mask_u8(mask_png_path)
    bin01 = (mask_u8 > 0).astype(np.uint8)

    px_per_mV = float(px_per_mV or estimate_px_per_mV_default())

    H, W = bin01.shape
    rois = _split_rois_3x4(H, W)

    leads_mV: dict[str, np.ndarray] = {}

    for lead_name, (y1, y2, x1, x2) in zip(STD_LEADS, rois):
        roi = bin01[y1:y2, x1:x2].astype(np.uint8)

        if roi.sum() < 30:
            leads_mV[lead_name] = np.zeros((target_len_samples,), dtype=np.float32)
            continue

        sk = skeletonize_bin(roi)
        y = extract_y_per_x_median(sk)
        y = fill_nans_linear(y)

        baseline = float(np.median(y))
        mv = ypix_to_mV(y, baseline_pix=baseline, px_per_mV=px_per_mV)

        mv = resample_1d(mv, target_len_samples)
        leads_mV[lead_name] = mv

    return leads_mV
