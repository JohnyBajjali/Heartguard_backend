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

def digitize_from_multiclass_mask(
    mask_png_path: str,
    target_len_samples: int,
    px_per_mV: float | None = None,
) -> dict[str, np.ndarray]:
    """
    mask values: 0 bg, 1..12 each lead class.
    Returns leads_mV dict with arrays length target_len_samples.
    """
    mask = load_mask_u8(mask_png_path)
    px_per_mV = float(px_per_mV or estimate_px_per_mV_default())

    H, W = mask.shape
    leads_mV: dict[str, np.ndarray] = {}

    for class_id, lead_name in enumerate(STD_LEADS, start=1):
        bin01 = (mask == class_id).astype(np.uint8)

        if bin01.sum() < 30:
            leads_mV[lead_name] = np.zeros((target_len_samples,), dtype=np.float32)
            continue

        sk = skeletonize_bin(bin01)
        y = extract_y_per_x_median(sk)
        y = fill_nans_linear(y)

        baseline = float(np.median(y))
        mv = ypix_to_mV(y, baseline_pix=baseline, px_per_mV=px_per_mV)

        mv = resample_1d(mv, target_len_samples)
        leads_mV[lead_name] = mv

    return leads_mV
