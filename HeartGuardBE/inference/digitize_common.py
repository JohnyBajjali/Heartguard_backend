from __future__ import annotations
import numpy as np
import cv2

try:
    from skimage.morphology import skeletonize
except Exception as e:
    skeletonize = None


def _require_skeletonize():
    if skeletonize is None:
        raise ImportError("scikit-image is required: pip install scikit-image")


def load_gray_u8(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    return img


def load_mask_u8(path: str) -> np.ndarray:
    m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if m is None:
        raise ValueError(f"Failed to read mask: {path}")
    if m.ndim == 3:
        m = m[:, :, 0]
    return m.astype(np.uint8)


def skeletonize_bin(bin01: np.ndarray) -> np.ndarray:
    _require_skeletonize()
    sk = skeletonize(bin01.astype(bool))
    return sk.astype(np.uint8)


def extract_y_per_x_median(skel01: np.ndarray) -> np.ndarray:
    """
    For each x, choose median y among skeleton pixels in that column.
    Missing columns -> NaN.
    """
    H, W = skel01.shape
    y = np.full((W,), np.nan, dtype=np.float32)
    for x in range(W):
        ys = np.where(skel01[:, x] > 0)[0]
        if len(ys) == 0:
            continue
        y[x] = float(np.median(ys))
    return y


def fill_nans_linear(y: np.ndarray) -> np.ndarray:
    x = np.arange(len(y), dtype=np.float32)
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        # degenerate, return zeros
        return np.nan_to_num(y, nan=0.0).astype(np.float32)
    return np.interp(x, x[mask], y[mask]).astype(np.float32)


def estimate_px_per_mV_default() -> float:
    """
    You MUST tune this once for your scan DPI/layout.
    If you have calibration pulse detection later, we can replace this.
    """
    return 40.0  # adjust if amplitude looks wrong


def ypix_to_mV(y_pix: np.ndarray, baseline_pix: float, px_per_mV: float) -> np.ndarray:
    """
    Pixel y grows downward; ECG amplitude grows upward -> invert about baseline.
    """
    return ((baseline_pix - y_pix) / px_per_mV).astype(np.float32)


def resample_1d(y: np.ndarray, target_len: int) -> np.ndarray:
    n = len(y)
    if n == target_len:
        return y.astype(np.float32)
    x_old = np.linspace(0, 1, num=n, dtype=np.float32)
    x_new = np.linspace(0, 1, num=target_len, dtype=np.float32)
    return np.interp(x_new, x_old, y).astype(np.float32)
