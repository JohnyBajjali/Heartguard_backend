import cv2
import numpy as np


def _decode_image(image_bytes: bytes):
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
    if img is None:
        raise ValueError("Invalid image data")
    return img


def _encode_jpeg(img_bgr, quality=92) -> bytes:
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise ValueError("Failed to encode image")
    return buf.tobytes()


def _largest_contour_bbox(mask: np.ndarray):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    # reject tiny regions
    if w * h < 0.10 * mask.shape[0] * mask.shape[1]:
        return None
    return x, y, w, h


def _trim_by_projection(img_bgr: np.ndarray, axis: int):
    """
    axis=0 -> trim top/bottom
    axis=1 -> trim left/right
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 60, 160)

    proj = edges.sum(axis=axis).astype(np.float32)

    k = 25
    kernel = np.ones(k, dtype=np.float32) / k
    proj_s = np.convolve(proj, kernel, mode="same")

    mx = float(proj_s.max() if proj_s.size else 0.0)
    if mx <= 1e-6:
        return img_bgr

    thr = 0.12 * mx
    idx = np.where(proj_s > thr)[0]
    if idx.size == 0:
        return img_bgr

    a = int(idx[0])
    b = int(idx[-1])

    margin = 8
    a = max(0, a - margin)
    b = min((img_bgr.shape[0] if axis == 1 else img_bgr.shape[1]) - 1, b + margin)

    if axis == 0:
        return img_bgr[a:b + 1, :]
    else:
        return img_bgr[:, a:b + 1]


def auto_crop_ecg_bytes(image_bytes: bytes) -> bytes:
    """
    ECG auto-crop (like your 2nd image):
    1) Detect ECG paper (pink/red grid) by HSV
    2) Crop to paper bounding box
    3) Remove ONLY the footer band (~10-12%) so rhythm strip is kept
    4) Trim borders safely using edge projection
    5) Return JPEG bytes
    """
    img = _decode_image(image_bytes)
    H, W = img.shape[:2]

    # 1) Detect ECG paper (pink/red) in HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # red/pink hue ranges: near 0 and near 180
    mask1 = ((h >= 0) & (h <= 12) & (s >= 10) & (v >= 80))
    mask2 = ((h >= 165) & (h <= 180) & (s >= 10) & (v >= 80))
    mask = (mask1 | mask2).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    bbox = _largest_contour_bbox(mask)
    if bbox is None:
        cropped = img
    else:
        x, y, w, h_ = bbox
        pad = 6
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(W, x + w + pad)
        y1 = min(H, y + h_ + pad)
        cropped = img[y0:y1, x0:x1]

    # 2) Remove footer band (fix: was 25% in your code -> too much)
    ph = cropped.shape[0]
    footer_ratio = 0.12  # ✅ keep rhythm strip, remove speed/gain/footer text
    cut_y = int(ph * (1.0 - footer_ratio))

    # safety: keep at least 88% (do not over cut)
    min_keep = int(ph * 0.88)
    cut_y = max(cut_y, min_keep)

    cropped = cropped[:cut_y, :]

    # 3) Trim borders
    cropped = _trim_by_projection(cropped, axis=0)  # small top/bottom cleanup
    cropped = _trim_by_projection(cropped, axis=1)  # left/right cleanup

    # safety fallback
    if cropped.shape[0] < 80 or cropped.shape[1] < 80:
        cropped = img

    return _encode_jpeg(cropped, quality=92)