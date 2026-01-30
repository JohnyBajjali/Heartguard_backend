from pathlib import Path
import cv2
import numpy as np

def denoise_ecg_grayscale(
    img_gray: np.ndarray,
    use_median: bool = True,
    median_ksize: int = 3,
    use_nlm: bool = False,
    nlm_h: int = 4,
    nlm_template: int = 7,
    nlm_search: int = 21,
) -> np.ndarray:
    """
    Very conservative denoise for ECG-paper images.
    - median(3) removes speckle with minimal edge loss
    - optional NLM with small h further reduces sensor noise
    """
    if img_gray.dtype != np.uint8:
        img_gray = img_gray.astype(np.uint8)

    out = img_gray

    if use_median:
        out = cv2.medianBlur(out, median_ksize)

    if use_nlm:
        out = cv2.fastNlMeansDenoising(
            out,
            None,
            h=nlm_h,
            templateWindowSize=nlm_template,
            searchWindowSize=nlm_search,
        )

    return out


def load_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    return img


def save_gray(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), img)
    if not ok:
        raise ValueError(f"Failed to write image: {path}")
