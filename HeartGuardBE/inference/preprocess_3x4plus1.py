from pathlib import Path
from .denoise import load_gray, save_gray, denoise_ecg_grayscale

def preprocess_3x4plus1(input_path: Path, out_png: Path) -> Path:
    """
    Safe preprocessing for 3x4+rhythm strip:
    - grayscale
    - conservative denoise
    """
    img = load_gray(input_path)

    # Conservative denoise: median only by default
    img = denoise_ecg_grayscale(
        img,
        use_median=True,
        median_ksize=3,
        use_nlm=False,   # keep False unless you really need more denoise
    )

    save_gray(out_png, img)
    return out_png
