from __future__ import annotations

from pathlib import Path
import numpy as np
import cv2
import SimpleITK as sitk


def _read_grayscale_png_u8(path: Path) -> np.ndarray:
    path = Path(path).resolve()
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read PNG: {path}")
    return img.astype(np.uint8)


def png_to_nii(png_path: Path, nii_path: Path) -> Path:
    """
    Converts a single-channel PNG to a 2D NIfTI (.nii.gz).
    nnU-Net expects case_0000.nii.gz for Dataset504.

    Notes:
    - We store as float32 (ok for nnU-Net).
    - This creates a 2D image; SimpleITK may load it back as (H,W) or (1,H,W).
    """
    png_path = Path(png_path).resolve()
    nii_path = Path(nii_path).resolve()
    nii_path.parent.mkdir(parents=True, exist_ok=True)

    img = _read_grayscale_png_u8(png_path)  # (H,W)

    sitk_img = sitk.GetImageFromArray(img.astype(np.float32))
    sitk.WriteImage(sitk_img, str(nii_path))
    return nii_path


def _load_nifti_as_array(nifti_path: Path) -> np.ndarray:
    """
    Loads a NIfTI and returns a 2D array (H,W) if possible.
    Handles common cases:
    - (H,W)
    - (1,H,W)  -> squeeze
    - (H,W,1)  -> squeeze
    """
    nifti_path = Path(nifti_path).resolve()
    img = sitk.ReadImage(str(nifti_path))
    arr = sitk.GetArrayFromImage(img)

    # Common: (1,H,W) for 2D written as "3D with z=1"
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]

    # Rare: (H,W,1) if something wrote channel-last
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array from NIfTI, got shape {arr.shape} from {nifti_path}")

    return arr


def nii_to_png_mask(nifti_pred_path: Path, out_png_path: Path, binary: bool) -> Path:
    """
    Converts nnU-Net prediction NIfTI to PNG mask.

    - binary=True: outputs 0/255 uint8 (for 3x4 model Dataset504)
    - binary=False: outputs 0..K uint8 labels (for multiclass if you ever output NIfTI)
    """
    nifti_pred_path = Path(nifti_pred_path).resolve()
    out_png_path = Path(out_png_path).resolve()
    out_png_path.parent.mkdir(parents=True, exist_ok=True)

    arr = _load_nifti_as_array(nifti_pred_path)

    if binary:
        m = (arr > 0).astype(np.uint8) * 255
    else:
        # Preserve class ids
        # nnU-Net segmentation NIfTI is usually int labels already, but be safe:
        m = np.rint(arr).astype(np.int32)
        m = np.clip(m, 0, 255).astype(np.uint8)

    ok = cv2.imwrite(str(out_png_path), m)
    if not ok:
        raise RuntimeError(f"Failed to write PNG: {out_png_path}")

    return out_png_path
