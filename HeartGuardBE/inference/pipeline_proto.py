from __future__ import annotations
from pathlib import Path
import uuid

from django.conf import settings
from django.core.files.storage import default_storage

from .preprocess_3x4plus1 import preprocess_3x4plus1
from .preprocess_3x4 import preprocess_3x4
from .nnunet_runner import run_nnunet
from .postprocess import png_to_nii, nii_to_png_mask

from .digitize_multiclass import digitize_from_multiclass_mask
from .digitize_binary_3x4 import digitize_from_binary_mask_3x4

from HeartGuardBE.protoecgnet_Model.preprocess import preprocess_single_ecg


def run_nnunet_and_digitize_to_proto_input(
    uploaded_abs_path: Path,
    layout: str,                 # "3x4plus1" or "3x4"
    fs_in: int = 500,
    fs_target: int = 100,
    target_len: int = 1000,
    digitize_len_samples: int = 5000,
) -> dict:
    """
    Returns:
      {
        "X": np.ndarray (1,1000,12),
        "fs_out": int,
        "mask_path": str,
      }
    """
    uploaded_abs_path = Path(uploaded_abs_path).resolve()
    work = uploaded_abs_path.parent
    job = uuid.uuid4().hex

    if layout == "3x4plus1":
        # preprocess (denoise already inside)
        pre_png = work / f"pre_503_{job}.png"
        preprocess_3x4plus1(uploaded_abs_path, pre_png)

        # nnunet predict 503 (png IO)
        pred = run_nnunet(pre_png, dataset_id=settings.NNUNET_DATASET_3X4PLUS1, is_png=True)

        # ensure png mask exists (prefer png output)
        if pred.suffix.lower() == ".png":
            mask_png = pred
        else:
            mask_png = work / f"mask_503_{job}.png"
            nii_to_png_mask(pred, mask_png, binary=False)

        # digitize multiclass
        leads_mV = digitize_from_multiclass_mask(str(mask_png), target_len_samples=digitize_len_samples)

    elif layout == "3x4":
        pre_png = work / f"pre_504_{job}.png"
        preprocess_3x4(uploaded_abs_path, pre_png)

        in_nii = work / f"case_{job}_0000.nii.gz"
        png_to_nii(pre_png, in_nii)

        pred = run_nnunet(in_nii, dataset_id=settings.NNUNET_DATASET_3X4, is_png=False)

        mask_png = work / f"mask_504_{job}.png"
        nii_to_png_mask(pred, mask_png, binary=True)

        leads_mV = digitize_from_binary_mask_3x4(str(mask_png), target_len_samples=digitize_len_samples)

    else:
        raise ValueError("layout must be '3x4' or '3x4plus1'")

    # feed proto preprocess (your file)
    X, fs_out = preprocess_single_ecg(
        leads_mV=leads_mV,
        fs_in=fs_in,
        fs_target=fs_target,
        target_len=target_len,
        remove_baseline=True,
    )

    return {
        "X": X,
        "fs_out": fs_out,
        "mask_path": str(mask_png),
    }
