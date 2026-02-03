from __future__ import annotations

from pathlib import Path
import json
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")  # server-safe

# -----------------------------
# Import shim: works when run directly OR as module
# -----------------------------
try:
    # if running as module: python -m HeartGuardBE.inference.final_3x4_pipeline_run
    from .vis_504_outputs import plot_like_original_3x4 as plot_like_original_3x4_vis
    from .preprocess_3x4 import preprocess_3x4
    from .nnunet_runner import run_nnunet
    from .postprocess import nii_to_png_mask
    from .digitize_binary_3x4 import digitize_from_binary_mask_3x4
except Exception:
    # if running directly: python HeartGuardBE/inference/final_3x4_pipeline_run.py
    from HeartGuardBE.inference.vis_504_outputs import plot_like_original_3x4 as plot_like_original_3x4_vis
    from HeartGuardBE.inference.preprocess_3x4 import preprocess_3x4
    from HeartGuardBE.inference.nnunet_runner import run_nnunet
    from HeartGuardBE.inference.postprocess import nii_to_png_mask
    from HeartGuardBE.inference.digitize_binary_3x4 import digitize_from_binary_mask_3x4

from HeartGuardBE.protoecgnet_Model.preprocess import preprocess_single_ecg
from HeartGuardBE.protoecgnet_Model.predictor import FusionPredictor


BASE = Path(__file__).resolve().parents[2] / "media"

SCP_CSV = Path(
    r"C:\Users\SS\Desktop\Grad_Proj\heartguard-backend\HeartGuardBE\protoecgnet_Model\assets\scp_statementsRegrouped2.csv"
)


# -----------------------------
# helpers
# -----------------------------
def pick_first_png(folder: Path) -> Path:
    folder = folder.resolve()
    png = next(folder.glob("*.png"), None)
    if png is None:
        raise FileNotFoundError(f"No PNG found in {folder}")
    return png


def save_json(obj, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _print_checkpoint_help():
    print("\n⚠️ ProtoECGNet checkpoints could not be loaded.")
    print("This is almost always because the .ckpt files are Git-LFS pointers (text files).")
    print("\nFix:")
    print("  git lfs install")
    print("  git lfs pull")
    print("\nQuick check:")
    print("  Get-Content path\\to\\last.ckpt -TotalCount 2")
    print("If you see 'version https://git-lfs.github.com/spec/v1' => it's a pointer, not weights.\n")


def _norm_code(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\ufeff", "")   # BOM
    s = s.replace("\u200b", "")   # zero width space
    s = s.replace("\u00a0", " ")  # non-breaking space
    return s.strip()


def load_scp_label_map(csv_path: Path) -> dict[str, str]:
    """
    Robustly load: SCP_CODE -> description

    IMPORTANT:
    Your CSV has duplicate blank column names ('') in the header.
    csv.DictReader breaks in that case (overwrites keys).
    So we use csv.reader and column indices.

    Convention for this file:
      col 0 = SCP code (blank header)
      col 'description' = human label you want
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"SCP CSV not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            raise RuntimeError("CSV is empty / missing header row")

        # code is first column in THIS file
        code_idx = 0

        # description column name is literally 'description'
        try:
            desc_idx = header.index("description")
        except ValueError:
            # fallback if someone renamed it
            # (you said actual full name is in 'description', so this is just safety)
            candidates = [i for i, h in enumerate(header) if "desc" in (h or "").lower()]
            if not candidates:
                raise RuntimeError(f"Could not find 'description' column. Header={header}")
            desc_idx = candidates[0]

        m: dict[str, str] = {}
        for row in reader:
            if not row:
                continue
            if len(row) <= max(code_idx, desc_idx):
                continue

            code = _norm_code(row[code_idx])
            if not code:
                continue

            desc = (row[desc_idx] or "").strip()
            m[code] = desc if desc else code

    return m


def topk_indices(probs: np.ndarray, k: int = 10) -> list[int]:
    probs = np.asarray(probs, dtype=np.float32).reshape(-1)
    return [int(i) for i in np.argsort(-probs)[:k]]


# -----------------------------
# main pipeline
# -----------------------------
if __name__ == "__main__":
    # INPUT
    src_dir = BASE / "test_in" / "504"
    src = pick_first_png(src_dir)

    # OUTPUT
    out_dir = BASE / "test_out" / "505"
    out_dir.mkdir(parents=True, exist_ok=True)

    px_per_mV = 40.0
    target_len = 1000
    fs_assumed = 100.0

    # Load label_map ONCE for JSON writing
    label_map = load_scp_label_map(SCP_CSV)
    print(f"✅ Loaded SCP label_map: {len(label_map)} entries")

    # Predictor (also gives you class order)
    predictor = FusionPredictor.instance()
    fusion_labels = list(predictor.fusion_labels)  # length should be 71

    # 1) preprocess
    pre = out_dir / f"pre_{src.stem}.png"
    preprocess_3x4(src, pre)
    print("✅ Preprocessed:", pre)

    # 2) nnUNet -> NIfTI
    pred_nii = run_nnunet(pre, dataset_id="504", is_png=False)
    print("✅ nnU-Net output:", pred_nii)

    if pred_nii.suffixes[-2:] != [".nii", ".gz"]:
        raise RuntimeError(f"Expected .nii.gz from 504, got: {pred_nii}")

    # 3) NIfTI -> binary mask
    mask_png = out_dir / f"mask_504_{src.stem}.png"
    nii_to_png_mask(pred_nii, mask_png, binary=True)
    print("✅ Binary mask:", mask_png)

    # 4) digitize
    debug_dir = out_dir / f"debug_digitize_{src.stem}"
    leads_mV, meta = digitize_from_binary_mask_3x4(
        str(mask_png),
        target_len_samples=target_len,
        px_per_mV=px_per_mV,
        debug_dir=str(debug_dir),
        return_meta=True,
    )
    print("✅ Digitized leads. Debug:", debug_dir)

    # 5) render digitized image
    like_png = out_dir / "digitized_like_original_3x4.png"
    plot_like_original_3x4_vis(
        leads_mV,
        like_png,
        cell_w_mm=62.0,
        cell_h_mm=30.0,
        row_gap_mm=4.0,
        col_gap_mm=4.0,
        auto_gain=True,
        never_clip=True,
        headroom=1.10,
        max_gain_mm_per_mV=7.0,
        out_width_px=1400,
    )
    print("✅ Saved:", like_png)

    # 6) proto preprocess
    X, fs_out = preprocess_single_ecg(
        leads_mV,
        fs_in=int(fs_assumed),
        fs_target=int(fs_assumed),
        target_len=target_len,
        remove_baseline=True,
    )
    print("✅ Proto input X:", X.shape, "fs_out:", fs_out)

    # 7) proto predict + JSON with description
    try:
        probs = predictor.predict_from_X(X)  # (71,)
        probs = np.asarray(probs, dtype=np.float32).reshape(-1)

        if probs.size != len(fusion_labels):
            print(f"⚠️ probs size={probs.size} but fusion_labels size={len(fusion_labels)}")

        idxs = topk_indices(probs, k=10)

        topk_json = []
        for i in idxs:
            code_raw = fusion_labels[i] if 0 <= i < len(fusion_labels) else f"Class{i}"
            code = _norm_code(code_raw)
            desc = label_map.get(code, code)  # fallback to code

            topk_json.append(
                {
                    "index": i,
                    "code": code,
                    "description": desc,   # ✅ what Flutter needs
                    "prob": float(probs[i]),
                    "prob_percent": round(float(probs[i]) * 100.0, 2),
                }
            )

        out_json = out_dir / "proto_topk.json"
        save_json(
            {
                "source_png": str(src),
                "digitized_png": str(like_png),
                "topk": topk_json,
                "label_map": label_map,  # ✅ optional: full map inside JSON for Flutter
            },
            out_json,
        )

        print("\n✅ Saved:", out_json)
        print("\n--- Top-10 (for quick sanity) ---")
        for r in topk_json:
            print(f"[{r['index']:>3}] {r['code']:<6} {r['description']} — {r['prob_percent']:>6.2f}%")

    except Exception as e:
        print("\n❌ Proto prediction step failed:")
        print("   ", repr(e))
        _print_checkpoint_help()
        print("✅ Pipeline outputs up to digitized image were saved.")

    print("\nDONE ✅")
