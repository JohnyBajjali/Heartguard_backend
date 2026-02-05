# HeartGuardBE/inference/nnunet_runner.py
from __future__ import annotations

import os
import shutil
import subprocess
import uuid
from pathlib import Path


def _repo_root_from_this_file() -> Path:
    # .../HeartGuardBE/inference/nnunet_runner.py -> repo root is 2 parents up from HeartGuardBE
    return Path(__file__).resolve().parents[2]


def _default_nnunet_results_dir() -> Path:
    # heartguard-backend/HeartGuardBE/nnUNet_results
    repo_root = _repo_root_from_this_file()
    return repo_root / "HeartGuardBE" / "nnUNet_results"


def _make_clean_dir(p: Path) -> None:
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)


def _full_suffix(p: Path) -> str:
    """
    Return full suffix including multi-part ones like '.nii.gz'
    Path.suffix would return only '.gz' which is wrong for NIfTI.
    """
    name = p.name.lower()
    if name.endswith(".nii.gz"):
        return ".nii.gz"
    # fallback: join all suffixes (works for .tar.gz etc)
    return "".join(p.suffixes) if p.suffixes else ""


def _pick_pred_file(out_dir: Path, case_id: str, is_png: bool) -> Path:
    """
    nnU-Net outputs:
      - PNG dataset -> out/<case_id>.png
      - NIfTI dataset -> out/<case_id>.nii.gz
    Prefer those exact expected outputs.
    """
    if is_png:
        expected = out_dir / f"{case_id}.png"
        if expected.exists():
            return expected
    else:
        expected = out_dir / f"{case_id}.nii.gz"
        if expected.exists():
            return expected

    # Fallback: pick first file that starts with case_id
    outputs = sorted(out_dir.glob("*"))
    if not outputs:
        raise RuntimeError(f"No outputs found in {out_dir}")

    for p in outputs:
        if p.name.startswith(case_id):
            return p

    return outputs[0]


def run_nnunet(
    input_file: Path,
    dataset_id: str,
    is_png: bool,
    out_root: Path | None = None,
    folds: list[str] | None = None,
    checkpoint: str = "checkpoint_best.pth",
) -> Path:
    """
    Runs nnUNetv2_predict for ONE case.

    - input_file: a PNG (for Dataset503) OR a .nii.gz (for Dataset504)
    - is_png=True  -> will create case_0000.png in temp input dir
    - is_png=False -> will create case_0000.nii.gz in temp input dir

    Returns:
      Path to the prediction file copied to media/test_out/<dataset_id>/pred_<job_id>.(png|nii.gz)
    """
    input_file = Path(input_file).resolve()
    if not input_file.exists():
        raise FileNotFoundError(str(input_file))

    if folds is None:
        folds = ["0", "1", "2", "3", "4"]

    repo_root = _repo_root_from_this_file()

    # Default stable output root:
    # heartguard-backend/media/test_out/<dataset_id>/
    if out_root is None:
        out_root = repo_root / "media" / "test_out" / str(dataset_id)
    out_root.mkdir(parents=True, exist_ok=True)

    job_id = uuid.uuid4().hex
    workdir = repo_root / "media" / "nnunet_runs" / f"job_{job_id}"
    in_dir = workdir / "in"
    out_dir = workdir / "out"

    _make_clean_dir(in_dir)
    _make_clean_dir(out_dir)

    case_id = "case"

    # nnU-Net expects: <case>_0000.(png|nii.gz)
    if is_png:
        nn_in = in_dir / f"{case_id}_0000.png"
    else:
        nn_in = in_dir / f"{case_id}_0000.nii.gz"

    shutil.copy2(str(input_file), str(nn_in))

    # Environment vars: point nnU-Net to your trained models
    env = os.environ.copy()
    env["nnUNet_raw"] = str(repo_root / "nnUNet_raw")
    env["nnUNet_preprocessed"] = str(repo_root / "nnUNet_preprocessed")
    env["nnUNet_results"] = str(_default_nnunet_results_dir())

    # Avoid torch.compile complications
    env["nnUNet_compile"] = "0"

    cmd = [
        "nnUNetv2_predict",
        "-i", str(in_dir),
        "-o", str(out_dir),
        "-d", str(dataset_id),
        "-c", "2d",
        "-f", *folds,
        "-chk", checkpoint,
        "-device", "cpu",   # ✅ ADD THIS
    ]

    print("=== Running nnU-Net ===")
    print("CMD:", " ".join(cmd))
    print("nnUNet_results =", env["nnUNet_results"])
    print("Input:", nn_in)
    print("Output dir:", out_dir)

    proc = subprocess.run(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )

    log_path = out_root / f"nnunet_{dataset_id}_{job_id}.log"
    log_path.write_text(proc.stdout, encoding="utf-8")

    if proc.returncode != 0:
        raise RuntimeError(
            f"nnUNet failed (return code {proc.returncode}). Log saved to:\n{log_path}"
        )

    # Pick prediction file from out_dir
    pred = _pick_pred_file(out_dir, case_id=case_id, is_png=is_png)

    # Copy prediction to stable location with CORRECT extension
    pred_ext = _full_suffix(pred)  # '.png' OR '.nii.gz'
    if not pred_ext:
        # ultra fallback
        pred_ext = pred.suffix

    stable_pred = out_root / f"pred_{job_id}{pred_ext}"
    shutil.copy2(str(pred), str(stable_pred))

    # Copy the exact input used for nnU-Net too (debug)
    in_ext = _full_suffix(nn_in)  # '.png' OR '.nii.gz'
    if not in_ext:
        in_ext = nn_in.suffix

    stable_in = out_root / f"input_{job_id}{in_ext}"
    shutil.copy2(str(nn_in), str(stable_in))

    print("✅ Saved input to:", stable_in)
    print("✅ Saved prediction to:", stable_pred)
    print("✅ Saved log to:", log_path)

    return stable_pred
