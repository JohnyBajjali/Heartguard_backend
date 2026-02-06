# HeartGuardBE/views.py
import json
import uuid
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import cv2
from lxml import etree

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from django.conf import settings
from django.core.exceptions import ValidationError
from django.http import HttpResponse, JsonResponse
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET

from django.core.files.base import ContentFile

from .models import Patient, ECGUpload, DoctorNote

from .protoecgnet_Model.philips_xml import read_hospital_ecg_xml_repbeats
from .protoecgnet_Model.aecg_hl7 import read_aecg_hl7_bytes
from .protoecgnet_Model.preprocess import preprocess_single_ecg
from .protoecgnet_Model.predictor import FusionPredictor

from HeartGuardBE.inference.postprocess import nii_to_png_mask
from HeartGuardBE.inference.preprocess_3x4 import preprocess_3x4
from HeartGuardBE.inference.nnunet_runner import run_nnunet
from HeartGuardBE.inference.digitize_binary_3x4 import digitize_from_binary_mask_3x4
from HeartGuardBE.inference.vis_504_outputs import plot_like_original_3x4 as plot_like_original_3x4_vis

from HeartGuardBE.inference.xml_preview import generate_xml_preview_png


# ----------------------------
# Utils
# ----------------------------
def normalize_name(name: str) -> str:
    return " ".join((name or "").strip().split()).lower()


def _media_root() -> Path:
    return Path(getattr(settings, "MEDIA_ROOT", Path(__file__).resolve().parents[1] / "media"))


def _abs_media_url(request, rel_url: str) -> str:
    return request.build_absolute_uri(rel_url)


def _patient_ecgs_from_ids(patient: Patient):
    """
    Return ECG list in the SAME order stored in patient.ecg_upload_ids (newest first).
    Includes prediction + preview fields for Flutter history.
    """
    ids = list(patient.ecg_upload_ids or [])
    if not ids:
        return []

    qs = ECGUpload.objects.filter(id__in=ids)
    by_id = {e.id: e for e in qs}

    out = []
    for eid in ids:
        ecg = by_id.get(eid)
        if not ecg:
            continue
        out.append({
            "id": ecg.id,
            "imageUrl": ecg.file.url,
            "uploaded_at": ecg.uploaded_at.isoformat(),

            "prediction_topk": ecg.prediction_topk,
            "prediction_meta": ecg.prediction_meta,
            "predicted_at": ecg.predicted_at.isoformat() if ecg.predicted_at else None,
            "preview_image_url": ecg.preview_image_url,
        })
    return out


# ===========================
# AUTO CROP (OpenCV)
# ===========================
def _auto_crop_ecg_bytes(image_bytes: bytes) -> bytes:
    """
    Reliable ECG auto-crop:
    - Detect paper region using HSV red-ish grid
    - Crop to bounding box
    - Remove a small footer band
    - Trim borders using edge projections
    - Return JPG bytes
    """
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")

    H, W = img.shape[:2]

    def encode_jpg(bgr, quality=92) -> bytes:
        ok, out = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        return out.tobytes() if ok else image_bytes

    def trim_by_projection(bgr, axis: int):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 60, 160)

        proj = edges.sum(axis=axis).astype(np.float32)
        k = 31
        ker = np.ones(k, dtype=np.float32) / k
        proj_s = np.convolve(proj, ker, mode="same")

        mx = float(proj_s.max() if proj_s.size else 0.0)
        if mx <= 1e-6:
            return bgr

        thr = 0.12 * mx
        idx = np.where(proj_s > thr)[0]
        if idx.size == 0:
            return bgr

        a = int(idx[0])
        b = int(idx[-1])

        margin = 10
        a = max(0, a - margin)
        if axis == 0:
            b = min(bgr.shape[1] - 1, b + margin)
            if (b - a + 1) > int(0.75 * bgr.shape[1]):
                return bgr[:, a:b + 1]
            return bgr
        else:
            b = min(bgr.shape[0] - 1, b + margin)
            if (b - a + 1) > int(0.75 * bgr.shape[0]):
                return bgr[a:b + 1, :]
            return bgr

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    mask1 = ((h >= 0) & (h <= 12) & (s >= 10) & (v >= 70))
    mask2 = ((h >= 165) & (h <= 180) & (s >= 10) & (v >= 70))
    mask = (mask1 | mask2).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return encode_jpg(img, 92)

    c = max(cnts, key=cv2.contourArea)
    x, y, w, h_ = cv2.boundingRect(c)

    if w * h_ < 0.12 * (W * H):
        return encode_jpg(img, 92)

    pad = 8
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(W, x + w + pad)
    y1 = min(H, y + h_ + pad)

    paper = img[y0:y1, x0:x1]
    ph, pw = paper.shape[:2]

    footer_ratio = 0.12
    cut_y = int(ph * (1.0 - footer_ratio))
    min_keep = int(ph * 0.88)
    cut_y = max(cut_y, min_keep)
    if 0 < cut_y < ph:
        paper = paper[:cut_y, :]

    paper = trim_by_projection(paper, axis=1)
    paper = trim_by_projection(paper, axis=0)

    paper = cv2.convertScaleAbs(paper, alpha=1.05, beta=0)

    if paper.shape[0] < 80 or paper.shape[1] < 80:
        return encode_jpg(img, 92)

    return encode_jpg(paper, 92)


# ===========================
# XML Preview helper
# ===========================
def _make_preview_for_xml(
    file_bytes: bytes,
    *,
    base_stem: str,
    request=None,
) -> Tuple[Optional[str], Optional[dict]]:
    """
    Create ECG preview PNG for Philips XML.
    Returns (preview_url, preview_meta) or (None, None) if fails.
    """
    try:
        previews_dir = _media_root() / "ecg_previews"
        previews_dir.mkdir(parents=True, exist_ok=True)

        out_png = previews_dir / f"{base_stem}_preview.png"
        meta = generate_xml_preview_png(file_bytes, out_png)

        rel_url = f"{settings.MEDIA_URL.rstrip('/')}/ecg_previews/{out_png.name}"
        if request is not None:
            return request.build_absolute_uri(rel_url), meta
        return rel_url, meta
    except Exception as e:
        print("⚠️ XML preview generation failed:", e)
        return None, None

def _make_preview_for_aecg(leads_mV, fs, *, base_stem: str, request=None):
    """
    Create preview PNG for aECG.
    Works even if leads_mV comes as (N,12) or (1,12,N) etc.
    Returns (preview_url, preview_meta) or (None, None) if fails.
    """
    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from django.conf import settings

        # ===== Normalize shape =====
        x = np.asarray(leads_mV)

        # Squeeze singleton dims: (1,12,N) -> (12,N)
        if x.ndim >= 3:
            x = np.squeeze(x)

        if x.ndim != 2:
            print("⚠️ aECG preview: unexpected ndim/shape:", x.ndim, getattr(x, "shape", None))
            return None, None

        # Ensure [12, N]
        if x.shape[0] != 12 and x.shape[1] == 12:
            x = x.T

        if x.shape[0] != 12:
            print("⚠️ aECG preview: expected 12 leads, got shape:", x.shape)
            # still plot what exists
        n_leads = x.shape[0]
        n = x.shape[1]

        # ===== Output path =====
        previews_dir = _media_root() / "ecg_previews"
        previews_dir.mkdir(parents=True, exist_ok=True)
        out_png = previews_dir / f"{base_stem}_preview.png"

        t = np.arange(n) / float(fs)

        lead_names = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
        lead_names = lead_names[:n_leads]

        fig_h = max(6, n_leads * 1.0)
        plt.figure(figsize=(14, fig_h))

        for i in range(n_leads):
            ax = plt.subplot(n_leads, 1, i + 1)
            ax.plot(t, x[i], linewidth=0.9)
            ax.set_ylabel(lead_names[i] if i < len(lead_names) else f"L{i+1}",
                          rotation=0, labelpad=18, fontsize=9)
            ax.grid(True, alpha=0.25)
            if i != n_leads - 1:
                ax.set_xticks([])
            else:
                ax.set_xlabel("Time (s)")

        plt.tight_layout()
        plt.savefig(out_png, dpi=160)
        plt.close()

        # Return relative URL so Flutter can prefix base URL
        rel_url = f"{settings.MEDIA_URL.rstrip('/')}/ecg_previews/{out_png.name}"
        meta = {"fs": float(fs), "n_leads": int(n_leads), "n_samples": int(n), "shape": list(x.shape)}

        return rel_url, meta

    except Exception as e:
        print("⚠️ aECG preview generation failed:", e)
        return None, None

def _make_preview_from_X(X, fs, *, base_stem: str, style: str = "3x4+rhythm"):
    """
    Create preview PNG from numeric X (output of preprocess_single_ecg).
    style:
      - "stacked": 12 stacked leads
      - "3x4+rhythm": ECG paper style (3x4) + rhythm strip (lead II)
    Returns (relative_url, meta) or (None, None).
    """
    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from django.conf import settings

        x = np.asarray(X)
        x = np.squeeze(x)

        if x.ndim != 2:
            print("⚠️ preview_from_X: unexpected shape:", getattr(x, "shape", None))
            return None, None

        # ensure [12, N]
        if x.shape[0] != 12 and x.shape[1] == 12:
            x = x.T

        if x.shape[0] < 1:
            print("⚠️ preview_from_X: empty leads")
            return None, None

        n_leads = x.shape[0]
        n = x.shape[1]

        previews_dir = _media_root() / "ecg_previews"
        previews_dir.mkdir(parents=True, exist_ok=True)
        out_png = previews_dir / f"{base_stem}_preview.png"

        lead_names = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
        lead_names = lead_names[:n_leads]

        # Helper: "ECG paper" grid
        def _ecg_grid(ax, seconds_span, mv_span):
            # small squares: 0.04s, 0.1mV
            ax.set_xlim(0, seconds_span)
            ax.set_ylim(-mv_span, mv_span)

            # major squares: 0.2s, 0.5mV
            ax.set_xticks(np.arange(0, seconds_span + 1e-9, 0.2))
            ax.set_xticks(np.arange(0, seconds_span + 1e-9, 0.04), minor=True)

            ax.set_yticks(np.arange(-mv_span, mv_span + 1e-9, 0.5))
            ax.set_yticks(np.arange(-mv_span, mv_span + 1e-9, 0.1), minor=True)

            ax.grid(which="major", alpha=0.25, linewidth=0.8)
            ax.grid(which="minor", alpha=0.10, linewidth=0.5)

            ax.tick_params(which="both", length=0)
            ax.set_xticklabels([])
            ax.set_yticklabels([])

        # ===== Choose output style =====
        if style == "stacked":
            t = np.arange(n) / float(fs)
            fig_h = max(6, n_leads * 1.0)
            plt.figure(figsize=(14, fig_h))
            for i in range(n_leads):
                ax = plt.subplot(n_leads, 1, i + 1)
                ax.plot(t, x[i], linewidth=0.9)
                ax.set_ylabel(lead_names[i] if i < len(lead_names) else f"L{i+1}",
                              rotation=0, labelpad=18, fontsize=9)
                ax.grid(True, alpha=0.25)
                if i != n_leads - 1:
                    ax.set_xticks([])
                else:
                    ax.set_xlabel("Time (s)")
            plt.tight_layout()
            plt.savefig(out_png, dpi=160)
            plt.close()

        else:
            # "3x4+rhythm" layout (like your friend's image)
            # We show 2.5s per small panel so 4 panels cover 10s
            total_seconds = n / float(fs)
            panel_seconds = min(2.5, total_seconds)  # 2.5s per lead cell
            panel_samples = int(panel_seconds * fs)

            # If your signal is exactly 10s at fs=100 => 1000 samples, panel_samples=250
            # Use first 10 seconds if longer
            max_seconds = 10.0
            max_samples = min(n, int(max_seconds * fs))

            # slice usable segment
            x_use = x[:, :max_samples]
            n_use = x_use.shape[1]
            total_seconds_use = n_use / float(fs)

            # recompute panel
            panel_seconds = min(2.5, total_seconds_use)
            panel_samples = max(1, int(panel_seconds * fs))

            # rhythm strip will use full width (up to 10s)
            rhythm_idx = 1 if n_leads >= 2 else 0  # II preferred

            # amplitude span (mV): use robust scaling
            # take percentile to avoid outliers
            amp = np.percentile(np.abs(x_use), 99)
            mv_span = max(0.5, min(2.0, float(amp * 1.2)))  # clamp nice range

            # Figure: 4 columns x 3 rows + 1 rhythm row
            fig = plt.figure(figsize=(14, 8), constrained_layout=False)
            gs = fig.add_gridspec(4, 4, height_ratios=[1, 1, 1, 1.4], hspace=0.30, wspace=0.15)

            # lead order like standard ECG
            order = [
                0, 3, 6, 9,     # I, aVR, V1, V4
                1, 4, 7, 10,    # II, aVL, V2, V5
                2, 5, 8, 11     # III, aVF, V3, V6
            ]
            order = [i for i in order if i < n_leads]

            # Plot 3x4
            for k, lead_i in enumerate(order[:12]):
                r = k // 4
                c = k % 4
                ax = fig.add_subplot(gs[r, c])

                start = 0
                end = min(panel_samples, n_use)
                t = np.arange(end - start) / float(fs)

                _ecg_grid(ax, seconds_span=(end - start) / float(fs), mv_span=mv_span)
                ax.plot(t, x_use[lead_i, start:end], linewidth=1.0)
                ax.text(0.02, 0.85, lead_names[lead_i] if lead_i < len(lead_names) else f"L{lead_i+1}",
                        transform=ax.transAxes, fontsize=10, fontweight="bold")

            # Rhythm strip (full width)
            axr = fig.add_subplot(gs[3, :])
            t = np.arange(n_use) / float(fs)
            _ecg_grid(axr, seconds_span=total_seconds_use, mv_span=mv_span)
            axr.plot(t, x_use[rhythm_idx, :], linewidth=1.0)
            axr.text(0.01, 0.85, "II", transform=axr.transAxes, fontsize=10, fontweight="bold")
            axr.set_xlabel("Time (s)")

            # small footer like report
            fig.suptitle("aECG Preview", fontsize=14, fontweight="bold", y=0.98)

            fig.tight_layout(rect=[0, 0, 1, 0.97])
            fig.savefig(out_png, dpi=170)
            plt.close(fig)

        rel_url = f"{settings.MEDIA_URL.rstrip('/')}/ecg_previews/{out_png.name}"
        meta = {"fs": float(fs), "n_leads": int(n_leads), "n_samples": int(n), "shape": list(x.shape), "style": style}
        return rel_url, meta

    except Exception as e:
        print("⚠️ preview_from_X failed:", e)
        return None, None

# ===========================
# Detect ECG format
# ===========================
def detect_ecg_format(file_bytes: bytes) -> str:
    parser = etree.XMLParser(recover=True, huge_tree=True)
    try:
        root = etree.fromstring(file_bytes, parser)
        if "annotatedecg" in root.tag.lower():
            return "aecg"
        if root.xpath('//*[local-name()="repbeats" or local-name()="repbeat"]'):
            return "philips"
    except Exception:
        pass

    try:
        buf = np.frombuffer(file_bytes, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is not None:
            return "image_3x4"
    except Exception:
        pass

    return "unknown"


# ===========================
# 3x4 image pipeline
# ===========================
def _predict_from_3x4_image_bytes(
    img_bytes: bytes,
    filename: str = "upload.png",
    *,
    layout: str = "3x4",
):
    dataset_id = "504"
    target_len = 1000
    px_per_mV = 40.0
    fs_assumed = 100.0

    job_id = uuid.uuid4().hex
    out_dir = _media_root() / "test_out" / "3x4_api" / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    src_path = out_dir / f"src_{Path(filename).stem}.png"
    buf = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image bytes (not a valid PNG/JPG).")
    cv2.imwrite(str(src_path), img)

    pre_png = out_dir / f"pre_{src_path.stem}.png"
    preprocess_3x4(src_path, pre_png)

    pred_nii = run_nnunet(pre_png, dataset_id=dataset_id, is_png=False)
    if pred_nii.suffixes[-2:] != [".nii", ".gz"]:
        raise RuntimeError(f"Expected .nii.gz from 504, got: {pred_nii}")

    mask_png = out_dir / f"mask_504_{src_path.stem}.png"
    nii_to_png_mask(pred_nii, mask_png, binary=True)

    debug_dir = out_dir / f"debug_digitize_{src_path.stem}"
    leads_mV, digi_meta = digitize_from_binary_mask_3x4(
        str(mask_png),
        target_len_samples=target_len,
        px_per_mV=px_per_mV,
        debug_dir=str(debug_dir),
        return_meta=True,
    )

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

    X, fs_out = preprocess_single_ecg(
        leads_mV,
        fs_in=int(fs_assumed),
        fs_target=int(fs_assumed),
        target_len=target_len,
        remove_baseline=True,
    )

    model = FusionPredictor.instance()
    probs = model.predict_from_X(X)
    probs = np.asarray(probs, dtype=np.float32).reshape(-1)

    labels = list(model.fusion_labels)
    idxs = np.argsort(-probs)[:10]
    topk = [{"label": labels[i], "p": float(probs[i])} for i in idxs]

    if isinstance(digi_meta, dict):
        for k in ("y_pix_by_lead", "y_path_by_lead", "y_by_lead", "x_by_lead", "points_by_lead"):
            digi_meta.pop(k, None)

    meta = {
        "format": "image",
        "layout": layout,  # "3x4" or "3x4+1" (stored even if pipeline same now)
        "dataset_id": dataset_id,
        "target_len": target_len,
        "px_per_mV": px_per_mV,
        "fs_assumed": fs_assumed,
        "fs_out": fs_out,
        "debug_paths": {
            "out_dir": str(out_dir),
            "src": str(src_path),
            "pre_png": str(pre_png),
            "pred_nii": str(pred_nii),
            "mask_png": str(mask_png),
            "debug_digitize": str(debug_dir),
            "digitized_like_original": str(like_png),
        },
    }
    return topk, meta


# ===========================
# API ENDPOINTS
# ===========================
@csrf_exempt
def patients_list_create(request):
    if request.method == "GET":
        patients = Patient.objects.all().order_by("-created_at")
        data = []
        for p in patients:
            data.append({
                "patient_id": p.patient_id,
                "full_name": p.full_name,
                "age": p.age,
                "gender": p.gender,
                "created_at": p.created_at.isoformat(),
                "ecg_count": len(p.ecg_upload_ids or []),
            })
        return JsonResponse({"patients": data}, status=200)

    if request.method == "POST":
        try:
            data = json.loads(request.body.decode("utf-8"))
        except Exception:
            return JsonResponse({"error": "Invalid JSON body"}, status=400)

        patient_id_raw = str(data.get("patient_id", "")).strip()
        full_name = str(data.get("full_name", "")).strip()
        gender = data.get("gender", None)
        age_raw = data.get("age", None)

        if not patient_id_raw:
            return JsonResponse({"error": "patient_id is required"}, status=400)
        if not patient_id_raw.isdigit():
            return JsonResponse({"error": "patient_id must be an integer"}, status=400)

        if not full_name:
            return JsonResponse({"error": "full_name is required"}, status=400)

        patient_id = int(patient_id_raw)

        if Patient.objects.filter(patient_id=patient_id).exists():
            return JsonResponse({"error": "patient_id already exists"}, status=409)

        if gender not in ("M", "F"):
            return JsonResponse({"error": "gender is required and must be 'M' or 'F'"}, status=400)

        if age_raw is None or str(age_raw).strip() == "":
            return JsonResponse({"error": "age is required"}, status=400)
        try:
            age = int(str(age_raw).strip())
        except Exception:
            return JsonResponse({"error": "age must be an integer"}, status=400)
        if age < 1 or age > 130:
            return JsonResponse({"error": "age must be between 1 and 130"}, status=400)

        patient = Patient(patient_id=patient_id, full_name=full_name, age=age, gender=gender)

        try:
            patient.full_clean()
        except ValidationError as e:
            return JsonResponse({"error": e.message_dict or str(e)}, status=400)

        patient.save()
        return JsonResponse({
            "patient": {
                "patient_id": patient.patient_id,
                "full_name": patient.full_name,
                "age": patient.age,
                "gender": patient.gender,
                "created_at": patient.created_at.isoformat(),
                "ecg_count": len(patient.ecg_upload_ids or []),
            }
        }, status=201)

    return JsonResponse({"error": "Method not allowed"}, status=405)


@csrf_exempt
def patient_delete(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    try:
        data = json.loads(request.body.decode("utf-8"))
    except Exception:
        return JsonResponse({"error": "Invalid JSON body"}, status=400)

    patient_id_raw = str(data.get("patient_id", "")).strip()
    if not patient_id_raw:
        return JsonResponse({"error": "patient_id is required"}, status=400)
    if not patient_id_raw.isdigit():
        return JsonResponse({"error": "patient_id must be an integer"}, status=400)

    patient_id = int(patient_id_raw)

    try:
        patient = Patient.objects.get(patient_id=patient_id)
    except Patient.DoesNotExist:
        return JsonResponse({"error": "patient not found"}, status=404)

    ids = list(patient.ecg_upload_ids or [])
    if ids:
        ecgs = ECGUpload.objects.filter(id__in=ids)
        for e in ecgs:
            try:
                e.file.delete(save=False)
            except Exception:
                pass
        ecgs.delete()

    patient.delete()
    return JsonResponse({"ok": True, "deleted_patient_id": patient_id}, status=200)


@csrf_exempt
def patient_login_or_create(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    try:
        data = json.loads(request.body.decode("utf-8"))
    except Exception:
        data = request.POST

    name = str(data.get("name", "")).strip()
    patient_id_raw = str(data.get("patient_id", "")).strip()

    if not name:
        return JsonResponse({"error": "name is required"}, status=400)
    if not patient_id_raw:
        return JsonResponse({"error": "patient_id is required"}, status=400)
    if not patient_id_raw.isdigit():
        return JsonResponse({"error": "patient_id must be an integer"}, status=400)

    patient_id = int(patient_id_raw)

    try:
        patient = Patient.objects.get(patient_id=patient_id)
        if normalize_name(patient.full_name) != normalize_name(name):
            return JsonResponse(
                {"error": "This ID is already registered to another person. Please use your own ID."},
                status=409,
            )
        created = False
    except Patient.DoesNotExist:
        patient = Patient(patient_id=patient_id, full_name=name)
        try:
            patient.full_clean()
        except ValidationError as e:
            return JsonResponse({"error": e.message_dict or str(e)}, status=400)
        patient.save()
        created = True

    ecgs = _patient_ecgs_from_ids(patient)

    return JsonResponse({
        "patient_id": patient.patient_id,
        "name": patient.full_name,
        "gender": patient.gender,
        "ecgs": ecgs,
        "created": created,
    }, status=200)


@csrf_exempt
def ecg_upload(request):
    """
    POST /api/ecg/upload/  (multipart)
    Saves file in ECGUpload and adds its id into Patient.ecg_upload_ids
    """
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    uploaded_file = request.FILES.get("file")
    source = (request.POST.get("source", "mobile") or "mobile").strip()
    patient_id_raw = (request.POST.get("patient_id", "") or "").strip()
    patient_name = (request.POST.get("patient_name", "") or "").strip()

    if not uploaded_file:
        return JsonResponse({"error": "No file uploaded"}, status=400)
    if not patient_name:
        return JsonResponse({"error": "patient_name is required"}, status=400)
    if not patient_id_raw:
        return JsonResponse({"error": "patient_id is required"}, status=400)
    if not patient_id_raw.isdigit():
        return JsonResponse({"error": "patient_id must be an integer"}, status=400)

    patient_id = int(patient_id_raw)

    # find/create patient
    try:
        patient = Patient.objects.get(patient_id=patient_id)
        if normalize_name(patient.full_name) != normalize_name(patient_name):
            return JsonResponse(
                {"error": "This ID is already registered to another person. Please use your own ID."},
                status=409,
            )
    except Patient.DoesNotExist:
        patient = Patient(patient_id=patient_id, full_name=patient_name)
        try:
            patient.full_clean()
        except ValidationError as e:
            return JsonResponse({"error": e.message_dict or str(e)}, status=400)
        patient.save()

    ext = (Path(uploaded_file.name).suffix.lower() or ".jpg")
    new_filename = f"{uuid.uuid4()}{ext}"

    file_bytes = uploaded_file.read()

    ecg = ECGUpload.objects.create(
        file=ContentFile(file_bytes, name=new_filename),
        source=source,
    )

    ids = list(patient.ecg_upload_ids or [])
    ids.insert(0, ecg.id)
    patient.ecg_upload_ids = ids[:500]
    patient.save(update_fields=["ecg_upload_ids"])

    preview_url = None
    preview_meta = None

    # XML preview
    if ext == ".xml":
        stem = f"ecg_{ecg.id}_{Path(new_filename).stem}"
        preview_url, preview_meta = _make_preview_for_xml(file_bytes, base_stem=stem, request=request)
        if preview_url:
            ecg.preview_image_url = preview_url
            ecg.save(update_fields=["preview_image_url"])

    return JsonResponse({
        "status": "ok",
        "ecg_id": ecg.id,
        "url": request.build_absolute_uri(ecg.file.url),
        "filename": new_filename,
        "source": ecg.source,
        "uploaded_at": ecg.uploaded_at.isoformat(),
        "patient_id": patient.patient_id,
        "patient_name": patient.full_name,
        "preview_image_url": preview_url,
        "preview_meta": preview_meta,
    }, status=201)


@csrf_exempt
def ecg_auto_crop(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    uploaded_file = request.FILES.get("file")
    if not uploaded_file:
        return JsonResponse({"error": "No file uploaded"}, status=400)

    try:
        cropped_jpg = _auto_crop_ecg_bytes(uploaded_file.read())
    except Exception as e:
        return JsonResponse({"error": f"auto crop failed: {e}"}, status=400)

    return HttpResponse(cropped_jpg, content_type="image/jpeg")


@require_GET
def list_ecgs_for_patient(request):
    """
    GET /api/ecg/list/?patient_id=123
    Returns LIST (safe=False) because Flutter expects List.
    """
    patient_id_raw = (request.GET.get("patient_id", "") or "").strip()
    if not patient_id_raw:
        return JsonResponse({"error": "patient_id is required"}, status=400)
    if not patient_id_raw.isdigit():
        return JsonResponse({"error": "patient_id must be an integer"}, status=400)

    patient_id = int(patient_id_raw)

    try:
        patient = Patient.objects.get(patient_id=patient_id)
    except Patient.DoesNotExist:
        return JsonResponse({"error": "patient not found"}, status=404)

    ecgs = _patient_ecgs_from_ids(patient)

    return JsonResponse([
        {
            "id": e["id"],
            "patient_id": patient.patient_id,
            "patient_name": patient.full_name,
            "url": request.build_absolute_uri(e["imageUrl"]),
            "uploaded_at": e["uploaded_at"],

            "prediction_topk": e.get("prediction_topk"),
            "prediction_meta": e.get("prediction_meta"),
            "predicted_at": e.get("predicted_at"),
            "preview_image_url": e.get("preview_image_url"),
        }
        for e in ecgs
    ], safe=False, status=200)


@csrf_exempt
def predict_ecg(request):
    """
    POST /api/ecg/predict/  (multipart)
    - Detect format: philips xml / aecg / image
    - Predict
    - If ecg_id provided, save prediction in ECGUpload row
    """
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)

    if "file" not in request.FILES:
        return JsonResponse({"error": "missing file field"}, status=400)

    # optional: link prediction to existing ECGUpload row
    ecg_obj = None
    ecg_id_raw = (request.POST.get("ecg_id", "") or "").strip()
    if ecg_id_raw.isdigit():
        try:
            ecg_obj = ECGUpload.objects.get(id=int(ecg_id_raw))
        except ECGUpload.DoesNotExist:
            ecg_obj = None

    layout = (request.POST.get("layout", "") or "").strip()  # "3x4" or "3x4+1"

    up = request.FILES["file"]
    file_bytes = up.read()
    fmt = detect_ecg_format(file_bytes)

    try:
        # ===== Philips XML =====
        if fmt == "philips":
            leads_mV, fs, meta = read_hospital_ecg_xml_repbeats(file_bytes)

            X, fs_out = preprocess_single_ecg(
                leads_mV,
                fs_in=fs,
                fs_target=100,
                target_len=1000,
                remove_baseline=True
            )

            model = FusionPredictor.instance()
            probs = model.predict_from_X(X)
            probs = np.asarray(probs).reshape(-1)

            labels = list(model.fusion_labels)
            idx = np.argsort(-probs)[:10]
            out = [{"label": labels[i], "p": float(probs[i])} for i in idx]

            preview_url, preview_meta = _make_preview_for_xml(
                file_bytes, base_stem=f"predict_{uuid.uuid4().hex}", request=request
            )

            resp = {
                "topk": out,
                "meta": {
                    "format": "philips",
                    "fs_in": fs,
                    "fs_out": fs_out,
                    "xml_meta": meta,
                },
                "preview_image_url": preview_url,
                "preview_meta": preview_meta,
            }

            if ecg_obj:
                ecg_obj.prediction_topk = resp["topk"]
                ecg_obj.prediction_meta = resp["meta"]
                ecg_obj.predicted_at = timezone.now()

                if preview_url:
                    ecg_obj.preview_image_url = preview_url
                    ecg_obj.save(update_fields=[
                        "prediction_topk", "prediction_meta", "predicted_at", "preview_image_url"
                    ])
                else:
                    ecg_obj.save(update_fields=[
                        "prediction_topk", "prediction_meta", "predicted_at"
                    ])

            return JsonResponse(resp, status=200)

        # ===== aECG HL7 =====
        elif fmt == "aecg":
            leads_mV, fs, meta = read_aecg_hl7_bytes(file_bytes)

            X, fs_out = preprocess_single_ecg(
                leads_mV,
                fs_in=fs,
                fs_target=100,
                target_len=1000,
                remove_baseline=True
            )

            model = FusionPredictor.instance()
            probs = model.predict_from_X(X)
            probs = np.asarray(probs).reshape(-1)

            labels = list(model.fusion_labels)
            idx = np.argsort(-probs)[:10]
            out = [{"label": labels[i], "p": float(probs[i])} for i in idx]

            # ✅ aECG preview (same structure as XML)
            # after X, fs_out = preprocess_single_ecg(...)
            preview_url, preview_meta = _make_preview_from_X(
                X,
                fs_out,
                base_stem=f"predict_{uuid.uuid4().hex}",
                style="3x4+rhythm"
                )

            resp = {
                "topk": out,
                "meta": {
                    "format": "aecg",
                    "fs_in": fs,
                    "fs_out": fs_out,
                    "aecg_meta": meta,
                },
                "preview_image_url": preview_url,
                "preview_meta": preview_meta,
            }

            if ecg_obj:
                ecg_obj.prediction_topk = resp["topk"]
                ecg_obj.prediction_meta = resp["meta"]
                ecg_obj.predicted_at = timezone.now()

                if preview_url:
                    ecg_obj.preview_image_url = preview_url
                    ecg_obj.save(update_fields=[
                        "prediction_topk", "prediction_meta", "predicted_at", "preview_image_url"
                    ])
                else:
                    ecg_obj.save(update_fields=[
                        "prediction_topk", "prediction_meta", "predicted_at"
                    ])

                    print("RESP KEYS:", resp.keys())
                    print("PREVIEW:", resp.get("preview_image_url"), resp.get("preview_url"), resp.get("preview_meta"))

            return JsonResponse(resp, status=200)

        # ===== Image (3x4 / 3x4+1) =====
        elif fmt == "image_3x4":
            used_layout = layout if layout in ("3x4", "3x4+1") else "3x4"

            # NOTE: currently pipeline is 3x4; for 3x4+1 we keep layout in meta but same pipeline.
            topk, meta = _predict_from_3x4_image_bytes(
                file_bytes,
                filename=getattr(up, "name", "upload.png"),
                layout=used_layout,
            )

            resp = {"topk": topk, "meta": meta}

            if ecg_obj:
                ecg_obj.prediction_topk = resp["topk"]
                ecg_obj.prediction_meta = resp["meta"]
                ecg_obj.predicted_at = timezone.now()
                ecg_obj.save(update_fields=["prediction_topk", "prediction_meta", "predicted_at"])

            return JsonResponse(resp, status=200)

        # ===== Unknown =====
        else:
            return JsonResponse(
                {"error": "Unknown format. Expected Philips XML, HL7 aECG, or ECG image.", "format": fmt},
                status=400
            )

    except Exception as e:
        return JsonResponse({"error": str(e), "format": fmt}, status=500)
    
def patient_notes(request, patient_id):
    pid_raw = str(patient_id).strip()
    if not pid_raw.isdigit():
        return JsonResponse({"error": "patient_id must be an integer"}, status=400)
    pid = int(pid_raw)

    try:
        patient = Patient.objects.get(patient_id=pid)
    except Patient.DoesNotExist:
        return JsonResponse({"error": "patient not found"}, status=404)

    if request.method == "GET":
        qs = DoctorNote.objects.filter(patient=patient).order_by("-created_at")
        notes = [{
            "id": n.id,
            "patient_id": patient.patient_id,
            "text": n.text,
            "created_at": n.created_at.isoformat(),
            "updated_at": n.updated_at.isoformat(),
        } for n in qs]
        return JsonResponse({"notes": notes}, status=200)

    if request.method == "POST":
        try:
            data = json.loads(request.body.decode("utf-8"))
        except Exception:
            return JsonResponse({"error": "Invalid JSON body"}, status=400)

        text = str(data.get("text", "")).strip()
        if not text:
            return JsonResponse({"error": "text is required"}, status=400)

        note = DoctorNote.objects.create(patient=patient, text=text)

        return JsonResponse({
            "note": {
                "id": note.id,
                "patient_id": patient.patient_id,
                "text": note.text,
                "created_at": note.created_at.isoformat(),
                "updated_at": note.updated_at.isoformat(),
            }
        }, status=201)

    return JsonResponse({"error": "Method not allowed"}, status=405)


@csrf_exempt
def patient_note_detail(request, patient_id, note_id):
    pid_raw = str(patient_id).strip()
    if not pid_raw.isdigit():
        return JsonResponse({"error": "patient_id must be an integer"}, status=400)
    pid = int(pid_raw)

    try:
        patient = Patient.objects.get(patient_id=pid)
    except Patient.DoesNotExist:
        return JsonResponse({"error": "patient not found"}, status=404)

    try:
        note = DoctorNote.objects.get(id=int(note_id), patient=patient)
    except Exception:
        return JsonResponse({"error": "note not found"}, status=404)

    if request.method in ("PATCH", "PUT"):
        try:
            data = json.loads(request.body.decode("utf-8"))
        except Exception:
            return JsonResponse({"error": "Invalid JSON body"}, status=400)

        text = data.get("text", None)
        if text is None:
            return JsonResponse({"error": "text is required"}, status=400)

        text = str(text).strip()
        if not text:
            return JsonResponse({"error": "text cannot be empty"}, status=400)

        note.text = text
        note.save(update_fields=["text", "updated_at"])

        return JsonResponse({
            "note": {
                "id": note.id,
                "patient_id": patient.patient_id,
                "text": note.text,
                "created_at": note.created_at.isoformat(),
                "updated_at": note.updated_at.isoformat(),
            }
        }, status=200)

    if request.method == "DELETE":
        note.delete()
        return JsonResponse({"ok": True, "deleted_note_id": int(note_id)}, status=200)

    return JsonResponse({"error": "Method not allowed"}, status=405)