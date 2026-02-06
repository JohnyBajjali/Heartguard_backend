# HeartGuardBE/views.py
import json
import uuid
from pathlib import Path
import numpy as np
from lxml import etree
from django.conf import settings
import cv2
import nibabel as nib

from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET
from django.core.exceptions import ValidationError

from .models import Patient, ECGUpload, DoctorNote  # ✅ CHANGED (added DoctorNote)
from .protoecgnet_Model.philips_xml import read_hospital_ecg_xml_repbeats
from .protoecgnet_Model.aecg_hl7 import read_aecg_hl7_bytes
from .protoecgnet_Model.preprocess import preprocess_single_ecg
from .protoecgnet_Model.predictor import FusionPredictor
from HeartGuardBE.protoecgnet_Model import predictor

from HeartGuardBE.inference.postprocess import nii_to_png_mask
from HeartGuardBE.inference.preprocess_3x4 import preprocess_3x4
from HeartGuardBE.inference.nnunet_runner import run_nnunet
from HeartGuardBE.inference.digitize_binary_3x4 import digitize_from_binary_mask_3x4
from HeartGuardBE.inference.vis_504_outputs import plot_like_original_3x4 as plot_like_original_3x4_vis


def normalize_name(name: str) -> str:
    return " ".join((name or "").strip().split()).lower()


def _patient_ecgs_from_ids(patient: "Patient"):
    """
    Return ECG list in the SAME order stored in patient.ecg_upload_ids (newest first).
    Skips missing ECG ids safely.
    """
    ids = list(patient.ecg_upload_ids or [])
    if not ids:
        return []

    qs = ECGUpload.objects.filter(id__in=ids)
    by_id = {e.id: e for e in qs}

    ecgs = []
    for eid in ids:
        ecg = by_id.get(eid)
        if not ecg:
            continue
        ecgs.append({
            "id": ecg.id,
            "imageUrl": ecg.file.url,
            "uploaded_at": ecg.uploaded_at.isoformat(),
        })
    return ecgs


# ===========================
# AUTO CROP (OpenCV) - FIXED
# ===========================
def _auto_crop_ecg_bytes(image_bytes: bytes) -> bytes:
    """
    Reliable ECG auto-crop (fixed):

    1) Detect ECG paper (pink/red grid) using HSV
    2) Crop to paper bounding box
    3) Remove ONLY bottom footer titles band (small % so rhythm strip is kept)
    4) Trim borders safely using edge projections (top/bottom + left/right)
    5) Return JPEG bytes
    """
    import numpy as np
    import cv2

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
    if img is None:
        raise ValueError("Could not decode image")

    H, W = img.shape[:2]

    def encode_jpg(bgr, quality=92):
        ok, out = cv2.imencode(
            ".jpg",
            bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
        )
        return out.tobytes() if ok else image_bytes

    def trim_by_projection(bgr, axis: int):
        """
        axis=0 => trim top/bottom (rows)
        axis=1 => trim left/right (cols)
        Uses edge projections so we don't cut ECG content.
        """
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 60, 160)

        proj = edges.sum(axis=axis).astype(np.float32)

        # smooth
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
        b = min((bgr.shape[0] - 1) if axis == 1 else (bgr.shape[1] - 1), b + margin)

        if axis == 0:
            # columns projection -> trim left/right
            left, right = a, b
            # keep at least 75% width
            if (right - left + 1) > int(0.75 * bgr.shape[1]):
                return bgr[:, left:right + 1]
            return bgr
        else:
            # rows projection -> trim top/bottom
            top, bottom = a, b
            # keep at least 75% height
            if (bottom - top + 1) > int(0.75 * bgr.shape[0]):
                return bgr[top:bottom + 1, :]
            return bgr

    # -------------------------
    # 1) Detect paper (pink/red)
    # -------------------------
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # ECG grid is usually red/pink => hue near 0 or near 180
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

    # sanity: paper must be reasonably large
    if w * h_ < 0.12 * (W * H):
        return encode_jpg(img, 92)

    pad = 8
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(W, x + w + pad)
    y1 = min(H, y + h_ + pad)

    paper = img[y0:y1, x0:x1]
    ph, pw = paper.shape[:2]

    # ---------------------------------------
    # 2) Remove footer titles (FIXED)
    # ---------------------------------------
    footer_ratio = 0.12
    cut_y = int(ph * (1.0 - footer_ratio))

    min_keep = int(ph * 0.88)
    cut_y = max(cut_y, min_keep)

    if 0 < cut_y < ph:
        paper = paper[:cut_y, :]

    # -------------------------------------------
    # 3) Trim borders safely (top/bottom + left/right)
    # -------------------------------------------
    paper = trim_by_projection(paper, axis=1)  # top/bottom
    paper = trim_by_projection(paper, axis=0)  # left/right

    paper = cv2.convertScaleAbs(paper, alpha=1.05, beta=0)

    if paper.shape[0] < 80 or paper.shape[1] < 80:
        return encode_jpg(img, 92)

    return encode_jpg(paper, 92)


# ===========================
# API ENDPOINTS
# ===========================
@csrf_exempt
def patients_list_create(request):
    """
    GET  /api/patients/          -> list patients
    POST /api/patients/          -> create patient
    """
    if request.method == "GET":
        patients = Patient.objects.all().order_by("-created_at")
        data = []
        for p in patients:
            data.append({
                "patient_id": p.patient_id,
                "full_name": p.full_name,
                "age": p.age,  # ✅ NEW
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
        age_raw = data.get("age", None)  # ✅ NEW

        if not patient_id_raw:
            return JsonResponse({"error": "patient_id is required"}, status=400)
        if not patient_id_raw.isdigit():
            return JsonResponse({"error": "patient_id must be an integer"}, status=400)

        if not full_name:
            return JsonResponse({"error": "full_name is required"}, status=400)

        patient_id = int(patient_id_raw)

        if Patient.objects.filter(patient_id=patient_id).exists():
            return JsonResponse({"error": "patient_id already exists"}, status=409)

        gender = data.get("gender", None)

        if gender not in ("M", "F"):
            return JsonResponse(
                {"error": "gender is required and must be 'M' or 'F'"},
                status=400,
            )

        if age_raw is None or str(age_raw).strip() == "":
            return JsonResponse({"error": "age is required"}, status=400)

        try:
            age = int(str(age_raw).strip())
        except Exception:
            return JsonResponse({"error": "age must be an integer"}, status=400)

        if age < 1 or age > 130:
            return JsonResponse({"error": "age must be between 1 and 130"}, status=400)

        patient = Patient(
            patient_id=patient_id,
            full_name=full_name,
            age=age,  # ✅ NEW
            gender=gender,
        )

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
            }}, status=201)

    return JsonResponse({"error": "Method not allowed"}, status=405)


@csrf_exempt
def patient_delete(request):
    """
    POST /api/patients/delete/
    Body JSON: {"patient_id": 123}
    Deletes patient + deletes ECGUpload rows referenced in patient.ecg_upload_ids.
    Also deletes ECG files (best-effort).
    """
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

    # delete all ECG uploads linked to this patient
    ids = list(patient.ecg_upload_ids or [])
    if ids:
        ecgs = ECGUpload.objects.filter(id__in=ids)
        # best-effort: delete files from storage
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
    """
    POST /api/patient/login_or_create/
    """
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    try:
        data = json.loads(request.body.decode())
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

    return JsonResponse(
        {
            "patient_id": patient.patient_id,
            "name": patient.full_name,
            "gender": patient.gender,
            "ecgs": ecgs,
            "created": created,
        },
        status=200,
    )


@csrf_exempt
def ecg_upload(request):
    """
    POST /api/ecg/upload/  (multipart)
    """
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    uploaded_file = request.FILES.get("file")
    source = request.POST.get("source", "mobile").strip()
    patient_id_raw = request.POST.get("patient_id", "").strip()
    patient_name = request.POST.get("patient_name", "").strip()

    if not uploaded_file:
        return JsonResponse({"error": "No file uploaded"}, status=400)

    if not patient_name:
        return JsonResponse({"error": "patient_name is required"}, status=400)

    if not patient_id_raw:
        return JsonResponse({"error": "patient_id is required"}, status=400)

    if not patient_id_raw.isdigit():
        return JsonResponse({"error": "patient_id must be an integer"}, status=400)

    patient_id = int(patient_id_raw)

    # Get or create patient + enforce same-name rule
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

    # Save ECG globally
    ext = Path(uploaded_file.name).suffix.lower() or ".jpg"
    new_filename = f"{uuid.uuid4()}{ext}"
    uploaded_file.name = new_filename

    ecg = ECGUpload.objects.create(
        file=uploaded_file,
        source=source,
    )

    ids = list(patient.ecg_upload_ids or [])
    ids.insert(0, ecg.id)
    patient.ecg_upload_ids = ids[:500]
    patient.save(update_fields=["ecg_upload_ids"])

    return JsonResponse(
        {
            "status": "ok",
            "ecg_id": ecg.id,
            "url": request.build_absolute_uri(ecg.file.url),
            "filename": new_filename,
            "source": ecg.source,
            "uploaded_at": ecg.uploaded_at.isoformat(),
            "patient_id": patient.patient_id,
            "patient_name": patient.full_name,
        },
        status=201,
    )


@csrf_exempt
def ecg_auto_crop(request):
    """
    POST /api/ecg/auto_crop/  (multipart)
    Fields:
      - file
    Returns:
      image/jpeg (cropped bytes)
    """
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
    """
    patient_id_raw = request.GET.get("patient_id", "").strip()
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

    return JsonResponse(
        [
            {
                "id": e["id"],
                "patient_id": patient.patient_id,
                "patient_name": patient.full_name,
                "url": request.build_absolute_uri(e["imageUrl"]),
                "uploaded_at": e["uploaded_at"],
            }
            for e in ecgs
        ],
        safe=False,
        status=200,
    )


def detect_ecg_format(file_bytes: bytes) -> str:
    # 1) try xml (philips/aecg)
    parser = etree.XMLParser(recover=True, huge_tree=True)
    try:
        root = etree.fromstring(file_bytes, parser)
        if "annotatedecg" in root.tag.lower():
            return "aecg"
        if root.xpath('//*[local-name()="repbeats" or local-name()="repbeat"]'):
            return "philips"
    except Exception:
        pass

    # 2) try image
    try:
        buf = np.frombuffer(file_bytes, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is not None:
            return "image_3x4"
    except Exception:
        pass

    return "unknown"


def _norm_code(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\ufeff", "")   # BOM
    s = s.replace("\u200b", "")   # zero width space
    s = s.replace("\u00a0", " ")  # non-breaking space
    return s.strip()


def _predict_from_3x4_image_bytes(img_bytes: bytes, filename: str = "upload.png"):
    dataset_id = "504"
    target_len = 1000
    px_per_mV = 40.0
    fs_assumed = 100.0

    media_root = Path(getattr(settings, "MEDIA_ROOT", Path(__file__).resolve().parents[1] / "media"))
    job_id = uuid.uuid4().hex

    out_dir = media_root / "test_out" / "3x4_api" / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # ============== (A) save src image ==============
    src_path = out_dir / f"src_{Path(filename).stem}.png"
    buf = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image bytes (not a valid PNG/JPG).")
    cv2.imwrite(str(src_path), img)

    # ============== (B) preprocess_3x4 ==============
    pre_png = out_dir / f"pre_{src_path.stem}.png"
    preprocess_3x4(src_path, pre_png)

    # ============== (C) nnU-Net (504) ==============
    pred_nii = run_nnunet(pre_png, dataset_id=dataset_id, is_png=False)

    if pred_nii.suffixes[-2:] != [".nii", ".gz"]:
        raise RuntimeError(f"Expected .nii.gz from 504, got: {pred_nii}")

    # ============== (D) nii -> binary mask png ==============
    mask_png = out_dir / f"mask_504_{src_path.stem}.png"
    nii_to_png_mask(pred_nii, mask_png, binary=True)

    # ============== (E) digitize  ==============
    debug_dir = out_dir / f"debug_digitize_{src_path.stem}"
    leads_mV, digi_meta = digitize_from_binary_mask_3x4(
        str(mask_png),
        target_len_samples=target_len,
        px_per_mV=px_per_mV,
        debug_dir=str(debug_dir),
        return_meta=True,
    )

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

    # ============== (F) proto preprocess ==============
    X, fs_out = preprocess_single_ecg(
        leads_mV,
        fs_in=int(fs_assumed),
        fs_target=int(fs_assumed),
        target_len=target_len,
        remove_baseline=True,
    )

    # ============== (G) proto predict + topk ==============
    predictor = FusionPredictor.instance()

    probs = predictor.predict_from_X(X)
    probs = np.asarray(probs, dtype=np.float32).reshape(-1)

    labels = list(predictor.fusion_labels)

    idxs = np.argsort(-probs)[:10]

    topk = [
        {"label": labels[i], "p": float(probs[i])}
        for i in idxs
    ]

    if isinstance(digi_meta, dict):
        for k in ("y_pix_by_lead", "y_path_by_lead", "y_by_lead", "x_by_lead", "points_by_lead"):
            digi_meta.pop(k, None)

    meta = {
        "format": "image_3x4",
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
        }
    }

    return topk, meta


@csrf_exempt
def predict_ecg(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)

    if "file" not in request.FILES:
        return JsonResponse({"error": "missing file field"}, status=400)

    up = request.FILES["file"]
    file_bytes = up.read()
    fmt = detect_ecg_format(file_bytes)

    try:
        # ===== XML / aECG =====
        if fmt == "philips":
            leads_mV, fs, meta = read_hospital_ecg_xml_repbeats(file_bytes)
            meta_key = "xml_meta"

            X, fs_out = preprocess_single_ecg(leads_mV, fs_in=fs, fs_target=100, target_len=1000, remove_baseline=True)
            predictor = FusionPredictor.instance()
            probs = predictor.predict_from_X(X)

            labels = predictor.fusion_labels
            idx = np.argsort(-probs)[:10]
            out = [{"label": labels[i], "p": float(probs[i])} for i in idx]

            return JsonResponse({
                "topk": out,
                "meta": {"fs_in": fs, "fs_out": fs_out, meta_key: meta, "format": fmt},
            }, status=200)

        if fmt == "aecg":
            leads_mV, fs, meta = read_aecg_hl7_bytes(file_bytes)
            meta_key = "aecg_meta"

            X, fs_out = preprocess_single_ecg(leads_mV, fs_in=fs, fs_target=100, target_len=1000, remove_baseline=True)
            predictor = FusionPredictor.instance()
            probs = predictor.predict_from_X(X)

            labels = predictor.fusion_labels
            idx = np.argsort(-probs)[:10]
            out = [{"label": labels[i], "p": float(probs[i])} for i in idx]

            return JsonResponse({
                "topk": out,
                "meta": {"fs_in": fs, "fs_out": fs_out, meta_key: meta, "format": fmt},
            }, status=200)

        # ===== IMAGE 3x4 =====
        if fmt == "image_3x4":
            topk, meta = _predict_from_3x4_image_bytes(file_bytes, filename=getattr(up, "name", "upload.png"))
            return JsonResponse({"topk": topk, "meta": meta}, status=200)

        return JsonResponse({"error": "Unknown format. Expected Philips XML, HL7 aECG, or 3x4 PNG/JPG."}, status=400)

    except Exception as e:
        return JsonResponse({"error": str(e), "format": fmt}, status=500)


# ===========================
# ✅ DOCTOR NOTES API (NEW)
# ===========================

@csrf_exempt
def patient_notes(request, patient_id):
    """
    GET  /api/patients/<patient_id>/notes/  -> list notes
    POST /api/patients/<patient_id>/notes/ -> create note
    JSON body: {"text": "..."}
    """
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
    """
    PATCH/PUT /api/patients/<patient_id>/notes/<note_id>/ -> edit note
    DELETE    /api/patients/<patient_id>/notes/<note_id>/ -> delete note (optional)
    JSON body: {"text": "..."}
    """
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