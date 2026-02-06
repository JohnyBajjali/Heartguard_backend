# HeartGuardBE/ecg_buckets.py
from __future__ import annotations

from pathlib import Path
import csv
from django.conf import settings

# Some rhythm / device codes are not always classified in diagnostic_class.
# We still want them grouped nicely.
EXTRA_BUCKETS = [
    {
        "key": "ARRHYTHMIA",
        "title": "ARRHYTHMIA",
        # We keep these conservative, only common ones used in PTB-XL style labeling.
        # If CSV contains more rhythm codes, they will be handled by "OTHER" unless you add them here.
        "labels": [
            "AFIB", "AFLT", "SVTAC", "PSVT", "STACH", "SBRAD",
            "PAC", "PVC", "BIGU", "TRIGU",
        ],
    },
    {
        "key": "PACEMAKER_DEVICE",
        "title": "PACEMAKER / DEVICE",
        "labels": ["PACE"],
    },
]

# These are the diagnostic classes used in your CSV regrouping.
CLASS_TO_BUCKET = {
    "MI": ("MI", "MI"),
    "STTC": ("STTC", "ISCHEMIA / ST-T"),
    "CD": ("CD", "CONDUCTION"),
    "HYP": ("HYP", "HYPERTROPHY"),
    "NORM": ("NORM", "NORMAL"),
}

def _csv_path() -> Path:
    """
    Default path inside backend repo:
      HeartGuardBE/resources/scp_statementsRegrouped2.csv
    """
    repo_root = Path(getattr(settings, "BASE_DIR", Path(__file__).resolve().parents[1]))
    return repo_root / "HeartGuardBE" / "resources" / "scp_statementsRegrouped2.csv"

def load_csv_codes() -> dict[str, dict]:
    """
    Returns mapping: code -> row dict (like: diagnostic_class, description, ...)
    """
    path = _csv_path()
    if not path.exists():
        # Fail safe: no CSV => empty map (still won't crash)
        print(f"⚠️ scp CSV not found at: {path}")
        return {}

    mapping: dict[str, dict] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        # We assume first column contains the code; in PTB-XL regrouping it's usually "statement"
        # We'll detect it robustly.
        headers = [h.strip() for h in (reader.fieldnames or [])]
        # Try common names:
        code_keys = ["statement", "scp_code", "code", headers[0] if headers else "statement"]

        for row in reader:
            # pick first available code field
            code = ""
            for k in code_keys:
                if k in row and row[k]:
                    code = str(row[k]).strip()
                    break
            if not code:
                continue
            # normalize
            code = code.replace("\ufeff", "").strip()
            mapping[code] = {kk: (vv.strip() if isinstance(vv, str) else vv) for kk, vv in row.items()}

    return mapping

def build_bucket_index() -> tuple[dict[str, tuple[str, str]], dict[str, dict]]:
    """
    Returns:
      label_to_bucket: code -> (bucket_key, bucket_title)
      codes_map: code -> CSV row data
    """
    codes_map = load_csv_codes()
    label_to_bucket: dict[str, tuple[str, str]] = {}

    # 1) From CSV diagnostic_class
    for code, row in codes_map.items():
        dc = (row.get("diagnostic_class") or "").strip()
        if dc in CLASS_TO_BUCKET:
            key, title = CLASS_TO_BUCKET[dc]
            label_to_bucket[code] = (key, title)

    # 2) Add extra manual buckets (arrhythmia/device)
    for b in EXTRA_BUCKETS:
        for code in b["labels"]:
            label_to_bucket[code] = (b["key"], b["title"])

    return label_to_bucket, codes_map

# Cache once (fast)
_LABEL_TO_BUCKET, _CODES_MAP = build_bucket_index()

def bucketize_topk(topk: list[dict]) -> list[dict]:
    """
    Input: topk = [{"label":"AFIB","p":0.9}, ...]
    Output: list of buckets:
      [{"key":"ARRHYTHMIA","title":"ARRHYTHMIA","items":[...topk items...]}, ...]
    """
    buckets: dict[str, dict] = {}

    def push(bucket_key: str, bucket_title: str, item: dict):
        if bucket_key not in buckets:
            buckets[bucket_key] = {"key": bucket_key, "title": bucket_title, "items": []}
        buckets[bucket_key]["items"].append(item)

    other_items = []

    for item in (topk or []):
        label = str(item.get("label", "")).strip()
        if not label:
            continue

        bt = _LABEL_TO_BUCKET.get(label)
        if bt:
            key, title = bt
            push(key, title, item)
        else:
            other_items.append(item)

    # deterministic order
    order = ["ARRHYTHMIA", "CD", "STTC", "HYP", "MI", "PACEMAKER_DEVICE", "NORM"]
    out = []
    for k in order:
        if k in buckets:
            out.append(buckets[k])

    # add any remaining buckets not in predefined order
    for k, v in buckets.items():
        if k not in order:
            out.append(v)

    # always include OTHER at end
    out.append({"key": "OTHER", "title": "OTHER", "items": other_items})

    return out