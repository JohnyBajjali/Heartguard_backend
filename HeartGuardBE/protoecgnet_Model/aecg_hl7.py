# HeartGuardBE/ml/aecg_hl7.py
import numpy as np
from lxml import etree

STD_LEADS = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]

# Map HL7 MDC lead codes -> standard lead names
MDC_TO_STD = {
    "MDC_ECG_LEAD_I": "I",
    "MDC_ECG_LEAD_II": "II",
    "MDC_ECG_LEAD_III": "III",
    "MDC_ECG_LEAD_AVR": "aVR",
    "MDC_ECG_LEAD_AVL": "aVL",
    "MDC_ECG_LEAD_AVF": "aVF",
    "MDC_ECG_LEAD_V1": "V1",
    "MDC_ECG_LEAD_V2": "V2",
    "MDC_ECG_LEAD_V3": "V3",
    "MDC_ECG_LEAD_V4": "V4",
    "MDC_ECG_LEAD_V5": "V5",
    "MDC_ECG_LEAD_V6": "V6",
}

def _parse_digits_text(txt: str) -> np.ndarray | None:
    if txt is None:
        return None
    txt = txt.strip()
    if not txt:
        return None
    arr = np.fromstring(txt, sep=" ", dtype=np.int32)
    if arr.size == 0:
        arr = np.array([int(x) for x in txt.split()], dtype=np.int32)
    return arr.astype(np.float32)

def _unit_to_seconds(value: float, unit: str) -> float:
    unit = (unit or "s").strip().lower()
    if unit == "s":
        return value
    if unit == "ms":
        return value * 1e-3
    if unit in ("us", "µs"):
        return value * 1e-6
    # unknown -> assume seconds
    return value

def _to_uV(v: float, unit: str) -> float:
    u = (unit or "uV").strip().lower()
    if u in ("uv", "µv"):
        return v
    if u == "mv":
        return v * 1000.0
    if u == "v":
        return v * 1e6
    return v  # assume uV

def read_aecg_hl7_bytes(aecg_bytes: bytes):
    """
    Parse HL7 aECG (.aECG / XML) bytes and extract 12-lead rhythm waveforms.

    Returns:
      leads_mV: dict {lead_name: np.ndarray (N,)} in mV
      fs: int sampling rate (Hz)
      meta: dict
    """
    parser = etree.XMLParser(recover=True, huge_tree=True)
    root = etree.fromstring(aecg_bytes, parser)

    # ---- sampling rate (from TIME_ABSOLUTE increment) ----
    inc_nodes = root.xpath(
        '//*[local-name()="sequence"]/*[local-name()="code" and @code="TIME_ABSOLUTE"]/..//*[local-name()="increment"]'
    )
    if not inc_nodes:
        inc_nodes = root.xpath('//*[local-name()="increment"]')

    if not inc_nodes:
        raise ValueError("Couldn't find <increment> to infer sampling rate (fs).")

    inc = inc_nodes[0]
    inc_val = float(inc.attrib.get("value", "0.001"))
    inc_unit = inc.attrib.get("unit", "s")
    dt = _unit_to_seconds(inc_val, inc_unit)
    if dt <= 0:
        raise ValueError(f"Bad increment dt: value={inc_val}, unit={inc_unit}")
    fs = int(round(1.0 / dt))

    # ---- lead sequences ----
    lead_seq_nodes = root.xpath(
        '//*[local-name()="sequence"]/*[local-name()="code" and starts-with(@code,"MDC_ECG_LEAD_")]/..'
    )
    if not lead_seq_nodes:
        raise ValueError("Couldn't find any lead sequences with MDC_ECG_LEAD_* codes.")

    leads_mV = {}
    debug = {}

    for seq in lead_seq_nodes:
        code_el = seq.xpath('./*[local-name()="code"]')
        if not code_el:
            continue
        mdc_code = (code_el[0].attrib.get("code") or "").strip()
        std_name = MDC_TO_STD.get(mdc_code, mdc_code)

        digits_el = seq.xpath('.//*[local-name()="digits"]')
        if not digits_el:
            continue

        digits = _parse_digits_text(digits_el[0].text)
        if digits is None:
            continue

        origin_el = seq.xpath('.//*[local-name()="origin"]')
        scale_el  = seq.xpath('.//*[local-name()="scale"]')

        origin_val, origin_unit = 0.0, "uV"
        if origin_el:
            origin_val = float(origin_el[0].attrib.get("value", "0"))
            origin_unit = origin_el[0].attrib.get("unit", "uV")

        scale_val, scale_unit = 1.0, "uV"
        if scale_el:
            scale_val = float(scale_el[0].attrib.get("value", "1"))
            scale_unit = scale_el[0].attrib.get("unit", "uV")

        origin_uV = _to_uV(origin_val, origin_unit)
        scale_uV  = _to_uV(scale_val,  scale_unit)

        sig_uV = origin_uV + digits * scale_uV
        sig_mV = (sig_uV / 1000.0).astype(np.float32)

        leads_mV[std_name] = sig_mV
        debug[std_name] = {"origin_uV": origin_uV, "scale_uV": scale_uV, "n": int(sig_mV.shape[0])}

    # ---- ensure 12 standard leads exist ----
    missing = [ld for ld in STD_LEADS if ld not in leads_mV]
    if missing:
        raise ValueError(f"Missing leads in aECG: {missing}")

    meta = {
        "fs_hz": fs,
        "lead_names": STD_LEADS,
        "n_samples_each": {k: int(leads_mV[k].shape[0]) for k in STD_LEADS},
        "origin_scale_uV_debug": {k: debug[k] for k in STD_LEADS},
    }
    return leads_mV, fs, meta
