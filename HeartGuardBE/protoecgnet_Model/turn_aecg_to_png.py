# turn_aecg_to_png.py
# Convert HL7 aECG (.aECG) -> clinical PNG (3x4 + long II + footer)
# Usage:
#   python turn_aecg_to_png.py --aecg "C:\path\file.aECG" --out "C:\path\out.png"
#   python turn_aecg_to_png .py --in_dir "C:\folder" --out_dir "C:\out_folder"

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import xml.etree.ElementTree as ET
import numpy as np

import matplotlib
matplotlib.use("Agg")  # IMPORTANT: set backend BEFORE pyplot
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Optional, Dict
from lxml import etree


HL7_NS = {"hl7": "urn:hl7-org:v3"}

# Map HL7 lead codes -> short clinical names
CODE_TO_SHORT = {
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
    # sometimes already short:
    "I": "I",
    "II": "II",
    "III": "III",
    "aVR": "aVR",
    "aVL": "aVL",
    "aVF": "aVF",
    "V1": "V1",
    "V2": "V2",
    "V3": "V3",
    "V4": "V4",
    "V5": "V5",
    "V6": "V6",
}

SHORT_LEADS = {"I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"}


# ----------------------------
# Time parsing (optional)
# ----------------------------
@dataclass(frozen=True)
class AcquisitionStamp:
    dt: datetime
    source: str


def _try_parse_yyyymmddhhmmss(v: str) -> Optional[datetime]:
    v = v.strip()
    # HL7 often: YYYYMMDDHHMMSS or YYYYMMDDHHMMSS+ZZZZ
    v = v.split("+")[0].split("-")[0]
    for fmt in ("%Y%m%d%H%M%S", "%Y%m%d%H%M"):
        try:
            return datetime.strptime(v, fmt)
        except ValueError:
            pass
    return None


def get_acquisition_stamp_aecg(root: ET.Element, file_stem: str) -> Optional[AcquisitionStamp]:
    # 1) try HL7 effectiveTime value
    for el in root.iter():
        tag = el.tag.split("}")[-1]
        if tag.lower() == "effectivetime":
            val = el.attrib.get("value", "")
            dt = _try_parse_yyyymmddhhmmss(val)
            if dt:
                return AcquisitionStamp(dt=dt, source="effectiveTime")

    # 2) fallback: from filename like 20250915163314
    dt2 = _try_parse_yyyymmddhhmmss(file_stem)
    if dt2:
        return AcquisitionStamp(dt=dt2, source="filename")

    return None


# ----------------------------
# aECG decoding
# ----------------------------
def parse_aecg_hl7(aecg_path: Path) -> Tuple[Dict[str, np.ndarray], float, Dict]:
    """
    Parse HL7 aECG XML and return:
      - leads_mV: dict { 'I','II',... } each np.ndarray in mV
      - fs: sampling frequency in Hz (float)
      - meta: debug info
    """
    tree = ET.parse(aecg_path)
    root = tree.getroot()

    leads_mV: Dict[str, np.ndarray] = {}
    fs: Optional[float] = None
    found_codes: List[str] = []

    for seq in root.findall(".//hl7:sequence", HL7_NS):
        code_el = seq.find(".//hl7:code", HL7_NS)
        if code_el is None:
            continue
        code = (code_el.attrib.get("code") or "").strip()
        if not code:
            continue

        found_codes.append(code)
        lead_name = CODE_TO_SHORT.get(code, None)
        if lead_name is None or lead_name not in SHORT_LEADS:
            continue

        digits_el = seq.find(".//hl7:digits", HL7_NS)
        if digits_el is None or not (digits_el.text and digits_el.text.strip()):
            continue

        # sampling rate from increment (seconds/sample)
        if fs is None:
            inc_el = seq.find(".//hl7:value/hl7:increment", HL7_NS)
            if inc_el is not None:
                unit = (inc_el.attrib.get("unit") or "").strip()
                inc_val = (inc_el.attrib.get("value") or "").strip()
                if unit == "s" and inc_val:
                    try:
                        inc = float(inc_val)
                        if inc > 0:
                            fs = 1.0 / inc
                    except Exception:
                        pass

        # scale/origin usually in microvolts (uV)
        scale_el = seq.find(".//hl7:scale", HL7_NS)
        origin_el = seq.find(".//hl7:origin", HL7_NS)
        scale_uV = float(scale_el.attrib.get("value", "1")) if scale_el is not None else 1.0
        origin_uV = float(origin_el.attrib.get("value", "0")) if origin_el is not None else 0.0

        # digits are whitespace-separated numbers
        digits_txt = " ".join(digits_el.text.split())
        digits = np.fromstring(digits_txt, sep=" ", dtype=np.float32)

        y_uV = origin_uV + scale_uV * digits
        y_mV = y_uV / 1000.0  # uV -> mV

        # keep the first occurrence if duplicated
        if lead_name not in leads_mV:
            leads_mV[lead_name] = y_mV

    if fs is None:
        fs = 1000.0  # fallback like your pipeline

    meta = {
        "fs_hz": fs,
        "lead_names": sorted(leads_mV.keys()),
        "n_samples_each": {k: int(v.size) for k, v in leads_mV.items()},
        "codes_found_debug": found_codes[:50],
    }

    if len(leads_mV) == 0:
        raise ValueError("No usable lead data found in this .aECG file (no matching <sequence><digits>).")

    return leads_mV, fs, meta


# ----------------------------
# Footer formatting (same style)
# ----------------------------
def loop_to_length(sig: np.ndarray, n: int) -> np.ndarray:
    if sig.size == 0:
        return np.zeros(n, dtype=np.float32)
    if sig.size >= n:
        return sig[:n]
    reps = int(np.ceil(n / sig.size))
    tiled = np.tile(sig, reps)
    return tiled[:n]


def compute_common_ylim(signals: Dict[str, np.ndarray], leads: Tuple[str, ...], pad_ratio: float = 0.15) -> Tuple[float, float]:
    vals = []
    for ld in leads:
        if ld in signals and signals[ld].size:
            vals.append(signals[ld])
    if not vals:
        return (-1.0, 1.0)
    x = np.concatenate(vals)
    p = np.percentile(np.abs(x), 99)
    if p <= 1e-6:
        p = float(np.max(np.abs(x)) + 1.0)
    lim = p * (1.0 + pad_ratio)
    return (-lim, lim)


def style_ecg_axes(ax, duration_s: float, ylim: Tuple[float, float], show_grid: bool = True):
    ax.set_xlim(0, duration_s)
    ax.set_ylim(*ylim)

    if show_grid:
        x_minor = 0.04
        x_major = 0.2
        ax.set_xticks(np.arange(0, duration_s + x_major, x_major))
        ax.set_xticks(np.arange(0, duration_s + x_minor, x_minor), minor=True)

        y0, y1 = ylim
        y_span = (y1 - y0)
        y_minor = y_span / 20.0
        y_major = y_minor * 5.0

        ax.set_yticks(np.arange(np.floor(y0 / y_major) * y_major, y1 + y_major, y_major))
        ax.set_yticks(np.arange(np.floor(y0 / y_minor) * y_minor, y1 + y_minor, y_minor), minor=True)

        ax.grid(True, which="major", linewidth=0.6)
        ax.grid(True, which="minor", linewidth=0.3)

    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _format_meas_line(m: Dict[str, Optional[float]]) -> str:
    def fmt_int(v):
        return "—" if v is None else f"{int(v)}"
    return (
        f"Rate: {fmt_int(m.get('Rate_bpm'))} bpm   "
        f"RR: {fmt_int(m.get('RR_ms'))} ms   "
        f"PR: {fmt_int(m.get('PR_ms'))} ms   "
        f"QRSd: {fmt_int(m.get('QRSd_ms'))} ms   "
        f"QT: {fmt_int(m.get('QT_ms'))} ms   "
        f"QTcB: {fmt_int(m.get('QTcB_ms'))} ms   "
        f"QTcF: {fmt_int(m.get('QTcF_ms'))} ms"
    )


def _format_axis_line(m: Dict[str, Optional[float]]) -> str:
    def fmt_deg(v):
        return "—" if v is None else f"{int(v)}°"
    return (
        f"Axis:  P {fmt_deg(m.get('AxisP_deg'))}   "
        f"QRS {fmt_deg(m.get('AxisQRS_deg'))}   "
        f"T {fmt_deg(m.get('AxisT_deg'))}"
    )


def _format_settings_line(s: Dict[str, Optional[float]]) -> str:
    def fmt(v, unit="", nd=2):
        if v is None:
            return "—"
        if nd == 0:
            return f"{int(round(v))}{unit}"
        return f"{v:.{nd}f}{unit}"

    speed = fmt(s.get("Speed_mm_s"), " mm/s", nd=0)
    gain = fmt(s.get("Gain_mm_mV"), " mm/mV", nd=0)
    hp = fmt(s.get("Highpass_Hz"), " Hz", nd=2)
    lp = fmt(s.get("Lowpass_Hz"), " Hz", nd=0)
    notch = fmt(s.get("Notch_Hz"), " Hz", nd=0)

    return f"Speed: {speed}    Gain: {gain}    Filters: {hp}–{lp}    Notch: {notch}"


def save_ecg_with_footer(
    lead_signals: Dict[str, np.ndarray],
    sampling_rate_hz: int,
    out_png: Path,
    title: Optional[str],
    meas: Dict[str, Optional[float]],
    settings: Dict[str, Optional[float]],
    statements: List[str],
    segment_seconds: float = 2.5,
    rhythm_seconds: float = 10.0,
    show_grid: bool = True,
):
    layout = [
        ("I",   "aVR", "V1", "V4"),
        ("II",  "aVL", "V2", "V5"),
        ("III", "aVF", "V3", "V6"),
    ]

    seg_n = int(round(segment_seconds * sampling_rate_hz))
    rhy_n = int(round(rhythm_seconds * sampling_rate_hz))

    needed_top = tuple([ld for row in layout for ld in row])
    looped_top = {ld: loop_to_length(lead_signals.get(ld, np.array([], dtype=np.float32)), seg_n) for ld in needed_top}
    looped_rhythm = loop_to_length(lead_signals.get("II", np.array([], dtype=np.float32)), rhy_n)

    common_ylim = compute_common_ylim(looped_top, needed_top, pad_ratio=0.15)

    fig = plt.figure(figsize=(18, 9))
    gs = GridSpec(
        5, 4,
        figure=fig,
        height_ratios=[1.0, 1.0, 1.0, 1.15, 0.70],
        hspace=0.18,
        wspace=0.08
    )

    for r in range(3):
        for c in range(4):
            lead = layout[r][c]
            ax = fig.add_subplot(gs[r, c])

            sig = looped_top.get(lead, None)
            if sig is None or sig.size == 0:
                ax.text(0.02, 0.5, f"{lead} (missing)", transform=ax.transAxes, va="center")
                style_ecg_axes(ax, segment_seconds, common_ylim, show_grid=show_grid)
                continue

            t = np.arange(seg_n) / float(sampling_rate_hz)
            ax.plot(t, sig)
            ax.text(0.01, 0.83, lead, transform=ax.transAxes, fontsize=12, fontweight="bold")
            style_ecg_axes(ax, segment_seconds, common_ylim, show_grid=show_grid)

    ax_r = fig.add_subplot(gs[3, :])
    t = np.arange(rhy_n) / float(sampling_rate_hz)
    ax_r.plot(t, looped_rhythm)
    ax_r.text(0.01, 0.83, "II", transform=ax_r.transAxes, fontsize=12, fontweight="bold")
    style_ecg_axes(ax_r, rhythm_seconds, common_ylim, show_grid=show_grid)

    ax_info = fig.add_subplot(gs[4, :])
    ax_info.axis("off")

    line1 = _format_meas_line(meas)
    line2 = _format_axis_line(meas)
    line3 = _format_settings_line(settings)

    y = 0.85
    ax_info.text(0.01, y, line1, fontsize=11, fontweight="bold", va="top")
    y -= 0.28
    ax_info.text(0.01, y, line2, fontsize=11, va="top")
    y -= 0.28
    ax_info.text(0.01, y, line3, fontsize=11, va="top")

    if statements:
        stmt_text = " | ".join(statements)
        ax_info.text(0.01, 0.05, f"Interpretation: {stmt_text}", fontsize=10, va="bottom")

    if title:
        fig.suptitle(title, fontsize=14)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# ----------------------------
# Main
# ----------------------------
def convert_one(aecg_path: Path, out_png: Optional[Path], segment_seconds: float, rhythm_seconds: float):
    leads_mV, fs, meta = parse_aecg_hl7(aecg_path)

    # aECG usually doesn't include Philips-like measurement/footer fields -> keep defaults
    meas = {
        "Rate_bpm": None,
        "RR_ms": None,
        "PR_ms": None,
        "QRSd_ms": None,
        "QT_ms": None,
        "QTcB_ms": None,
        "QTcF_ms": None,
        "AxisP_deg": None,
        "AxisQRS_deg": None,
        "AxisT_deg": None,
    }
    # you can set defaults if you want:
    settings = {
        "Speed_mm_s": 25,
        "Gain_mm_mV": 10,
        "Highpass_Hz": None,
        "Lowpass_Hz": None,
        "Notch_Hz": None,
    }
    statements: List[str] = []

    stamp = get_acquisition_stamp_aecg(ET.parse(aecg_path).getroot(), aecg_path.stem)
    stamp_str = stamp.dt.strftime("%Y-%m-%d %H:%M:%S") if stamp else "Unknown time"
    title = f"aECG — {stamp_str}"

    out_png = out_png if out_png else aecg_path.with_suffix(".png")

    save_ecg_with_footer(
        lead_signals=leads_mV,
        sampling_rate_hz=int(round(fs)),
        out_png=out_png,
        title=title,
        meas=meas,
        settings=settings,
        statements=statements,
        segment_seconds=segment_seconds,
        rhythm_seconds=rhythm_seconds,
        show_grid=True,
    )

    print("Saved PNG:", out_png)
    # optional debug:
    # print("Meta:", meta)

from typing import Optional, Dict
from lxml import etree

def extract_aecg_parameters(file_bytes: bytes) -> Dict[str, Optional[float]]:
    root = etree.fromstring(file_bytes)

    def get(tag: str) -> Optional[float]:
        # use xpath (supports local-name())
        nodes = root.xpath(
            f"//*[local-name()='Parameters']/*[local-name()='{tag}']/text()"
        )
        if not nodes:
            return None
        try:
            return float(str(nodes[0]).strip())
        except Exception:
            return None

    hr = get("HR")
    rr = int(round(60000.0 / hr)) if hr and hr > 0 else None

    qtc = get("QTcInterval")  # sometimes only one QTc exists

    def to_int(x):
        return int(round(x)) if x is not None else None

    return {
        "Rate_bpm": to_int(hr),
        "RR_ms": rr,
        "PR_ms": to_int(get("PRInterval")),
        "QRSd_ms": to_int(get("QRSDuration")),
        "QT_ms": to_int(get("QTInterval")),
        "QTcB_ms": to_int(qtc),
        "QTcF_ms": None,
        "AxisP_deg": to_int(get("PAxis")),
        "AxisQRS_deg": to_int(get("QRSAxis")),
        "AxisT_deg": to_int(get("TAxis")),
    }

def main():
    ap = argparse.ArgumentParser(description="Convert HL7 aECG (.aECG) to clinical PNG (3x4 + long II + footer).")
    ap.add_argument("--aecg", default="", help="Convert a specific .aECG file directly")
    ap.add_argument("--in_dir", default="", help="Folder containing .aECG files")
    ap.add_argument("--out", default="", help="Output PNG path for single file (optional)")
    ap.add_argument("--out_dir", default="", help="Output folder for batch mode")
    ap.add_argument("--segment_seconds", type=float, default=2.5)
    ap.add_argument("--rhythm_seconds", type=float, default=10.0)
    args = ap.parse_args()

    if args.aecg:
        aecg_path = Path(args.aecg)
        if not aecg_path.exists():
            raise FileNotFoundError(f"aECG not found: {aecg_path}")
        out_png = Path(args.out) if args.out else None
        convert_one(aecg_path, out_png, args.segment_seconds, args.rhythm_seconds)
        return

    if args.in_dir:
        in_dir = Path(args.in_dir)
        if not in_dir.exists():
            raise FileNotFoundError(f"in_dir not found: {in_dir}")
        out_dir = Path(args.out_dir) if args.out_dir else (in_dir / "aecg_png")
        out_dir.mkdir(parents=True, exist_ok=True)

        files = sorted([p for p in in_dir.rglob("*.aecg")] + [p for p in in_dir.rglob("*.aECG")])
        print(f"Found {len(files)} aECG files.")
        for i, p in enumerate(files, 1):
            try:
                out_png = out_dir / (p.stem + ".png")
                convert_one(p, out_png, args.segment_seconds, args.rhythm_seconds)
                print(f"[{i}/{len(files)}] OK  {p.name}")
            except Exception as e:
                print(f"[{i}/{len(files)}] FAIL {p.name}: {e}")
        return

    raise ValueError("Provide either --aecg for a single file OR --in_dir for batch mode.")


if __name__ == "__main__":
    main()
