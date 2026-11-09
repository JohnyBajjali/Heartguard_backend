# HeartGuardBE/inference/xml_preview.py
# Philips XML -> ECG preview PNG (3x4 + rhythm strip + footer)
#
# This is a Django-friendly refactor of your script:
# ✅ Keeps the SAME plotting/layout logic and output image structure
# ❌ Removes argparse / folder scanning / main()
# ✅ Exposes one function: generate_xml_preview_png(xml_bytes, out_png)

from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import xml.etree.ElementTree as ET

import numpy as np

# IMPORTANT for backend servers (no GUI display)
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

PHILIPS_NS = {"p": "http://www3.medical.philips.com"}


# ----------------------------
# Time parsing + searching
# ----------------------------
@dataclass(frozen=True)
class AcquisitionStamp:
    dt: datetime
    source: str


def parse_datetime_flexible(s: str) -> datetime:
    s = s.strip()
    fmts = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %I:%M:%S %p",
        "%m/%d/%Y %I:%M:%S %p",
        "%m/%d/%Y %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
    ]
    for f in fmts:
        try:
            return datetime.strptime(s, f)
        except ValueError:
            pass
    raise ValueError(f"Unrecognized datetime format: {s!r}")


def get_acquisition_stamp(root: ET.Element) -> Optional[AcquisitionStamp]:
    def read_date_time(node_xpath: str) -> Optional[AcquisitionStamp]:
        node = root.find(node_xpath, PHILIPS_NS)
        if node is None:
            return None
        d = node.attrib.get("date", "").strip()
        t = node.attrib.get("time", "").strip()
        if not d or not t:
            return None
        try:
            dt = datetime.strptime(f"{d} {t}", "%Y-%m-%d %H:%M:%S")
            return AcquisitionStamp(dt=dt, source=node.tag.split("}")[-1])
        except ValueError:
            return None

    stamp = read_date_time(".//p:dataacquisition")
    if stamp:
        return stamp

    stamp = read_date_time(".//p:reportinfo")
    if stamp:
        return stamp

    # last resort scan
    for el in root.iter():
        d = el.attrib.get("date")
        t = el.attrib.get("time")
        if d and t:
            try:
                dt = datetime.strptime(f"{d} {t}", "%Y-%m-%d %H:%M:%S")
                return AcquisitionStamp(dt=dt, source=el.tag.split("}")[-1])
            except ValueError:
                pass

    return None


# ----------------------------
# Philips waveform decoding
# ----------------------------
def read_sampling_rate(root: ET.Element, default_hz: int = 500) -> int:
    sr = root.find(".//p:samplingrate", PHILIPS_NS)
    if sr is not None and sr.text and sr.text.strip().isdigit():
        return int(sr.text.strip())
    return default_hz


def decode_repbeat_waveforms(root: ET.Element) -> Dict[str, np.ndarray]:
    """
    Philips sample:
      <waveforms><repbeats><repbeat leadname="II"><waveform>BASE64</waveform>...
    Data: int16 little-endian
    """
    repbeats = root.find(".//p:waveforms/p:repbeats", PHILIPS_NS)
    if repbeats is None:
        raise ValueError("No <waveforms><repbeats> found in this XML.")

    out: Dict[str, np.ndarray] = {}
    for rb in repbeats.findall("p:repbeat", PHILIPS_NS):
        lead = rb.attrib.get("leadname")
        wf = rb.find("p:waveform", PHILIPS_NS)
        if not lead or wf is None or not (wf.text and wf.text.strip()):
            continue

        b64 = "".join(wf.text.split())
        raw = base64.b64decode(b64)
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32)
        out[lead] = samples

    # Keeping your warning behaviour (not fatal)
    if len(out) < 12:
        print(f"Warning: extracted {len(out)} leads (expected 12). Leads: {sorted(out.keys())}")

    return out


# ----------------------------
# Extract measurements + report info
# ----------------------------
def _first_text(root: ET.Element, xpaths: List[str]) -> Optional[str]:
    for xp in xpaths:
        el = root.find(xp, PHILIPS_NS)
        if el is not None and el.text and el.text.strip():
            return el.text.strip()
    return None


def _first_int(root: ET.Element, xpaths: List[str]) -> Optional[int]:
    s = _first_text(root, xpaths)
    if s is None:
        return None
    try:
        return int(float(s))
    except Exception:
        return None


def _first_float(root: ET.Element, xpaths: List[str]) -> Optional[float]:
    s = _first_text(root, xpaths)
    if s is None:
        return None
    try:
        return float(s)
    except Exception:
        return None


def extract_core_measurements(root: ET.Element) -> Dict[str, Optional[float]]:
    """
    Pulls:
      Rate, RR, PR, QRSd, QT, QTcB, QTcF, Axes P/QRS/T
    """
    hr = _first_int(
        root,
        [
            ".//p:interpretations/p:interpretation/p:globalmeasurements/p:heartrate",
            ".//p:internalmeasurements/p:crossleadmeasurements/p:heartrate",
        ],
    )

    # PR interval (Philips uses 'print' for PR interval, in ms)
    pr = _first_int(
        root,
        [
            ".//p:internalmeasurements/p:crossleadmeasurements/p:meanprint",
            ".//p:internalmeasurements/p:leadmeasurements/p:leadmeasurement/p:print",
        ],
    )

    # QRS duration (ms)
    qrsd = _first_int(
        root,
        [
            ".//p:internalmeasurements/p:crossleadmeasurements/p:meanqrsdur",
            ".//p:internalmeasurements/p:leadmeasurements/p:leadmeasurement/p:qrsdur",
        ],
    )

    # QT interval (ms)
    qt = _first_int(
        root,
        [
            ".//p:internalmeasurements/p:crossleadmeasurements/p:meanqtint",
            ".//p:internalmeasurements/p:leadmeasurements/p:leadmeasurement/p:qtint",
        ],
    )

    # QTc
    qtc_b = _first_int(
        root,
        [
            ".//p:interpretations/p:interpretation/p:globalmeasurements/p:qtcb",
            ".//p:internalmeasurements/p:crossleadmeasurements/p:meanqtc",
        ],
    )
    qtc_f = _first_int(
        root,
        [
            ".//p:interpretations/p:interpretation/p:globalmeasurements/p:qtcf",
        ],
    )

    # Axes (degrees)
    p_axis = _first_int(root, [".//p:internalmeasurements/p:crossleadmeasurements/p:pfrontaxis"])
    qrs_axis = _first_int(root, [".//p:internalmeasurements/p:crossleadmeasurements/p:qrsfrontaxis"])
    t_axis = _first_int(root, [".//p:internalmeasurements/p:crossleadmeasurements/p:tfrontaxis"])

    rr_ms = None
    if hr and hr > 0:
        rr_ms = int(round(60000.0 / hr))

    return {
        "Rate_bpm": hr,
        "RR_ms": rr_ms,
        "PR_ms": pr,
        "QRSd_ms": qrsd,
        "QT_ms": qt,
        "QTcB_ms": qtc_b,
        "QTcF_ms": qtc_f,
        "AxisP_deg": p_axis,
        "AxisQRS_deg": qrs_axis,
        "AxisT_deg": t_axis,
    }


def extract_report_settings(root: ET.Element) -> Dict[str, Optional[float]]:
    """
    Speed, Gain, Filters (like print footer):
      timegain = mm/s
      overallgain = mm/mV
      highpassfiltersetting, lowpassfiltersetting, notchfiltersetting
    """
    speed = _first_float(root, [".//p:reportinfo/p:reportgain/p:timegain"])
    gain = _first_float(root, [".//p:reportinfo/p:reportgain/p:amplitudegain/p:overallgain"])
    hp = _first_float(root, [".//p:reportinfo/p:reportbandwidth/p:highpassfiltersetting"])
    lp = _first_float(root, [".//p:reportinfo/p:reportbandwidth/p:lowpassfiltersetting"])
    notch = _first_float(root, [".//p:reportinfo/p:reportbandwidth/p:notchfiltersetting"])

    return {
        "Speed_mm_s": speed,
        "Gain_mm_mV": gain,
        "Highpass_Hz": hp,
        "Lowpass_Hz": lp,
        "Notch_Hz": notch,
    }


def extract_statements(root: ET.Element, max_lines: int = 4) -> List[str]:
    """
    Pull interpretation statements (if any).
    We'll show a few lines in the footer area.
    """
    stmts: List[str] = []
    for el in root.iter():
        tag = el.tag.split("}")[-1]
        if tag == "statement" and el.text and el.text.strip():
            s = " ".join(el.text.strip().split())
            stmts.append(s)

    # de-dup while preserving order
    seen = set()
    out: List[str] = []
    for s in stmts:
        if s not in seen:
            out.append(s)
            seen.add(s)

    return out[:max_lines]


# ----------------------------
# Plot helpers (LOOPING)
# ----------------------------
def loop_to_length(sig: np.ndarray, n: int) -> np.ndarray:
    """Repeat (tile) waveform until it reaches exactly n samples."""
    if sig.size == 0:
        return np.zeros(n, dtype=np.float32)
    if sig.size >= n:
        return sig[:n]
    reps = int(np.ceil(n / sig.size))
    tiled = np.tile(sig, reps)
    return tiled[:n]


def compute_common_ylim(
    signals: Dict[str, np.ndarray],
    leads: Tuple[str, ...],
    pad_ratio: float = 0.15,
) -> Tuple[float, float]:
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

    # ECG time grid @ 25mm/s:
    # small box = 0.04s, big box = 0.2s
    if show_grid:
        x_minor = 0.04
        x_major = 0.2

        ax.set_xticks(np.arange(0, duration_s + x_major, x_major))
        ax.set_xticks(np.arange(0, duration_s + x_minor, x_minor), minor=True)

        # Y grid (generic)
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
    """
    Exact layout like your photo + footer info panel.
    (This function is unchanged in structure from your script.)
    """
    layout = [
        ("I", "aVR", "V1", "V4"),
        ("II", "aVL", "V2", "V5"),
        ("III", "aVF", "V3", "V6"),
    ]

    seg_n = int(round(segment_seconds * sampling_rate_hz))
    rhy_n = int(round(rhythm_seconds * sampling_rate_hz))

    needed_top = tuple([ld for row in layout for ld in row])
    looped_top = {
        ld: loop_to_length(lead_signals.get(ld, np.array([], dtype=np.float32)), seg_n)
        for ld in needed_top
    }
    looped_rhythm = loop_to_length(lead_signals.get("II", np.array([], dtype=np.float32)), rhy_n)

    common_ylim = compute_common_ylim(looped_top, needed_top, pad_ratio=0.15)

    # 5 rows: 3 lead rows + rhythm + footer info
    fig = plt.figure(figsize=(18, 9))
    gs = GridSpec(
        5,
        4,
        figure=fig,
        height_ratios=[1.0, 1.0, 1.0, 1.15, 0.70],
        hspace=0.18,
        wspace=0.08,
    )

    # --- 3x4 leads ---
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

    # --- Rhythm strip Lead II ---
    ax_r = fig.add_subplot(gs[3, :])
    t = np.arange(rhy_n) / float(sampling_rate_hz)
    ax_r.plot(t, looped_rhythm)
    ax_r.text(0.01, 0.83, "II", transform=ax_r.transAxes, fontsize=12, fontweight="bold")
    style_ecg_axes(ax_r, rhythm_seconds, common_ylim, show_grid=show_grid)

    # --- Footer info row ---
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
# Django-friendly API
# ----------------------------
def generate_xml_preview_png(
    xml_bytes: bytes,
    out_png: Path,
    *,
    segment_seconds: float = 2.5,
    rhythm_seconds: float = 10.0,
    show_grid: bool = True,
    max_statement_lines: int = 6,
) -> dict:
    """
    Generate ECG preview PNG from Philips XML bytes.
    Saves PNG to out_png and returns metadata dict.

    ✅ Does NOT change the output image structure (uses save_ecg_with_footer)
    ✅ Designed to be called from Django views
    """
    root = ET.fromstring(xml_bytes)

    sampling_rate = read_sampling_rate(root, default_hz=500)
    leads = decode_repbeat_waveforms(root)

    meas = extract_core_measurements(root)
    settings = extract_report_settings(root)
    stmts = extract_statements(root, max_lines=max_statement_lines)

    stamp = get_acquisition_stamp(root)
    stamp_str = stamp.dt.strftime("%Y-%m-%d %H:%M:%S") if stamp else "Unknown time"
    title = f"Philips ECG — {stamp_str}"

    save_ecg_with_footer(
        lead_signals=leads,
        sampling_rate_hz=sampling_rate,
        out_png=Path(out_png),
        title=title,
        meas=meas,
        settings=settings,
        statements=stmts,
        segment_seconds=segment_seconds,
        rhythm_seconds=rhythm_seconds,
        show_grid=show_grid,
    )

    return {
        "preview_generated": True,
        "sampling_rate_hz": sampling_rate,
        "stamp": stamp_str,
        "measurements": meas,
        "settings": settings,
        "statements": stmts,
        "title": title,
        "out_png": str(out_png),
    }