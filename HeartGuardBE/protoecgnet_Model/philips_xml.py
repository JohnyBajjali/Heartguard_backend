# HeartGuardBE/ml/philips_xml.py
import base64
import numpy as np
from lxml import etree

STD_LEADS = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]

def read_hospital_ecg_xml_repbeats(xml_bytes: bytes):
    parser = etree.XMLParser(recover=True, huge_tree=True)
    root = etree.fromstring(xml_bytes, parser)

    fs_list = [el.text.strip() for el in root.xpath('//*[local-name()="samplingrate"]')
               if el.text and el.text.strip()]
    fs = int(float(fs_list[0])) if fs_list else 500

    og_list = [el.text.strip() for el in root.xpath('//*[local-name()="overallgain"]')
               if el.text and el.text.strip()]
    gain_uV_per_lsb = float(og_list[0]) if og_list else 10.0

    repbeats = root.xpath('//*[local-name()="repbeats"]')
    if not repbeats:
        raise ValueError("Couldn't find <repbeats> in the XML.")
    repbeats = repbeats[0]

    repbeat_nodes = repbeats.xpath('./*[local-name()="repbeat"]')
    if not repbeat_nodes:
        raise ValueError("Found <repbeats> but no <repbeat> children.")

    leads_raw = {}
    for rb in repbeat_nodes:
        lead = (rb.attrib.get("lead")
                or rb.attrib.get("leadname")
                or rb.attrib.get("leadName")
                or rb.attrib.get("name"))

        wf_nodes = rb.xpath('./*[local-name()="waveform"]')
        if not lead or not wf_nodes or wf_nodes[0].text is None:
            continue

        b64 = wf_nodes[0].text.strip()
        raw_bytes = base64.b64decode(b64)
        if len(raw_bytes) % 2 != 0:
            raw_bytes = raw_bytes[:-1]

        sig = np.frombuffer(raw_bytes, dtype="<i2").astype(np.float32)
        leads_raw[lead] = sig

    if not leads_raw:
        raise ValueError("No lead waveforms extracted from <repbeats>.")

    leads_mV = {k: (v * gain_uV_per_lsb / 1000.0) for k, v in leads_raw.items()}

    missing = [ld for ld in STD_LEADS if ld not in leads_mV]
    if missing:
        raise ValueError(f"Missing leads in XML: {missing}")

    meta = {
        "fs_hz": fs,
        "gain_uV_per_lsb": gain_uV_per_lsb,
        "lead_names": STD_LEADS,
        "n_samples_each": {k: int(leads_mV[k].shape[0]) for k in STD_LEADS},
    }
    return leads_mV, fs, meta
