# HeartGuardBE/protoecgnet_Model/preprocess.py
import numpy as np
from scipy.signal import butter, filtfilt, resample_poly

STD_LEADS = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]

def baseline_highpass_per_lead(X, fs, cutoff=0.5, order=1):
    b, a = butter(order, cutoff / (fs / 2), btype='high', analog=False)
    Y = np.empty_like(X, dtype=np.float32)
    for j in range(X.shape[1]):
        Y[:, j] = filtfilt(b, a, X[:, j]).astype(np.float32)
    return Y

def tile_to_len_crossfade(X, target_len=1000, fade=20):
    T, C = X.shape
    reps = int(np.ceil(target_len / T))
    out = []
    prev = None

    for _ in range(reps):
        seg = X.copy()
        if prev is not None and fade > 0:
            f = min(fade, len(prev), len(seg))
            w = np.linspace(0, 1, f).reshape(-1, 1).astype(np.float32)
            prev[-f:] = prev[-f:] * (1 - w) + seg[:f] * w
            seg = seg[f:]
        if prev is not None:
            out.append(prev)
        prev = seg

    out.append(prev)
    Y = np.vstack(out)
    return Y[:target_len].astype(np.float32)

def preprocess_single_ecg(leads_mV: dict, fs_in: int,
                          fs_target=100, target_len=1000,
                          remove_baseline=True):
    # stack -> (T, 12)
    leads_list = [leads_mV[ld] for ld in STD_LEADS]
    T = min(len(x) for x in leads_list)
    X = np.stack([x[:T] for x in leads_list], axis=1).astype(np.float32)

    # resample
    fs_out = fs_in
    if fs_in != fs_target:
        X = resample_poly(X, up=fs_target, down=fs_in, axis=0).astype(np.float32)
        fs_out = fs_target

    if remove_baseline:
        X = baseline_highpass_per_lead(X, fs_out, cutoff=0.5, order=1)

    X = tile_to_len_crossfade(X, target_len=target_len, fade=20)
    X = X[None, :, :]  # (1,1000,12)
    return X.astype(np.float32), fs_out
