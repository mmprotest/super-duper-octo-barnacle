from __future__ import annotations

import numpy as np

try:
    from scipy.signal import resample_poly
except Exception:  # pragma: no cover - optional dependency safety
    resample_poly = None



def resample_audio(audio: np.ndarray, src_rate: int, target_rate: int) -> np.ndarray:
    if src_rate == target_rate:
        return audio.astype(np.float32, copy=False)
    if resample_poly is None:
        raise RuntimeError("scipy is required for audio resampling")
    gcd = np.gcd(src_rate, target_rate)
    up = target_rate // gcd
    down = src_rate // gcd
    return resample_poly(audio, up, down).astype(np.float32)
