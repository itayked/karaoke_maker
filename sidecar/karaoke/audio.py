"""Audio utilities: loudness normalization, duration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf


def get_duration(path: Path) -> float:
    info = sf.info(str(path))
    return float(info.frames) / float(info.samplerate)


def loudness_normalize(in_path: Path, out_path: Path, target_lufs: float = -23.0) -> Path:
    """Normalize integrated loudness. Falls back to peak normalization on degenerate input."""
    data, rate = sf.read(str(in_path))
    if data.ndim == 1:
        meas = data
    else:
        meas = data.mean(axis=1)

    try:
        meter = pyln.Meter(rate)
        loudness = meter.integrated_loudness(meas)
        if not np.isfinite(loudness):
            raise ValueError("non-finite loudness")
        normalized = pyln.normalize.loudness(data, loudness, target_lufs)
    except Exception:
        peak = float(np.max(np.abs(data))) or 1.0
        normalized = data * (0.7 / peak)

    sf.write(str(out_path), normalized, rate)
    return out_path
