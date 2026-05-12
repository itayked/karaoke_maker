"""Silero VAD wrapper.

Used as a fallback: if direct alignment on the full vocals track returns very few
matches, we trim the audio to the [first_voiced - pad, last_voiced + pad] range
and retry alignment, then shift timestamps back to the original timeline.

We deliberately do NOT splice out internal instrumental gaps — that would break
timestamp continuity for words sung across them. Only the head/tail gets trimmed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

_VAD_SR = 16_000


@dataclass
class VoicedRange:
    start: float  # seconds, in the original audio's timeline
    end: float


def detect_voiced_range(audio_path: Path, *, pad_s: float = 0.5) -> VoicedRange | None:
    """Return (first_voiced_start, last_voiced_end) in original-timeline seconds, or None."""
    try:
        from silero_vad import get_speech_timestamps, load_silero_vad, read_audio
    except ImportError:
        return None

    model = load_silero_vad()
    wav = read_audio(str(audio_path), sampling_rate=_VAD_SR)
    ts = get_speech_timestamps(wav, model, sampling_rate=_VAD_SR)
    if not ts:
        return None

    duration = len(wav) / _VAD_SR
    start = max(0.0, ts[0]["start"] / _VAD_SR - pad_s)
    end = min(duration, ts[-1]["end"] / _VAD_SR + pad_s)
    return VoicedRange(start=start, end=end)


def trim_to_range(audio_path: Path, out_path: Path, rng: VoicedRange) -> Path:
    """Write a copy of `audio_path` containing only samples in [rng.start, rng.end]."""
    data, rate = sf.read(str(audio_path))
    a = int(rng.start * rate)
    b = int(rng.end * rate)
    sliced = data[a:b] if data.ndim == 1 else data[a:b, :]
    sf.write(str(out_path), np.asarray(sliced), rate)
    return out_path
