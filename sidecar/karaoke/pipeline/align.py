"""Alignment interface + stable-whisper implementation. Step 2.

The aligner is swappable. New aligners implement the `Aligner` protocol;
nothing in `postprocess.py` should depend on a concrete implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from karaoke.log import log
from karaoke.models import AlignedWord


class Aligner(Protocol):
    """Audio + lyrics text -> flat list of word-level timestamps.

    Implementations must:
      - return words in order, in seconds, with start <= end
      - set `confidence` in [0, 1] where available (else 1.0)
      - leave `interpolated=False`; interpolation is postprocess's job
    """

    def align(
        self,
        audio_path: Path,
        lyrics_text: str,
        language: str = "he",
    ) -> list[AlignedWord]: ...


class StableWhisperAligner:
    """stable-ts (formerly stable-whisper) aligner.

    The Whisper model is heavy — instantiate once and reuse across jobs.
    """

    def __init__(self, model_size: str = "large-v3", device: str = "cuda"):
        import stable_whisper

        log("align.load_model", model=model_size, device=device)
        self._model = stable_whisper.load_model(model_size, device=device)
        self._device = device

    def align(
        self,
        audio_path: Path,
        lyrics_text: str,
        language: str = "he",
    ) -> list[AlignedWord]:
        log("align.start", audio=str(audio_path), chars=len(lyrics_text))
        result = self._model.align(str(audio_path), lyrics_text, language=language)

        out: list[AlignedWord] = []
        for seg in result.segments:
            for w in seg.words:
                text = (w.word or "").strip()
                if not text:
                    continue
                start = float(w.start)
                end = float(w.end)
                if end < start:
                    end = start
                confidence = _word_confidence(w)
                out.append(
                    AlignedWord(
                        text=text,
                        start=start,
                        end=end,
                        confidence=confidence,
                        interpolated=False,
                    )
                )

        log("align.done", words=len(out))
        return out


def _word_confidence(word: Any) -> float:
    for attr in ("probability", "confidence", "score"):
        v = getattr(word, attr, None)
        if v is not None:
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            return max(0.0, min(1.0, fv))
    return 1.0
