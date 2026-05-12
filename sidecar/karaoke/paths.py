"""Path resolution for bundled vs user-data vs dev resources.

In a packaged installation Electron sets `KARAOKE_RESOURCES_DIR` to
`<install>/resources/app/resources` (or wherever extraResources lands). The
sidecar prefers files there over user-data and over Whisper's dev cache.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

_WHISPER_FILENAMES = {
    "small": "whisper-small.pt",
    "medium": "whisper-medium.pt",
    "large-v3": "whisper-large-v3.pt",
}


def resources_dir() -> Path | None:
    p = os.environ.get("KARAOKE_RESOURCES_DIR")
    return Path(p) if p else None


def bundled_models_dir() -> Path | None:
    r = resources_dir()
    return (r / "models") if r else None


def bundled_ffmpeg() -> Path | None:
    r = resources_dir()
    if r is None:
        return None
    candidate = r / "bin" / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    return candidate if candidate.exists() else None


def whisper_model_locations(size: str, user_models_dir: Path) -> Iterable[Path]:
    """Yield candidate locations for a Whisper model, in lookup order:
        1. bundled resources/models/whisper-<size>.pt
        2. <user_models_dir>/whisper-<size>.pt          (downloaded by us)
        3. ~/.cache/whisper/<size>.pt                   (dev-mode fallback)
    """
    fname = _WHISPER_FILENAMES.get(size)
    if fname is None:
        return

    b = bundled_models_dir()
    if b is not None:
        yield b / fname
    yield user_models_dir / fname
    yield Path.home() / ".cache" / "whisper" / f"{size}.pt"


def find_whisper_model(size: str, user_models_dir: Path) -> Path | None:
    for p in whisper_model_locations(size, user_models_dir):
        if p.exists():
            return p
    return None


def ensure_ffmpeg_on_path() -> None:
    """Prepend the bundled FFmpeg directory to PATH so yt-dlp/Demucs find it
    even when the user has no system ffmpeg installed. No-op in dev.
    """
    ff = bundled_ffmpeg()
    if ff is None:
        return
    bin_dir = str(ff.parent)
    current = os.environ.get("PATH", "")
    sep = os.pathsep
    if bin_dir in current.split(sep):
        return
    os.environ["PATH"] = bin_dir + sep + current
