"""yt-dlp wrapper: audio download + metadata extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yt_dlp

from karaoke.log import log


class DownloadError(RuntimeError):
    pass


def extract_metadata(url: str) -> dict[str, Any]:
    """Fetch title / thumbnail / duration_hint / uploader without downloading.

    Returns a flat dict suitable for embedding in library meta.json. Best-effort:
    on yt-dlp errors returns a minimal dict so the caller can still proceed (the
    audio download will surface the underlying error a moment later).
    """
    try:
        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True}) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as e:
        log("download.metadata_failed", level="WARN", error=str(e))
        return {"title": None, "thumbnail": None, "duration_hint": None}

    if info is None:
        return {"title": None, "thumbnail": None, "duration_hint": None}

    return {
        "title": info.get("title"),
        "thumbnail": info.get("thumbnail"),
        "duration_hint": info.get("duration"),
        "uploader": info.get("uploader"),
        "video_id": info.get("id"),
    }


def download_audio(url: str, out_dir: Path, *, socket_timeout: int = 30) -> Path:
    """Download `url` as a 16-bit wav into `out_dir`. Returns the wav path.

    yt-dlp doesn't expose a wall-clock timeout, but `socket_timeout` plus its
    own retries cover hung downloads. Hard wall-clock enforcement is handled
    by the queue layer in step 4.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_template = str(out_dir / "audio.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": out_template,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "retries": 3,
        "fragment_retries": 3,
        "socket_timeout": socket_timeout,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "0",
            }
        ],
    }

    log("download.start", url=url)
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except yt_dlp.utils.DownloadError as e:
        raise DownloadError(f"yt-dlp failed: {e}") from e

    wav_path = out_dir / "audio.wav"
    if not wav_path.exists():
        candidates = list(out_dir.glob("audio.*"))
        raise DownloadError(
            f"expected {wav_path} after extraction; found: {[p.name for p in candidates]}"
        )

    log("download.done", path=str(wav_path), bytes=wav_path.stat().st_size)
    return wav_path
