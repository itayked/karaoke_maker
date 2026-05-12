"""Whisper model downloads with progress + resume.

Endpoints:
    POST /models/download {size} -> 202 if started or already running, 409 if
        a different download is in flight.
    GET  /models/status reports a `ModelStatus` with state="downloading",
        progress 0..1, bytes_downloaded, bytes_total — folded in by app.py.

Only one download runs at a time (good enough for v1; users only ever trigger
this from the first-launch flow). Resumes via HTTP Range when a `.part` file
exists from a prior attempt.
"""

from __future__ import annotations

import threading
import urllib.request
from pathlib import Path

from karaoke.log import log
from karaoke.server.schemas import ModelStatus

_CHUNK = 1 << 18  # 256 KiB
_FILENAMES = {
    "small": "whisper-small.pt",
    "medium": "whisper-medium.pt",
    "large-v3": "whisper-large-v3.pt",
}


def _whisper_url(size: str) -> str:
    """Look up the model URL from openai-whisper's own table — keeps us
    aligned with whatever stable-ts/whisper would have downloaded internally.
    """
    import whisper  # part of stable-ts's runtime deps
    return whisper._MODELS[size]


class ModelDownloadManager:
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self._lock = threading.Lock()
        self._active: dict[str, dict] = {}  # size -> {bytes_done, bytes_total, error?}
        self._thread: threading.Thread | None = None

    # ---- public API ----

    def start(self, size: str) -> str:
        """Returns one of: 'started', 'already-running', 'already-installed',
        'busy', 'unsupported'."""
        if size not in _FILENAMES:
            return "unsupported"
        dest = self.models_dir / _FILENAMES[size]
        if dest.exists():
            return "already-installed"
        with self._lock:
            if size in self._active and "error" not in self._active[size]:
                return "already-running"
            if self._thread is not None and self._thread.is_alive():
                return "busy"
            self._active[size] = {"bytes_done": 0, "bytes_total": 0}
            t = threading.Thread(
                target=self._run, args=(size, dest), daemon=True, name=f"download-{size}"
            )
            self._thread = t
            t.start()
        return "started"

    def get_progress(self, size: str) -> ModelStatus | None:
        with self._lock:
            st = self._active.get(size)
            if st is None:
                return None
            bytes_done = st["bytes_done"]
            bytes_total = st["bytes_total"] or None
            err = st.get("error")
            if err:
                # Surface as 'missing' once the user reads it once; downloads
                # are restartable.
                return ModelStatus(
                    size=size,
                    state="missing",
                    progress=0.0,
                    bytes_downloaded=0,
                    bytes_total=None,
                )
        progress = bytes_done / bytes_total if bytes_total else 0.0
        return ModelStatus(
            size=size,
            state="downloading",
            progress=round(min(1.0, max(0.0, progress)), 4),
            bytes_downloaded=bytes_done,
            bytes_total=bytes_total,
        )

    # ---- worker ----

    def _run(self, size: str, dest: Path) -> None:
        log("models.download_start", size=size, dest=str(dest))
        part = dest.with_suffix(dest.suffix + ".part")
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            url = _whisper_url(size)
        except Exception as e:
            log("models.url_lookup_failed", level="ERROR", size=size, error=str(e))
            self._mark_error(size, f"url lookup failed: {e}")
            return

        try:
            start = part.stat().st_size if part.exists() else 0
            headers = {"User-Agent": "karaoke-sidecar/0.1"}
            if start > 0:
                headers["Range"] = f"bytes={start}-"
                log("models.download_resume", size=size, start=start)

            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                content_length = int(resp.headers.get("Content-Length") or 0)
                # If we resumed, Content-Length is the remaining bytes; add the
                # part we already have on disk to make total absolute.
                total = (content_length + start) if start > 0 and resp.status == 206 else (
                    content_length or 0
                )
                with self._lock:
                    self._active[size]["bytes_total"] = total
                    self._active[size]["bytes_done"] = start

                mode = "ab" if start > 0 and resp.status == 206 else "wb"
                done = start if mode == "ab" else 0
                with open(part, mode) as f:
                    while True:
                        chunk = resp.read(_CHUNK)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        with self._lock:
                            self._active[size]["bytes_done"] = done

            part.replace(dest)
            with self._lock:
                self._active.pop(size, None)
            log("models.download_done", size=size, bytes=done)
        except Exception as e:
            log(
                "models.download_failed",
                level="ERROR",
                size=size,
                error=f"{type(e).__name__}: {e}",
            )
            self._mark_error(size, str(e))

    def _mark_error(self, size: str, msg: str) -> None:
        with self._lock:
            st = self._active.setdefault(size, {"bytes_done": 0, "bytes_total": 0})
            st["error"] = msg
