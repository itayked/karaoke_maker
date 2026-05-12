"""On-disk library: one folder per song under <library_dir>/<song_id>/."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

from karaoke.models import PipelineResult
from karaoke.server.schemas import LibraryEntry


def _entry_dir(library_dir: Path, song_id: str) -> Path:
    return library_dir / song_id


class LibraryStore:
    def __init__(self, library_dir: Path):
        self.library_dir = library_dir
        self.library_dir.mkdir(parents=True, exist_ok=True)

    # ---------- persistence ----------

    def song_dir(self, song_id: str) -> Path:
        """Where the orchestrator should land the instrumental for this song."""
        d = _entry_dir(self.library_dir, song_id)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save(
        self,
        song_id: str,
        result: PipelineResult,
        *,
        youtube_url: str,
        metadata: dict,
        model_used: str,
    ) -> LibraryEntry:
        d = _entry_dir(self.library_dir, song_id)
        d.mkdir(parents=True, exist_ok=True)

        # The orchestrator already wrote `instrumental.wav` into `d` — we only
        # have to persist the metadata + lyrics here.
        result.audio_url = f"/library/{song_id}/audio"
        (d / "lyrics.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")

        entry = LibraryEntry(
            song_id=song_id,
            title=metadata.get("title"),
            youtube_url=youtube_url,
            thumbnail_url=metadata.get("thumbnail"),
            duration_seconds=result.duration_seconds,
            model_used=model_used,
            created_at=time.time(),
        )
        (d / "meta.json").write_text(entry.model_dump_json(indent=2), encoding="utf-8")
        return entry

    # ---------- read ----------

    def list(self) -> list[LibraryEntry]:
        entries: list[LibraryEntry] = []
        if not self.library_dir.exists():
            return entries
        for sub in self.library_dir.iterdir():
            if not sub.is_dir():
                continue
            meta_path = sub / "meta.json"
            if not meta_path.exists():
                continue
            try:
                entries.append(LibraryEntry.model_validate_json(meta_path.read_text("utf-8")))
            except Exception:
                continue
        entries.sort(key=lambda e: e.created_at, reverse=True)
        return entries

    def get_lyrics(self, song_id: str) -> PipelineResult | None:
        d = _entry_dir(self.library_dir, song_id)
        lyrics_path = d / "lyrics.json"
        if not lyrics_path.exists():
            return None
        return PipelineResult.model_validate_json(lyrics_path.read_text("utf-8"))

    def get_meta(self, song_id: str) -> LibraryEntry | None:
        d = _entry_dir(self.library_dir, song_id)
        meta_path = d / "meta.json"
        if not meta_path.exists():
            return None
        return LibraryEntry.model_validate_json(meta_path.read_text("utf-8"))

    def get_audio_path(self, song_id: str) -> Path | None:
        d = _entry_dir(self.library_dir, song_id)
        p = d / "instrumental.wav"
        return p if p.exists() else None

    # ---------- delete ----------

    def delete(self, song_id: str) -> bool:
        d = _entry_dir(self.library_dir, song_id)
        if not d.exists():
            return False
        shutil.rmtree(d, ignore_errors=True)
        return True
