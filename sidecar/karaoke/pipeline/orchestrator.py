"""End-to-end pipeline orchestration.

The orchestrator owns its own work directory: it creates one under
`tempfile.gettempdir()`, runs the pipeline, moves the instrumental to a
caller-supplied destination, then cleans up — on both success and failure
paths. Callers don't see the work dir at all, so they can't forget to
remove it.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from karaoke.audio import get_duration, loudness_normalize
from karaoke.log import log
from karaoke.models import PipelineResult
from karaoke.pipeline.align import Aligner
from karaoke.pipeline.download import download_audio, extract_metadata
from karaoke.pipeline.postprocess import PipelineError, assemble_lines, match_rate
from karaoke.pipeline.separate import separate
from karaoke.vad import detect_voiced_range, trim_to_range

_MATCH_RATE_RETRY_THRESHOLD = 0.5

StageCallback = Callable[[str, float], None]


@dataclass
class PipelineOutputs:
    instrumental_path: Path  # stable: lives in `dest_dir`, not the work dir
    result: PipelineResult
    metadata: dict[str, Any] = field(default_factory=dict)


def _noop(_stage: str, _progress: float) -> None:
    pass


def _run_alignment_with_fallback(
    aligner: Aligner,
    vocals_norm: Path,
    lyrics: str,
    work_dir: Path,
) -> list:
    aligned = aligner.align(vocals_norm, lyrics, language="he")
    rate = match_rate(lyrics, aligned)
    log("align.match_rate", rate=round(rate, 3), words=len(aligned))

    if rate >= _MATCH_RATE_RETRY_THRESHOLD:
        return aligned

    log("align.vad_retry", reason="low match rate", rate=round(rate, 3))
    rng = detect_voiced_range(vocals_norm)
    if rng is None:
        log("align.vad_skip", reason="no voiced range detected")
        return aligned

    trimmed = trim_to_range(vocals_norm, work_dir / "vocals_trimmed.wav", rng)
    aligned_trim = aligner.align(trimmed, lyrics, language="he")
    for w in aligned_trim:
        w.start += rng.start
        w.end += rng.start
    new_rate = match_rate(lyrics, aligned_trim)
    log("align.vad_match_rate", rate=round(new_rate, 3), words=len(aligned_trim))

    return aligned_trim if new_rate > rate else aligned


def run_pipeline(
    youtube_url: str,
    lyrics: str,
    dest_dir: Path,
    *,
    aligner: Aligner,
    device: str = "cuda",
    on_stage: StageCallback = _noop,
) -> PipelineOutputs:
    """Run yt-dlp → Demucs → align → postprocess and land the instrumental in
    `dest_dir/instrumental.wav`. The orchestrator's own scratch dir is removed
    on success and on every failure path.
    """
    if not lyrics.strip():
        raise PipelineError("lyrics is empty")

    dest_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix="karaoke_"))
    log("pipeline.work_dir", path=str(work_dir))

    try:
        on_stage("downloading", 0.05)
        metadata = extract_metadata(youtube_url)
        wav = download_audio(youtube_url, work_dir / "download")

        on_stage("separating", 0.3)
        vocals, no_vocals = separate(wav, work_dir / "separated", device=device)

        on_stage("aligning", 0.6)
        vocals_norm = loudness_normalize(vocals, work_dir / "vocals_norm.wav")
        aligned = _run_alignment_with_fallback(aligner, vocals_norm, lyrics, work_dir)
        if not aligned:
            raise PipelineError(
                "aligner returned zero words — vocals stem may be silent or the "
                "lyrics may not match the audio"
            )

        duration = get_duration(no_vocals)
        lines = assemble_lines(lyrics, aligned, duration)

        on_stage("finalizing", 0.95)
        instrumental_dest = dest_dir / "instrumental.wav"
        # shutil.move handles cross-volume cases; on same-volume it's a rename.
        shutil.move(str(no_vocals), str(instrumental_dest))

        result = PipelineResult(
            audio_url="",
            duration_seconds=duration,
            lines=lines,
        )
        return PipelineOutputs(
            instrumental_path=instrumental_dest,
            result=result,
            metadata=metadata,
        )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def sweep_stale_work_dirs(max_age_seconds: int = 3600) -> int:
    """Defense in depth: nuke leftover karaoke_* dirs in tempdir on sidecar boot.

    Returns number of dirs removed. Called from the FastAPI lifespan.
    """
    import time

    tmp = Path(tempfile.gettempdir())
    if not tmp.exists():
        return 0
    cutoff = time.time() - max_age_seconds
    removed = 0
    for d in tmp.glob("karaoke_*"):
        try:
            if d.is_dir() and d.stat().st_mtime < cutoff:
                shutil.rmtree(d, ignore_errors=True)
                removed += 1
        except OSError:
            continue
    if removed:
        log("pipeline.sweep_stale", removed=removed)
    return removed
