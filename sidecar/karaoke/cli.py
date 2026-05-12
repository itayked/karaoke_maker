"""Standalone CLI — full pipeline end-to-end, no HTTP layer.

Usage:
    python -m karaoke.cli <youtube_url> <lyrics_file> [--out OUT_DIR]

Outputs:
    <OUT_DIR>/<job_id>/instrumental.wav
    <OUT_DIR>/<job_id>/lyrics.json
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
import uuid
from pathlib import Path

from karaoke.log import log
from karaoke.pipeline.align import StableWhisperAligner
from karaoke.pipeline.orchestrator import run_pipeline
from karaoke.pipeline.postprocess import PipelineError


def run(youtube_url: str, lyrics_file: Path, out_dir: Path) -> Path:
    lyrics = lyrics_file.read_text(encoding="utf-8-sig")
    if not lyrics.strip():
        raise PipelineError(f"lyrics file is empty: {lyrics_file}")
    job_id = uuid.uuid4().hex[:12]
    song_dir = out_dir / job_id
    song_dir.mkdir(parents=True, exist_ok=True)

    log("cli.start", job_id=job_id, url=youtube_url, out=str(song_dir))

    device = os.environ.get("DEVICE", "cuda")
    aligner = StableWhisperAligner(
        model_size=os.environ.get("WHISPER_MODEL_SIZE", "large-v3"),
        device=device,
    )

    try:
        outputs = run_pipeline(youtube_url, lyrics, song_dir, aligner=aligner, device=device)
        outputs.result.audio_url = str(outputs.instrumental_path.as_posix())
        (song_dir / "lyrics.json").write_text(
            outputs.result.model_dump_json(indent=2), encoding="utf-8"
        )
        log(
            "cli.done",
            job_id=job_id,
            audio=str(outputs.instrumental_path),
            lines=len(outputs.result.lines),
            duration=round(outputs.result.duration_seconds, 2),
        )
        return song_dir
    except Exception as e:
        log("cli.failed", level="ERROR", job_id=job_id, error=str(e), trace=traceback.format_exc())
        raise


def main() -> None:
    parser = argparse.ArgumentParser(prog="karaoke.cli")
    parser.add_argument("youtube_url")
    parser.add_argument("lyrics_file", type=Path)
    parser.add_argument("--out", type=Path, default=Path("./out"))
    args = parser.parse_args()

    if not args.lyrics_file.exists():
        print(f"lyrics file not found: {args.lyrics_file}", file=sys.stderr)
        sys.exit(2)

    run(args.youtube_url, args.lyrics_file, args.out)


if __name__ == "__main__":
    main()
