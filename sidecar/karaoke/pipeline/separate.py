"""Demucs htdemucs two-stems separation via subprocess. Step 2."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from karaoke.log import log


class SeparationError(RuntimeError):
    pass


def separate(
    audio_path: Path,
    out_dir: Path,
    *,
    timeout_s: int = 600,
    device: str = "cuda",
) -> tuple[Path, Path]:
    """Run Demucs htdemucs --two-stems=vocals. Returns (vocals_path, no_vocals_path).

    Demucs writes to `{out_dir}/htdemucs/{stem_name}/{vocals,no_vocals}.wav`,
    where `{stem_name}` is the input filename without extension.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = audio_path.stem

    if not shutil.which("demucs"):
        raise SeparationError("`demucs` CLI not on PATH — is the venv active?")

    cmd = [
        "demucs",
        "-n",
        "htdemucs",
        "--two-stems=vocals",
        "-d",
        device,
        "-o",
        str(out_dir),
        str(audio_path),
    ]

    log("separate.start", cmd=" ".join(cmd))
    try:
        proc = subprocess.run(
            cmd,
            timeout=timeout_s,
            capture_output=True,
            text=True,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise SeparationError(f"demucs timed out after {timeout_s}s") from e

    if proc.returncode != 0:
        stderr_tail = "\n".join((proc.stderr or "").splitlines()[-50:])
        stdout_tail = "\n".join((proc.stdout or "").splitlines()[-50:])
        log(
            "separate.failed",
            level="ERROR",
            returncode=proc.returncode,
            stderr_tail=stderr_tail,
            stdout_tail=stdout_tail,
        )
        raise SeparationError(
            f"demucs exited {proc.returncode}\n"
            f"--- stderr (last 50 lines) ---\n{stderr_tail or '<empty>'}\n"
            f"--- stdout (last 50 lines) ---\n{stdout_tail or '<empty>'}"
        )

    vocals = out_dir / "htdemucs" / stem / "vocals.wav"
    no_vocals = out_dir / "htdemucs" / stem / "no_vocals.wav"
    if not vocals.exists() or not no_vocals.exists():
        raise SeparationError(
            f"expected stems under {out_dir / 'htdemucs' / stem}; got: "
            f"{list((out_dir / 'htdemucs' / stem).glob('*')) if (out_dir / 'htdemucs' / stem).exists() else 'no dir'}"
        )

    log("separate.done", vocals=str(vocals), no_vocals=str(no_vocals))
    return vocals, no_vocals
