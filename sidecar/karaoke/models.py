"""Pydantic models for job state and lyrics output."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel


class JobStatus(str, Enum):
    QUEUED = "queued"
    LOADING_MODEL = "loading_model"
    DOWNLOADING = "downloading"
    SEPARATING = "separating"
    ALIGNING = "aligning"
    COMPLETE = "complete"
    FAILED = "failed"


class AlignedWord(BaseModel):
    """One word after alignment. Internal type — `confidence` is stripped before frontend output."""

    text: str
    start: float
    end: float
    confidence: float = 1.0
    interpolated: bool = False


class OutputWord(BaseModel):
    text: str
    start: float
    end: float


class OutputLine(BaseModel):
    is_empty: bool
    start: float | None = None
    end: float | None = None
    words: list[OutputWord] = []


class PipelineResult(BaseModel):
    audio_url: str
    duration_seconds: float
    lines: list[OutputLine]


class JobRecord(BaseModel):
    id: str
    status: JobStatus
    stage: str | None = None
    progress: float = 0.0
    youtube_url: str
    lyrics: str
    audio_url: str | None = None
    lyrics_data: PipelineResult | None = None
    error: str | None = None
    failed_stage: Literal[
        "download", "separate", "align", "postprocess", None
    ] = None
    created_at: float
    updated_at: float
