"""Request / response schemas for the sidecar HTTP API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from karaoke.models import JobStatus, PipelineResult


class CreateJobRequest(BaseModel):
    youtube_url: str = Field(..., min_length=1)
    lyrics: str = Field(..., min_length=1)


class CreateJobResponse(BaseModel):
    job_id: str


class JobState(BaseModel):
    id: str
    status: JobStatus
    stage: str
    progress: float = 0.0
    youtube_url: str
    title: str | None = None
    error: str | None = None
    failed_stage: str | None = None
    created_at: float
    updated_at: float


class LibraryEntry(BaseModel):
    song_id: str
    title: str | None = None
    youtube_url: str
    thumbnail_url: str | None = None
    duration_seconds: float
    model_used: str | None = None
    created_at: float


class LibrarySongDetail(BaseModel):
    song_id: str
    audio_url: str
    lyrics_data: PipelineResult
    meta: LibraryEntry


class HealthResponse(BaseModel):
    status: Literal["ok"]
    cuda_available: bool
    gpu_name: str | None = None
    vram_mb: int | None = None
    selected_model: str
    aligner_ready: bool


class ModelStatus(BaseModel):
    """One entry per Whisper model size."""

    size: str
    state: Literal["installed", "missing", "downloading"]
    progress: float = 0.0
    bytes_downloaded: int = 0
    bytes_total: int | None = None


class ModelsStatusResponse(BaseModel):
    models: list[ModelStatus]
    selected: str


class ModelDownloadRequest(BaseModel):
    size: str = Field(..., description="small | medium | large-v3")
