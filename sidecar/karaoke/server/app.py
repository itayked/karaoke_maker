"""FastAPI app factory."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from karaoke.config import settings
from karaoke.paths import (
    bundled_models_dir,
    ensure_ffmpeg_on_path,
    find_whisper_model,
)
from karaoke.pipeline.align import Aligner, StableWhisperAligner
from karaoke.pipeline.orchestrator import sweep_stale_work_dirs
from karaoke.server.downloads import ModelDownloadManager
from karaoke.server.health import probe_gpu, select_model_size
from karaoke.server.jobs import JobBusy, JobManager, PipelineFn
from karaoke.server.library import LibraryStore
from karaoke.server.schemas import (
    CreateJobRequest,
    CreateJobResponse,
    HealthResponse,
    JobState,
    LibraryEntry,
    LibrarySongDetail,
    ModelDownloadRequest,
    ModelsStatusResponse,
    ModelStatus,
)


_WHISPER_MODEL_FILES = {
    "small": "whisper-small.pt",
    "medium": "whisper-medium.pt",
    "large-v3": "whisper-large-v3.pt",
}


def create_app(
    *,
    library_dir: Path | None = None,
    models_dir: Path | None = None,
    aligner_factory: Callable[[], Aligner] | None = None,
    pipeline_fn: PipelineFn | None = None,
    selected_model: str | None = None,
    device: str | None = None,
) -> FastAPI:
    """Build a FastAPI app. Defaults route through `karaoke.config.settings`;
    tests inject everything explicitly with mock paths and a fake pipeline_fn.
    """

    library_dir = library_dir or settings.library_dir
    models_dir = models_dir or settings.models_dir
    library_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Make the bundled FFmpeg available to yt-dlp / Demucs subprocesses.
    ensure_ffmpeg_on_path()

    gpu = probe_gpu()
    selected = selected_model or select_model_size(gpu)
    device = device or ("cuda" if gpu.cuda_available else "cpu")

    library = LibraryStore(library_dir)
    downloads = ModelDownloadManager(models_dir)

    def _default_aligner_factory() -> Aligner:
        return StableWhisperAligner(model_size=selected, device=device)

    jobs = JobManager(
        library,
        aligner_factory=aligner_factory or _default_aligner_factory,
        model_name=selected,
        device=device,
        pipeline_fn=pipeline_fn,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        loop = asyncio.get_running_loop()
        jobs.bind_loop(loop)
        # Defense in depth: sweep any leaked karaoke_* temp dirs from prior runs.
        sweep_stale_work_dirs()
        # Warm the aligner in a background thread so the FastAPI app is reachable
        # while Whisper loads. /health.aligner_ready flips to true when done.
        jobs.start_warming()
        gc_task = loop.create_task(jobs.gc_loop())
        try:
            yield
        finally:
            gc_task.cancel()
            jobs.shutdown()

    app = FastAPI(title="Karaoke sidecar", version="0.1.0", lifespan=lifespan)
    # Sidecar binds only to 127.0.0.1; Electron renderer loads from file:// origin
    # and needs cross-origin fetch + EventSource. Wide-open CORS is fine.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.jobs = jobs
    app.state.library = library
    app.state.models_dir = models_dir
    app.state.selected_model = selected
    app.state.gpu = gpu

    # ---------------- health ----------------

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            cuda_available=gpu.cuda_available,
            gpu_name=gpu.name,
            vram_mb=gpu.vram_mb,
            selected_model=selected,
            aligner_ready=jobs.aligner_ready,
        )

    # ---------------- jobs ----------------

    @app.post("/jobs", response_model=CreateJobResponse, status_code=201)
    def create_job(req: CreateJobRequest) -> CreateJobResponse:
        try:
            state = jobs.submit(req.youtube_url, req.lyrics)
        except JobBusy as e:
            raise HTTPException(status_code=409, detail=str(e)) from e
        return CreateJobResponse(job_id=state.id)

    @app.get("/jobs/{job_id}", response_model=JobState)
    def get_job(job_id: str) -> JobState:
        state = jobs.get(job_id)
        if state is None:
            raise HTTPException(status_code=404, detail="job not found")
        return state

    @app.get("/jobs/{job_id}/events")
    async def job_events(job_id: str, request: Request) -> StreamingResponse:
        state = jobs.get(job_id)
        if state is None:
            raise HTTPException(status_code=404, detail="job not found")

        async def event_stream() -> AsyncIterator[bytes]:
            last_serialized: str | None = None
            terminal = {"complete", "failed"}
            while True:
                if await request.is_disconnected():
                    return
                current = jobs.get(job_id)
                if current is None:
                    return
                payload = current.model_dump_json()
                if payload != last_serialized:
                    yield f"data: {payload}\n\n".encode("utf-8")
                    last_serialized = payload
                if current.status.value in terminal:
                    return
                ev = jobs.event_for(job_id)
                if ev is not None:
                    try:
                        await asyncio.wait_for(ev.wait(), timeout=1.0)
                    except asyncio.TimeoutError:
                        pass
                    ev.clear()
                else:
                    await asyncio.sleep(0.25)

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # ---------------- library ----------------

    @app.get("/library", response_model=list[LibraryEntry])
    def list_library() -> list[LibraryEntry]:
        return library.list()

    @app.get("/library/{song_id}", response_model=LibrarySongDetail)
    def get_song(song_id: str) -> LibrarySongDetail:
        meta = library.get_meta(song_id)
        lyrics = library.get_lyrics(song_id)
        if meta is None or lyrics is None:
            raise HTTPException(status_code=404, detail="song not found")
        return LibrarySongDetail(
            song_id=song_id,
            audio_url=f"/library/{song_id}/audio",
            lyrics_data=lyrics,
            meta=meta,
        )

    @app.get("/library/{song_id}/audio")
    def get_audio(song_id: str) -> FileResponse:
        p = library.get_audio_path(song_id)
        if p is None:
            raise HTTPException(status_code=404, detail="audio not found")
        return FileResponse(p, media_type="audio/wav", filename=f"{song_id}.wav")

    @app.delete("/library/{song_id}", status_code=204)
    def delete_song(song_id: str):
        ok = library.delete(song_id)
        if not ok:
            raise HTTPException(status_code=404, detail="song not found")
        return None

    # ---------------- models ----------------

    def _scan_models() -> list[ModelStatus]:
        out: list[ModelStatus] = []
        for size in _WHISPER_MODEL_FILES:
            # Active download takes priority over filesystem state.
            progress = downloads.get_progress(size)
            if progress is not None:
                out.append(progress)
                continue
            found = find_whisper_model(size, models_dir)
            out.append(
                ModelStatus(
                    size=size,
                    state="installed" if found is not None else "missing",
                )
            )
        return out

    @app.get("/models/status", response_model=ModelsStatusResponse)
    def models_status() -> ModelsStatusResponse:
        return ModelsStatusResponse(models=_scan_models(), selected=selected)

    @app.post("/models/download", status_code=202)
    def models_download(req: ModelDownloadRequest):
        outcome = downloads.start(req.size)
        if outcome == "unsupported":
            raise HTTPException(status_code=400, detail=f"unsupported size: {req.size}")
        if outcome == "busy":
            raise HTTPException(
                status_code=409, detail="a different model download is already running"
            )
        # 'started', 'already-running', 'already-installed' all return 202 with a hint.
        return {"size": req.size, "outcome": outcome}

    return app
