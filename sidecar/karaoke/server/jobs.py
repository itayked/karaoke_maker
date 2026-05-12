"""Job runner: small FIFO queue (max 4 active) + warm-on-startup aligner.

- Pipeline execution runs in `ThreadPoolExecutor(max_workers=1)` so jobs run
  one at a time in submission order. Up to 3 jobs may be pending behind the
  running one; a 4th POST returns 409.
- The Whisper aligner is loaded once in a background thread kicked off from
  the FastAPI lifespan (`start_warming`). A job that runs before warming
  completes sits in `loading_model` until the aligner is ready, then transitions
  through `downloading` → `separating` → `aligning` → `complete`.
- Job state is in-memory. Successful jobs are GC'd 5 minutes after completion;
  failed jobs after 1 hour, so the user has time to inspect the error.
"""

from __future__ import annotations

import asyncio
import shutil
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from karaoke.log import log
from karaoke.models import JobStatus
from karaoke.pipeline.align import Aligner
from karaoke.pipeline.orchestrator import PipelineOutputs, run_pipeline
from karaoke.pipeline.postprocess import PipelineError
from karaoke.server.library import LibraryStore
from karaoke.server.schemas import JobState


_FAILED_JOB_TTL_S = 60 * 60        # 1 hour
_COMPLETE_JOB_TTL_S = 5 * 60       # 5 minutes
_GC_INTERVAL_S = 60                # sweep every minute

# Including the running one. Brief: "up to 3 pending" + the one in flight = 4.
MAX_ACTIVE_JOBS = 4

_NON_TERMINAL_STATUSES = frozenset(
    {
        JobStatus.QUEUED,
        JobStatus.LOADING_MODEL,
        JobStatus.DOWNLOADING,
        JobStatus.SEPARATING,
        JobStatus.ALIGNING,
    }
)


PipelineFn = Callable[..., PipelineOutputs]


def _default_pipeline_fn(
    youtube_url: str,
    lyrics: str,
    work_dir: Path,
    *,
    aligner: Aligner,
    device: str,
    on_stage: Callable[[str, float], None],
) -> PipelineOutputs:
    return run_pipeline(
        youtube_url,
        lyrics,
        work_dir,
        aligner=aligner,
        device=device,
        on_stage=on_stage,
    )


class JobBusy(Exception):
    """Raised by `JobManager.submit` when the active-jobs queue is full."""


class JobManager:
    def __init__(
        self,
        library: LibraryStore,
        *,
        aligner_factory: Callable[[], Aligner],
        model_name: str,
        device: str = "cuda",
        pipeline_fn: PipelineFn | None = None,
    ):
        self._library = library
        self._aligner_factory = aligner_factory
        self._model_name = model_name
        self._device = device
        self._pipeline_fn: PipelineFn = pipeline_fn or _default_pipeline_fn

        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="karaoke-job")
        self._jobs: dict[str, JobState] = {}
        self._lock = threading.Lock()

        # Aligner warming. Cross-thread signal: threading.Event (set from warm thread,
        # waited from worker thread). Also exposed for /health via `aligner_ready`.
        self._aligner: Aligner | None = None
        self._aligner_lock = threading.Lock()
        self._aligner_ready = threading.Event()
        self._aligner_load_error: str | None = None
        self._warming_started = False

        # asyncio.Event-per-job for SSE wakeups (lives on the loop bound at startup).
        self._events: dict[str, asyncio.Event] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

    # ---------- lifecycle ----------

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def start_warming(self) -> None:
        """Kick off aligner load in a background daemon thread. Idempotent."""
        with self._aligner_lock:
            if self._warming_started:
                return
            self._warming_started = True

        def _warm() -> None:
            log("aligner.warm_start", model=self._model_name, device=self._device)
            try:
                aligner = self._aligner_factory()
                with self._aligner_lock:
                    self._aligner = aligner
                log("aligner.warm_done", model=self._model_name)
            except Exception as e:
                self._aligner_load_error = f"{type(e).__name__}: {e}"
                log(
                    "aligner.warm_failed",
                    level="ERROR",
                    error=self._aligner_load_error,
                    trace=traceback.format_exc(),
                )
            finally:
                self._aligner_ready.set()

        threading.Thread(target=_warm, daemon=True, name="karaoke-aligner-warm").start()

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    @property
    def aligner_ready(self) -> bool:
        return self._aligner is not None

    # ---------- inspection ----------

    def get(self, job_id: str) -> JobState | None:
        with self._lock:
            state = self._jobs.get(job_id)
        return state.model_copy() if state else None

    def event_for(self, job_id: str) -> asyncio.Event | None:
        return self._events.get(job_id)

    def _active_count(self) -> int:
        # Caller holds _lock.
        return sum(1 for s in self._jobs.values() if s.status in _NON_TERMINAL_STATUSES)

    # ---------- create ----------

    def submit(self, youtube_url: str, lyrics: str) -> JobState:
        with self._lock:
            if self._active_count() >= MAX_ACTIVE_JOBS:
                raise JobBusy(
                    f"queue is full ({MAX_ACTIVE_JOBS} active jobs); wait for one to finish"
                )
            now = time.time()
            job_id = uuid.uuid4().hex[:12]
            state = JobState(
                id=job_id,
                status=JobStatus.QUEUED,
                stage="queued",
                progress=0.0,
                youtube_url=youtube_url,
                created_at=now,
                updated_at=now,
            )
            self._jobs[job_id] = state

        self._events[job_id] = asyncio.Event()
        self._executor.submit(self._run, job_id, youtube_url, lyrics)
        return state.model_copy()

    # ---------- worker thread ----------

    def _run(self, job_id: str, youtube_url: str, lyrics: str) -> None:
        log("job.start", job_id=job_id, url=youtube_url)
        # Orchestrator owns its scratch dir lifetime; we hand it the song's
        # library folder so the instrumental lands there directly.
        dest_dir = self._library.song_dir(job_id)

        try:
            aligner = self._wait_for_aligner(job_id)

            def on_stage(stage: str, progress: float) -> None:
                self._update(
                    job_id,
                    status=_STATUS_FOR_STAGE.get(stage, JobStatus.ALIGNING),
                    stage=stage,
                    progress=progress,
                )

            outputs = self._pipeline_fn(
                youtube_url,
                lyrics,
                dest_dir,
                aligner=aligner,
                device=self._device,
                on_stage=on_stage,
            )

            entry = self._library.save(
                job_id,
                outputs.result,
                youtube_url=youtube_url,
                metadata=outputs.metadata,
                model_used=self._model_name,
            )
            self._update(
                job_id,
                status=JobStatus.COMPLETE,
                stage="complete",
                progress=1.0,
                title=entry.title,
            )
            log("job.complete", job_id=job_id, duration=outputs.result.duration_seconds)
        except PipelineError as e:
            log("job.failed", level="ERROR", job_id=job_id, error=str(e))
            self._update(
                job_id,
                status=JobStatus.FAILED,
                stage="failed",
                error=str(e),
                failed_stage=self._current_stage(job_id),
            )
            # The orchestrator already cleaned its scratch dir; the song folder
            # in the library is empty, so drop it too.
            shutil.rmtree(dest_dir, ignore_errors=True)
        except Exception as e:
            log(
                "job.failed",
                level="ERROR",
                job_id=job_id,
                error=str(e),
                trace=traceback.format_exc(),
            )
            self._update(
                job_id,
                status=JobStatus.FAILED,
                stage="failed",
                error=f"{type(e).__name__}: {e}",
                failed_stage=self._current_stage(job_id),
            )
            shutil.rmtree(dest_dir, ignore_errors=True)

    def _wait_for_aligner(self, job_id: str) -> Aligner:
        """Block until the aligner is loaded, transitioning state if needed."""
        if not self._warming_started:
            # Caller forgot to warm; warm on demand so we don't deadlock.
            self.start_warming()

        if not self._aligner_ready.is_set():
            self._update(job_id, status=JobStatus.LOADING_MODEL, stage="loading_model")
            self._aligner_ready.wait()

        with self._aligner_lock:
            aligner = self._aligner
        if aligner is None:
            raise PipelineError(
                f"aligner failed to load: {self._aligner_load_error or 'unknown error'}"
            )
        return aligner

    def _current_stage(self, job_id: str) -> str | None:
        with self._lock:
            state = self._jobs.get(job_id)
            return state.stage if state else None

    # ---------- state mutation + SSE wake ----------

    def _update(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            state = self._jobs.get(job_id)
            if state is None:
                return
            updated = state.model_copy(update={**fields, "updated_at": time.time()})
            self._jobs[job_id] = updated
        loop = self._loop
        ev = self._events.get(job_id)
        if loop is not None and ev is not None:
            try:
                loop.call_soon_threadsafe(ev.set)
            except RuntimeError:
                # Loop closed during shutdown; no listener anyway.
                pass

    # ---------- GC ----------

    async def gc_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(_GC_INTERVAL_S)
                self._gc()
            except asyncio.CancelledError:
                return
            except Exception as e:
                log("job.gc_error", level="ERROR", error=str(e))

    def _gc(self) -> None:
        now = time.time()
        complete_cutoff = now - _COMPLETE_JOB_TTL_S
        failed_cutoff = now - _FAILED_JOB_TTL_S
        to_drop: list[str] = []
        with self._lock:
            for jid, state in self._jobs.items():
                if state.status == JobStatus.COMPLETE and state.updated_at < complete_cutoff:
                    to_drop.append(jid)
                elif state.status == JobStatus.FAILED and state.updated_at < failed_cutoff:
                    to_drop.append(jid)
            for jid in to_drop:
                self._jobs.pop(jid, None)
                self._events.pop(jid, None)
        if to_drop:
            log("job.gc_swept", count=len(to_drop))


_STATUS_FOR_STAGE: dict[str, JobStatus] = {
    "downloading": JobStatus.DOWNLOADING,
    "separating": JobStatus.SEPARATING,
    "aligning": JobStatus.ALIGNING,
    "finalizing": JobStatus.ALIGNING,
}
