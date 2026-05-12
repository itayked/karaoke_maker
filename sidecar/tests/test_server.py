"""HTTP API tests for the sidecar.

The pipeline is mocked: we inject a `pipeline_fn` so we cover routing,
state transitions, queue behaviour, library persistence, and error paths
without touching torch / Demucs / yt-dlp.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Callable

import pytest
from fastapi.testclient import TestClient

from karaoke.models import PipelineResult
from karaoke.pipeline.align import Aligner
from karaoke.pipeline.orchestrator import PipelineOutputs
from karaoke.pipeline.postprocess import PipelineError
from karaoke.server import jobs as jobs_module
from karaoke.server.app import create_app


# ---- shared fixtures ----------------------------------------------------------


class _FakeAligner:
    def align(self, *a, **kw):
        return []


def _instant_aligner() -> Aligner:
    return _FakeAligner()


def _ok_pipeline(
    youtube_url, lyrics, dest_dir: Path, *, aligner, device, on_stage
) -> PipelineOutputs:
    on_stage("downloading", 0.05)
    on_stage("separating", 0.3)
    on_stage("aligning", 0.6)

    dest_dir.mkdir(parents=True, exist_ok=True)
    instrumental = dest_dir / "instrumental.wav"
    instrumental.write_bytes(b"RIFF....WAVE")
    result = PipelineResult(
        audio_url="",
        duration_seconds=12.34,
        lines=[
            {
                "is_empty": False,
                "start": 0.0,
                "end": 1.0,
                "words": [{"text": "שלום", "start": 0.0, "end": 1.0}],
            }
        ],
    )
    return PipelineOutputs(
        instrumental_path=instrumental,
        result=result,
        metadata={"title": "Fake song", "thumbnail": "https://x/y.jpg", "video_id": "abc"},
    )


def _failing_pipeline(*a, **kw):
    raise PipelineError("simulated alignment failure")


def _make_client(
    tmp_path: Path,
    *,
    pipeline_fn: Callable = _ok_pipeline,
    aligner_factory: Callable[[], Aligner] = _instant_aligner,
) -> TestClient:
    app = create_app(
        library_dir=tmp_path / "library",
        models_dir=tmp_path / "models",
        aligner_factory=aligner_factory,
        pipeline_fn=pipeline_fn,
        selected_model="small",
        device="cpu",
    )
    return TestClient(app)


def _wait_until_complete(client: TestClient, job_id: str, *, timeout: float = 5.0) -> dict:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        r = client.get(f"/jobs/{job_id}")
        assert r.status_code == 200
        last = r.json()
        if last["status"] in ("complete", "failed"):
            return last
        time.sleep(0.02)
    pytest.fail(f"job {job_id} did not finish within {timeout}s; last={last}")


def _wait_for_status(
    client: TestClient, job_id: str, status: str, *, timeout: float = 5.0
) -> dict:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = client.get(f"/jobs/{job_id}").json()
        if last["status"] == status:
            return last
        time.sleep(0.02)
    pytest.fail(f"job {job_id} never reached status={status}; last={last}")


# ---- health -------------------------------------------------------------------


def test_health_ok(tmp_path):
    with _make_client(tmp_path) as c:
        body = c.get("/health").json()
        assert body["status"] == "ok"
        assert body["selected_model"] == "small"
        # Warming started in the lifespan; the fake aligner loads instantly,
        # but we don't synchronize — assert it eventually becomes ready.
        deadline = time.time() + 2.0
        while time.time() < deadline:
            if c.get("/health").json()["aligner_ready"]:
                return
            time.sleep(0.02)
        pytest.fail("aligner never became ready")


# ---- create job ---------------------------------------------------------------


def test_create_job_requires_fields(tmp_path):
    with _make_client(tmp_path) as c:
        assert c.post("/jobs", json={"youtube_url": "", "lyrics": ""}).status_code == 422


def test_create_job_runs_to_completion_and_persists(tmp_path):
    with _make_client(tmp_path) as c:
        r = c.post("/jobs", json={"youtube_url": "https://yt/x", "lyrics": "שלום"})
        assert r.status_code == 201
        job_id = r.json()["job_id"]
        assert len(job_id) == 12

        body = _wait_until_complete(c, job_id)
        assert body["status"] == "complete"
        assert body["progress"] == 1.0
        assert body["title"] == "Fake song"

        lib = c.get("/library").json()
        assert any(e["song_id"] == job_id for e in lib)

        detail = c.get(f"/library/{job_id}").json()
        assert detail["lyrics_data"]["audio_url"] == f"/library/{job_id}/audio"
        assert detail["lyrics_data"]["duration_seconds"] == 12.34
        assert detail["meta"]["thumbnail_url"] == "https://x/y.jpg"


# ---- bug #1 regression: stage transitions are visible during execution --------


def test_stage_transitions_visible_during_execution(tmp_path):
    """Regression: GET /jobs/{id} must surface `downloading` while download runs,
    `separating` while Demucs runs, etc. Previous version stayed 'queued' the
    whole time on real audio runs."""
    gate_after_downloading = threading.Event()
    gate_after_separating = threading.Event()
    gate_after_aligning = threading.Event()

    def staged_pipeline(youtube_url, lyrics, dest_dir, *, aligner, device, on_stage):
        on_stage("downloading", 0.05)
        assert gate_after_downloading.wait(timeout=5.0)
        on_stage("separating", 0.3)
        assert gate_after_separating.wait(timeout=5.0)
        on_stage("aligning", 0.6)
        assert gate_after_aligning.wait(timeout=5.0)
        return _ok_pipeline(youtube_url, lyrics, dest_dir,
                            aligner=aligner, device=device, on_stage=on_stage)

    with _make_client(tmp_path, pipeline_fn=staged_pipeline) as c:
        r = c.post("/jobs", json={"youtube_url": "u", "lyrics": "ל"})
        job_id = r.json()["job_id"]

        s = _wait_for_status(c, job_id, "downloading")
        assert s["stage"] == "downloading"
        assert s["progress"] == pytest.approx(0.05)

        gate_after_downloading.set()
        s = _wait_for_status(c, job_id, "separating")
        assert s["progress"] == pytest.approx(0.3)

        gate_after_separating.set()
        s = _wait_for_status(c, job_id, "aligning")
        assert s["progress"] == pytest.approx(0.6)

        gate_after_aligning.set()
        _wait_until_complete(c, job_id)


# ---- bug #2: aligner warms at startup, jobs see `loading_model` if not ready --


def test_job_shows_loading_model_until_aligner_ready(tmp_path):
    """If a job hits the queue before warming completes, its state should be
    `loading_model`, not `queued`."""
    aligner_gate = threading.Event()

    def slow_factory() -> Aligner:
        # Block until the test releases — simulates the 2:14 model download.
        aligner_gate.wait(timeout=5.0)
        return _FakeAligner()

    with _make_client(tmp_path, aligner_factory=slow_factory) as c:
        r = c.post("/jobs", json={"youtube_url": "u", "lyrics": "ל"})
        job_id = r.json()["job_id"]

        # Worker thread should park in `loading_model` since the aligner blocks.
        _wait_for_status(c, job_id, "loading_model")
        # /health reflects aligner-not-ready while warming.
        assert c.get("/health").json()["aligner_ready"] is False

        aligner_gate.set()
        _wait_until_complete(c, job_id)
        assert c.get("/health").json()["aligner_ready"] is True


# ---- bug #3: small FIFO queue, 4th job rejected ------------------------------


def test_queue_admits_three_pending_rejects_fourth(tmp_path):
    pipeline_gate = threading.Event()

    def blocking_pipeline(youtube_url, lyrics, dest_dir, *, aligner, device, on_stage):
        on_stage("downloading", 0.05)
        assert pipeline_gate.wait(timeout=5.0)
        return _ok_pipeline(youtube_url, lyrics, dest_dir,
                            aligner=aligner, device=device, on_stage=on_stage)

    with _make_client(tmp_path, pipeline_fn=blocking_pipeline) as c:
        # Job 1 starts running (blocks at on_stage downloading).
        ids = []
        r = c.post("/jobs", json={"youtube_url": "u1", "lyrics": "ל"})
        assert r.status_code == 201
        ids.append(r.json()["job_id"])
        _wait_for_status(c, ids[0], "downloading")

        # Jobs 2, 3, 4 should queue (status=queued).
        for i in range(2, 5):
            r = c.post("/jobs", json={"youtube_url": f"u{i}", "lyrics": "ל"})
            assert r.status_code == 201, f"job {i}: {r.status_code} {r.text}"
            ids.append(r.json()["job_id"])

        # Verify the pending ones are still `queued`.
        for jid in ids[1:]:
            assert c.get(f"/jobs/{jid}").json()["status"] == "queued"

        # 5th must be rejected — 1 running + 3 pending = MAX_ACTIVE_JOBS.
        r5 = c.post("/jobs", json={"youtube_url": "u5", "lyrics": "ל"})
        assert r5.status_code == 409

        # Drain everything.
        pipeline_gate.set()
        for jid in ids:
            _wait_until_complete(c, jid, timeout=10.0)

        # Now a new submission goes through.
        rN = c.post("/jobs", json={"youtube_url": "uN", "lyrics": "ל"})
        assert rN.status_code == 201


# ---- bug #4: successful jobs retained 5 min, failed retained 1h ---------------


def test_complete_job_retained_then_gc(tmp_path, monkeypatch):
    # Shrink both TTLs so the test runs fast.
    monkeypatch.setattr(jobs_module, "_COMPLETE_JOB_TTL_S", 0.05)
    monkeypatch.setattr(jobs_module, "_FAILED_JOB_TTL_S", 0.05)

    with _make_client(tmp_path) as c:
        r = c.post("/jobs", json={"youtube_url": "u", "lyrics": "ל"})
        job_id = r.json()["job_id"]
        body = _wait_until_complete(c, job_id)
        assert body["status"] == "complete"

        # Immediately after completion the job state is still queryable.
        assert c.get(f"/jobs/{job_id}").status_code == 200

        # Force a GC sweep without waiting on the 60s loop.
        jobs = c.app.state.jobs
        time.sleep(0.1)
        jobs._gc()

        # Now the in-memory entry is gone.
        assert c.get(f"/jobs/{job_id}").status_code == 404

        # But the library entry remains — that's the durable copy.
        assert c.get(f"/library/{job_id}").status_code == 200


def test_failed_job_retained_after_complete_gc_window(tmp_path, monkeypatch):
    # Successful TTL very small; failed TTL still long. Verify a failed job
    # survives a sweep that would have evicted a successful one.
    monkeypatch.setattr(jobs_module, "_COMPLETE_JOB_TTL_S", 0.05)
    monkeypatch.setattr(jobs_module, "_FAILED_JOB_TTL_S", 60 * 60)

    with _make_client(tmp_path, pipeline_fn=_failing_pipeline) as c:
        r = c.post("/jobs", json={"youtube_url": "u", "lyrics": "ל"})
        job_id = r.json()["job_id"]
        _wait_until_complete(c, job_id)
        time.sleep(0.1)
        c.app.state.jobs._gc()
        body = c.get(f"/jobs/{job_id}").json()
        assert body["status"] == "failed"


# ---- bug #5: whisper cache fallback ------------------------------------------


def test_models_status_detects_whisper_cache(tmp_path, monkeypatch):
    fake_home = tmp_path / "home"
    cache = fake_home / ".cache" / "whisper"
    cache.mkdir(parents=True)
    (cache / "medium.pt").write_bytes(b"x")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))

    with _make_client(tmp_path) as c:
        body = c.get("/models/status").json()
        by_size = {m["size"]: m for m in body["models"]}
        assert by_size["medium"]["state"] == "installed"
        assert by_size["small"]["state"] == "missing"


# ---- legacy paths still work -------------------------------------------------


def test_failed_job_records_error(tmp_path):
    with _make_client(tmp_path, pipeline_fn=_failing_pipeline) as c:
        r = c.post("/jobs", json={"youtube_url": "u", "lyrics": "ל"})
        job_id = r.json()["job_id"]
        body = _wait_until_complete(c, job_id)
        assert body["status"] == "failed"
        assert "simulated alignment failure" in body["error"]


def test_job_404_for_unknown_id(tmp_path):
    with _make_client(tmp_path) as c:
        assert c.get("/jobs/missing-id-1").status_code == 404
        assert c.get("/jobs/missing-id-1/events").status_code == 404


def test_library_404s(tmp_path):
    with _make_client(tmp_path) as c:
        assert c.get("/library/none").status_code == 404
        assert c.get("/library/none/audio").status_code == 404
        assert c.delete("/library/none").status_code == 404


def test_library_audio_and_delete(tmp_path):
    with _make_client(tmp_path) as c:
        r = c.post("/jobs", json={"youtube_url": "u", "lyrics": "ל"})
        job_id = r.json()["job_id"]
        _wait_until_complete(c, job_id)

        audio = c.get(f"/library/{job_id}/audio")
        assert audio.status_code == 200
        assert audio.headers["content-type"].startswith("audio/wav")
        assert audio.content.startswith(b"RIFF")

        assert c.delete(f"/library/{job_id}").status_code == 204
        assert c.get(f"/library/{job_id}").status_code == 404


def test_sse_streams_terminal_state(tmp_path):
    with _make_client(tmp_path) as c:
        r = c.post("/jobs", json={"youtube_url": "u", "lyrics": "ל"})
        job_id = r.json()["job_id"]
        _wait_until_complete(c, job_id)

        with c.stream("GET", f"/jobs/{job_id}/events") as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")
            for line in resp.iter_lines():
                if not line:
                    continue
                assert line.startswith("data: ")
                payload = json.loads(line[len("data: "):])
                assert payload["id"] == job_id
                assert payload["status"] == "complete"
                break


def test_models_download_validates_size(tmp_path):
    with _make_client(tmp_path) as c:
        r = c.post("/models/download", json={"size": "not-a-size"})
        assert r.status_code == 400


def test_models_download_already_installed(tmp_path):
    # Pre-create the medium model file so the manager short-circuits.
    (tmp_path / "models").mkdir(parents=True, exist_ok=True)
    (tmp_path / "models" / "whisper-medium.pt").write_bytes(b"x")
    with _make_client(tmp_path) as c:
        r = c.post("/models/download", json={"size": "medium"})
        assert r.status_code == 202
        assert r.json()["outcome"] == "already-installed"
