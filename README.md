# Hebrew Karaoke (Desktop App)

YouTube URL + Hebrew lyrics → instrumental audio + word-level timed karaoke. Packaged as a standalone Windows desktop app (Electron + Python sidecar).

## Repo layout

```
sidecar/                 Python package + local HTTP server (step 2+)
└── karaoke/             pipeline: download → separate → align → postprocess
    └── pipeline/
app/                     Electron app — main + renderer (step 4+)
installer/               electron-builder + NSIS config (step 7)
preview.html             standalone QA viewer for CLI output
Karaoke_old.py           original Colab prototype (reference only)
```

## Architecture

Karaoke.exe is a single Electron process that spawns a Python sidecar at launch:

- The sidecar (FastAPI on `127.0.0.1`, ephemeral port chosen at startup, port printed to stdout) exposes the pipeline as a local HTTP API.
- The Electron main process owns the sidecar lifecycle (spawn / health-check every 5s / kill on exit).
- The renderer is plain HTML/CSS/JS and talks to the sidecar over `fetch`. Progress streams via Server-Sent Events.
- Processed songs live on disk under `%APPDATA%\Karaoke\library\<song_id>\` — playable later without reprocessing. Models live under `%APPDATA%\Karaoke\models\`.

## Defaults chosen for the open questions

| Decision | Choice | Rationale |
|---|---|---|
| Sidecar framework | **FastAPI** | Async, automatic schema, easy to `curl` during dev. |
| Progress transport | **HTTP + SSE** (`GET /jobs/{id}/events`) | Simpler than WebSocket; one-way is all we need. |
| Renderer framework | **Vanilla JS** | Itay does the design; no build step. |
| Model download source | **Hugging Face Hub** | Stable URLs, resumable downloads, well-cached. |
| Embedded Python for installer | **python-build-standalone** (Astral) | Real `site-packages`, works with `venv`, no embeddable-zip quirks. |
| Python version | 3.11 | PyTorch + stable-ts wheel coverage. |
| Package manager (dev) | uv | pip works fine too. |
| Redis client (worker) | **`redis-py`** sync | Worker is single-process; no async benefit. |
| Long-audio chunking | only if duration > 10 min, 8-min windows + 5s overlap | Hebrew songs are almost always < 6 min. |

## Implementation status

Following the pivot brief.

- [x] **Step 1** — repo reorganization: `worker/` → `sidecar/`, package renamed `karaoke`, web/Redis/R2/upload code removed. Bugs 1/2/3 from prior verification already fixed in the surviving code.
- [x] **Step 2** — FastAPI sidecar wrapping the pipeline (single in-process worker, in-memory job state, library persistence to `%APPDATA%`). **STOP** for manual `curl` verification.
- [x] **Step 3** — Electron skeleton: spawn sidecar, single-screen UI (URL+lyrics → progress → karaoke player). **STOP** for end-to-end UX check.
- [x] **Step 4** — Concrete design integration (Rubik typography, palette switcher, clip-path word fill).
- [x] **Step 5** — first-launch GPU detection + large-v3 download flow (folded into the same shell).
- [x] **Step 6** — electron-builder packaging, NSIS installer, bundled Python + models + FFmpeg, INSTALL.md.
- [ ] Future — logs viewer, settings page, library delete, code signing.

## Setup (dev)

### Prerequisites
- Python 3.11
- Node 20+ (for the Electron app, step 3+)
- ffmpeg on PATH
- NVIDIA GPU with CUDA 12.x driver (CPU works but ~10x slower)

### Sidecar (Python)

```bash
cd sidecar
uv venv && source .venv/Scripts/activate     # Windows
# or: python -m venv .venv

# Install CUDA torch FIRST (the requirements.txt entry is a generic fallback).
uv pip install torch==2.4.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

uv pip install -r requirements.txt

# Standalone CLI — full pipeline end-to-end, no HTTP layer:
python -m karaoke.cli "<youtube_url>" path/to/lyrics.txt --out ./out

# HTTP sidecar (step 2):
python -m karaoke.server
# Prints `KARAOKE_SIDECAR_PORT=<n>` to stdout, then listens on 127.0.0.1:<n>.

# Tests:
python -m pytest tests/ -q
```

### Sidecar endpoints (step 2)

| Verb   | Path                       | Notes |
|---|---|---|
| GET    | `/health`                  | `{status, cuda_available, gpu_name, vram_mb, selected_model, aligner_ready}` |
| POST   | `/jobs`                    | body `{youtube_url, lyrics}` → `{job_id}`; 409 if a job is already running |
| GET    | `/jobs/{id}`               | current `JobState` |
| GET    | `/jobs/{id}/events`        | SSE stream of state changes; closes on terminal status |
| GET    | `/library`                 | list of `LibraryEntry` (newest first) |
| GET    | `/library/{song_id}`       | `{song_id, audio_url, lyrics_data, meta}` |
| GET    | `/library/{song_id}/audio` | streams the instrumental WAV |
| DELETE | `/library/{song_id}`       | removes from disk |
| GET    | `/models/status`           | filesystem scan of `%APPDATA%\Karaoke\models\` |
| POST   | `/models/download`         | **stub — 501**; real download lands in step 6 |

Library lives under `%APPDATA%\Karaoke\library\<song_id>\` (Windows) or `~/.local/share/Karaoke/library/<song_id>/` elsewhere.

Verify CUDA is visible:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

### Electron app

```bash
cd app
npm install
npm start          # or: npm run dev
```

### Building the Windows installer

```bash
cd app
npm install
npm run prepare-build    # downloads python-build-standalone + torch CUDA + models + ffmpeg
npm run build            # invokes electron-builder, produces dist/KaraokeSetup-x.y.z.exe
# or both in one shot:
npm run dist
```

What `prepare-build` does (idempotent — re-runs skip work already done; safe to resume after a failed download):

1. Downloads python-build-standalone 3.11 (Windows x64 install-only) and extracts to `build-staging/python/`. Falls back to GitHub releases API for the latest 3.11 install_only asset if the pinned URL has been retired.
2. Bootstraps pip in that interpreter.
3. Installs `torch==2.4.1 + torchaudio==2.4.1` from `download.pytorch.org/whl/cu121`.
4. Installs the rest of `sidecar/requirements.txt` (torch/torchaudio/pytest/ruff stripped — `torch` because we already installed the CUDA build, dev tools because they don't ship).
5. Copies `sidecar/karaoke/` into `build-staging/sidecar/` (no `.venv`, no `tests/`).
6. Downloads the Whisper `small` `.pt` file (only model bundled — medium and large-v3 download on first launch when VRAM warrants).
7. Pre-warms Demucs htdemucs + Silero VAD into `build-staging/models/`.
8. Downloads FFmpeg "release essentials" zip from gyan.dev, extracts `ffmpeg.exe` into `build-staging/bin/`.
9. **Slims** the staging: removes `torch/lib/*.lib` static archives, `torch/include` headers, `tcl/`+`tkinter/` (no Python GUI), `Lib/ensurepip`+`Lib/idlelib`+`Lib/test`, `.pdb` debug symbols, and all `__pycache__/` directories. Roughly 1GB saved.

`electron-builder` then picks the staged tree up via `extraResources` and produces an **NSIS web installer**:
- `app/dist/KaraokeSetup-x.y.z.exe` — small stub (~5–10MB) that goes to GitHub Releases
- `app/dist/Karaoke-x.y.z-x64.nsis.7z` — the actual payload (~3–4GB) that hosts on Google Drive or similar

The stub installer fetches the payload at install time from the URL configured in `package.json → build.nsisWeb.appPackageUrl`. After every rebuild **update that URL** to point at wherever the new payload is hosted before publishing.

**Expected**: build time ~15–25 min on first run. Stub <10MB, payload ~3.5GB.

**Prerequisites**: Node 20+, `tar` on PATH (Windows 10+ ships it as a built-in), an internet connection. **No system Python required** — `prepare-build` brings its own.

End-user install instructions for non-technical friends live in `INSTALL.md` (Hebrew).

On launch the main process spawns the sidecar from `../sidecar/.venv/Scripts/python.exe -m karaoke.server`, reads `KARAOKE_SIDECAR_PORT=<n>` off its stdout, and exposes the port to the renderer over IPC. Renderer hits the sidecar over HTTP (`fetch` + `EventSource`); CORS is wide-open since the sidecar is bound to 127.0.0.1.

Override the Python executable for ad-hoc dev:
```bash
KARAOKE_PYTHON=/some/other/python npm start
```

**Flow** (step 3 — single-screen on purpose; library/settings come later):
1. Form view: YouTube URL + Hebrew lyrics + Create button. Button disabled until the sidecar port is received.
2. Progress view: spinner + stage label. Subscribes to `GET /jobs/{id}/events` (SSE). On `failed`, error is shown with a "try again" button.
3. Player view: `<audio>` instrumental + RTL lyrics, current line scaled up, current word highlighted. A "back" button in the corner returns to the form.

The shell intentionally has no styling beyond a flat dark layout. The visual design lands separately.

## Reference

`Karaoke_old.py` — original single-file Colab prototype, kept for reference.
