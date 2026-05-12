"use strict";

// Karaoke renderer.
//   form    -> POST /jobs
//   progress (SSE)  -> on complete -> player
//                   -> on failed   -> error + retry
//   player  <- back button -> form
//
// Player visuals follow the "Concrete" design (Rubik 900 typographic poster,
// clip-path word fill, palette-driven). The player consumes the same
// PipelineResult schema the sidecar emits — no data reshaping.

// ─── palettes ────────────────────────────────────────────────────────────────

const PALETTES = [
  { key: "saffron",  name: "Saffron · Ink",   bg: "#EBBE3F", ink: "#14110D", warm: "rgba(255,239,180,0.5)",  cool: "rgba(56,38,8,0.10)"  },
  { key: "eggshell", name: "Eggshell · Ink",  bg: "#F2EAD8", ink: "#14110D", warm: "rgba(255,250,238,0.55)", cool: "rgba(56,38,8,0.06)" },
  { key: "ink",      name: "Ink · Saffron",   bg: "#14110D", ink: "#EBBE3F", warm: "rgba(235,190,63,0.05)",  cool: "rgba(0,0,0,0.35)"    },
  { key: "tomato",   name: "Tomato · Bone",   bg: "#E04A2E", ink: "#F4EAD5", warm: "rgba(255,210,180,0.18)", cool: "rgba(40,8,4,0.18)"   },
  { key: "sky",      name: "Sky · Marine",    bg: "#BCD7E6", ink: "#0C1A2A", warm: "rgba(255,255,255,0.45)", cool: "rgba(12,26,42,0.10)" },
  { key: "sage",     name: "Sage · Olive",    bg: "#C8CFA6", ink: "#1E2412", warm: "rgba(255,253,225,0.35)", cool: "rgba(30,36,18,0.10)" },
  { key: "plum",     name: "Pink · Plum",     bg: "#EE7AAC", ink: "#2A0A1F", warm: "rgba(255,210,225,0.30)", cool: "rgba(42,10,31,0.16)" },
  { key: "concrete", name: "Concrete · Chalk",bg: "#C7BFB1", ink: "#1A1714", warm: "rgba(245,240,230,0.35)", cool: "rgba(26,23,20,0.10)" },
];

function applyPalette(p) {
  const root = document.documentElement;
  root.style.setProperty("--cc-bg",   p.bg);
  root.style.setProperty("--cc-ink",  p.ink);
  root.style.setProperty("--cc-warm", p.warm);
  root.style.setProperty("--cc-cool", p.cool);
  document.getElementById("palette-btn-bg").style.background = p.bg;
  document.getElementById("palette-btn-ink").style.background = p.ink;
  for (const chip of document.querySelectorAll(".concrete__palette-chip")) {
    chip.setAttribute("data-on", chip.dataset.key === p.key ? "1" : "0");
  }
  state.palette = p;
  try { localStorage.setItem("karaoke.palette", p.key); } catch (_) {}
}

function buildPaletteRow() {
  const row = document.getElementById("palette-row");
  row.innerHTML = "";
  for (const p of PALETTES) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "concrete__palette-chip";
    btn.dataset.key = p.key;
    btn.title = `${p.bg} · ${p.ink}`;
    btn.setAttribute("role", "option");
    btn.setAttribute("aria-label", p.name);
    btn.innerHTML = `<span style="background:${p.bg}"></span><span style="background:${p.ink}"></span>`;
    btn.addEventListener("click", () => {
      applyPalette(p);
      document.getElementById("palette").classList.remove("open");
    });
    row.appendChild(btn);
  }
}

function loadInitialPalette() {
  let key = null;
  try { key = localStorage.getItem("karaoke.palette"); } catch (_) {}
  return PALETTES.find((p) => p.key === key) || PALETTES[0];
}

// ─── findActive — same logic as the design, adapted to our schema ───────────

function findActive(lines, t) {
  let result = { activeLine: -1, activeWord: -1, wordProgress: 0 };
  for (let i = 0; i < lines.length; i++) {
    const ln = lines[i];
    if (ln.is_empty) continue;
    const lineStart = ln.start, lineEnd = ln.end;
    if (t < lineStart) break;
    if (t <= lineEnd) {
      for (let j = 0; j < ln.words.length; j++) {
        const w = ln.words[j];
        if (t < w.start) return { activeLine: i, activeWord: j, wordProgress: 0 };
        if (t <= w.end) return { activeLine: i, activeWord: j, wordProgress: (t - w.start) / (w.end - w.start) };
      }
      return { activeLine: i, activeWord: ln.words.length, wordProgress: 1 };
    }
    result = { activeLine: i, activeWord: ln.words.length, wordProgress: 1 };
  }
  return result;
}

function fmtTime(sec) {
  const total = Math.max(0, Math.round(sec || 0));
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${s.toString().padStart(2, "0")}`;
}

// ─── state ──────────────────────────────────────────────────────────────────

const state = {
  sidecarPort: null,
  currentEventSource: null,
  palette: PALETTES[0],
  // Player
  data: null,           // PipelineResult: { audio_url, duration_seconds, lines }
  lineEls: [],          // DOM ref per line (including empty placeholders)
  activeLine: -1,
  activeWord: -1,
  activeBlockEl: null,  // the .cc-word__block inside the active word
  lastTimeShown: -1,
  audio: null,
  scrubDrag: null,
};

const STAGE_LABELS = {
  queued:         { big: "בתור",            sub: "ממתין" },
  loading_model:  { big: "טוען מודל",        sub: "פעם ראשונה — דקות ספורות" },
  downloading:    { big: "מוריד מ-YouTube", sub: "שלב 1 / 3" },
  separating:     { big: "מפריד ערוצים",    sub: "Demucs — שלב 2 / 3" },
  aligning:       { big: "מסנכרן מילים",    sub: "Whisper — שלב 3 / 3" },
  finalizing:     { big: "מסיים",           sub: "כמעט שם" },
  complete:       { big: "מוכן",            sub: "" },
  failed:         { big: "נכשל",            sub: "" },
};

// ─── view switching ─────────────────────────────────────────────────────────

function showView(name) {
  for (const v of ["welcome", "form", "progress", "player"]) {
    document.getElementById("view-" + v).classList.toggle("active", v === name);
  }
}

function resetProgressView() {
  document.getElementById("progress-error").classList.remove("active");
  document.getElementById("progress-error").textContent = "";
  document.getElementById("reset-btn").classList.remove("active");
  document.getElementById("stage-label").textContent = "מכין";
  document.getElementById("stage-sub").textContent = "שלב";
}

function showError(msg) {
  document.getElementById("stage-label").textContent = "נכשל";
  document.getElementById("stage-sub").textContent = "";
  const err = document.getElementById("progress-error");
  err.textContent = msg;
  err.classList.add("active");
  document.getElementById("reset-btn").classList.add("active");
}

function setStage(state_obj) {
  const label = STAGE_LABELS[state_obj.stage] || STAGE_LABELS[state_obj.status] || { big: state_obj.stage, sub: "" };
  document.getElementById("stage-label").textContent = label.big;
  document.getElementById("stage-sub").textContent = label.sub;
}

// ─── bootstrap ──────────────────────────────────────────────────────────────

async function bootstrap() {
  buildPaletteRow();
  applyPalette(loadInitialPalette());

  state.sidecarPort = await window.karaoke.getSidecarPort();
  document.getElementById("status-info").textContent = `sidecar 127.0.0.1:${state.sidecarPort}`;
  document.getElementById("submit-btn").disabled = false;

  window.karaoke.onSidecarReady((port) => {
    state.sidecarPort = port;
    document.getElementById("status-info").textContent = `sidecar 127.0.0.1:${port}`;
  });

  const firstLaunch = await window.karaoke.isFirstLaunch();
  if (firstLaunch) {
    await runFirstLaunchFlow();
  }
}

// ─── first-launch flow ──────────────────────────────────────────────────────

async function runFirstLaunchFlow() {
  showView("welcome");
  const status = document.getElementById("welcome-status");
  const detail = document.getElementById("welcome-detail");
  const bar = document.getElementById("welcome-progress");
  const fill = document.getElementById("welcome-progress-fill");
  const cont = document.getElementById("welcome-continue");

  status.textContent = "בודק חומרה...";
  detail.textContent = "";
  bar.classList.remove("active");
  cont.classList.remove("active");

  let health;
  try {
    const r = await fetch(`http://127.0.0.1:${state.sidecarPort}/health`);
    health = await r.json();
  } catch {
    health = { cuda_available: false, vram_mb: 0 };
  }

  const vram = health.vram_mb || 0;
  // Pick the model to download (if any) based on VRAM. small is bundled and
  // always available; medium and large-v3 are fetched on demand.
  let downloadSize = null;
  let downloadLabel = null;
  if (health.cuda_available && vram >= 6000) {
    downloadSize = "large-v3";
    downloadLabel = "מוריד את המודל הגדול ל-GPU שלך (≈2.9 GB)";
  } else if (health.cuda_available && vram >= 4000) {
    downloadSize = "medium";
    downloadLabel = "מוריד מודל בינוני ל-GPU שלך (≈1.5 GB)";
  }

  if (downloadSize) {
    status.textContent = downloadLabel;
    detail.textContent = `${health.gpu_name} · ${(vram / 1024).toFixed(1)} GB VRAM`;
    bar.classList.add("active");
    fill.style.width = "0%";
    try {
      await downloadModel(downloadSize, fill, detail);
      status.textContent = "מוכן";
      detail.textContent = `המודל (${downloadSize}) הותקן בהצלחה`;
    } catch (e) {
      status.textContent = "ההורדה נכשלה";
      detail.textContent = `${e.message} — נמשיך עם המודל המובנה. אפשר להוריד שוב מאוחר יותר.`;
    }
    bar.classList.remove("active");
  } else if (health.cuda_available) {
    status.textContent = "מוכן";
    detail.textContent = `${health.gpu_name || "GPU"} · ${(vram / 1024).toFixed(1)} GB VRAM — משתמש במודל ה-small המובנה`;
  } else {
    status.textContent = "מוכן";
    detail.textContent = "לא נמצא GPU של NVIDIA — האפליקציה תרוץ על מעבד (איטי יותר). השתמש במודל ה-small המובנה.";
  }

  cont.classList.add("active");
  await new Promise((resolve) => {
    cont.addEventListener("click", resolve, { once: true });
  });
  await window.karaoke.markInitialized();
  showView("form");
}

async function downloadModel(size, fillEl, detailEl) {
  const port = state.sidecarPort;
  // Kick off the download. 'already-installed' / 'already-running' are OK.
  const start = await fetch(`http://127.0.0.1:${port}/models/download`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ size }),
  });
  if (!start.ok) throw new Error(`POST /models/download ${start.status}`);

  // Poll progress until installed.
  while (true) {
    await new Promise((r) => setTimeout(r, 600));
    const r = await fetch(`http://127.0.0.1:${port}/models/status`);
    const body = await r.json();
    const m = body.models.find((x) => x.size === size);
    if (!m) throw new Error(`models/status missing ${size} entry`);
    if (m.state === "installed") return;
    if (m.state === "downloading") {
      const pct = Math.round((m.progress || 0) * 100);
      fillEl.style.width = `${pct}%`;
      const mb = (m.bytes_downloaded / 1e6).toFixed(0);
      const total = m.bytes_total ? (m.bytes_total / 1e6).toFixed(0) : "?";
      detailEl.textContent = `מוריד... ${pct}% · ${mb} / ${total} MB`;
    } else {
      // 'missing' here = the manager hit an error and gave up; report it.
      throw new Error("download stalled or failed — see sidecar logs");
    }
  }
}

// ─── submit + SSE ───────────────────────────────────────────────────────────

async function submitJob() {
  const url = document.getElementById("url-input").value.trim();
  const lyrics = document.getElementById("lyrics-input").value.trim();
  if (!url || !lyrics) return;

  resetProgressView();
  showView("progress");

  try {
    const resp = await fetch(`http://127.0.0.1:${state.sidecarPort}/jobs`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ youtube_url: url, lyrics }),
    });
    if (!resp.ok) {
      const body = await resp.text();
      throw new Error(`POST /jobs ${resp.status}: ${body}`);
    }
    const { job_id } = await resp.json();
    streamJob(job_id);
  } catch (e) {
    showError(e.message);
  }
}

function streamJob(jobId) {
  if (state.currentEventSource) state.currentEventSource.close();
  const es = new EventSource(`http://127.0.0.1:${state.sidecarPort}/jobs/${jobId}/events`);
  state.currentEventSource = es;

  es.onmessage = (ev) => {
    let s;
    try { s = JSON.parse(ev.data); } catch { return; }
    setStage(s);
    if (s.status === "complete") {
      es.close();
      state.currentEventSource = null;
      loadPlayer(jobId);
    } else if (s.status === "failed") {
      es.close();
      state.currentEventSource = null;
      showError(s.error || "alignment failed");
    }
  };
  es.onerror = () => { /* EventSource retries on its own */ };
}

// ─── player ─────────────────────────────────────────────────────────────────

async function loadPlayer(songId) {
  try {
    const resp = await fetch(`http://127.0.0.1:${state.sidecarPort}/library/${songId}`);
    if (!resp.ok) throw new Error(`GET /library/${songId} ${resp.status}`);
    const detail = await resp.json();

    state.data = detail.lyrics_data;
    document.getElementById("player-title").textContent = (detail.meta && detail.meta.title) || "";

    buildLyricsDom();
    teardownAudio();
    state.audio = document.getElementById("audio");
    state.audio.src = `http://127.0.0.1:${state.sidecarPort}/library/${songId}/audio`;
    state.audio.currentTime = 0;
    wireAudioEvents();

    showView("player");
    state.audio.play().catch(() => {});  // autoplay may be blocked; user can hit play
    state.audio.addEventListener("loadedmetadata", () => {
      const total = state.audio.duration || state.data.duration_seconds;
      document.getElementById("time-total").textContent = fmtTime(total);
      document.getElementById("bar").setAttribute("aria-valuemax", String(Math.round(total)));
    }, { once: true });
  } catch (e) {
    showError(`could not load song: ${e.message}`);
  }
}

function teardownAudio() {
  const a = document.getElementById("audio");
  try { a.pause(); } catch (_) {}
  a.removeAttribute("src");
  a.load();
}

function wireAudioEvents() {
  const a = state.audio;
  a.addEventListener("play",  () => { swapPlayIcon(true); });
  a.addEventListener("pause", () => { swapPlayIcon(false); });
}

function swapPlayIcon(playing) {
  document.getElementById("play-icon").style.display = playing ? "none" : "";
  document.getElementById("pause-icon").style.display = playing ? "" : "none";
  document.getElementById("play-btn").setAttribute("aria-label", playing ? "השהיה" : "ניגון");
}

// Build all line/word DOM once per song. Subsequent updates only mutate
// classes on lines + rebuild the single current line's word spans.
function buildLyricsDom() {
  const lyricsEl = document.getElementById("lyrics");
  lyricsEl.innerHTML = "";
  lyricsEl.style.transform = "translateY(0)";
  state.lineEls = [];
  state.activeLine = -1;
  state.activeWord = -1;
  state.activeBlockEl = null;

  for (let i = 0; i < state.data.lines.length; i++) {
    const ln = state.data.lines[i];
    const el = document.createElement("div");
    if (ln.is_empty) {
      el.className = "cc-empty";
      el.setAttribute("aria-hidden", "true");
    } else {
      el.className = "cc-line cc-line--upcoming";
      el.setAttribute("data-rel", "2");
      renderLineSpans(el, ln, /*isCurrent=*/false, /*activeWord=*/-1);
    }
    state.lineEls.push(el);
    lyricsEl.appendChild(el);
  }
}

function renderLineSpans(el, ln, isCurrent, activeWord) {
  // Replace contents of `el` with span structure for this line.
  el.replaceChildren();
  for (let j = 0; j < ln.words.length; j++) {
    const word = ln.words[j];
    const span = document.createElement("span");
    span.className = "cc-word";
    if (isCurrent) {
      if (j < activeWord) {
        span.classList.add("cc-word--sung");
        span.textContent = word.text;
      } else if (j === activeWord) {
        span.classList.add("cc-word--active");
        const base = document.createElement("span");
        base.className = "cc-word__base";
        base.textContent = word.text;
        const block = document.createElement("span");
        block.className = "cc-word__block";
        block.setAttribute("aria-hidden", "true");
        block.style.clipPath = "inset(0 0 0 100%)";
        const inverse = document.createElement("span");
        inverse.className = "cc-word__inverse";
        inverse.textContent = word.text;
        block.appendChild(inverse);
        span.append(base, block);
        state.activeBlockEl = block;
      } else {
        span.classList.add("cc-word--upcoming");
        span.textContent = word.text;
      }
    } else {
      span.textContent = word.text;
    }
    el.appendChild(span);
  }
}

function setLineRelClass(idx, activeLine) {
  const el = state.lineEls[idx];
  if (!el) return;
  if (state.data.lines[idx].is_empty) return;
  el.classList.remove("cc-line--past", "cc-line--upcoming", "cc-line--current");
  const rel = idx - activeLine;
  if (rel === 0) {
    el.classList.add("cc-line--current");
    el.setAttribute("data-rel", "0");
  } else if (rel < 0) {
    el.classList.add("cc-line--past");
    el.setAttribute("data-rel", String(Math.max(-2, rel)));
  } else {
    el.classList.add("cc-line--upcoming");
    el.setAttribute("data-rel", String(Math.min(2, rel)));
  }
}

function recenterLyrics() {
  if (state.activeLine < 0) return;
  const el = state.lineEls[state.activeLine];
  if (!el) return;
  // Offset to bring the active line's vertical center onto the stage's center.
  // .concrete__lyrics has `top: 50%`, so translating by -lineCenter aligns it.
  const lineCenter = el.offsetTop + el.offsetHeight / 2;
  document.getElementById("lyrics").style.transform = `translateY(${-lineCenter}px)`;
}

// rAF tick — single source of truth, reads audio.currentTime per frame.
function tick() {
  requestAnimationFrame(tick);
  if (!state.data || !state.audio) return;
  const t = state.audio.currentTime || 0;
  const { activeLine, activeWord, wordProgress } = findActive(state.data.lines, t);

  // Line change: update all affected lines' rel classes, rebuild the new
  // current line's spans, recenter.
  if (activeLine !== state.activeLine) {
    const prev = state.activeLine;
    state.activeLine = activeLine;
    state.activeWord = -1;
    state.activeBlockEl = null;

    // Update prev line back to past/upcoming with plain spans.
    if (prev >= 0 && state.lineEls[prev] && !state.data.lines[prev].is_empty) {
      renderLineSpans(state.lineEls[prev], state.data.lines[prev], false, -1);
      setLineRelClass(prev, activeLine);
    }
    // Update neighbours that may have moved between -1/-2 or +1/+2 brackets.
    for (let k = activeLine - 2; k <= activeLine + 2; k++) {
      if (k < 0 || k >= state.lineEls.length || k === activeLine) continue;
      setLineRelClass(k, activeLine);
    }
    // Promote the new current line.
    if (activeLine >= 0 && state.lineEls[activeLine] && !state.data.lines[activeLine].is_empty) {
      setLineRelClass(activeLine, activeLine);
    }
    recenterLyrics();
  }

  // Word change within the current line: rebuild only the current line.
  if (activeLine >= 0 && activeWord !== state.activeWord) {
    state.activeWord = activeWord;
    const el = state.lineEls[activeLine];
    if (el && !state.data.lines[activeLine].is_empty) {
      renderLineSpans(el, state.data.lines[activeLine], true, activeWord);
    }
  }

  // Within-word progress: update clip-path every frame.
  if (state.activeBlockEl && activeLine === state.activeLine && activeWord === state.activeWord) {
    const pct = Math.max(0, Math.min(100, Math.round(wordProgress * 100)));
    state.activeBlockEl.style.clipPath = `inset(0 0 0 ${100 - pct}%)`;
  }

  // Scrub bar + time (cheap, but skip when nothing changed).
  const secs = Math.floor(t);
  if (secs !== state.lastTimeShown) {
    state.lastTimeShown = secs;
    document.getElementById("time-current").textContent = fmtTime(t);
  }
  const total = state.audio.duration || state.data.duration_seconds || 1;
  const pct = (t / total) * 100;
  document.getElementById("bar-fill").style.width = `${pct}%`;
  document.getElementById("bar-knob").style.right = `calc(${pct}% - 7px)`;
  document.getElementById("bar").setAttribute("aria-valuenow", String(Math.round(t)));
}
requestAnimationFrame(tick);

// ─── transport + scrub ──────────────────────────────────────────────────────

function togglePlay() {
  if (!state.audio) return;
  if (state.audio.paused) state.audio.play().catch(() => {});
  else state.audio.pause();
}

function jumpLine(dir) {
  if (!state.data || !state.audio) return;
  const lines = state.data.lines;
  let target = null;
  if (dir < 0) {
    for (let i = state.activeLine - 1; i >= 0; i--) {
      if (!lines[i].is_empty) { target = i; break; }
    }
  } else {
    for (let i = state.activeLine + 1; i < lines.length; i++) {
      if (!lines[i].is_empty) { target = i; break; }
    }
  }
  if (target != null) state.audio.currentTime = lines[target].start;
  else if (dir < 0) state.audio.currentTime = 0;
}

function seekFromClientX(clientX) {
  if (!state.audio) return;
  const bar = document.getElementById("bar");
  const rect = bar.getBoundingClientRect();
  const fromLeft = Math.max(0, Math.min(rect.width, clientX - rect.left));
  // RTL: right edge = 0, left edge = duration
  const frac = 1 - fromLeft / rect.width;
  const total = state.audio.duration || state.data.duration_seconds || 0;
  state.audio.currentTime = frac * total;
}

function wireScrub() {
  const bar = document.getElementById("bar");

  bar.addEventListener("pointerdown", (e) => {
    if (!state.audio) return;
    e.preventDefault();
    state.scrubDrag = { wasPlaying: !state.audio.paused, pointerId: e.pointerId };
    state.audio.pause();
    try { bar.setPointerCapture(e.pointerId); } catch (_) {}
    seekFromClientX(e.clientX);
  });
  bar.addEventListener("pointermove", (e) => {
    if (!state.scrubDrag) return;
    seekFromClientX(e.clientX);
  });
  const release = (e) => {
    if (!state.scrubDrag) return;
    try { bar.releasePointerCapture(state.scrubDrag.pointerId); } catch (_) {}
    if (state.scrubDrag.wasPlaying && state.audio) state.audio.play().catch(() => {});
    state.scrubDrag = null;
  };
  bar.addEventListener("pointerup",     release);
  bar.addEventListener("pointercancel", release);
  // Keyboard accessibility on the slider.
  bar.addEventListener("keydown", (e) => {
    if (!state.audio) return;
    const step = e.shiftKey ? 5 : 1;
    if (e.key === "ArrowRight") { e.preventDefault(); state.audio.currentTime = Math.max(0, state.audio.currentTime - step); }
    else if (e.key === "ArrowLeft") { e.preventDefault(); state.audio.currentTime = Math.min(state.audio.duration || 0, state.audio.currentTime + step); }
  });
}

// ─── palette popover ────────────────────────────────────────────────────────

function wirePalettePopover() {
  const wrap = document.getElementById("palette");
  const btn = document.getElementById("palette-btn");
  btn.addEventListener("click", (e) => {
    e.stopPropagation();
    const isOpen = wrap.classList.toggle("open");
    btn.setAttribute("aria-expanded", String(isOpen));
  });
  document.addEventListener("pointerdown", (e) => {
    if (!wrap.contains(e.target)) {
      wrap.classList.remove("open");
      btn.setAttribute("aria-expanded", "false");
    }
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      wrap.classList.remove("open");
      btn.setAttribute("aria-expanded", "false");
    }
  });
}

// ─── boot ───────────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("submit-btn").addEventListener("click", submitJob);
  document.getElementById("reset-btn").addEventListener("click", () => {
    resetProgressView();
    showView("form");
  });
  document.getElementById("back-btn").addEventListener("click", () => {
    teardownAudio();
    state.data = null;
    state.lineEls = [];
    state.activeLine = -1;
    state.activeWord = -1;
    state.activeBlockEl = null;
    showView("form");
  });
  document.getElementById("play-btn").addEventListener("click", togglePlay);
  document.getElementById("prev-btn").addEventListener("click", () => jumpLine(-1));
  document.getElementById("next-btn").addEventListener("click", () => jumpLine(1));
  wireScrub();
  wirePalettePopover();
  bootstrap();
});
