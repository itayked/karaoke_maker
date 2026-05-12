"use strict";

// Build-prep script. Populates ../../build-staging/{python,sidecar,models,bin}
// so electron-builder can pick the lot up via extraResources.
//
// Steps (each is skipped when output already looks good — idempotent on
// re-runs, so you can resume after a failed download without re-doing the
// expensive bits):
//   1. Download + extract python-build-standalone (CPython 3.11, Windows x64)
//   2. Bootstrap pip in that interpreter
//   3. Install CUDA torch wheels (cu121)
//   4. Install sidecar/requirements.txt
//   5. Stage sidecar source (just karaoke/ + pyproject.toml — not .venv, not tests)
//   6. Pre-download Whisper small + medium .pt files
//   7. Pre-warm Demucs htdemucs + Silero VAD into HF / torch caches inside
//      the staged python's user_data, then copy those caches into models/
//   8. Download FFmpeg static build, extract ffmpeg.exe into bin/
//
// Run from the app/ dir:  npm run prepare-build

const fs = require("fs");
const path = require("path");
const https = require("https");
const { spawnSync } = require("child_process");
const crypto = require("crypto");

// ---- knobs ----------------------------------------------------------------

// Pinned to a known-good 3.11 install_only release. python-build-standalone
// retires older releases periodically, so if this URL 404s we fall through to
// `resolvePythonStandaloneUrl()` which queries the GitHub releases API for the
// latest 3.11 install_only Windows-x64 asset.
const PYTHON_STANDALONE_URL =
  "https://github.com/astral-sh/python-build-standalone/releases/download/" +
  "20260510/cpython-3.11.15+20260510-x86_64-pc-windows-msvc-install_only.tar.gz";

// Used by the fallback resolver — match install_only but NOT install_only_stripped.
const PYTHON_STANDALONE_ASSET_RE =
  /^cpython-3\.11\.\d+\+\d+-x86_64-pc-windows-msvc-install_only\.tar\.gz$/;

const TORCH_INDEX = "https://download.pytorch.org/whl/cu121";
const TORCH_PACKAGES = ["torch==2.4.1", "torchaudio==2.4.1"];

// Only small is bundled. medium + large-v3 are downloaded on first launch
// when the user's VRAM warrants them (see renderer's first-launch flow).
// Keeps the installer payload small enough for NSIS's 32-bit mmap ceiling
// and avoids shipping a 1.5GB model most low-VRAM users won't actually use.
const WHISPER_MODELS = ["small"];

const FFMPEG_URL =
  "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip";

// ---- paths ----------------------------------------------------------------

const APP_DIR     = path.resolve(__dirname, "..");
const REPO_DIR    = path.resolve(APP_DIR, "..");
const STAGE_DIR   = path.join(REPO_DIR, "build-staging");
const PY_DIR      = path.join(STAGE_DIR, "python");
const SIDECAR_DIR = path.join(STAGE_DIR, "sidecar");
const MODELS_DIR  = path.join(STAGE_DIR, "models");
const BIN_DIR     = path.join(STAGE_DIR, "bin");
const TMP_DIR     = path.join(REPO_DIR, "build-tmp");

const PY_EXE = path.join(PY_DIR, "python.exe");

// ---- helpers --------------------------------------------------------------

function log(...args) {
  const ts = new Date().toISOString().slice(11, 19);
  console.log(`[${ts}]`, ...args);
}

function ensureDir(p) {
  fs.mkdirSync(p, { recursive: true });
}

function exists(p) {
  try { fs.accessSync(p); return true; } catch { return false; }
}

function run(cmd, args, opts = {}) {
  log(`$ ${cmd} ${args.join(" ")}`);
  const res = spawnSync(cmd, args, { stdio: "inherit", ...opts });
  if (res.error) throw res.error;
  if (res.status !== 0) {
    throw new Error(`${cmd} exited with ${res.status}`);
  }
}

function py(args, opts = {}) {
  run(PY_EXE, args, opts);
}

function fetchJson(url, attempt = 1) {
  return new Promise((resolve, reject) => {
    https
      .get(url, { headers: { "User-Agent": "karaoke-build/0.1", "Accept": "application/vnd.github+json" } }, (res) => {
        if ([301, 302, 303, 307, 308].includes(res.statusCode)) {
          res.resume();
          return fetchJson(res.headers.location, attempt).then(resolve, reject);
        }
        if (res.statusCode !== 200) {
          res.resume();
          if (attempt < 3) {
            return setTimeout(() => fetchJson(url, attempt + 1).then(resolve, reject), 2000);
          }
          return reject(new Error(`HTTP ${res.statusCode} on ${url}`));
        }
        let body = "";
        res.setEncoding("utf8");
        res.on("data", (c) => { body += c; });
        res.on("end", () => {
          try { resolve(JSON.parse(body)); } catch (e) { reject(e); }
        });
      })
      .on("error", (err) => {
        if (attempt < 3) setTimeout(() => fetchJson(url, attempt + 1).then(resolve, reject), 2000);
        else reject(err);
      });
  });
}

async function resolvePythonStandaloneUrl() {
  log(
    "python: pinned URL 404'd — querying GitHub releases API for the latest " +
    "install_only Windows-x64 asset"
  );
  const release = await fetchJson(
    "https://api.github.com/repos/astral-sh/python-build-standalone/releases/latest"
  );
  log(`python: latest release is ${release.tag_name} (${release.published_at})`);
  const asset = (release.assets || []).find((a) => PYTHON_STANDALONE_ASSET_RE.test(a.name));
  if (!asset) {
    throw new Error(
      `no asset matching ${PYTHON_STANDALONE_ASSET_RE} in release ${release.tag_name}`
    );
  }
  return asset.browser_download_url;
}

// HEAD an URL to confirm it's actually downloadable before we commit to it.
function headOk(url, attempt = 1) {
  return new Promise((resolve) => {
    const req = https.request(url, { method: "HEAD", headers: { "User-Agent": "karaoke-build/0.1" } }, (res) => {
      if ([301, 302, 303, 307, 308].includes(res.statusCode)) {
        res.resume();
        return headOk(res.headers.location, attempt).then(resolve);
      }
      res.resume();
      resolve(res.statusCode >= 200 && res.statusCode < 300);
    });
    req.on("error", () => {
      if (attempt < 2) setTimeout(() => headOk(url, attempt + 1).then(resolve), 1000);
      else resolve(false);
    });
    req.end();
  });
}

function downloadTo(url, dest, attempt = 1) {
  return new Promise((resolve, reject) => {
    log(`download (try ${attempt}): ${url}`);
    const tmp = dest + ".part";
    if (exists(tmp)) fs.unlinkSync(tmp);
    const file = fs.createWriteStream(tmp);
    const get = (u) => {
      https.get(u, (res) => {
        if ([301, 302, 303, 307, 308].includes(res.statusCode)) {
          res.resume();
          return get(res.headers.location);
        }
        if (res.statusCode !== 200) {
          file.close(() => fs.unlinkSync(tmp));
          return reject(new Error(`HTTP ${res.statusCode} on ${u}`));
        }
        const total = parseInt(res.headers["content-length"] || "0", 10);
        let got = 0;
        let lastPctLogged = -1;
        res.on("data", (chunk) => {
          got += chunk.length;
          if (total) {
            const pct = Math.floor((got / total) * 100);
            if (pct !== lastPctLogged && pct % 5 === 0) {
              lastPctLogged = pct;
              process.stdout.write(`\r  ${pct}% (${(got / 1e6).toFixed(1)} / ${(total / 1e6).toFixed(1)} MB)`);
            }
          }
        });
        res.pipe(file);
        file.on("finish", () => {
          file.close(() => {
            process.stdout.write("\n");
            fs.renameSync(tmp, dest);
            resolve();
          });
        });
      }).on("error", (err) => {
        file.close(() => { try { fs.unlinkSync(tmp); } catch {} });
        if (attempt < 3) {
          log(`retrying after error: ${err.message}`);
          setTimeout(() => downloadTo(url, dest, attempt + 1).then(resolve, reject), 2000);
        } else {
          reject(err);
        }
      });
    };
    get(url);
  });
}

function sha256File(p) {
  const buf = fs.readFileSync(p);
  return crypto.createHash("sha256").update(buf).digest("hex");
}

// ---- steps ----------------------------------------------------------------

async function resolvePythonUrl() {
  if (await headOk(PYTHON_STANDALONE_URL)) {
    log(`python: using pinned URL ${PYTHON_STANDALONE_URL}`);
    return PYTHON_STANDALONE_URL;
  }
  const resolved = await resolvePythonStandaloneUrl();
  log(`python: using resolved URL ${resolved}`);
  return resolved;
}

async function stepPython() {
  if (exists(PY_EXE)) {
    log("python: standalone interpreter already staged");
    return;
  }
  ensureDir(TMP_DIR);
  const tgz = path.join(TMP_DIR, "cpython.tar.gz");
  if (!exists(tgz)) {
    const url = await resolvePythonUrl();
    await downloadTo(url, tgz);
  } else {
    log(`python: tarball cached at ${tgz}, skipping download`);
  }
  log("python: extracting...");
  ensureDir(PY_DIR);
  // python-build-standalone tarball has a `python/` top-level dir; we want
  // its contents directly in PY_DIR. `--strip-components=1` handles that.
  //
  // On Windows the `tar` on PATH is usually GNU tar (from Git Bash), which
  // by default interprets `S:\path` as `host:path` and tries to DNS-resolve
  // "S". `--force-local` opts out of that; forward-slash paths help too.
  // The MS-shipped bsdtar accepts `--force-local` as a no-op flag, so this
  // works on either implementation.
  const posix = (p) => p.replace(/\\/g, "/");
  run("tar", [
    "--force-local",
    "-xzf",
    posix(tgz),
    "-C",
    posix(PY_DIR),
    "--strip-components=1",
  ]);
  if (!exists(PY_EXE)) {
    throw new Error(`python.exe not found in extracted tarball: ${PY_DIR}`);
  }
}

function stepPipBootstrap() {
  // python-build-standalone "install_only" already includes pip via ensurepip.
  log("python: ensurepip + upgrade pip");
  py(["-m", "ensurepip", "--upgrade"]);
  py(["-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools"]);
}

function stepTorch() {
  // Check torch is already installed (cu121 build).
  const probe = spawnSync(PY_EXE, ["-c", "import torch; print(torch.__version__)"]);
  if (probe.status === 0) {
    const ver = probe.stdout.toString().trim();
    if (ver.includes("+cu121") || ver.startsWith("2.4.1")) {
      log(`torch: already installed (${ver})`);
      return;
    }
  }
  log("torch: installing CUDA wheels");
  py(["-m", "pip", "install", "--index-url", TORCH_INDEX, ...TORCH_PACKAGES]);
}

function stepRequirements() {
  const req = path.join(REPO_DIR, "sidecar", "requirements.txt");
  // Filter out the torch lines (we just installed CUDA-specific ones above)
  // and any dev-only entries we don't ship.
  const lines = fs.readFileSync(req, "utf8")
    .split(/\r?\n/)
    .filter(l => l && !l.startsWith("#"))
    .filter(l => !/^torch(\b|==)/.test(l))
    .filter(l => !/^torchaudio(\b|==)/.test(l))
    .filter(l => !/^pytest/.test(l))
    .filter(l => !/^ruff/.test(l));
  const tmpReq = path.join(TMP_DIR, "requirements-pruned.txt");
  ensureDir(TMP_DIR);
  fs.writeFileSync(tmpReq, lines.join("\n"));
  log(`pip install -r ${tmpReq}`);
  py(["-m", "pip", "install", "-r", tmpReq]);
}

function stepSidecarSource() {
  log("sidecar: copying source");
  ensureDir(SIDECAR_DIR);
  const src = path.join(REPO_DIR, "sidecar", "karaoke");
  const dst = path.join(SIDECAR_DIR, "karaoke");
  rmIfExists(dst);
  copyDirSync(src, dst);
}

async function stepWhisperModels() {
  ensureDir(MODELS_DIR);
  // Use python-side whisper._MODELS for canonical URLs so we don't drift.
  const code = `
import json, whisper
print(json.dumps({k: whisper._MODELS[k] for k in ${JSON.stringify(WHISPER_MODELS)}}))
`;
  const res = spawnSync(PY_EXE, ["-c", code], { encoding: "utf8" });
  if (res.status !== 0) throw new Error("whisper URL lookup failed:\n" + res.stderr);
  const urls = JSON.parse(res.stdout.trim().split("\n").pop());

  for (const size of WHISPER_MODELS) {
    const dest = path.join(MODELS_DIR, `whisper-${size}.pt`);
    if (exists(dest)) { log(`whisper: ${size} already staged`); continue; }
    await downloadTo(urls[size], dest);
  }
}

function stepDemucsAndVad() {
  // Pre-warm by importing — Demucs htdemucs lazy-downloads on first use; we
  // force it now into a known cache dir, then copy to MODELS_DIR.
  ensureDir(MODELS_DIR);
  const torchHubCache = path.join(MODELS_DIR, "torch-hub");
  const demucsCache = path.join(MODELS_DIR, "demucs-models");
  ensureDir(torchHubCache);
  ensureDir(demucsCache);

  const env = {
    ...process.env,
    TORCH_HOME: torchHubCache,             // Silero VAD downloads here via torch.hub
    XDG_CACHE_HOME: demucsCache,           // demucs/dora uses XDG_CACHE_HOME on Windows too
  };

  log("demucs: warming htdemucs weights");
  py([
    "-c",
    "from demucs.pretrained import get_model; m = get_model('htdemucs'); print('demucs loaded')",
  ], { env });

  log("silero: warming VAD weights");
  py([
    "-c",
    "from silero_vad import load_silero_vad; load_silero_vad(); print('silero loaded')",
  ], { env });
}

async function stepFfmpeg() {
  ensureDir(BIN_DIR);
  const target = path.join(BIN_DIR, "ffmpeg.exe");
  if (exists(target)) { log("ffmpeg: already staged"); return; }

  ensureDir(TMP_DIR);
  const zip = path.join(TMP_DIR, "ffmpeg.zip");
  if (!exists(zip)) {
    await downloadTo(FFMPEG_URL, zip);
  }
  log("ffmpeg: extracting via python (built-in zipfile)");
  const extractScript = `
import os, sys, zipfile, shutil
src = sys.argv[1]; dst_dir = sys.argv[2]
with zipfile.ZipFile(src) as z:
    members = [m for m in z.namelist() if m.endswith('/bin/ffmpeg.exe')]
    if not members:
        sys.exit('ffmpeg.exe not found in zip')
    member = members[0]
    with z.open(member) as src_f, open(os.path.join(dst_dir, 'ffmpeg.exe'), 'wb') as dst_f:
        shutil.copyfileobj(src_f, dst_f)
print('ffmpeg.exe extracted')
`;
  py(["-c", extractScript, zip, BIN_DIR]);
  if (!exists(target)) throw new Error("ffmpeg.exe extraction failed");
}

function dirSize(p) {
  let total = 0;
  try {
    for (const entry of fs.readdirSync(p, { withFileTypes: true })) {
      const f = path.join(p, entry.name);
      if (entry.isDirectory()) total += dirSize(f);
      else if (entry.isFile()) {
        try { total += fs.statSync(f).size; } catch {}
      }
    }
  } catch {}
  return total;
}

function fmtBytes(n) {
  if (n >= 1e9) return (n / 1e9).toFixed(2) + " GB";
  if (n >= 1e6) return Math.round(n / 1e6) + " MB";
  if (n >= 1e3) return Math.round(n / 1e3) + " KB";
  return n + " B";
}

function walkFiles(root, fn) {
  if (!exists(root)) return;
  for (const entry of fs.readdirSync(root, { withFileTypes: true })) {
    const f = path.join(root, entry.name);
    if (entry.isDirectory()) walkFiles(f, fn);
    else if (entry.isFile()) fn(f);
  }
}

function rmAllNamed(root, name) {
  if (!exists(root)) return 0;
  let count = 0;
  const stack = [root];
  while (stack.length) {
    const dir = stack.pop();
    let entries;
    try { entries = fs.readdirSync(dir, { withFileTypes: true }); } catch { continue; }
    for (const entry of entries) {
      const p = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        if (entry.name === name) {
          fs.rmSync(p, { recursive: true, force: true });
          count++;
        } else {
          stack.push(p);
        }
      }
    }
  }
  return count;
}

function stepSlim() {
  // Order matters: report total size before and after, listing each cut.
  // Targets are runtime-irrelevant artefacts:
  //   - torch's .lib static-link archives (compile-time only)
  //   - C++ headers under torch/include
  //   - .pdb debug symbols
  //   - Tk/tkinter (we never use a Python GUI)
  //   - Lib/test, Lib/idlelib, Lib/ensurepip (stdlib bits we don't ship)
  //   - All __pycache__/ directories (regenerated on first use)
  const before = dirSize(STAGE_DIR);
  log(`slim: build-staging size before = ${fmtBytes(before)}`);

  const torchDir = path.join(PY_DIR, "Lib", "site-packages", "torch");
  const torchLib = path.join(torchDir, "lib");

  // 1. torch/lib/*.lib  — static archives (linking only)
  if (exists(torchLib)) {
    let count = 0, bytes = 0;
    for (const name of fs.readdirSync(torchLib)) {
      if (name.endsWith(".lib")) {
        const p = path.join(torchLib, name);
        try { bytes += fs.statSync(p).size; fs.unlinkSync(p); count++; } catch {}
      }
    }
    log(`slim:   torch/lib/*.lib removed (${count} files, ${fmtBytes(bytes)})`);
  }

  // 2. torch/include — C++ headers, never used at runtime by Python imports
  const torchInc = path.join(torchDir, "include");
  if (exists(torchInc)) {
    const s = dirSize(torchInc);
    fs.rmSync(torchInc, { recursive: true, force: true });
    log(`slim:   torch/include removed (${fmtBytes(s)})`);
  }

  // 3. tcl / tkinter
  for (const p of [
    path.join(PY_DIR, "tcl"),
    path.join(PY_DIR, "Lib", "tkinter"),
  ]) {
    if (exists(p)) {
      const s = dirSize(p);
      fs.rmSync(p, { recursive: true, force: true });
      log(`slim:   ${path.relative(PY_DIR, p)} removed (${fmtBytes(s)})`);
    }
  }
  for (const f of ["tcl86t.dll", "tk86t.dll"]) {
    const p = path.join(PY_DIR, "DLLs", f);
    if (exists(p)) {
      const s = fs.statSync(p).size;
      fs.unlinkSync(p);
      log(`slim:   DLLs/${f} removed (${fmtBytes(s)})`);
    }
  }

  // 4. Lib/ensurepip + Lib/idlelib (no longer needed after build)
  for (const sub of ["ensurepip", "idlelib", "test", "turtledemo"]) {
    const p = path.join(PY_DIR, "Lib", sub);
    if (exists(p)) {
      const s = dirSize(p);
      fs.rmSync(p, { recursive: true, force: true });
      log(`slim:   Lib/${sub} removed (${fmtBytes(s)})`);
    }
  }

  // 5. .pdb debug symbols anywhere under python/
  let pdbCount = 0, pdbBytes = 0;
  walkFiles(PY_DIR, (f) => {
    if (f.endsWith(".pdb")) {
      try { pdbBytes += fs.statSync(f).size; fs.unlinkSync(f); pdbCount++; } catch {}
    }
  });
  if (pdbCount) log(`slim:   ${pdbCount} .pdb files removed (${fmtBytes(pdbBytes)})`);

  // 6. __pycache__ directories — regenerated on first use
  const pycacheCount = rmAllNamed(PY_DIR, "__pycache__");
  log(`slim:   ${pycacheCount} __pycache__ dirs removed`);

  const after = dirSize(STAGE_DIR);
  log(
    `slim: build-staging size after  = ${fmtBytes(after)} ` +
    `(saved ${fmtBytes(before - after)})`
  );
}

// ---- utility ---------------------------------------------------------------

function rmIfExists(p) {
  if (exists(p)) fs.rmSync(p, { recursive: true, force: true });
}

function copyDirSync(src, dst) {
  fs.mkdirSync(dst, { recursive: true });
  for (const entry of fs.readdirSync(src, { withFileTypes: true })) {
    if (entry.name === "__pycache__" || entry.name === ".pytest_cache") continue;
    const s = path.join(src, entry.name);
    const d = path.join(dst, entry.name);
    if (entry.isDirectory()) copyDirSync(s, d);
    else fs.copyFileSync(s, d);
  }
}

// ---- main ------------------------------------------------------------------

(async () => {
  ensureDir(STAGE_DIR);
  ensureDir(TMP_DIR);
  log("staging dir:", STAGE_DIR);

  await stepPython();
  stepPipBootstrap();
  stepTorch();
  stepRequirements();
  stepSidecarSource();
  await stepWhisperModels();
  stepDemucsAndVad();
  await stepFfmpeg();
  stepSlim();

  log("done. build-staging is ready for `electron-builder`.");
})().catch((err) => {
  console.error("prepare-build failed:", err);
  process.exit(1);
});
