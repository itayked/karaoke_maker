"use strict";

// Main process:
//   1. Spawn the Python sidecar as a child process.
//   2. Read KARAOKE_SIDECAR_PORT=<n> from its stdout.
//   3. Open a BrowserWindow and expose the port to the renderer via IPC.
//   4. Health-check the sidecar every 5s; auto-restart on death.
//   5. Kill the sidecar on app exit (SIGTERM, then SIGKILL after 5s).

const { app, BrowserWindow, ipcMain } = require("electron");
const { spawn } = require("child_process");
const fs = require("fs");
const path = require("path");

const HEALTH_INTERVAL_MS = 5000;
const HEALTH_TIMEOUT_MS = 3000;
const SHUTDOWN_GRACE_MS = 5000;

let mainWindow = null;
let sidecarProc = null;
let sidecarPort = null;
let healthTimer = null;
let shuttingDown = false;
let portWaiters = []; // resolvers for renderer ipc 'get-sidecar-port' calls

function resolveSidecarSpec() {
  // Packaged: electron-builder lands extraResources at `process.resourcesPath`.
  //   <resourcesPath>/python/python.exe
  //   <resourcesPath>/sidecar/karaoke/...
  //   <resourcesPath>/models/...
  //   <resourcesPath>/bin/ffmpeg.exe
  // Dev: fall back to the sidecar/.venv interpreter at the repo root.
  const isWin = process.platform === "win32";
  const venvDir = isWin ? "Scripts" : "bin";
  const pyName = isWin ? "python.exe" : "python";

  const packaged = app.isPackaged;
  let pythonExe;
  let sidecarCwd;
  let resourcesDir;

  if (packaged) {
    const r = process.resourcesPath;
    pythonExe = path.join(r, "python", pyName);
    sidecarCwd = path.join(r, "sidecar");
    resourcesDir = r;
  } else {
    const projectRoot = path.resolve(__dirname, "..");
    pythonExe = path.join(projectRoot, "sidecar", ".venv", venvDir, pyName);
    sidecarCwd = path.join(projectRoot, "sidecar");
    resourcesDir = ""; // unset in dev — sidecar falls back to dev paths
  }

  return {
    command: process.env.KARAOKE_PYTHON || pythonExe,
    args: ["-m", "karaoke.server"],
    cwd: sidecarCwd,
    env: resourcesDir ? { ...process.env, KARAOKE_RESOURCES_DIR: resourcesDir } : process.env,
  };
}

function spawnSidecar() {
  const spec = resolveSidecarSpec();
  console.log(`[main] spawning sidecar: ${spec.command} ${spec.args.join(" ")}`);

  const proc = spawn(spec.command, spec.args, {
    cwd: spec.cwd,
    env: spec.env || process.env,
    stdio: ["ignore", "pipe", "pipe"],
    windowsHide: true,
  });

  let stdoutBuf = "";
  let portFound = false;
  proc.stdout.on("data", (chunk) => {
    const s = chunk.toString();
    process.stdout.write(`[sidecar] ${s}`);
    if (portFound) return;
    stdoutBuf += s;
    const m = stdoutBuf.match(/KARAOKE_SIDECAR_PORT=(\d+)/);
    if (m) {
      portFound = true;
      sidecarPort = parseInt(m[1], 10);
      console.log(`[main] sidecar port: ${sidecarPort}`);
      flushPortWaiters();
      if (mainWindow && !mainWindow.isDestroyed()) {
        mainWindow.webContents.send("sidecar-ready", sidecarPort);
      }
    }
  });

  proc.stderr.on("data", (chunk) => {
    process.stderr.write(`[sidecar:err] ${chunk}`);
  });

  proc.on("exit", (code, signal) => {
    console.log(`[main] sidecar exit code=${code} signal=${signal}`);
    sidecarProc = null;
    sidecarPort = null;
    if (!shuttingDown) {
      console.log("[main] restarting sidecar in 2s...");
      setTimeout(() => {
        sidecarProc = spawnSidecar();
      }, 2000);
    }
  });

  return proc;
}

function flushPortWaiters() {
  const waiters = portWaiters;
  portWaiters = [];
  for (const resolve of waiters) resolve(sidecarPort);
}

function startHealthCheck() {
  healthTimer = setInterval(async () => {
    if (!sidecarPort) return;
    const controller = new AbortController();
    const t = setTimeout(() => controller.abort(), HEALTH_TIMEOUT_MS);
    try {
      const resp = await fetch(`http://127.0.0.1:${sidecarPort}/health`, {
        signal: controller.signal,
      });
      if (!resp.ok) throw new Error(`status ${resp.status}`);
    } catch (e) {
      console.error(`[main] health check failed: ${e.message}`);
      // The process 'exit' handler does the restart if the proc actually died;
      // a transient hung response is harmless to log and move on.
    } finally {
      clearTimeout(t);
    }
  }, HEALTH_INTERVAL_MS);
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1100,
    height: 800,
    backgroundColor: "#0e0e10",
    autoHideMenuBar: true,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });
  mainWindow.loadFile(path.join(__dirname, "renderer", "index.html"));
  mainWindow.on("closed", () => {
    mainWindow = null;
  });

  mainWindow.webContents.on("did-finish-load", () => {
    if (sidecarPort && mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send("sidecar-ready", sidecarPort);
    }
  });
}

// ---- IPC ----

ipcMain.handle("get-sidecar-port", () => {
  if (sidecarPort) return sidecarPort;
  return new Promise((resolve) => portWaiters.push(resolve));
});

// First-launch sentinel — file is dropped only after a successful welcome flow.
function sentinelPath() {
  return path.join(app.getPath("appData"), "Karaoke", ".initialized");
}

ipcMain.handle("is-first-launch", () => {
  try {
    return !fs.existsSync(sentinelPath());
  } catch {
    return true;
  }
});

ipcMain.handle("mark-initialized", () => {
  try {
    const p = sentinelPath();
    fs.mkdirSync(path.dirname(p), { recursive: true });
    fs.writeFileSync(p, new Date().toISOString());
    return true;
  } catch (e) {
    console.error("[main] mark-initialized failed:", e.message);
    return false;
  }
});

// ---- lifecycle ----

app.whenReady().then(() => {
  sidecarProc = spawnSidecar();
  startHealthCheck();
  createWindow();
});

app.on("window-all-closed", () => {
  app.quit();
});

app.on("before-quit", (e) => {
  if (shuttingDown) return;
  shuttingDown = true;
  if (healthTimer) clearInterval(healthTimer);
  if (!sidecarProc) return;

  e.preventDefault();
  console.log("[main] sending SIGTERM to sidecar");
  try {
    sidecarProc.kill("SIGTERM");
  } catch (err) {
    console.error(`[main] SIGTERM failed: ${err.message}`);
  }
  const force = setTimeout(() => {
    if (!sidecarProc) return;
    console.log("[main] forcing SIGKILL");
    try {
      sidecarProc.kill("SIGKILL");
    } catch (_) {}
  }, SHUTDOWN_GRACE_MS);

  const onExit = () => {
    clearTimeout(force);
    sidecarProc = null;
    app.quit();
  };
  if (sidecarProc) sidecarProc.once("exit", onExit);
  else onExit();
});
