"use strict";

// Tiny bridge exposed to the renderer as `window.karaoke`.

const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("karaoke", {
  getSidecarPort: () => ipcRenderer.invoke("get-sidecar-port"),
  onSidecarReady: (cb) =>
    ipcRenderer.on("sidecar-ready", (_event, port) => cb(port)),
  isFirstLaunch: () => ipcRenderer.invoke("is-first-launch"),
  markInitialized: () => ipcRenderer.invoke("mark-initialized"),
});
