"""Sidecar entrypoint.

Picks an ephemeral port, prints `KARAOKE_SIDECAR_PORT=<n>` to stdout for the
Electron parent to read, then starts uvicorn. Use:

    python -m karaoke.server
"""

from __future__ import annotations

import socket
import sys

import uvicorn

from karaoke.server.app import create_app


def _pick_port() -> int:
    """Reserve a free localhost port. Tiny TOCTOU window but fine in practice."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
    finally:
        s.close()


def main() -> None:
    port = _pick_port()
    # Print BEFORE uvicorn starts logging so the parent finds it deterministically.
    print(f"KARAOKE_SIDECAR_PORT={port}", flush=True)

    app = create_app()
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        access_log=False,
    )


if __name__ == "__main__":
    main()
    sys.exit(0)
