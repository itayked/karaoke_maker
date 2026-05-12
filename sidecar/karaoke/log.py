"""Structured JSON logger to stdout. One record per line."""

from __future__ import annotations

import json
import sys
import time
from typing import Any


def log(event: str, *, level: str = "INFO", **fields: Any) -> None:
    record: dict[str, Any] = {
        "ts": round(time.time(), 3),
        "level": level,
        "event": event,
    }
    record.update(fields)
    sys.stdout.write(json.dumps(record, ensure_ascii=False) + "\n")
    sys.stdout.flush()
