"""Karaoke sidecar configuration.

Most settings have sensible defaults; env vars are mainly for dev overrides.
The desktop app sets paths under %APPDATA%\\Karaoke; here we fall back to
platform-appropriate locations so the package can also run standalone (CLI,
tests, manual sidecar bring-up).
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_app_data_dir() -> Path:
    # %APPDATA%\Karaoke on Windows; ~/.local/share/Karaoke elsewhere (for dev/CI).
    appdata = os.environ.get("APPDATA")
    if appdata:
        return Path(appdata) / "Karaoke"
    return Path.home() / ".local" / "share" / "Karaoke"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    whisper_model_size: str = Field("large-v3", alias="WHISPER_MODEL_SIZE")
    device: str = Field("cuda", alias="DEVICE")

    app_data_dir: Path = Field(default_factory=_default_app_data_dir, alias="KARAOKE_APP_DATA")
    tmp_dir: Path = Field(Path("./tmp"), alias="KARAOKE_TMP_DIR")
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    @property
    def library_dir(self) -> Path:
        return self.app_data_dir / "library"

    @property
    def models_dir(self) -> Path:
        return self.app_data_dir / "models"

    @property
    def logs_dir(self) -> Path:
        return self.app_data_dir / "logs"


settings = Settings()  # type: ignore[call-arg]
