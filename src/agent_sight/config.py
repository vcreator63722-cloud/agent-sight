"""Runtime configuration loaded from environment variables (prefix ``AGENT_SIGHT_``) or ``.env``."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

QualityPreset = Literal["best", "1080p", "720p", "480p", "360p", "worst"]


def _default_cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base) if base else Path.home() / ".cache"
    return root / "agent-sight"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AGENT_SIGHT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    cache_dir: Path = Field(default_factory=_default_cache_dir)
    cache_ttl_seconds: int = Field(default=7 * 24 * 3600, ge=0)
    output_dir: Path = Field(default_factory=lambda: Path.cwd() / "downloads")
    max_download_mb: int = Field(default=2048, ge=1)
    default_quality: QualityPreset = "720p"

    ffmpeg_path: Path | None = None
    ffprobe_path: Path | None = None

    whisper_enabled: bool = True
    whisper_model: str = "small"
    whisper_device: str = "auto"
    whisper_compute_type: str = "auto"
    whisper_max_duration: int = Field(default=900, ge=0)

    cookies_file: Path | None = None
    user_agent: str | None = None

    min_request_interval: float = Field(default=1.0, ge=0)
    max_retries: int = Field(default=3, ge=0, le=10)
    socket_timeout: float = Field(default=30.0, gt=0)
    ffmpeg_timeout: float = Field(default=600.0, gt=0)

    # Videos up to this length get a local low-res proxy for accurate scene detection.
    # Longer ones fall back to sparse remote seeking to bound time and bandwidth.
    proxy_max_duration: int = Field(default=1800, ge=0)
    proxy_max_height: int = Field(default=360, ge=144)

    allow_private_hosts: bool = False
    max_html_bytes: int = Field(default=5 * 1024 * 1024, ge=1024)

    @field_validator("cache_dir", "output_dir", "ffmpeg_path", "ffprobe_path", "cookies_file")
    @classmethod
    def _expand(cls, v: Path | None) -> Path | None:
        return v.expanduser() if v is not None else None

    @property
    def max_download_bytes(self) -> int:
        return self.max_download_mb * 1024 * 1024

    @property
    def db_path(self) -> Path:
        return self.cache_dir / "cache.sqlite3"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings() -> None:
    """Clear the cached settings (used by tests and after env changes)."""
    get_settings.cache_clear()
