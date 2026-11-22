"""Abstract platform adapter plus shared politeness helpers (rate limiting, retries)."""

from __future__ import annotations

import logging
import random
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

from agent_sight.config import Settings
from agent_sight.errors import AgentError
from agent_sight.models import DownloadRequest, NormalizedURL, VideoMetadata

log = logging.getLogger("agent_sight")
T = TypeVar("T")


@dataclass(frozen=True)
class SubtitleTrack:
    language: str
    ext: str
    content: str
    automatic: bool


@dataclass(frozen=True)
class StreamSource:
    """A URL that ffmpeg can open directly, with the headers the platform requires."""

    url: str
    headers: dict[str, str] = field(default_factory=dict)
    height: int | None = None
    is_hls: bool = False


class RateLimiter:
    """Enforces a minimum interval between requests to the same host (process-wide)."""

    def __init__(self, min_interval: float) -> None:
        self.min_interval = min_interval
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()

    def wait(self, host: str) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            ready_at = self._last.get(host, 0.0) + self.min_interval
            delay = max(0.0, ready_at - now)
            self._last[host] = max(now, ready_at)
        if delay:
            time.sleep(delay)


_limiters: dict[float, RateLimiter] = {}


def get_rate_limiter(min_interval: float) -> RateLimiter:
    return _limiters.setdefault(min_interval, RateLimiter(min_interval))


def with_retries(
    fn: Callable[[], T],
    *,
    retries: int,
    base_delay: float = 1.5,
    max_delay: float = 30.0,
    convert: Callable[[BaseException], AgentError],
) -> T:
    """Run ``fn`` retrying only retryable AgentErrors, with exponential backoff and jitter."""
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as exc:
            err = convert(exc)
            if not err.retryable or attempt >= retries:
                err.details.setdefault("attempts", attempt + 1)
                raise err from exc
            delay = min(max_delay, base_delay * (2**attempt)) * (0.75 + random.random() / 2)
            log.warning("%s (attempt %d/%d), retrying in %.1fs", err.code, attempt + 1, retries + 1, delay)
            time.sleep(delay)
            attempt += 1


class PlatformAdapter(ABC):
    name: str = "base"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.limiter = get_rate_limiter(settings.min_request_interval)

    @abstractmethod
    def can_handle(self, url: NormalizedURL) -> bool: ...

    @abstractmethod
    def probe(self, url: NormalizedURL) -> VideoMetadata:
        """Fetch metadata without downloading media."""

    @abstractmethod
    def list_subtitles(self, url: NormalizedURL, languages: list[str]) -> list[SubtitleTrack]:
        """Return native/auto caption tracks for the preferred languages (best first)."""

    @abstractmethod
    def stream_source(self, url: NormalizedURL, max_height: int) -> StreamSource:
        """Resolve a directly seekable video stream (used for frame sampling without downloading)."""

    @abstractmethod
    def download(self, url: NormalizedURL, request: DownloadRequest, dest_dir: Path) -> list[Path]:
        """Download media described by ``request`` into ``dest_dir``; return written files."""
