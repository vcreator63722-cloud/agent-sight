"""Download orchestration: size limits, output-directory confinement, caching and proxies."""

from __future__ import annotations

import logging
from pathlib import Path

from agent_sight.adapters.base import PlatformAdapter
from agent_sight.cache import Cache, make_key
from agent_sight.config import Settings
from agent_sight.downloader.ffprobe_utils import probe_media
from agent_sight.errors import AgentError, ErrorCode
from agent_sight.models import DownloadRequest, DownloadResult, MediaInfo, NormalizedURL

log = logging.getLogger("agent_sight.download")
_PRESETS = (360, 480, 720, 1080)


def resolve_output_dir(settings: Settings, requested: str | Path | None, *, confine: bool) -> Path:
    """Pick the output directory; with ``confine`` it must stay inside ``settings.output_dir``."""
    root = settings.output_dir.expanduser().resolve()
    if requested is None or str(requested).strip() == "":
        target = root
    else:
        req = Path(requested).expanduser()
        target = (root / req).resolve() if confine and not req.is_absolute() else req.resolve()
    if confine and target != root and root not in target.parents:
        raise AgentError(
            ErrorCode.INVALID_ARGUMENT,
            "Output directory must be inside the configured download directory.",
            f"Pass a relative sub-directory name or omit it. Downloads go to {root}.",
        )
    target.mkdir(parents=True, exist_ok=True)
    return target


class MediaDownloader:
    def __init__(self, settings: Settings, cache: Cache) -> None:
        self.settings = settings
        self.cache = cache

    def download(
        self,
        url: NormalizedURL,
        adapter: PlatformAdapter,
        request: DownloadRequest,
        *,
        output_dir: str | Path | None = None,
        confine: bool = False,
        refresh: bool = False,
    ) -> DownloadResult:
        dest = resolve_output_dir(self.settings, output_dir, confine=confine)
        params = request.model_dump(exclude={"url"}) | {"dest": str(dest)}
        key = make_key(url.normalized, "download", params)
        if not refresh and (hit := self.cache.get(key)):
            result = DownloadResult.model_validate(hit)
            if all(Path(f).exists() for f in result.files):
                result.cached = True
                return result

        files = adapter.download(url, request, dest)
        total = sum(f.stat().st_size for f in files)
        if total > self.settings.max_download_bytes:
            for f in files:
                f.unlink(missing_ok=True)
            raise AgentError(
                ErrorCode.FILE_TOO_LARGE,
                f"Downloaded {total / 1e6:.0f} MB, above the {self.settings.max_download_mb} MB limit.",
            )
        media: MediaInfo | None = None
        if request.mode in ("video", "audio"):
            try:
                media = probe_media(files[0], settings=self.settings)
            except AgentError as err:
                log.info("post-download probe failed: %s", err.message)
            if media is not None and not media.duration and not media.streams:
                for f in files:
                    f.unlink(missing_ok=True)
                raise AgentError(
                    ErrorCode.FFMPEG_FAILED, "The download produced an empty media file.",
                    "The server may not support range requests for this file; retry with "
                    "--mode video (full file) and extract locally.",
                )
        result = DownloadResult(
            url=url.normalized, mode=request.mode, files=[str(f) for f in files], size_bytes=total,
            media=media, start=request.start, end=request.end,
        )
        self.cache.set(key, url.normalized, "download", result.model_dump(), file_path=files[0])
        return result

    def _proxy_key(self, url: NormalizedURL) -> tuple[str, int]:
        height = next((h for h in _PRESETS if h >= self.settings.proxy_max_height), 1080)
        return make_key(url.normalized, "proxy", {"h": height}), height

    def cached_proxy(self, url: NormalizedURL) -> Path | None:
        key, _ = self._proxy_key(url)
        hit = self.cache.get(key)
        return Path(hit["path"]) if hit and Path(hit["path"]).exists() else None

    def proxy(self, url: NormalizedURL, adapter: PlatformAdapter, *, refresh: bool = False) -> Path:
        """Low-resolution local copy (with audio) used for scene detection and transcription."""
        key, height = self._proxy_key(url)
        if not refresh and (cached := self.cached_proxy(url)):
            return cached
        work = self.cache.work_dir(key)
        request = DownloadRequest(
            url=url.normalized, mode="video", quality=f"{height}p",  # type: ignore[arg-type]
            container="mp4", filename="proxy",
        )
        files = adapter.download(url, request, work)
        path = next((f for f in files if f.suffix.lower() in (".mp4", ".webm", ".mkv", ".mov")), files[0])
        self.cache.set(key, url.normalized, "proxy", {"path": str(path)}, file_path=path)
        return path

    def audio_for_asr(self, url: NormalizedURL, adapter: PlatformAdapter) -> Path:
        key = make_key(url.normalized, "asr-audio", {})
        if (hit := self.cache.get(key)) and Path(hit["path"]).exists():
            return Path(hit["path"])
        work = self.cache.work_dir(key)
        request = DownloadRequest(url=url.normalized, mode="audio", audio_format="best", filename="audio")
        files = adapter.download(url, request, work)
        self.cache.set(key, url.normalized, "asr-audio", {"path": str(files[0])}, file_path=files[0])
        return files[0]
