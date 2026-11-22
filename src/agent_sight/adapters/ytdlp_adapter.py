"""Primary extractor built on the yt-dlp Python API (1800+ sites)."""

from __future__ import annotations

import copy
import logging
import os
import re
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any

from agent_sight.adapters.base import PlatformAdapter, StreamSource, SubtitleTrack, with_retries
from agent_sight.config import Settings
from agent_sight.downloader.ffprobe_utils import ffmpeg_dir_for_ytdlp
from agent_sight.errors import AgentError, ErrorCode, from_extractor_error
from agent_sight.models import (
    Chapter,
    DownloadRequest,
    FormatSummary,
    NormalizedURL,
    Platform,
    VideoMetadata,
)

log = logging.getLogger("agent_sight.ytdlp")

_HEIGHTS = {"1080p": 1080, "720p": 720, "480p": 480, "360p": 360}
_SUB_EXT_PREFERENCE = ("json3", "vtt", "srt")
_INFO_TTL = 15 * 60  # signed stream URLs expire; keep raw info dicts briefly in-process


class _QuietLogger:
    def debug(self, msg: str) -> None:
        if not msg.startswith("[debug] "):
            log.debug(msg)

    def info(self, msg: str) -> None:
        log.debug(msg)

    def warning(self, msg: str) -> None:
        log.info(msg)

    def error(self, msg: str) -> None:
        log.debug(msg)


def detect_js_runtimes() -> dict[str, dict[str, str]]:
    """Find JS runtimes yt-dlp can use (PATH plus the active virtualenv's bin directory)."""
    bindir = Path(sys.prefix) / ("Scripts" if os.name == "nt" else "bin")
    found: dict[str, dict[str, str]] = {}
    for name in ("deno", "node", "bun", "quickjs"):
        path = shutil.which(name) or shutil.which(name, path=str(bindir))
        if path:
            found[name] = {"path": path}
    return found


def format_selector(quality: str | None, mode: str) -> str:
    if mode == "audio":
        return "ba/b"
    if quality in (None, "best"):
        return "bv*+ba/b"
    if quality == "worst":
        return "wv*+wa/w"
    h = _HEIGHTS[quality]
    return f"bv*[height<={h}]+ba/b[height<={h}]/wv*+ba/w"


def _lang_matches(key: str, wanted: str) -> bool:
    k, w = key.lower(), wanted.lower()
    return k == w or k.startswith(w + "-") or k.split("-")[0] == w.split("-")[0]


class YtDlpAdapter(PlatformAdapter):
    name = "yt-dlp"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._info_cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ plumbing

    def can_handle(self, url: NormalizedURL) -> bool:
        return url.platform != Platform.DIRECT

    def _opts(self, **extra: Any) -> dict[str, Any]:
        s = self.settings
        opts: dict[str, Any] = {
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            "logger": _QuietLogger(),
            "noplaylist": True,
            "extract_flat": "in_playlist",
            "socket_timeout": s.socket_timeout,
            "retries": s.max_retries,
            "fragment_retries": s.max_retries,
            "extractor_retries": 1,
            "sleep_interval_requests": min(s.min_request_interval, 2.0) or None,
            "color": {"stdout": "never", "stderr": "never"},
        }
        runtimes = detect_js_runtimes()
        if runtimes:
            opts["js_runtimes"] = runtimes
        ffdir = ffmpeg_dir_for_ytdlp(s)
        if ffdir:
            opts["ffmpeg_location"] = ffdir
        if s.cookies_file:
            if not s.cookies_file.is_file():
                raise AgentError(
                    ErrorCode.INVALID_ARGUMENT,
                    "AGENT_SIGHT_COOKIES_FILE does not point to a readable file.",
                    "Fix or unset AGENT_SIGHT_COOKIES_FILE (path is not shown to protect privacy).",
                )
            opts["cookiefile"] = str(s.cookies_file)
        if s.user_agent:
            opts["http_headers"] = {"User-Agent": s.user_agent}
        opts.update(extra)
        return opts

    def _ydl(self, **extra: Any) -> Any:
        from yt_dlp import YoutubeDL

        return YoutubeDL(self._opts(**extra))  # type: ignore[arg-type]

    def _raw_info(self, url: NormalizedURL, *, refresh: bool = False) -> dict[str, Any]:
        """Extract (unprocessed) info once and reuse it for metadata, subs and streams."""
        key = url.normalized
        with self._lock:
            hit = self._info_cache.get(key)
        if hit and not refresh and time.time() - hit[0] < _INFO_TTL:
            return hit[1]

        def run() -> dict[str, Any]:
            self.limiter.wait(url.host)
            with self._ydl() as ydl:
                info = ydl.extract_info(url.normalized, download=False, process=False)
                return self._resolve_single(ydl, info, url)

        info = with_retries(
            run, retries=self.settings.max_retries,
            convert=lambda e: from_extractor_error(e, url.normalized),
        )
        with self._lock:
            self._info_cache[key] = (time.time(), info)
        return info

    def _resolve_single(self, ydl: Any, info: dict[str, Any], url: NormalizedURL) -> dict[str, Any]:
        for _ in range(3):
            kind = info.get("_type", "video")
            if kind in ("url", "url_transparent"):
                info = ydl.extract_info(info["url"], download=False, process=False)
                continue
            if kind == "playlist":
                entries = list(info.get("entries") or [])
                if len(entries) == 1 or (entries and url.platform in (Platform.X, Platform.REDDIT)):
                    info = entries[0]
                    continue
                sample = [e.get("url") or e.get("webpage_url") for e in entries[:5] if e]
                raise AgentError(
                    ErrorCode.UNSUPPORTED_URL,
                    f"URL is a playlist/channel with {len(entries)} entries, not a single video.",
                    "Pick one entry URL from details.sample_entries and call the tool again.",
                    details={"sample_entries": sample, "entry_count": len(entries)},
                )
            return info
        raise AgentError(ErrorCode.EXTRACTION_FAILED, "Too many redirects while resolving the URL.")

    def _processed(self, url: NormalizedURL, **opts: Any) -> dict[str, Any]:
        """Run format selection on a copy of the cached raw info with the given options."""
        raw = copy.deepcopy(self._raw_info(url))

        def run() -> dict[str, Any]:
            with self._ydl(**opts) as ydl:
                return ydl.process_ie_result(raw, download=False)  # type: ignore[no-any-return]

        return with_retries(
            run, retries=0, convert=lambda e: from_extractor_error(e, url.normalized)
        )

    # ------------------------------------------------------------------ metadata

    def probe(self, url: NormalizedURL) -> VideoMetadata:
        raw = self._raw_info(url)
        if raw.get("is_live") or raw.get("live_status") in ("is_live", "is_upcoming"):
            meta = self._to_metadata(url, raw, processed=None)
            meta.is_live = True
            return meta
        formats = raw.get("formats") or []
        if formats and all(f.get("has_drm") for f in formats):
            meta = self._to_metadata(url, raw, processed=None)
            meta.has_drm = True
            return meta
        try:
            processed = self._processed(url, format=format_selector(self.settings.default_quality, "video"))
        except AgentError as err:
            if err.code in (ErrorCode.EXTRACTION_FAILED, ErrorCode.UNSUPPORTED_URL):
                processed = None
            else:
                raise
        return self._to_metadata(url, raw, processed)

    def _to_metadata(
        self, url: NormalizedURL, raw: dict[str, Any], processed: dict[str, Any] | None
    ) -> VideoMetadata:
        info = processed or raw
        duration = info.get("duration")
        chapters = [
            Chapter(start=float(c.get("start_time") or 0), end=c.get("end_time"),
                    title=str(c.get("title") or f"Chapter {i + 1}"))
            for i, c in enumerate(info.get("chapters") or [])
        ]
        subs = sorted((info.get("subtitles") or {}).keys())
        subs = [s for s in subs if s != "live_chat"]
        autos = sorted(k for k in (info.get("automatic_captions") or {}) if k.endswith("-orig"))
        if not autos and info.get("automatic_captions"):
            autos = sorted(info["automatic_captions"].keys())[:10]
        estimated = None
        if processed:
            req = processed.get("requested_formats") or [processed]
            sizes = [f.get("filesize") or f.get("filesize_approx") for f in req]
            if all(sizes):
                estimated = int(sum(sizes))  # type: ignore[arg-type]
        desc = info.get("description")
        return VideoMetadata(
            url=url.normalized,
            platform=url.platform,
            extractor=str(info.get("extractor_key") or info.get("ie_key") or "unknown"),
            id=str(info["id"]) if info.get("id") is not None else url.video_id,
            title=info.get("title") or info.get("fulltitle"),
            uploader=info.get("uploader") or info.get("uploader_id"),
            channel=info.get("channel"),
            upload_date=info.get("upload_date"),
            duration=float(duration) if duration else None,
            view_count=info.get("view_count"),
            like_count=info.get("like_count"),
            comment_count=info.get("comment_count"),
            description=(desc[:2000] + "…") if desc and len(desc) > 2000 else desc,
            tags=list(info.get("tags") or [])[:25],
            thumbnail=info.get("thumbnail"),
            width=info.get("width"),
            height=info.get("height"),
            fps=info.get("fps"),
            is_live=bool(info.get("is_live")),
            age_limit=info.get("age_limit"),
            chapters=chapters,
            subtitle_languages=subs,
            auto_caption_languages=autos,
            formats=self._summarize_formats(raw.get("formats") or []),
            has_drm=any(f.get("has_drm") for f in raw.get("formats") or []),
            estimated_size_bytes=estimated,
        )

    @staticmethod
    def _summarize_formats(formats: list[dict[str, Any]], limit: int = 12) -> list[FormatSummary]:
        """Keep the best format per (kind, height) so metadata stays small."""
        best: dict[tuple[str, int | None], dict[str, Any]] = {}
        for f in formats:
            if f.get("has_drm") or (f.get("format_note") or "").lower() == "storyboard":
                continue
            v, a = f.get("vcodec"), f.get("acodec")
            has_v = v not in (None, "none")
            has_a = a not in (None, "none")
            kind = "video+audio" if has_v and has_a else "video" if has_v else "audio" if has_a else "unknown"
            if kind == "unknown" and f.get("ext") in ("mhtml", None):
                continue
            k = (kind, f.get("height"))
            if k not in best or (f.get("tbr") or 0) > (best[k].get("tbr") or 0):
                best[k] = f
        out = [
            FormatSummary(
                format_id=str(f.get("format_id") or f.get("ext") or "default"),
                ext=f.get("ext"),
                kind=kind,  # type: ignore[arg-type]
                height=h,
                fps=f.get("fps"),
                vcodec=f.get("vcodec"),
                acodec=f.get("acodec"),
                filesize=f.get("filesize") or f.get("filesize_approx"),
                tbr=f.get("tbr"),
                protocol=f.get("protocol"),
            )
            for (kind, h), f in best.items()
        ]
        out.sort(key=lambda x: (x.kind != "video+audio", -(x.height or 0)))
        return out[:limit]

    # ------------------------------------------------------------------ subtitles

    def list_subtitles(self, url: NormalizedURL, languages: list[str]) -> list[SubtitleTrack]:
        raw = self._raw_info(url)
        native: dict[str, list[dict[str, Any]]] = {
            k: v for k, v in (raw.get("subtitles") or {}).items() if k != "live_chat"
        }
        auto: dict[str, list[dict[str, Any]]] = raw.get("automatic_captions") or {}
        orig_lang = (raw.get("language") or "").lower() or None

        candidates: list[tuple[str, list[dict[str, Any]], bool]] = []
        for want in languages:
            candidates += [(k, v, False) for k, v in native.items() if _lang_matches(k, want)]
        for want in languages:
            ordered = sorted(
                (k for k in auto if _lang_matches(k, want)), key=lambda k: not k.endswith("-orig")
            )
            candidates += [(k, auto[k], True) for k in ordered]
        candidates += [(k, v, False) for k, v in native.items()]
        orig = [k for k in auto if k.endswith("-orig")] or ([orig_lang] if orig_lang in auto else [])
        candidates += [(k, auto[k], True) for k in orig if k]

        seen: set[tuple[str, bool]] = set()
        for lang, tracks, automatic in candidates:
            if (lang, automatic) in seen:
                continue
            seen.add((lang, automatic))
            track = self._pick_track(tracks)
            if not track:
                continue
            try:
                content = self._fetch_text(url, track["url"])
            except AgentError as err:
                if err.code in (ErrorCode.RATE_LIMITED, ErrorCode.LOGIN_REQUIRED):
                    raise
                log.info("subtitle fetch failed for %s: %s", lang, err.message)
                continue
            if content.strip():
                return [SubtitleTrack(
                    language=lang.removesuffix("-orig"), ext=track["ext"], content=content,
                    automatic=automatic,
                )]
        return []

    @staticmethod
    def _pick_track(tracks: list[dict[str, Any]]) -> dict[str, Any] | None:
        by_ext: dict[str, dict[str, Any]] = {}
        for t in tracks:
            if not t.get("url"):
                continue
            # Generic extractors often omit "ext"; infer it from the URL path.
            ext = t.get("ext") or Path(t["url"].split("?", 1)[0]).suffix.lstrip(".").lower() or "vtt"
            by_ext.setdefault(ext, {**t, "ext": ext})
        for ext in _SUB_EXT_PREFERENCE:
            if ext in by_ext:
                return by_ext[ext]
        return None

    def _fetch_text(self, url: NormalizedURL, track_url: str) -> str:
        def run() -> str:
            self.limiter.wait(url.host)
            with self._ydl() as ydl:
                data: bytes = ydl.urlopen(track_url).read()
            return data.decode("utf-8", errors="replace")

        return with_retries(
            run, retries=self.settings.max_retries,
            convert=lambda e: from_extractor_error(e, url.normalized),
        )

    # ------------------------------------------------------------------ streams

    def stream_source(self, url: NormalizedURL, max_height: int) -> StreamSource:
        info = self._processed(
            url,
            format=(
                f"bv*[height<={max_height}][vcodec!=none]/b[height<={max_height}]/wv*/w"
            ),
            format_sort=[f"res:{max_height}", "proto"],
        )
        fmt = (info.get("requested_formats") or [info])[0]
        stream_url = fmt.get("url")
        if not stream_url:
            raise AgentError(ErrorCode.NO_VIDEO_STREAM, "No directly playable video stream found.")
        if fmt.get("vcodec") == "none":
            raise AgentError(ErrorCode.NO_VIDEO_STREAM, "This media only has audio.")
        protocol = str(fmt.get("protocol") or "")
        return StreamSource(
            url=stream_url,
            headers=dict(fmt.get("http_headers") or info.get("http_headers") or {}),
            height=fmt.get("height"),
            is_hls="m3u8" in protocol,
        )

    # ------------------------------------------------------------------ downloads

    def download(self, url: NormalizedURL, request: DownloadRequest, dest_dir: Path) -> list[Path]:
        from yt_dlp.utils import download_range_func

        dest_dir.mkdir(parents=True, exist_ok=True)
        raw = self._raw_info(url)
        if raw.get("is_live") or raw.get("live_status") == "is_live":
            raise AgentError(ErrorCode.LIVE_STREAM, "The video is currently live.")
        formats = raw.get("formats") or []
        if request.mode in ("video", "audio") and formats and all(f.get("has_drm") for f in formats):
            raise AgentError(ErrorCode.DRM_PROTECTED, "All available formats are DRM-protected.")

        stem = request.filename or _safe_stem(raw.get("title"), raw.get("id"))
        if request.is_clip:
            stem += f"_{_ts(request.start or 0)}-{_ts(request.end) if request.end else 'end'}"
        opts: dict[str, Any] = {
            "outtmpl": {"default": str(dest_dir / f"{stem}.%(ext)s")},
            "max_filesize": self.settings.max_download_bytes,
            "overwrites": True,
            "restrictfilenames": True,
        }
        pps: list[dict[str, Any]] = []
        if request.mode == "video":
            opts["format"] = format_selector(request.quality or self.settings.default_quality, "video")
            if request.container:
                opts["merge_output_format"] = request.container
                if request.container == "mp4":
                    opts["format_sort"] = ["vcodec:h264", "acodec:aac", "ext:mp4:m4a"]
        elif request.mode == "audio":
            if request.audio_format == "m4a":
                opts["format"] = "ba[ext=m4a]/ba/b"
            else:
                opts["format"] = "ba/b"
            if request.audio_format != "best":
                pps.append({"key": "FFmpegExtractAudio", "preferredcodec": request.audio_format})
        elif request.mode == "subtitles":
            opts.update(
                skip_download=True, writesubtitles=True, writeautomaticsub=True,
                subtitleslangs=request.subtitle_languages or ["en"], subtitlesformat="vtt/srt/best",
            )
        elif request.mode == "thumbnail":
            opts.update(skip_download=True, writethumbnail=True)
            pps.append({"key": "FFmpegThumbnailsConvertor", "format": "jpg", "when": "before_dl"})
        if pps:
            opts["postprocessors"] = pps
        if request.is_clip and request.mode in ("video", "audio"):
            end = request.end if request.end is not None else float("inf")
            opts["download_ranges"] = download_range_func(None, [(request.start or 0.0, end)])
            opts["force_keyframes_at_cuts"] = True

        if request.mode in ("video", "audio"):
            self._check_size(url, opts, raw, request)

        def run() -> None:
            self.limiter.wait(url.host)
            with self._ydl(**opts) as ydl:
                ydl.process_ie_result(copy.deepcopy(raw), download=True)

        started = time.time() - 2
        with_retries(
            run, retries=self.settings.max_retries,
            convert=lambda e: from_extractor_error(e, url.normalized),
        )
        # Only files written by this run (an earlier full download may share the stem).
        files = sorted(
            p for p in dest_dir.glob(f"{glob_escape(stem)}.*")
            if p.is_file() and p.stat().st_mtime >= started
            and not p.name.endswith((".part", ".ytdl", ".temp")) and ".part-" not in p.name
        )
        if not files:
            if request.mode == "subtitles":
                raise AgentError(ErrorCode.NO_TRANSCRIPT, "No subtitles exist for the requested languages.")
            raise AgentError(
                ErrorCode.FILE_TOO_LARGE if opts.get("max_filesize") else ErrorCode.EXTRACTION_FAILED,
                "Download produced no file (it may exceed the size limit).",
            )
        return files

    def _check_size(
        self, url: NormalizedURL, opts: dict[str, Any], raw: dict[str, Any], request: DownloadRequest
    ) -> None:
        try:
            info = self._processed(url, format=opts["format"], format_sort=opts.get("format_sort", []))
        except AgentError:
            return  # The real download will surface a precise error.
        req = info.get("requested_formats") or [info]
        sizes = [f.get("filesize") or f.get("filesize_approx") for f in req]
        if not all(sizes):
            return
        total = float(sum(sizes))  # type: ignore[arg-type]
        duration = raw.get("duration")
        if request.is_clip and duration:
            end = request.end if request.end is not None else duration
            total *= max(0.0, min(end, duration) - (request.start or 0)) / duration
        if total > self.settings.max_download_bytes:
            raise AgentError(
                ErrorCode.FILE_TOO_LARGE,
                f"Estimated size {total / 1e6:.0f} MB exceeds the "
                f"{self.settings.max_download_mb} MB limit.",
                details={"estimated_bytes": int(total)},
            )


def _safe_stem(title: Any, vid: Any) -> str:
    base = re.sub(r"[^\w\- ]+", "", str(title or "video"), flags=re.UNICODE).strip()
    base = re.sub(r"\s+", "_", base)[:80] or "video"
    return f"{base}-{vid}" if vid else base


def _ts(seconds: float) -> str:
    whole = int(seconds)
    return f"{whole // 3600:02d}{whole % 3600 // 60:02d}{whole % 60:02d}"


def glob_escape(text: str) -> str:
    return re.sub(r"([\[\]*?])", r"[\1]", text)
