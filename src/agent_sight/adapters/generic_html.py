"""Fallback adapter for direct media links and plain HTML pages with ``<video>`` tags.

Used for ``.mp4/.webm/.m3u8`` URLs and for pages yt-dlp has no extractor for. It reads
``<video>/<source>/<track>`` tags, OpenGraph/Twitter card meta tags and JSON-LD
``contentUrl`` values. Encrypted HLS (``#EXT-X-KEY`` with a key format other than
plain AES-128 identity) is rejected as DRM.
"""

from __future__ import annotations

import html
import re
import threading
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import httpx

from agent_sight.adapters.base import PlatformAdapter, StreamSource, SubtitleTrack, with_retries
from agent_sight.adapters.router import MEDIA_EXTENSIONS, check_public_host
from agent_sight.config import Settings
from agent_sight.downloader.ffprobe_utils import header_args, probe_media, run_ffmpeg
from agent_sight.errors import AgentError, ErrorCode
from agent_sight.models import DownloadRequest, NormalizedURL, Platform, VideoMetadata

_MEDIA_RE = re.compile(
    r"""https?:(?:\\?/){2}[^\s"'<>]+?\.(?:mp4|m4v|webm|mov|m3u8)(?:\?[^\s"'<>]*)?(?=["'\s<>]|$)""",
    re.I,
)
_CONTENT_URL = re.compile(r'"(?:contentUrl|embedUrl)"\s*:\s*"([^"]+)"')
_DRM_KEY = re.compile(r"#EXT-X-(?:SESSION-)?KEY:[^\n]*KEYFORMAT=\"(?!identity)", re.I)


@dataclass
class _PageInfo:
    media: list[str] = field(default_factory=list)
    tracks: list[tuple[str, str]] = field(default_factory=list)  # (lang, url)
    title: str | None = None
    description: str | None = None
    thumbnail: str | None = None


class _VideoTagParser(HTMLParser):
    def __init__(self, base: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base = base
        self.info = _PageInfo()
        self._in_title = False
        self._title_parts: list[str] = []
        self._in_video = False

    def _add(self, url: str | None) -> None:
        if not url or url.startswith(("blob:", "data:", "javascript:")):
            return
        full = urljoin(self.base, html.unescape(url.strip()))
        if full not in self.info.media:
            self.info.media.append(full)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = {k.lower(): (v or "") for k, v in attrs}
        if tag == "video":
            self._in_video = True
            self._add(a.get("src"))
            if a.get("poster") and not self.info.thumbnail:
                self.info.thumbnail = urljoin(self.base, a["poster"])
        elif tag == "source" and (self._in_video or "video" in a.get("type", "")):
            self._add(a.get("src"))
        elif tag == "track" and a.get("kind", "subtitles") in ("subtitles", "captions") and a.get("src"):
            self.info.tracks.append((a.get("srclang") or "und", urljoin(self.base, a["src"])))
        elif tag == "meta":
            prop = (a.get("property") or a.get("name") or "").lower()
            content = a.get("content")
            if prop in ("og:video", "og:video:url", "og:video:secure_url", "twitter:player:stream"):
                self._add(content)
            elif prop == "og:title" and content:
                self.info.title = content
            elif prop in ("og:description", "description") and content and not self.info.description:
                self.info.description = content
            elif prop == "og:image" and content and not self.info.thumbnail:
                self.info.thumbnail = urljoin(self.base, content)
        elif tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "video":
            self._in_video = False
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._title_parts.append(data)

    def close(self) -> None:
        super().close()
        if not self.info.title and self._title_parts:
            self.info.title = " ".join("".join(self._title_parts).split()) or None


def extract_page_info(page_html: str, base_url: str) -> _PageInfo:
    parser = _VideoTagParser(base_url)
    try:
        parser.feed(page_html)
        parser.close()
    except Exception:
        pass
    info = parser.info
    for m in _CONTENT_URL.finditer(page_html):
        url = m.group(1).replace("\\/", "/")
        if urlsplit(url).path.lower().endswith(MEDIA_EXTENSIONS):
            parser._add(url)
    for m in _MEDIA_RE.finditer(page_html):
        parser._add(m.group(0).replace("\\/", "/"))
    # Prefer progressive files over manifests, keep discovery order otherwise.
    info.media.sort(key=lambda u: urlsplit(u).path.lower().endswith((".m3u8", ".mpd")))
    return info


class GenericHTMLAdapter(PlatformAdapter):
    name = "generic-html"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._pages: dict[str, _PageInfo] = {}
        self._lock = threading.Lock()

    def can_handle(self, url: NormalizedURL) -> bool:
        return url.platform in (Platform.DIRECT, Platform.GENERIC)

    # ------------------------------------------------------------------ HTTP

    def _headers(self) -> dict[str, str]:
        return {"User-Agent": self.settings.user_agent or "Mozilla/5.0 (agent-sight)"}

    def _guard(self, url: str) -> None:
        if not self.settings.allow_private_hosts:
            check_public_host(urlsplit(url).hostname or "")

    def _client(self) -> httpx.Client:
        def on_request(request: httpx.Request) -> None:
            self._guard(str(request.url))

        return httpx.Client(
            follow_redirects=True, timeout=self.settings.socket_timeout, headers=self._headers(),
            event_hooks={"request": [on_request]},
        )

    def _convert(self, exc: BaseException) -> AgentError:
        if isinstance(exc, AgentError):
            return exc
        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
            if status in (401, 403):
                return AgentError(ErrorCode.ACCESS_DENIED, f"HTTP {status} from {exc.request.url.host}.")
            if status == 404:
                return AgentError(ErrorCode.NOT_FOUND, "HTTP 404: page or media not found.")
            if status == 429:
                return AgentError(ErrorCode.RATE_LIMITED, "HTTP 429: too many requests.")
            if status >= 500:
                return AgentError(ErrorCode.NETWORK_ERROR, f"HTTP {status} from server.")
            return AgentError(ErrorCode.EXTRACTION_FAILED, f"HTTP {status}.")
        if isinstance(exc, httpx.TransportError):
            return AgentError(ErrorCode.NETWORK_ERROR, f"Network error: {exc}")
        return AgentError(ErrorCode.INTERNAL_ERROR, f"{type(exc).__name__}: {exc}")

    def _get_text(self, url: str, limit: int) -> tuple[str, str]:
        def run() -> tuple[str, str]:
            self.limiter.wait(urlsplit(url).hostname or "")
            with self._client() as client, client.stream("GET", url) as resp:
                resp.raise_for_status()
                chunks: list[bytes] = []
                size = 0
                for chunk in resp.iter_bytes():
                    size += len(chunk)
                    if size > limit:
                        break
                    chunks.append(chunk)
                return b"".join(chunks).decode(resp.encoding or "utf-8", "replace"), str(resp.url)

        return with_retries(run, retries=self.settings.max_retries, convert=self._convert)

    def _page(self, url: NormalizedURL) -> _PageInfo:
        with self._lock:
            if url.normalized in self._pages:
                return self._pages[url.normalized]
        if url.platform == Platform.DIRECT:
            info = _PageInfo(media=[url.normalized])
        else:
            text, final = self._get_text(url.normalized, self.settings.max_html_bytes)
            info = extract_page_info(text, final)
            if not info.media:
                raise AgentError(
                    ErrorCode.UNSUPPORTED_URL,
                    "No playable <video>, og:video or media link found on the page.",
                )
        with self._lock:
            self._pages[url.normalized] = info
        return info

    def _media_url(self, url: NormalizedURL) -> str:
        media = self._page(url).media[0]
        self._guard(media)
        if urlsplit(media).path.lower().endswith(".m3u8"):
            playlist, _ = self._get_text(media, 2 * 1024 * 1024)
            if _DRM_KEY.search(playlist) or "SAMPLE-AES" in playlist:
                raise AgentError(ErrorCode.DRM_PROTECTED, "The HLS stream is encrypted with DRM.")
        if urlsplit(media).path.lower().endswith(".mpd"):
            raise AgentError(
                ErrorCode.UNSUPPORTED_URL, "DASH manifests are only supported through yt-dlp."
            )
        return media

    # ------------------------------------------------------------------ adapter API

    def probe(self, url: NormalizedURL) -> VideoMetadata:
        page = self._page(url)
        media_url = self._media_url(url)
        info = probe_media(media_url, headers=self._headers(), settings=self.settings)
        video = info.video
        name = Path(urlsplit(media_url).path).name
        return VideoMetadata(
            url=url.normalized,
            platform=url.platform,
            extractor=self.name,
            id=url.video_id or Path(name).stem or None,
            title=page.title or name or None,
            description=page.description,
            duration=info.duration,
            thumbnail=page.thumbnail,
            width=video.width if video else None,
            height=video.height if video else None,
            fps=video.fps if video else None,
            subtitle_languages=sorted({lang for lang, _ in page.tracks}),
            estimated_size_bytes=info.size_bytes
            or (int(info.bitrate * info.duration / 8) if info.bitrate and info.duration else None),
        )

    def list_subtitles(self, url: NormalizedURL, languages: list[str]) -> list[SubtitleTrack]:
        tracks = self._page(url).tracks
        if not tracks:
            return []
        wanted = [lang.split("-")[0].lower() for lang in languages]
        tracks = sorted(
            tracks, key=lambda t: wanted.index(t[0].split("-")[0].lower())
            if t[0].split("-")[0].lower() in wanted else len(wanted),
        )
        lang, track_url = tracks[0]
        self._guard(track_url)
        content, _ = self._get_text(track_url, 5 * 1024 * 1024)
        ext = "srt" if track_url.lower().split("?")[0].endswith(".srt") else "vtt"
        return [SubtitleTrack(language=lang, ext=ext, content=content, automatic=False)]

    def stream_source(self, url: NormalizedURL, max_height: int) -> StreamSource:
        media = self._media_url(url)
        return StreamSource(
            url=media, headers=self._headers(),
            is_hls=urlsplit(media).path.lower().endswith(".m3u8"),
        )

    def download(self, url: NormalizedURL, request: DownloadRequest, dest_dir: Path) -> list[Path]:
        dest_dir.mkdir(parents=True, exist_ok=True)
        page = self._page(url)
        fallback = Path(urlsplit(page.media[0]).path).stem if page.media else None
        stem = request.filename or re.sub(
            r"[^\w\-]+", "_", (page.title or url.video_id or fallback or "video"))[:80].strip("_") or "video"
        if request.is_clip:
            stem += f"_{int(request.start or 0)}-{int(request.end) if request.end else 'end'}"

        if request.mode == "subtitles":
            tracks = self.list_subtitles(url, request.subtitle_languages)
            if not tracks:
                raise AgentError(ErrorCode.NO_TRANSCRIPT, "The page exposes no <track> subtitles.")
            out = dest_dir / f"{stem}.{tracks[0].language}.{tracks[0].ext}"
            out.write_text(tracks[0].content, encoding="utf-8")
            return [out]

        media = self._media_url(url)
        hdr = header_args(self._headers())
        limit = str(self.settings.max_download_bytes)

        if request.mode == "thumbnail":
            out = dest_dir / f"{stem}.jpg"
            run_ffmpeg(["-y", *hdr, "-ss", "1", "-i", media, "-frames:v", "1", "-q:v", "2", str(out)],
                       settings=self.settings)
            return [out]

        is_hls = urlsplit(media).path.lower().endswith(".m3u8")
        if request.mode == "video" and not request.is_clip and not is_hls:
            ext = Path(urlsplit(media).path).suffix or ".mp4"
            out = dest_dir / f"{stem}{ext}"
            self._stream_to_file(media, out)
            return [out]

        seek: list[str] = []
        if request.start:
            seek += ["-ss", f"{request.start:.3f}"]
        if request.end is not None:
            seek += ["-to", f"{request.end:.3f}"] if not request.start else ["-t", f"{request.end - request.start:.3f}"]

        if request.mode == "audio":
            codec, ext = {
                "m4a": (["-c:a", "aac", "-b:a", "160k"], "m4a"),
                "mp3": (["-c:a", "libmp3lame", "-q:a", "3"], "mp3"),
                "opus": (["-c:a", "libopus", "-b:a", "96k"], "opus"),
                "wav": (["-c:a", "pcm_s16le"], "wav"),
                "best": (["-c:a", "copy"], "mka"),
            }[request.audio_format]
            out = dest_dir / f"{stem}.{ext}"
            run_ffmpeg(["-y", *hdr, *seek, "-i", media, "-vn", *codec, "-fs", limit, str(out)],
                       settings=self.settings)
            return [out]

        container = request.container or "mp4"
        out = dest_dir / f"{stem}.{container}"
        if request.is_clip:
            codec = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-c:a", "aac"]
            if container == "webm":
                codec = ["-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "33", "-c:a", "libopus"]
        else:
            codec = ["-c", "copy"] + (["-bsf:a", "aac_adtstoasc"] if container == "mp4" else [])
        run_ffmpeg(["-y", *hdr, *seek, "-i", media, *codec, "-fs", limit, str(out)],
                   settings=self.settings)
        return [out]

    def _stream_to_file(self, media: str, out: Path) -> None:
        limit = self.settings.max_download_bytes
        tmp = out.with_suffix(out.suffix + ".part")

        def run() -> None:
            self.limiter.wait(urlsplit(media).hostname or "")
            with self._client() as client, client.stream("GET", media) as resp:
                resp.raise_for_status()
                declared = int(resp.headers.get("content-length") or 0)
                if declared > limit:
                    raise AgentError(
                        ErrorCode.FILE_TOO_LARGE,
                        f"File is {declared / 1e6:.0f} MB, above the {limit / 1e6:.0f} MB limit.",
                    )
                size = 0
                with tmp.open("wb") as fh:
                    for chunk in resp.iter_bytes(1 << 16):
                        size += len(chunk)
                        if size > limit:
                            raise AgentError(ErrorCode.FILE_TOO_LARGE, "Download exceeded the size limit.")
                        fh.write(chunk)
            tmp.replace(out)

        try:
            with_retries(run, retries=self.settings.max_retries, convert=self._convert)
        finally:
            tmp.unlink(missing_ok=True)
