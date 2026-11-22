"""``AgentSight``: the single facade used by the CLI and the MCP server.

Progressive disclosure, cheapest first:

1. ``probe``          metadata only, no media bytes
2. ``digest``         metadata + budgeted transcript + scene timeline + one contact sheet
3. ``watch_segment``  dense frames + transcript for one time range
4. ``download``       the exact file, format or clip the agent needs
"""

from __future__ import annotations

import logging
import shlex
from collections.abc import Callable
from pathlib import Path
from typing import Literal, TypeVar

from agent_sight.adapters.base import PlatformAdapter, StreamSource
from agent_sight.adapters.generic_html import GenericHTMLAdapter
from agent_sight.adapters.router import check_public_host, normalize_url
from agent_sight.adapters.ytdlp_adapter import YtDlpAdapter
from agent_sight.audio import transcriber
from agent_sight.audio.transcripts import (
    merge_segments,
    paginate,
    parse_captions,
    render_timestamped,
)
from agent_sight.cache import Cache, make_key
from agent_sight.config import Settings, get_settings
from agent_sight.downloader.ffprobe_utils import extract_audio_for_asr, probe_media
from agent_sight.downloader.media_downloader import MediaDownloader
from agent_sight.errors import AgentError, ErrorCode
from agent_sight.models import (
    ContactSheet,
    Digest,
    DownloadRequest,
    DownloadResult,
    Frame,
    FramesResult,
    NextAction,
    NormalizedURL,
    Platform,
    ProbeResult,
    SceneCut,
    ScenesResult,
    SegmentResult,
    Transcript,
    TranscriptResult,
    VideoMetadata,
    estimate_text_tokens,
    format_timestamp,
)
from agent_sight.vision.contact_sheet import build_contact_sheets
from agent_sight.vision.frame_extractor import MAX_FRAMES, extract_frames, plan_timestamps
from agent_sight.vision.scene_detector import (
    detect_scenes_ffmpeg,
    detect_scenes_pyscenedetect,
    pyscenedetect_available,
    select_keyframes,
)

log = logging.getLogger("agent_sight")
T = TypeVar("T")
SourceKind = Literal["stream", "proxy", "local"]

# Errors after which the next adapter in the chain is tried.
_FALLBACK = {ErrorCode.UNSUPPORTED_URL, ErrorCode.EXTRACTION_FAILED, ErrorCode.NOT_FOUND}


def cli_hint(*parts: str | float | int | None) -> str:
    return " ".join(
        ["agent-sight", *(shlex.quote(str(p)) if not str(p).startswith("--") else str(p)
                          for p in parts if p is not None)]
    )


class AgentSight:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.cache = Cache(self.settings.cache_dir, self.settings.cache_ttl_seconds)
        self.ytdlp = YtDlpAdapter(self.settings)
        self.generic = GenericHTMLAdapter(self.settings)
        self.downloader = MediaDownloader(self.settings, self.cache)
        self._adapter_by_url: dict[str, PlatformAdapter] = {}

    # ================================================================== resolution

    def resolve(self, url: str) -> NormalizedURL:
        nurl = normalize_url(url)
        if not self.settings.allow_private_hosts:
            check_public_host(nurl.host)
        return nurl

    def _chain(self, nurl: NormalizedURL) -> list[PlatformAdapter]:
        if nurl.normalized in self._adapter_by_url:
            return [self._adapter_by_url[nurl.normalized]]
        if nurl.platform == Platform.DIRECT:
            return [self.generic]
        if nurl.platform == Platform.GENERIC:
            return [self.ytdlp, self.generic]
        return [self.ytdlp]

    def _run(self, nurl: NormalizedURL, fn: Callable[[PlatformAdapter], T]) -> T:
        chain = self._chain(nurl)
        last: AgentError | None = None
        for i, adapter in enumerate(chain):
            try:
                result = fn(adapter)
                self._adapter_by_url[nurl.normalized] = adapter
                return result
            except AgentError as err:
                last = err
                if err.code in _FALLBACK and i < len(chain) - 1:
                    log.info("%s failed (%s); trying %s", adapter.name, err.code, chain[i + 1].name)
                    continue
                raise
        assert last is not None
        raise last

    def _adapter(self, nurl: NormalizedURL) -> PlatformAdapter:
        self.metadata(nurl)
        return self._adapter_by_url.get(nurl.normalized) or self._chain(nurl)[0]

    # ================================================================== stage 1: probe

    def metadata(self, nurl: NormalizedURL, *, refresh: bool = False) -> tuple[VideoMetadata, bool]:
        key = make_key(nurl.normalized, "probe")
        if not refresh and (hit := self.cache.get(key)):
            adapter = self.ytdlp if hit["adapter"] == self.ytdlp.name else self.generic
            self._adapter_by_url.setdefault(nurl.normalized, adapter)
            return VideoMetadata.model_validate(hit["metadata"]), True

        def probe(adapter: PlatformAdapter) -> tuple[str, VideoMetadata]:
            return adapter.name, adapter.probe(nurl)

        name, meta = self._run(nurl, probe)
        if meta.duration is None and not meta.is_live and not meta.has_drm:
            self._fill_from_stream(nurl, meta)
        self.cache.set(key, nurl.normalized, "probe", {"adapter": name, "metadata": meta.model_dump()})
        return meta, False

    def _fill_from_stream(self, nurl: NormalizedURL, meta: VideoMetadata) -> None:
        """Generic extractors often omit duration/size; read them from the stream itself."""
        adapter = self._adapter_by_url.get(nurl.normalized) or self._chain(nurl)[0]
        try:
            src = adapter.stream_source(nurl, self.settings.proxy_max_height * 2)
            info = probe_media(src.url, headers=src.headers, settings=self.settings, timeout=30)
        except AgentError as err:
            log.info("could not probe stream for duration: %s", err.message)
            return
        meta.duration = info.duration
        if (v := info.video) is not None:
            meta.width = meta.width or v.width
            meta.height = meta.height or v.height
            meta.fps = meta.fps or v.fps
        meta.estimated_size_bytes = meta.estimated_size_bytes or info.size_bytes

    def probe(self, url: str, *, refresh: bool = False) -> ProbeResult:
        nurl = self.resolve(url)
        meta, cached = self.metadata(nurl, refresh=refresh)
        if meta.subtitle_languages:
            source: Literal["native", "auto", "whisper", "none"] = "native"
        elif meta.auto_caption_languages:
            source = "auto"
        elif self.settings.whisper_enabled and transcriber.available_backend():
            source = "whisper"
        else:
            source = "none"
        est = None
        if meta.duration and source != "none":
            est = int(meta.duration / 60 * 150 * 1.35)  # ~150 spoken words/min, ~1.35 tokens/word
        return ProbeResult(
            metadata=meta, has_transcript=source in ("native", "auto"), transcript_source=source,
            estimated_transcript_tokens=est, next_actions=self._probe_actions(meta, source),
            cached=cached,
        )

    def _probe_actions(self, meta: VideoMetadata, source: str) -> list[NextAction]:
        u = meta.url
        acts = [NextAction(
            tool="digest_video", reason="Skim the whole video (transcript + contact sheet) cheaply.",
            cli=cli_hint("digest", u, "--json"),
        )]
        if source != "none":
            acts.append(NextAction(
                tool="get_transcript", reason="Read the spoken content with timestamps.",
                cli=cli_hint("transcript", u, "--json"),
            ))
        if meta.has_drm:
            return [NextAction(
                tool="none", reason="Media is DRM-protected; only metadata is available.",
                cli=cli_hint("probe", u, "--json"),
            )]
        acts.append(NextAction(
            tool="download_media", reason="Save the file or a clip locally.",
            cli=cli_hint("download", u, "--quality", "720p", "--json"),
        ))
        return acts

    # ================================================================== transcripts

    def get_transcript_obj(
        self, nurl: NormalizedURL, *, languages: list[str], allow_whisper: bool, refresh: bool = False
    ) -> tuple[Transcript | None, bool]:
        """Native/auto captions first, then (optionally) local Whisper. Both are cached."""
        key = make_key(nurl.normalized, "transcript", {"langs": languages})
        hit = None if refresh else self.cache.get(key)
        if hit is not None and hit.get("segments") is not None:
            return Transcript.model_validate(hit), True
        if hit is None:
            self.metadata(nurl)
            tracks = self._adapter(nurl).list_subtitles(nurl, languages)
            native: Transcript | None = None
            if tracks:
                t = tracks[0]
                segs = parse_captions(t.content, t.ext)
                if segs:
                    native = Transcript(
                        language=t.language, source="auto" if t.automatic else "native",
                        segments=segs,
                    )
            # A "no captions" marker is cached too, so repeat calls skip the network.
            self.cache.set(key, nurl.normalized, "transcript",
                           native.model_dump() if native else {"segments": None})
            if native:
                return native, False
        if not allow_whisper:
            return None, hit is not None
        wkey = make_key(nurl.normalized, "whisper", {"model": self.settings.whisper_model})
        if not refresh and (whit := self.cache.get(wkey)):
            return Transcript.model_validate(whit), True
        transcript = self._whisper(nurl, self._adapter(nurl), languages[0] if languages else None)
        self.cache.set(wkey, nurl.normalized, "whisper", transcript.model_dump(), ttl=0)
        return transcript, False

    def _whisper(self, nurl: NormalizedURL, adapter: PlatformAdapter, language: str | None) -> Transcript:
        if transcriber.available_backend() is None:
            raise AgentError(
                ErrorCode.NO_TRANSCRIPT,
                "The video has no captions and no local Whisper backend is installed.",
                "pip install 'agent-sight[whisper]' to enable local transcription, or use "
                "digest/watch for visual analysis.",
            )
        media = self.downloader.cached_proxy(nurl) or self.downloader.audio_for_asr(nurl, adapter)
        wav = self.cache.work_dir(make_key(nurl.normalized, "asr-wav")) / "audio16k.wav"
        if not wav.exists():
            extract_audio_for_asr(media, wav, settings=self.settings)
        return transcriber.transcribe(wav, language=language, settings=self.settings)

    def transcript(
        self,
        url: str,
        *,
        language: str = "en",
        start: float | None = None,
        end: float | None = None,
        fmt: Literal["timestamped", "plain", "segments"] = "timestamped",
        max_tokens: int = 4000,
        allow_whisper: bool | None = None,
        refresh: bool = False,
    ) -> TranscriptResult:
        nurl = self.resolve(url)
        whisper = self.settings.whisper_enabled if allow_whisper is None else allow_whisper
        transcript, cached = self.get_transcript_obj(
            nurl, languages=_languages(language), allow_whisper=whisper, refresh=refresh
        )
        if transcript is None:
            raise AgentError(ErrorCode.NO_TRANSCRIPT, "No captions are available for this video.")
        _check_range(start, end, None)
        sliced = transcript.slice(start, end)
        # Raw cues for "segments"; paragraph blocks (fewer timestamps, fewer tokens) otherwise.
        blocks = sliced.segments if fmt == "segments" else merge_segments(sliced.segments, window=20.0)
        page, next_start = paginate(blocks, max_tokens)
        text = None
        if fmt == "timestamped":
            text = render_timestamped(page)
        elif fmt == "plain":
            text = " ".join(s.text for s in page)
        return TranscriptResult(
            url=nurl.normalized, language=transcript.language, source=transcript.source,
            start=start, end=end if next_start is None else next_start, format=fmt,
            text=text, segments=page if fmt == "segments" else None,
            segment_count=len(page),
            estimated_tokens=estimate_text_tokens(text or " ".join(s.text for s in page)),
            truncated=next_start is not None, next_start=next_start, cached=cached,
        )

    # ================================================================== visual sources

    def _visual_source(
        self, nurl: NormalizedURL, meta: VideoMetadata, *, full_decode: bool
    ) -> tuple[str, dict[str, str], SourceKind]:
        adapter = self._adapter(nurl)
        if cached := self.downloader.cached_proxy(nurl):
            return str(cached), {}, "proxy"
        short = meta.duration is not None and meta.duration <= self.settings.proxy_max_duration
        if full_decode and short:
            return str(self.downloader.proxy(nurl, adapter)), {}, "proxy"
        try:
            src: StreamSource = adapter.stream_source(nurl, self.settings.proxy_max_height * 2)
            return src.url, src.headers, "stream"
        except AgentError as err:
            if short and err.code not in (ErrorCode.DRM_PROTECTED, ErrorCode.LOGIN_REQUIRED):
                return str(self.downloader.proxy(nurl, adapter)), {}, "proxy"
            raise

    def _guard_visual(self, meta: VideoMetadata) -> None:
        if meta.has_drm and not meta.formats:
            raise AgentError(ErrorCode.DRM_PROTECTED, "The video is DRM-protected; frames are unavailable.")
        if meta.is_live:
            raise AgentError(ErrorCode.LIVE_STREAM, "The video is live right now.")

    def _frames_dir(self, nurl: NormalizedURL) -> Path:
        return self.cache.work_dir(make_key(nurl.normalized, "frames"))

    def _extract(
        self, nurl: NormalizedURL, meta: VideoMetadata, timestamps: list[float], width: int,
        scores: dict[float, float | None] | None = None,
    ) -> tuple[list[Frame], SourceKind]:
        source, headers, kind = self._visual_source(nurl, meta, full_decode=False)
        try:
            frames = extract_frames(
                source, timestamps, self._frames_dir(nurl), width=width, headers=headers,
                settings=self.settings, scores=scores,
            )
        except AgentError:
            if kind != "stream":
                raise
            # Signed stream URLs expire or refuse seeks: fall back to a local proxy if feasible.
            if meta.duration and meta.duration <= self.settings.proxy_max_duration:
                source = str(self.downloader.proxy(nurl, self._adapter(nurl)))
                kind = "proxy"
                frames = extract_frames(
                    source, timestamps, self._frames_dir(nurl), width=width,
                    settings=self.settings, scores=scores,
                )
            else:
                raise
        return frames, kind

    def _sheets(
        self, nurl: NormalizedURL, frames: list[Frame], *, header: str | None, columns: int = 3,
        rows: int = 3, tile: int = 384,
    ) -> list[ContactSheet]:
        key = make_key(nurl.normalized, "sheet", {
            "ts": [f.timestamp for f in frames], "c": columns, "r": rows, "t": tile, "h": header,
        })
        return build_contact_sheets(
            frames, self.cache.work_dir(key), columns=columns, rows=rows, tile_long_edge=tile,
            header=header,
        )

    # ================================================================== scenes

    def scenes(
        self,
        url: str,
        *,
        threshold: float = 0.3,
        max_scenes: int = 60,
        start: float | None = None,
        end: float | None = None,
        method: Literal["auto", "ffmpeg", "pyscenedetect", "uniform"] = "auto",
        refresh: bool = False,
    ) -> ScenesResult:
        nurl = self.resolve(url)
        meta, _ = self.metadata(nurl)
        self._guard_visual(meta)
        result = self._scenes(nurl, meta, threshold=threshold, start=start, end=end,
                              method=method, refresh=refresh)
        result.scenes = sorted(result.scenes, key=lambda s: -(s.score or 0))[:max_scenes]
        result.scenes.sort(key=lambda s: s.timestamp)
        return result

    def _scenes(
        self, nurl: NormalizedURL, meta: VideoMetadata, *, threshold: float, start: float | None,
        end: float | None, method: str, refresh: bool = False,
    ) -> ScenesResult:
        key = make_key(nurl.normalized, "scenes", {"t": threshold, "s": start, "e": end, "m": method})
        if not refresh and (hit := self.cache.get(key)):
            res = ScenesResult.model_validate(hit)
            res.cached = True
            return res
        duration = meta.duration
        short = duration is not None and duration <= self.settings.proxy_max_duration
        chosen = method
        if method == "auto":
            chosen = "ffmpeg" if short else "uniform"
        cuts: list[tuple[float, float]] = []
        if chosen in ("ffmpeg", "pyscenedetect"):
            source, headers, kind = self._visual_source(nurl, meta, full_decode=True)
            if chosen == "pyscenedetect":
                if kind == "stream" or not pyscenedetect_available():
                    chosen = "ffmpeg"
                else:
                    cuts = detect_scenes_pyscenedetect(Path(source))
                    cuts = [(t, sc) for t, sc in cuts
                            if (start is None or t >= start) and (end is None or t <= end)]
            if chosen == "ffmpeg":
                cuts = detect_scenes_ffmpeg(
                    source, threshold=threshold, start=start, end=end, headers=headers,
                    settings=self.settings,
                )
        res = ScenesResult(
            url=nurl.normalized,
            method="ffmpeg-scene" if chosen == "ffmpeg" else chosen,  # type: ignore[arg-type]
            threshold=threshold if chosen != "uniform" else None, duration=duration,
            scenes=[SceneCut(timestamp=t, label=format_timestamp(t), score=sc) for t, sc in cuts],
        )
        self.cache.set(key, nurl.normalized, "scenes", res.model_dump())
        return res

    # ================================================================== frames / watch

    def frames(
        self,
        url: str,
        *,
        timestamps: list[float] | None = None,
        start: float | None = None,
        end: float | None = None,
        count: int = 9,
        width: int = 640,
        contact_sheet: bool = True,
        columns: int = 3,
    ) -> FramesResult:
        nurl = self.resolve(url)
        meta, _ = self.metadata(nurl)
        self._guard_visual(meta)
        if timestamps:
            for t in timestamps:
                _check_range(t, None, meta.duration)
            ts = sorted(set(timestamps))
        else:
            lo, hi = _bounds(start, end, meta.duration)
            ts = plan_timestamps(lo, hi, count)
        frames, kind = self._extract(nurl, meta, ts, width)
        sheets = self._sheets(nurl, frames, header=_header(meta), columns=columns) if contact_sheet else []
        tokens = sum(s.estimated_tokens for s in sheets) if sheets else sum(
            (f.width * f.height + 749) // 750 for f in frames)
        return FramesResult(url=nurl.normalized, frames=frames, contact_sheets=sheets,
                            estimated_image_tokens=tokens, source=kind)

    def watch_segment(
        self,
        url: str,
        *,
        start: float,
        end: float,
        frames: int = 6,
        include_transcript: bool = True,
        language: str = "en",
        width: int = 640,
        contact_sheet: bool = True,
    ) -> SegmentResult:
        nurl = self.resolve(url)
        meta, _ = self.metadata(nurl)
        self._guard_visual(meta)
        lo, hi = _bounds(start, end, meta.duration)
        count = max(1, min(frames, MAX_FRAMES))
        cuts: list[tuple[float, float]] = []
        scene_key = make_key(nurl.normalized, "scenes", {"t": 0.3, "s": None, "e": None, "m": "auto"})
        if hit := self.cache.get(scene_key):
            cuts = [(s["timestamp"], s.get("score") or 0.0) for s in hit["scenes"]]
        picks = select_keyframes(cuts, start=lo, end=hi, max_frames=count,
                                 min_gap=min(1.0, (hi - lo) / (count * 2)))
        ts = [t for t, _ in picks] or plan_timestamps(lo, hi, count)
        scores = {t: sc for t, sc in picks}
        extracted, kind = self._extract(nurl, meta, ts, width, scores)
        label = f"{format_timestamp(lo)}-{format_timestamp(hi)}"
        sheets: list[ContactSheet] = []
        if contact_sheet:
            cols = 3 if len(extracted) > 4 else min(2, len(extracted))
            sheets = self._sheets(nurl, extracted, header=f"{_header(meta)}  {label}",
                                  columns=cols, rows=3, tile=480 if len(extracted) <= 4 else 384)
        text = source = None
        if include_transcript:
            try:
                tr, _ = self.get_transcript_obj(nurl, languages=_languages(language), allow_whisper=False)
                if tr:
                    text = render_timestamped(merge_segments(tr.slice(lo, hi).segments, window=15.0))
                    source = tr.source
            except AgentError as err:
                log.info("transcript unavailable for segment: %s", err.message)
        img_tokens = sum(s.estimated_tokens for s in sheets) if sheets else sum(
            (f.width * f.height + 749) // 750 for f in extracted)
        return SegmentResult(
            url=nurl.normalized, start=lo, end=hi, label=label, frames=extracted,
            contact_sheets=sheets, transcript=text or None, transcript_source=source,
            estimated_text_tokens=estimate_text_tokens(text or ""),
            estimated_image_tokens=img_tokens, source=kind,
        )

    # ================================================================== digest

    def digest(
        self,
        url: str,
        *,
        token_budget: int = 1500,
        max_frames: int = 9,
        language: str = "en",
        include_frames: bool = True,
        allow_whisper: bool | None = None,
        refresh: bool = False,
    ) -> Digest:
        from agent_sight.pipeline.digest import build_digest

        nurl = self.resolve(url)
        return build_digest(
            self, nurl, token_budget=token_budget, max_frames=max_frames, language=language,
            include_frames=include_frames, allow_whisper=allow_whisper, refresh=refresh,
        )

    # ================================================================== download

    def download(
        self, request: DownloadRequest, *, output_dir: str | Path | None = None,
        confine: bool = False, refresh: bool = False,
    ) -> DownloadResult:
        nurl = self.resolve(request.url)
        meta, _ = self.metadata(nurl)
        if request.start is not None or request.end is not None:
            _check_range(request.start, request.end, meta.duration)
        if meta.is_live:
            raise AgentError(ErrorCode.LIVE_STREAM, "The video is live right now.")
        adapter = self._adapter(nurl)
        return self.downloader.download(
            nurl, adapter, request, output_dir=output_dir, confine=confine, refresh=refresh
        )

    def media_info(self, path_or_url: str) -> dict[str, object]:
        return probe_media(path_or_url, settings=self.settings).model_dump()


# ====================================================================== helpers


def _languages(language: str) -> list[str]:
    langs = [lang.strip() for lang in language.split(",") if lang.strip()]
    return langs or ["en"]


def _header(meta: VideoMetadata) -> str:
    title = meta.title or meta.id or meta.url
    return f"{title} ({format_timestamp(meta.duration)})" if meta.duration else title


def _check_range(start: float | None, end: float | None, duration: float | None) -> None:
    if start is not None and start < 0:
        raise AgentError(ErrorCode.INVALID_TIME_RANGE, "start must be >= 0.")
    if start is not None and end is not None and end <= start:
        raise AgentError(ErrorCode.INVALID_TIME_RANGE, f"end ({end}) must be greater than start ({start}).")
    if duration is not None:
        if start is not None and start >= duration:
            raise AgentError(
                ErrorCode.INVALID_TIME_RANGE,
                f"start ({start:.1f}s) is past the end of the video ({duration:.1f}s).",
                details={"duration": duration},
            )
        if end is not None and end > duration + 1:
            raise AgentError(
                ErrorCode.INVALID_TIME_RANGE,
                f"end ({end:.1f}s) is past the end of the video ({duration:.1f}s).",
                details={"duration": duration},
            )


def _bounds(start: float | None, end: float | None, duration: float | None) -> tuple[float, float]:
    _check_range(start, end, duration)
    lo = start or 0.0
    if end is not None:
        hi = min(end, duration) if duration else end
    elif duration:
        hi = duration
    else:
        raise AgentError(
            ErrorCode.INVALID_TIME_RANGE,
            "The video duration is unknown; pass an explicit end time.",
        )
    # Keep the final frame slightly before the end to avoid empty decodes.
    if duration and hi >= duration:
        hi = max(lo + 0.1, duration - 0.1)
    return lo, hi
