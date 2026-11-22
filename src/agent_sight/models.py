"""Strict Pydantic v2 schemas shared by the CLI, the MCP server and the pipeline."""

from __future__ import annotations

import math
import re
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agent_sight.errors import AgentError, ErrorCode

# --------------------------------------------------------------------------------------
# Time helpers
# --------------------------------------------------------------------------------------

_HMS = re.compile(r"^(?:(\d+):)?(\d{1,2}):(\d{1,2}(?:\.\d+)?)$")
_UNITS = re.compile(r"^(?:(\d+(?:\.\d+)?)h)?(?:(\d+(?:\.\d+)?)m)?(?:(\d+(?:\.\d+)?)s)?$", re.I)


def parse_timestamp(value: str | float | int | None) -> float | None:
    """Parse ``90``, ``"90.5"``, ``"1:30"``, ``"01:02:03.5"``, ``"1h2m3s"`` into seconds."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise AgentError(ErrorCode.INVALID_TIME_RANGE, f"Invalid timestamp: {value!r}")
    if isinstance(value, (int, float)):
        seconds = float(value)
    else:
        text = value.strip().lower()
        if not text:
            return None
        try:
            seconds = float(text)
        except ValueError:
            m = _HMS.match(text)
            if m:
                h, mnt, s = m.groups()
                seconds = int(h or 0) * 3600 + int(mnt) * 60 + float(s)
            else:
                u = _UNITS.match(text)
                if not u or not any(u.groups()):
                    raise AgentError(
                        ErrorCode.INVALID_TIME_RANGE, f"Invalid timestamp: {value!r}"
                    ) from None
                h, mnt, s = (float(g) if g else 0.0 for g in u.groups())
                seconds = h * 3600 + mnt * 60 + s
    if math.isnan(seconds) or math.isinf(seconds) or seconds < 0:
        raise AgentError(ErrorCode.INVALID_TIME_RANGE, f"Invalid timestamp: {value!r}")
    return seconds


def format_timestamp(seconds: float | None, *, millis: bool = False) -> str:
    """Format seconds as ``H:MM:SS`` (or ``M:SS`` below an hour)."""
    if seconds is None:
        return "?"
    seconds = max(0.0, float(seconds))
    whole = int(seconds)
    h, rem = divmod(whole, 3600)
    m, s = divmod(rem, 60)
    base = f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
    if millis:
        base += f".{int(round((seconds - whole) * 1000)) % 1000:03d}"
    return base


def estimate_text_tokens(text: str) -> int:
    """Cheap, model-agnostic token estimate (~4 characters per token)."""
    return max(1, math.ceil(len(text) / 4)) if text else 0


def estimate_image_tokens(width: int, height: int) -> int:
    """Approximate vision-token cost of an image (Anthropic formula: w*h/750)."""
    return max(1, math.ceil(width * height / 750))


# --------------------------------------------------------------------------------------
# Base
# --------------------------------------------------------------------------------------


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True, use_enum_values=True)


class Platform(StrEnum):
    YOUTUBE = "youtube"
    INSTAGRAM = "instagram"
    TIKTOK = "tiktok"
    X = "x"
    REDDIT = "reddit"
    VIMEO = "vimeo"
    TWITCH = "twitch"
    FACEBOOK = "facebook"
    DIRECT = "direct"
    GENERIC = "generic"


class NormalizedURL(StrictModel):
    original: str
    normalized: str
    platform: Platform
    video_id: str | None = None
    start_hint: float | None = Field(
        default=None, description="Start offset carried in the URL (e.g. YouTube ?t=)."
    )
    host: str


# --------------------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------------------


class Chapter(StrictModel):
    start: float
    end: float | None = None
    title: str


class FormatSummary(StrictModel):
    format_id: str
    ext: str | None = None
    kind: Literal["video+audio", "video", "audio", "unknown"] = "unknown"
    height: int | None = None
    fps: float | None = None
    vcodec: str | None = None
    acodec: str | None = None
    filesize: int | None = None
    tbr: float | None = None
    protocol: str | None = None


class VideoMetadata(StrictModel):
    url: str
    platform: Platform
    extractor: str
    id: str | None = None
    title: str | None = None
    uploader: str | None = None
    channel: str | None = None
    upload_date: str | None = None
    duration: float | None = None
    view_count: int | None = None
    like_count: int | None = None
    comment_count: int | None = None
    description: str | None = None
    tags: list[str] = Field(default_factory=list)
    thumbnail: str | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    is_live: bool = False
    age_limit: int | None = None
    chapters: list[Chapter] = Field(default_factory=list)
    subtitle_languages: list[str] = Field(default_factory=list)
    auto_caption_languages: list[str] = Field(default_factory=list)
    formats: list[FormatSummary] = Field(default_factory=list)
    has_drm: bool = False
    estimated_size_bytes: int | None = Field(
        default=None, description="Approximate size of the default-quality download."
    )


class NextAction(StrictModel):
    tool: str
    reason: str
    cli: str


class ProbeResult(StrictModel):
    metadata: VideoMetadata
    has_transcript: bool
    transcript_source: Literal["native", "auto", "whisper", "none"]
    estimated_transcript_tokens: int | None = None
    next_actions: list[NextAction] = Field(default_factory=list)
    cached: bool = False


# --------------------------------------------------------------------------------------
# Transcripts
# --------------------------------------------------------------------------------------


class Word(StrictModel):
    start: float
    end: float
    text: str
    probability: float | None = None


class TranscriptSegment(StrictModel):
    start: float
    end: float
    text: str
    words: list[Word] | None = None

    @model_validator(mode="after")
    def _check_order(self) -> TranscriptSegment:
        if self.end < self.start:
            self.end = self.start
        return self


class Transcript(StrictModel):
    language: str | None = None
    source: Literal["native", "auto", "whisper"]
    segments: list[TranscriptSegment] = Field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.segments).strip()

    def slice(self, start: float | None = None, end: float | None = None) -> Transcript:
        lo = start if start is not None else 0.0
        hi = end if end is not None else math.inf
        segs = [s for s in self.segments if s.end >= lo and s.start <= hi]
        return Transcript(language=self.language, source=self.source, segments=segs)


class TranscriptResult(StrictModel):
    url: str
    language: str | None
    source: Literal["native", "auto", "whisper"]
    start: float | None = None
    end: float | None = None
    format: Literal["timestamped", "plain", "segments"]
    text: str | None = None
    segments: list[TranscriptSegment] | None = None
    segment_count: int
    estimated_tokens: int
    truncated: bool = False
    next_start: float | None = Field(
        default=None, description="If truncated, pass this as --start to continue reading."
    )
    cached: bool = False


# --------------------------------------------------------------------------------------
# Vision
# --------------------------------------------------------------------------------------


class Frame(StrictModel):
    timestamp: float
    label: str
    path: str
    width: int
    height: int
    scene_score: float | None = None


class ContactSheet(StrictModel):
    path: str
    width: int
    height: int
    columns: int
    rows: int
    timestamps: list[float]
    labels: list[str]
    estimated_tokens: int


class SceneCut(StrictModel):
    timestamp: float
    label: str
    score: float | None = None


class ScenesResult(StrictModel):
    url: str
    method: Literal["ffmpeg-scene", "pyscenedetect", "uniform"]
    threshold: float | None = None
    duration: float | None = None
    scenes: list[SceneCut]
    cached: bool = False


class FramesResult(StrictModel):
    url: str
    frames: list[Frame]
    contact_sheets: list[ContactSheet] = Field(default_factory=list)
    estimated_image_tokens: int
    source: Literal["stream", "proxy", "local"]


# --------------------------------------------------------------------------------------
# Digest and segment watching
# --------------------------------------------------------------------------------------


class TranscriptExcerpt(StrictModel):
    start: float
    end: float
    label: str
    text: str


class TimelineEntry(StrictModel):
    start: float
    end: float | None = None
    label: str
    kind: Literal["chapter", "scene", "transcript"]
    text: str


class Digest(StrictModel):
    """Compact overview. ``timeline`` merges chapters, scene cuts and transcript excerpts."""

    url: str
    title: str | None
    platform: Platform
    uploader: str | None = None
    duration: float | None
    duration_label: str
    upload_date: str | None = None
    view_count: int | None = None
    description_excerpt: str | None = None
    transcript_source: Literal["native", "auto", "whisper", "none"]
    transcript_language: str | None = None
    transcript_coverage: float = Field(
        default=0.0, description="Fraction of the full transcript included (0..1)."
    )
    scene_method: str | None = None
    scene_count: int = 0
    contact_sheets: list[ContactSheet] = Field(default_factory=list)
    timeline: list[TimelineEntry] = Field(default_factory=list)
    estimated_text_tokens: int
    estimated_image_tokens: int
    token_budget: int
    warnings: list[str] = Field(default_factory=list)
    next_actions: list[NextAction] = Field(default_factory=list)


class SegmentResult(StrictModel):
    url: str
    start: float
    end: float
    label: str
    frames: list[Frame]
    contact_sheets: list[ContactSheet] = Field(default_factory=list)
    transcript: str | None = None
    transcript_source: str | None = None
    estimated_text_tokens: int
    estimated_image_tokens: int
    source: Literal["stream", "proxy", "local"]


# --------------------------------------------------------------------------------------
# Downloads
# --------------------------------------------------------------------------------------

DownloadMode = Literal["video", "audio", "subtitles", "thumbnail"]


class DownloadRequest(StrictModel):
    url: str
    mode: DownloadMode = "video"
    quality: Literal["best", "1080p", "720p", "480p", "360p", "worst"] | None = None
    start: float | None = None
    end: float | None = None
    audio_format: Literal["m4a", "mp3", "opus", "wav", "best"] = "m4a"
    container: Literal["mp4", "webm", "mkv"] | None = "mp4"
    subtitle_languages: list[str] = Field(default_factory=lambda: ["en"])
    filename: str | None = Field(default=None, description="Optional file stem (no directories).")

    @field_validator("filename")
    @classmethod
    def _safe_filename(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if "/" in v or "\\" in v or v.startswith(".") or "\x00" in v:
            raise ValueError("filename must be a bare file stem without path separators")
        return v

    @model_validator(mode="after")
    def _check_range(self) -> DownloadRequest:
        if self.start is not None and self.end is not None and self.end <= self.start:
            raise ValueError("end must be greater than start")
        return self

    @property
    def is_clip(self) -> bool:
        return self.start is not None or self.end is not None


class StreamInfo(StrictModel):
    index: int
    type: Literal["video", "audio", "subtitle", "data", "unknown"]
    codec: str | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    sample_rate: int | None = None
    channels: int | None = None
    bitrate: int | None = None
    language: str | None = None


class MediaInfo(StrictModel):
    path: str
    format_name: str | None = None
    duration: float | None = None
    size_bytes: int | None = None
    bitrate: int | None = None
    streams: list[StreamInfo] = Field(default_factory=list)

    @property
    def video(self) -> StreamInfo | None:
        return next((s for s in self.streams if s.type == "video"), None)

    @property
    def audio(self) -> StreamInfo | None:
        return next((s for s in self.streams if s.type == "audio"), None)


class DownloadResult(StrictModel):
    url: str
    mode: DownloadMode
    files: list[str]
    size_bytes: int
    media: MediaInfo | None = None
    start: float | None = None
    end: float | None = None
    cached: bool = False


# --------------------------------------------------------------------------------------
# Doctor / envelopes
# --------------------------------------------------------------------------------------


class CheckResult(StrictModel):
    name: str
    status: Literal["ok", "warn", "fail"]
    detail: str
    fix: str | None = None
    required: bool = True


class DoctorReport(StrictModel):
    ok: bool
    checks: list[CheckResult]
    summary: str


class ErrorPayload(StrictModel):
    error_code: str
    message: str
    agent_recovery_hint: str
    retryable: bool
    details: dict[str, Any] = Field(default_factory=dict)


class Envelope(BaseModel):
    """Uniform JSON shape for every CLI command: ``{"ok", "command", "data" | "error"}``."""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    command: str
    data: Any | None = None
    error: ErrorPayload | None = None
