"""Typed errors that carry a stable code, a human message and a recovery hint for agents."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Any


class ErrorCode(StrEnum):
    INVALID_URL = "INVALID_URL"
    INVALID_ARGUMENT = "INVALID_ARGUMENT"
    INVALID_TIME_RANGE = "INVALID_TIME_RANGE"
    UNSUPPORTED_URL = "UNSUPPORTED_URL"
    DEPENDENCY_MISSING = "DEPENDENCY_MISSING"
    LOGIN_REQUIRED = "LOGIN_REQUIRED"
    ACCESS_DENIED = "ACCESS_DENIED"
    PRIVATE_CONTENT = "PRIVATE_CONTENT"
    AGE_RESTRICTED = "AGE_RESTRICTED"
    GEO_RESTRICTED = "GEO_RESTRICTED"
    DRM_PROTECTED = "DRM_PROTECTED"
    PAYWALLED = "PAYWALLED"
    NOT_FOUND = "NOT_FOUND"
    LIVE_STREAM = "LIVE_STREAM"
    RATE_LIMITED = "RATE_LIMITED"
    NETWORK_ERROR = "NETWORK_ERROR"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    NO_TRANSCRIPT = "NO_TRANSCRIPT"
    NO_VIDEO_STREAM = "NO_VIDEO_STREAM"
    EXTRACTION_FAILED = "EXTRACTION_FAILED"
    FFMPEG_FAILED = "FFMPEG_FAILED"
    TRANSCRIPTION_FAILED = "TRANSCRIPTION_FAILED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


# CLI exit codes. Agents can branch on these without parsing output.
EXIT_OK = 0
EXIT_INTERNAL = 1
EXIT_USAGE = 2
EXIT_DEPENDENCY = 3
EXIT_ACCESS = 4
EXIT_NOT_FOUND = 5
EXIT_TRANSIENT = 6
EXIT_PROCESSING = 7
EXIT_LIMIT = 8

_EXIT_CODES: dict[ErrorCode, int] = {
    ErrorCode.INVALID_URL: EXIT_USAGE,
    ErrorCode.INVALID_ARGUMENT: EXIT_USAGE,
    ErrorCode.INVALID_TIME_RANGE: EXIT_USAGE,
    ErrorCode.UNSUPPORTED_URL: EXIT_USAGE,
    ErrorCode.DEPENDENCY_MISSING: EXIT_DEPENDENCY,
    ErrorCode.LOGIN_REQUIRED: EXIT_ACCESS,
    ErrorCode.ACCESS_DENIED: EXIT_ACCESS,
    ErrorCode.PRIVATE_CONTENT: EXIT_ACCESS,
    ErrorCode.AGE_RESTRICTED: EXIT_ACCESS,
    ErrorCode.GEO_RESTRICTED: EXIT_ACCESS,
    ErrorCode.DRM_PROTECTED: EXIT_ACCESS,
    ErrorCode.PAYWALLED: EXIT_ACCESS,
    ErrorCode.NOT_FOUND: EXIT_NOT_FOUND,
    ErrorCode.NO_TRANSCRIPT: EXIT_NOT_FOUND,
    ErrorCode.NO_VIDEO_STREAM: EXIT_NOT_FOUND,
    ErrorCode.LIVE_STREAM: EXIT_USAGE,
    ErrorCode.RATE_LIMITED: EXIT_TRANSIENT,
    ErrorCode.NETWORK_ERROR: EXIT_TRANSIENT,
    ErrorCode.FILE_TOO_LARGE: EXIT_LIMIT,
    ErrorCode.EXTRACTION_FAILED: EXIT_PROCESSING,
    ErrorCode.FFMPEG_FAILED: EXIT_PROCESSING,
    ErrorCode.TRANSCRIPTION_FAILED: EXIT_PROCESSING,
    ErrorCode.INTERNAL_ERROR: EXIT_INTERNAL,
}

_RETRYABLE = {ErrorCode.RATE_LIMITED, ErrorCode.NETWORK_ERROR}

_DEFAULT_HINTS: dict[ErrorCode, str] = {
    ErrorCode.INVALID_URL: "Pass a full http(s) URL to a public video page or media file.",
    ErrorCode.INVALID_ARGUMENT: "Check the argument values against `agent-sight <command> --help`.",
    ErrorCode.INVALID_TIME_RANGE: (
        "Use timestamps like 90, 1:30 or 00:01:30.5 with start < end and end <= duration "
        "(call probe to read the duration)."
    ),
    ErrorCode.UNSUPPORTED_URL: (
        "No video was found at this URL. Pass the direct page of a single video, "
        "or a direct .mp4/.webm/.m3u8 link."
    ),
    ErrorCode.DEPENDENCY_MISSING: "Run `agent-sight doctor --json` and apply the listed fix commands.",
    ErrorCode.LOGIN_REQUIRED: (
        "The platform requires sign-in. If you own or are permitted to view this content, the "
        "machine owner can set AGENT_SIGHT_COOKIES_FILE to an exported cookies.txt. Otherwise "
        "stop and tell the user the content is not publicly accessible."
    ),
    ErrorCode.ACCESS_DENIED: (
        "The server refused access (HTTP 401/403): hotlink protection, a network policy or "
        "missing authorisation. Do not try to bypass it; ask the user for an accessible link."
    ),
    ErrorCode.PRIVATE_CONTENT: (
        "The video is private or deleted. Do not retry; ask the user for a public link."
    ),
    ErrorCode.AGE_RESTRICTED: (
        "Age-restricted content needs an authorised signed-in session (AGENT_SIGHT_COOKIES_FILE) "
        "configured by the machine owner. Otherwise report it as inaccessible."
    ),
    ErrorCode.GEO_RESTRICTED: (
        "The video is blocked in this region. Do not attempt to bypass it; report it to the user."
    ),
    ErrorCode.DRM_PROTECTED: (
        "The media is DRM-protected. agent-sight does not circumvent DRM. Use metadata from "
        "probe only, or ask the user for a DRM-free source."
    ),
    ErrorCode.PAYWALLED: "The content is behind a paywall and is out of scope. Report it to the user.",
    ErrorCode.NOT_FOUND: "Check the URL for typos; the video may have been removed.",
    ErrorCode.LIVE_STREAM: (
        "Live streams are not supported while live. Retry after the broadcast ends and a VOD exists."
    ),
    ErrorCode.RATE_LIMITED: (
        "The platform is throttling requests. Wait 30-120 seconds before retrying, and reuse "
        "cached results (they are returned automatically)."
    ),
    ErrorCode.NETWORK_ERROR: "Transient network failure. Retry once; if it persists run `agent-sight doctor --network`.",
    ErrorCode.FILE_TOO_LARGE: (
        "Request a lower quality (e.g. --quality 480p), audio-only (--mode audio), or a time "
        "range (--start/--end), or raise AGENT_SIGHT_MAX_DOWNLOAD_MB."
    ),
    ErrorCode.NO_TRANSCRIPT: (
        "No captions exist. Install the whisper extra (`pip install 'agent-sight[whisper]'`) for "
        "local transcription, or rely on visual tools (digest, watch)."
    ),
    ErrorCode.NO_VIDEO_STREAM: "The media has no video track. Use transcript or download --mode audio.",
    ErrorCode.EXTRACTION_FAILED: (
        "The extractor failed. Update yt-dlp (`pip install -U yt-dlp`) and retry; site changes "
        "often break extractors until the next release."
    ),
    ErrorCode.FFMPEG_FAILED: "ffmpeg failed. Run `agent-sight doctor` to verify ffmpeg, then retry.",
    ErrorCode.TRANSCRIPTION_FAILED: "Local transcription failed. Try a smaller AGENT_SIGHT_WHISPER_MODEL.",
    ErrorCode.INTERNAL_ERROR: "Unexpected error. Re-run with --verbose and report the issue.",
}


class AgentError(Exception):
    """An error designed to be read by an AI agent as well as a human."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        hint: str | None = None,
        *,
        details: dict[str, Any] | None = None,
        retryable: bool | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint or _DEFAULT_HINTS.get(code, "")
        self.details = details or {}
        self.retryable = code in _RETRYABLE if retryable is None else retryable

    @property
    def exit_code(self) -> int:
        return _EXIT_CODES.get(self.code, EXIT_INTERNAL)

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_code": self.code.value,
            "message": self.message,
            "agent_recovery_hint": self.hint,
            "retryable": self.retryable,
            "details": self.details,
        }

    def __repr__(self) -> str:
        return f"AgentError({self.code.value}, {self.message!r})"


# Ordered: the first matching pattern wins. Patterns cover yt-dlp and common HTTP failures.
_PATTERNS: list[tuple[re.Pattern[str], ErrorCode]] = [
    (re.compile(r"\bDRM\b|drm protected|widevine|playready|fairplay", re.I), ErrorCode.DRM_PROTECTED),
    (re.compile(r"private video|video is private|this video has been removed|account .* terminated", re.I), ErrorCode.PRIVATE_CONTENT),
    (re.compile(r"confirm your age|age[- ]restricted|inappropriate for some users", re.I), ErrorCode.AGE_RESTRICTED),
    (re.compile(r"available in your country|geo.?restrict|blocked it in your country|not available from your location", re.I), ErrorCode.GEO_RESTRICTED),
    (re.compile(r"members[- ]only|join this channel|requires payment|purchase|rental|premium", re.I), ErrorCode.PAYWALLED),
    (re.compile(r"sign in|log ?in|login required|cookies|authentication|not a bot|rate-limit reached or login", re.I), ErrorCode.LOGIN_REQUIRED),
    (re.compile(r"HTTP Error 429|too many requests|rate.?limit", re.I), ErrorCode.RATE_LIMITED),
    (re.compile(r"HTTP Error 40[13]|403 Forbidden|401 Unauthorized", re.I), ErrorCode.ACCESS_DENIED),
    (re.compile(r"is live|live event will begin|premieres in|is_live", re.I), ErrorCode.LIVE_STREAM),
    (re.compile(r"HTTP Error 404|not found|does not exist|video unavailable|no longer available", re.I), ErrorCode.NOT_FOUND),
    (re.compile(r"unsupported url|no video formats found|no video could be found|no media found", re.I), ErrorCode.UNSUPPORTED_URL),
    (re.compile(r"timed out|timeout|connection (reset|refused|aborted)|temporary failure|name or service not known|network is unreachable|HTTP Error 5\d\d|IncompleteRead|ssl", re.I), ErrorCode.NETWORK_ERROR),
    (re.compile(r"max_filesize|larger than max-filesize|file is larger", re.I), ErrorCode.FILE_TOO_LARGE),
]
_DEPENDENCY = re.compile(r"ffmpeg is not installed|ffprobe and ffmpeg not found|ffmpeg not found", re.I)

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def classify_error_message(message: str) -> ErrorCode:
    clean = _ANSI.sub("", message)
    if _DEPENDENCY.search(clean):
        return ErrorCode.DEPENDENCY_MISSING
    for pattern, code in _PATTERNS:
        if pattern.search(clean):
            return code
    return ErrorCode.EXTRACTION_FAILED


def from_extractor_error(exc: BaseException, url: str | None = None) -> AgentError:
    """Convert a yt-dlp (or similar) exception into an AgentError."""
    if isinstance(exc, AgentError):
        return exc
    raw = _ANSI.sub("", str(exc)).strip()
    raw = re.sub(r"^ERROR:\s*", "", raw)
    code = classify_error_message(raw)
    details: dict[str, Any] = {"extractor_message": raw[:500]}
    if url:
        details["url"] = url
    hint = None
    if code == ErrorCode.LOGIN_REQUIRED and re.search(r"not a bot", raw, re.I):
        hint = (
            "The platform is bot-checking this network (common on cloud/datacenter IPs). Retry "
            "later or from another network. Only if the user owns an account they may use, the "
            "machine owner can set AGENT_SIGHT_COOKIES_FILE. Do not try to bypass the check."
        )
    return AgentError(code, raw.splitlines()[0][:300] if raw else code.value, hint, details=details)
