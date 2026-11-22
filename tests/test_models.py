from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent_sight.errors import (
    EXIT_ACCESS,
    EXIT_TRANSIENT,
    AgentError,
    ErrorCode,
    classify_error_message,
    from_extractor_error,
)
from agent_sight.models import (
    DownloadRequest,
    Transcript,
    TranscriptSegment,
    estimate_image_tokens,
    format_timestamp,
    parse_timestamp,
)


@pytest.mark.parametrize(
    ("raw", "seconds"),
    [(90, 90.0), ("90", 90.0), ("90.5", 90.5), ("1:30", 90.0), ("01:02:03.5", 3723.5),
     ("1h2m3s", 3723.0), ("2m", 120.0), ("45s", 45.0), (None, None), ("", None)],
)
def test_parse_timestamp(raw: object, seconds: float | None) -> None:
    assert parse_timestamp(raw) == seconds  # type: ignore[arg-type]


@pytest.mark.parametrize("raw", ["abc", "-5", "1:2:3:4", "nan", True])
def test_parse_timestamp_invalid(raw: object) -> None:
    with pytest.raises(AgentError) as exc:
        parse_timestamp(raw)  # type: ignore[arg-type]
    assert exc.value.code == ErrorCode.INVALID_TIME_RANGE


def test_format_timestamp() -> None:
    assert format_timestamp(5) == "0:05"
    assert format_timestamp(3723.4) == "1:02:03"
    assert format_timestamp(None) == "?"
    assert format_timestamp(1.25, millis=True) == "0:01.250"


def test_image_tokens() -> None:
    assert estimate_image_tokens(750, 1) == 1
    assert estimate_image_tokens(1168, 664) == 1035


def test_download_request_validation() -> None:
    assert DownloadRequest(url="https://x.com/a/status/1", start=1, end=5).is_clip
    with pytest.raises(ValidationError):
        DownloadRequest(url="u", start=10, end=5)
    with pytest.raises(ValidationError):
        DownloadRequest(url="u", filename="../etc/passwd")
    with pytest.raises(ValidationError):
        DownloadRequest(url="u", mode="everything")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        DownloadRequest(url="u", unknown_field=1)  # type: ignore[call-arg]


def test_transcript_slice() -> None:
    t = Transcript(source="native", segments=[
        TranscriptSegment(start=0, end=2, text="a"),
        TranscriptSegment(start=5, end=7, text="b"),
        TranscriptSegment(start=10, end=12, text="c"),
    ])
    assert [s.text for s in t.slice(4, 8).segments] == ["b"]
    assert t.text == "a b c"


@pytest.mark.parametrize(
    ("message", "code"),
    [
        ("ERROR: [youtube] abc: Private video. Sign in if you've been granted access", ErrorCode.PRIVATE_CONTENT),
        ("ERROR: [youtube] abc: Sign in to confirm your age", ErrorCode.AGE_RESTRICTED),
        ("ERROR: [youtube] abc: Sign in to confirm you're not a bot", ErrorCode.LOGIN_REQUIRED),
        ("This video is DRM protected", ErrorCode.DRM_PROTECTED),
        ("The uploader has not made this video available in your country", ErrorCode.GEO_RESTRICTED),
        ("HTTP Error 429: Too Many Requests", ErrorCode.RATE_LIMITED),
        ("HTTP Error 403: Forbidden", ErrorCode.ACCESS_DENIED),
        ("Unsupported URL: https://example.com", ErrorCode.UNSUPPORTED_URL),
        ("Join this channel to get access to members-only content", ErrorCode.PAYWALLED),
        ("You have requested downloading the video partially, but ffmpeg is not installed", ErrorCode.DEPENDENCY_MISSING),
        ("Read timed out", ErrorCode.NETWORK_ERROR),
        ("something completely different", ErrorCode.EXTRACTION_FAILED),
    ],
)
def test_classify(message: str, code: ErrorCode) -> None:
    assert classify_error_message(message) == code


def test_agent_error_shape() -> None:
    err = from_extractor_error(Exception("\x1b[0;31mERROR:\x1b[0m HTTP Error 429: Too Many Requests"), "u")
    d = err.to_dict()
    assert d["error_code"] == "RATE_LIMITED"
    assert d["retryable"] is True
    assert d["agent_recovery_hint"]
    assert err.exit_code == EXIT_TRANSIENT
    assert AgentError(ErrorCode.DRM_PROTECTED, "x").exit_code == EXIT_ACCESS
