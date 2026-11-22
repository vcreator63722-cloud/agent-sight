from __future__ import annotations

import pytest

from agent_sight.adapters.router import check_public_host, normalize_url
from agent_sight.errors import AgentError, ErrorCode


@pytest.mark.parametrize(
    ("url", "platform", "normalized", "vid"),
    [
        ("https://youtu.be/dQw4w9WgXcQ?si=abc", "youtube", "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://m.youtube.com/watch?v=dQw4w9WgXcQ&feature=share", "youtube", "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/shorts/dQw4w9WgXcQ", "youtube", "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("youtube.com/embed/dQw4w9WgXcQ", "youtube", "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.instagram.com/reels/C1a2b3c4d5/?igsh=xyz", "instagram", "https://www.instagram.com/reel/C1a2b3c4d5/", "C1a2b3c4d5"),
        ("https://www.instagram.com/someuser/p/ABC123/", "instagram", "https://www.instagram.com/p/ABC123/", "ABC123"),
        ("https://www.tiktok.com/@user.name/video/7234567890123456789?is_from_webapp=1", "tiktok", "https://www.tiktok.com/@user.name/video/7234567890123456789", "7234567890123456789"),
        ("https://twitter.com/jack/status/20?s=20", "x", "https://x.com/jack/status/20", "20"),
        ("https://mobile.x.com/i/web/status/1234567890", "x", "https://x.com/i/status/1234567890", "1234567890"),
        ("https://old.reddit.com/r/videos/comments/abc123/some_title/", "reddit", "https://www.reddit.com/r/videos/comments/abc123/", "abc123"),
        ("https://v.redd.it/xyz789", "reddit", "https://v.redd.it/xyz789", "xyz789"),
        ("https://player.vimeo.com/video/76979871?h=abc", "vimeo", "https://vimeo.com/76979871/abc", "76979871"),
        ("https://vimeo.com/76979871", "vimeo", "https://vimeo.com/76979871", "76979871"),
        ("https://www.twitch.tv/somechannel/clip/FunnyClipSlug-abc", "twitch", "https://clips.twitch.tv/FunnyClipSlug-abc", "FunnyClipSlug-abc"),
        ("https://cdn.example.com/v/movie.mp4?sig=abc&utm_source=x", "direct", "https://cdn.example.com/v/movie.mp4?sig=abc&utm_source=x", None),
        ("https://example.com/stream/master.m3u8", "direct", "https://example.com/stream/master.m3u8", None),
        ("https://example.com/watch/page?id=5&utm_source=news&fbclid=1", "generic", "https://example.com/watch/page?id=5", None),
    ],
)
def test_normalize(url: str, platform: str, normalized: str, vid: str | None) -> None:
    n = normalize_url(url)
    assert n.platform == platform
    assert n.normalized == normalized
    assert n.video_id == vid


def test_youtube_start_hint() -> None:
    assert normalize_url("https://youtu.be/dQw4w9WgXcQ?t=1m30s").start_hint == 90
    assert normalize_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42").start_hint == 42


def test_share_links_share_cache_identity() -> None:
    a = normalize_url("https://youtu.be/dQw4w9WgXcQ?si=1")
    b = normalize_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ&feature=youtu.be")
    assert a.normalized == b.normalized


@pytest.mark.parametrize(
    "bad", ["", "   ", "ftp://example.com/a.mp4", "file:///etc/passwd", "javascript:alert(1)",
            "https://user:pass@example.com/v.mp4", "http://"],
)
def test_rejects_invalid(bad: str) -> None:
    with pytest.raises(AgentError) as exc:
        normalize_url(bad)
    assert exc.value.code == ErrorCode.INVALID_URL


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "10.0.0.8", "169.254.169.254", "::1"])
def test_ssrf_guard(host: str) -> None:
    with pytest.raises(AgentError) as exc:
        check_public_host(host)
    assert exc.value.code == ErrorCode.INVALID_URL
