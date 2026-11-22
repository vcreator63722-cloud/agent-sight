"""URL validation, normalization and platform detection.

Normalization produces a canonical URL per video so that cache keys are stable across
share links (``youtu.be``, ``/shorts/``, tracking parameters, mobile hosts, ...).
"""

from __future__ import annotations

import ipaddress
import re
import socket
from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit

from agent_sight.errors import AgentError, ErrorCode
from agent_sight.models import NormalizedURL, Platform, parse_timestamp

MEDIA_EXTENSIONS = (".mp4", ".m4v", ".webm", ".mov", ".mkv", ".m3u8", ".mpd", ".ts", ".ogv")
_TRACKING_PREFIXES = ("utm_",)
_TRACKING_PARAMS = {
    "si", "feature", "igshid", "igsh", "fbclid", "gclid", "ref", "ref_src", "ref_url",
    "s", "is_from_webapp", "sender_device", "share_id", "_r", "_t", "pp", "context",
}
_YT_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")


def _strip_tracking(query: str, keep: set[str] | None = None) -> str:
    pairs = parse_qs(query, keep_blank_values=False)
    clean = {
        k: v for k, v in pairs.items()
        if (keep is None or k in keep)
        and k not in _TRACKING_PARAMS
        and not k.startswith(_TRACKING_PREFIXES)
    }
    return urlencode(sorted(clean.items()), doseq=True)


def _host(netloc: str) -> str:
    host = netloc.rsplit("@", 1)[-1].split(":", 1)[0].lower().rstrip(".")
    return host[4:] if host.startswith("www.") else host


def _matches(host: str, *domains: str) -> bool:
    return any(host == d or host.endswith("." + d) for d in domains)


def _first(query: dict[str, list[str]], *keys: str) -> str | None:
    for k in keys:
        if query.get(k):
            return query[k][0]
    return None


def _yt_start(query: dict[str, list[str]]) -> float | None:
    raw = _first(query, "t", "start")
    if raw is None:
        return None
    try:
        return parse_timestamp(raw)
    except AgentError:
        return None


def check_public_host(host: str) -> None:
    """Reject loopback/private/link-local targets to prevent SSRF via the MCP server."""
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return  # Let the extractor surface a proper network error.
    for info in infos:
        addr = ipaddress.ip_address(info[4][0])
        if (
            addr.is_private or addr.is_loopback or addr.is_link_local
            or addr.is_reserved or addr.is_multicast or addr.is_unspecified
        ):
            raise AgentError(
                ErrorCode.INVALID_URL,
                f"Refusing to fetch {host}: it resolves to a non-public address ({addr}).",
                "Only public URLs are allowed. The machine owner can set "
                "AGENT_SIGHT_ALLOW_PRIVATE_HOSTS=true for trusted local media servers.",
            )


def normalize_url(url: str) -> NormalizedURL:
    if not isinstance(url, str) or not url.strip():
        raise AgentError(ErrorCode.INVALID_URL, "URL is empty.")
    raw = url.strip()
    if "://" not in raw and re.match(r"^[\w.-]+\.[a-z]{2,}(/|$)", raw, re.I):
        raw = "https://" + raw
    parts = urlsplit(raw)
    if parts.scheme.lower() not in ("http", "https"):
        raise AgentError(
            ErrorCode.INVALID_URL,
            f"Unsupported URL scheme {parts.scheme!r}; only http and https are allowed.",
        )
    if not parts.netloc:
        raise AgentError(ErrorCode.INVALID_URL, f"URL has no host: {url!r}")
    if "@" in parts.netloc:
        raise AgentError(
            ErrorCode.INVALID_URL,
            "URLs with embedded credentials are not accepted.",
            "Remove user:password@ from the URL. Use AGENT_SIGHT_COOKIES_FILE for authorised access.",
        )

    host = _host(parts.netloc)
    path = re.sub(r"/{2,}", "/", parts.path) or "/"
    query = parse_qs(parts.query)

    def build(
        platform: Platform,
        normalized: str,
        video_id: str | None = None,
        start: float | None = None,
    ) -> NormalizedURL:
        return NormalizedURL(
            original=url, normalized=normalized, platform=platform, video_id=video_id,
            start_hint=start, host=host,
        )

    # ---------------------------------------------------------------- YouTube
    if _matches(host, "youtube.com", "youtube-nocookie.com", "youtu.be"):
        vid: str | None = None
        if host == "youtu.be":
            vid = path.strip("/").split("/")[0]
        elif path.startswith(("/shorts/", "/embed/", "/live/", "/v/", "/e/")):
            vid = path.split("/")[2] if len(path.split("/")) > 2 else None
        elif path.startswith("/watch"):
            vid = _first(query, "v")
        if vid and _YT_ID.match(vid):
            return build(
                Platform.YOUTUBE, f"https://www.youtube.com/watch?v={vid}", vid, _yt_start(query)
            )
        q = _strip_tracking(parts.query, keep={"list", "v"})
        return build(
            Platform.YOUTUBE, urlunsplit(("https", "www.youtube.com", path, q, "")), None
        )

    # ---------------------------------------------------------------- Instagram
    if _matches(host, "instagram.com", "instagr.am"):
        m = re.match(r"^/(?:[\w.]+/)?(p|reel|reels|tv)/([\w-]+)", path)
        if m:
            kind = "reel" if m.group(1) in ("reel", "reels") else m.group(1)
            return build(
                Platform.INSTAGRAM, f"https://www.instagram.com/{kind}/{m.group(2)}/", m.group(2)
            )
        return build(Platform.INSTAGRAM, urlunsplit(("https", "www.instagram.com", path, "", "")))

    # ---------------------------------------------------------------- TikTok
    if _matches(host, "tiktok.com"):
        m = re.match(r"^/@([\w.-]+)/(?:video|photo)/(\d+)", path)
        if m:
            return build(
                Platform.TIKTOK, f"https://www.tiktok.com/@{m.group(1)}/video/{m.group(2)}",
                m.group(2),
            )
        m = re.match(r"^/(?:embed(?:/v2)?|v)/(\d+)", path)
        if m:
            return build(
                Platform.TIKTOK, f"https://www.tiktok.com/embed/v2/{m.group(1)}", m.group(1)
            )
        # vm.tiktok.com / vt.tiktok.com short links: keep as-is, the extractor follows them.
        return build(Platform.TIKTOK, urlunsplit(("https", parts.netloc.lower(), path, "", "")))

    # ---------------------------------------------------------------- X / Twitter
    if _matches(host, "x.com", "twitter.com", "fxtwitter.com", "vxtwitter.com", "fixupx.com"):
        m = re.match(r"^/(?:([\w]+)|i/web)/status(?:es)?/(\d+)", path)
        if m:
            user = m.group(1) or "i"
            return build(Platform.X, f"https://x.com/{user}/status/{m.group(2)}", m.group(2))
        return build(Platform.X, urlunsplit(("https", "x.com", path, "", "")))

    # ---------------------------------------------------------------- Reddit
    if _matches(host, "reddit.com", "redd.it"):
        if host == "v.redd.it":
            vid = path.strip("/").split("/")[0]
            return build(Platform.REDDIT, f"https://v.redd.it/{vid}", vid)
        if host == "redd.it":
            pid = path.strip("/").split("/")[0]
            return build(Platform.REDDIT, f"https://www.reddit.com/comments/{pid}/", pid)
        m = re.match(r"^/r/(\w+)/comments/(\w+)", path)
        if m:
            return build(
                Platform.REDDIT,
                f"https://www.reddit.com/r/{m.group(1)}/comments/{m.group(2)}/",
                m.group(2),
            )
        return build(Platform.REDDIT, urlunsplit(("https", "www.reddit.com", path, "", "")))

    # ---------------------------------------------------------------- Vimeo
    if _matches(host, "vimeo.com"):
        m = re.match(r"^/(?:video/|channels/[\w-]+/|groups/[\w-]+/videos/)?(\d+)(?:/(\w+))?", path)
        if m:
            vid = m.group(1)
            unlisted_hash = m.group(2) or _first(query, "h")
            suffix = f"/{unlisted_hash}" if unlisted_hash else ""
            return build(Platform.VIMEO, f"https://vimeo.com/{vid}{suffix}", vid)
        return build(Platform.VIMEO, urlunsplit(("https", "vimeo.com", path, "", "")))

    # ---------------------------------------------------------------- Twitch clips / VODs
    if _matches(host, "twitch.tv"):
        if host == "clips.twitch.tv":
            slug = path.strip("/").split("/")[0]
            return build(Platform.TWITCH, f"https://clips.twitch.tv/{slug}", slug)
        m = re.match(r"^/[\w]+/clip/([\w-]+)", path)
        if m:
            return build(Platform.TWITCH, f"https://clips.twitch.tv/{m.group(1)}", m.group(1))
        m = re.match(r"^/videos/(\d+)", path)
        if m:
            return build(Platform.TWITCH, f"https://www.twitch.tv/videos/{m.group(1)}", m.group(1))
        return build(Platform.TWITCH, urlunsplit(("https", "www.twitch.tv", path, "", "")))

    # ---------------------------------------------------------------- Facebook (public videos)
    if _matches(host, "facebook.com", "fb.watch"):
        q = _strip_tracking(parts.query, keep={"v"})
        return build(
            Platform.FACEBOOK, urlunsplit(("https", parts.netloc.lower(), path, q, ""))
        )

    # ---------------------------------------------------------------- Direct media / generic
    scheme = parts.scheme.lower()
    netloc = parts.netloc.lower()
    if path.lower().endswith(MEDIA_EXTENSIONS):
        # Query strings on direct media are often signatures: keep them untouched.
        return build(Platform.DIRECT, urlunsplit((scheme, netloc, path, parts.query, "")))
    q = _strip_tracking(parts.query)
    return build(Platform.GENERIC, urlunsplit((scheme, netloc, path, q, "")))


def detect_platform(url: str) -> Platform:
    return Platform(normalize_url(url).platform)
