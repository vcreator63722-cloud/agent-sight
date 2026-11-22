"""ffmpeg/ffprobe discovery, subprocess helpers and stream inspection."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

from agent_sight.config import Settings, get_settings
from agent_sight.errors import AgentError, ErrorCode
from agent_sight.models import MediaInfo, StreamInfo

_FFMPEG_HINT = (
    "Install ffmpeg: macOS `brew install ffmpeg`, Debian/Ubuntu `sudo apt-get install -y ffmpeg`, "
    "Fedora/Amazon Linux `sudo dnf install -y ffmpeg`, Windows `winget install Gyan.FFmpeg`, "
    "or the bundled binary `pip install 'agent-sight[ffmpeg]'`."
)


def _bundled_ffmpeg() -> str | None:
    try:
        import imageio_ffmpeg  # type: ignore[import-not-found]

        exe: str = imageio_ffmpeg.get_ffmpeg_exe()
        return exe if exe and os.path.exists(exe) else None
    except Exception:
        return None


@lru_cache(maxsize=4)
def _resolve(name: str, configured: str | None) -> str | None:
    if configured:
        return configured if os.path.exists(configured) else None
    found = shutil.which(name)
    if found:
        return found
    if name == "ffmpeg":
        return _bundled_ffmpeg()
    return None


def find_ffmpeg(settings: Settings | None = None) -> str | None:
    s = settings or get_settings()
    return _resolve("ffmpeg", str(s.ffmpeg_path) if s.ffmpeg_path else None)


def find_ffprobe(settings: Settings | None = None) -> str | None:
    s = settings or get_settings()
    return _resolve("ffprobe", str(s.ffprobe_path) if s.ffprobe_path else None)


def require_ffmpeg(settings: Settings | None = None) -> str:
    exe = find_ffmpeg(settings)
    if not exe:
        raise AgentError(ErrorCode.DEPENDENCY_MISSING, "ffmpeg was not found.", _FFMPEG_HINT)
    return exe


def ffmpeg_dir_for_ytdlp(settings: Settings | None = None) -> str | None:
    """Return a directory yt-dlp can use as ``ffmpeg_location``.

    yt-dlp identifies binaries by file name, so a bundled binary with a versioned name is
    exposed through a stable ``ffmpeg`` symlink inside the cache directory.
    """
    s = settings or get_settings()
    exe = find_ffmpeg(s)
    if not exe:
        return None
    exe_path = Path(exe)
    if exe_path.stem == "ffmpeg":
        return str(exe_path.parent)
    link_dir = s.cache_dir / "bin"
    # Some yt-dlp checks (e.g. partial downloads) look only at PATH, ignoring ffmpeg_location.
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if str(link_dir) not in path_entries:
        os.environ["PATH"] = os.pathsep.join([str(link_dir), *path_entries])
    link_dir.mkdir(parents=True, exist_ok=True)
    link = link_dir / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    try:
        if link.is_symlink() or link.exists():
            if link.resolve() == exe_path.resolve():
                return str(link_dir)
            link.unlink()
        link.symlink_to(exe_path)
    except OSError:
        shutil.copy2(exe_path, link)
    probe = find_ffprobe(s)
    if probe:
        plink = link_dir / ("ffprobe.exe" if os.name == "nt" else "ffprobe")
        if not plink.exists():
            try:
                plink.symlink_to(probe)
            except OSError:
                pass
    return str(link_dir)


def header_args(headers: Mapping[str, str] | None) -> list[str]:
    """ffmpeg input options for HTTP headers (must precede ``-i``)."""
    if not headers:
        return []
    args: list[str] = []
    ua = next((v for k, v in headers.items() if k.lower() == "user-agent"), None)
    if ua:
        args += ["-user_agent", ua]
    rest = "".join(f"{k}: {v}\r\n" for k, v in headers.items() if k.lower() != "user-agent")
    if rest:
        args += ["-headers", rest]
    return args


def run_ffmpeg(
    args: Sequence[str],
    *,
    timeout: float | None = None,
    settings: Settings | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    s = settings or get_settings()
    exe = require_ffmpeg(s)
    cmd = [exe, "-hide_banner", "-nostdin", *args]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout or s.ffmpeg_timeout,
            errors="replace",
        )
    except subprocess.TimeoutExpired as exc:
        raise AgentError(
            ErrorCode.FFMPEG_FAILED,
            f"ffmpeg timed out after {exc.timeout:.0f}s.",
            "Use a shorter time range or fewer frames, or raise AGENT_SIGHT_FFMPEG_TIMEOUT.",
        ) from exc
    if check and proc.returncode != 0:
        raise ffmpeg_error(proc.stderr, "ffmpeg failed.")
    return proc


def ffmpeg_error(stderr: str, message: str) -> AgentError:
    """Map ffmpeg/ffprobe stderr to the most specific error code."""
    tail = "\n".join(stderr.strip().splitlines()[-6:])
    low = tail.lower()
    code = ErrorCode.FFMPEG_FAILED
    if "403 forbidden" in low or "401 unauthorized" in low or "http error 403" in low or "http error 401" in low:
        code = ErrorCode.ACCESS_DENIED
    elif "404 not found" in low or "no such file" in low:
        code = ErrorCode.NOT_FOUND
    elif "connection" in low and any(w in low for w in ("refused", "timed out", "reset")):
        code = ErrorCode.NETWORK_ERROR
    elif "drm" in low or "encrypted" in low:
        code = ErrorCode.DRM_PROTECTED
    elif "invalid data found" in low:
        code = ErrorCode.UNSUPPORTED_URL
    return AgentError(code, message, details={"stderr_tail": tail[-1500:]})


def _num(v: object) -> float | None:
    try:
        if v in (None, "", "N/A"):
            return None
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _fps(rate: str | None) -> float | None:
    if not rate or rate in ("0/0", "N/A"):
        return None
    if "/" in rate:
        n, d = rate.split("/", 1)
        try:
            return round(float(n) / float(d), 3) if float(d) else None
        except ValueError:
            return None
    return _num(rate)


def probe_media(
    source: str | Path,
    *,
    headers: Mapping[str, str] | None = None,
    settings: Settings | None = None,
    timeout: float = 60,
) -> MediaInfo:
    """Inspect a local file or remote URL. Uses ffprobe, or parses ``ffmpeg -i`` if absent."""
    s = settings or get_settings()
    src = str(source)
    probe = find_ffprobe(s)
    if probe:
        cmd = [
            probe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams",
            *header_args(headers), src,
        ]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout, errors="replace"
            )
        except subprocess.TimeoutExpired as exc:
            raise AgentError(ErrorCode.NETWORK_ERROR, "ffprobe timed out.") from exc
        if proc.returncode != 0:
            raise ffmpeg_error(proc.stderr, "ffprobe could not read the media.")
        return _parse_ffprobe_json(src, json.loads(proc.stdout or "{}"))
    proc = run_ffmpeg([*header_args(headers), "-i", src], settings=s, timeout=timeout, check=False)
    info = _parse_ffmpeg_banner(src, proc.stderr)
    if info.duration is None and not info.streams:
        raise ffmpeg_error(proc.stderr, "ffmpeg could not read the media.")
    return info


def _int(v: Any) -> int | None:
    return int(v) if str(v or "").isdigit() else None


def _parse_ffprobe_json(src: str, data: dict[str, Any]) -> MediaInfo:
    fmt: dict[str, Any] = data.get("format") or {}
    streams: list[StreamInfo] = []
    for st in data.get("streams") or []:
        ctype = st.get("codec_type")
        kind = ctype if ctype in ("video", "audio", "subtitle", "data") else "unknown"
        tags = st.get("tags") or {}
        streams.append(
            StreamInfo(
                index=int(st.get("index", len(streams))),
                type=kind,  # type: ignore[arg-type]
                codec=st.get("codec_name"),
                width=st.get("width"),
                height=st.get("height"),
                fps=_fps(st.get("avg_frame_rate") or st.get("r_frame_rate")),
                sample_rate=_int(st.get("sample_rate")),
                channels=st.get("channels"),
                bitrate=_int(st.get("bit_rate")),
                language=tags.get("language") if isinstance(tags, dict) else None,
            )
        )
    return MediaInfo(
        path=src,
        format_name=fmt.get("format_name"),
        duration=_num(fmt.get("duration")),
        size_bytes=_int(fmt.get("size")) or _local_size(src),
        bitrate=_int(fmt.get("bit_rate")),
        streams=streams,
    )


_DUR = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_BITRATE = re.compile(r"bitrate:\s*(\d+)\s*kb/s")
_INPUT = re.compile(r"Input #0,\s*([^,]+(?:,[^,]+)*?),\s*from")
_STREAM = re.compile(r"Stream #0:(\d+)(?:\[[^\]]*\])?(?:\((\w+)\))?:\s*(Video|Audio|Subtitle|Data):\s*([\w-]+)(.*)")


def _parse_ffmpeg_banner(src: str, stderr: str) -> MediaInfo:
    duration = None
    if m := _DUR.search(stderr):
        duration = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    bitrate = int(m.group(1)) * 1000 if (m := _BITRATE.search(stderr)) else None
    fmt = m.group(1).strip() if (m := _INPUT.search(stderr)) else None
    streams: list[StreamInfo] = []
    for line in stderr.splitlines():
        sm = _STREAM.search(line)
        if not sm:
            continue
        idx, lang, kind, codec, rest = sm.groups()
        width = height = None
        if wh := re.search(r"\b(\d{2,5})x(\d{2,5})\b", rest):
            width, height = int(wh.group(1)), int(wh.group(2))
        fps = float(f.group(1)) if (f := re.search(r"([\d.]+)\s*fps", rest)) else None
        sr = int(r.group(1)) if (r := re.search(r"(\d+)\s*Hz", rest)) else None
        ch = None
        if "stereo" in rest:
            ch = 2
        elif "mono" in rest:
            ch = 1
        elif c := re.search(r"(\d+)\.(\d) ?(?:\(|,)", rest):
            ch = int(c.group(1)) + int(c.group(2))
        br = int(b.group(1)) * 1000 if (b := re.search(r"(\d+)\s*kb/s", rest)) else None
        streams.append(
            StreamInfo(
                index=int(idx), type=kind.lower(), codec=codec, width=width, height=height,  # type: ignore[arg-type]
                fps=fps, sample_rate=sr, channels=ch, bitrate=br,
                language=lang if lang and lang != "und" else None,
            )
        )
    return MediaInfo(
        path=src, format_name=fmt, duration=duration, size_bytes=_local_size(src),
        bitrate=bitrate, streams=streams,
    )


def _local_size(src: str) -> int | None:
    try:
        return os.path.getsize(src) if os.path.exists(src) else None
    except OSError:
        return None


def extract_audio_for_asr(
    source: str | Path, dest: Path, *, headers: Mapping[str, str] | None = None,
    settings: Settings | None = None,
) -> Path:
    """Decode to 16 kHz mono WAV, the input Whisper models expect."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(
        ["-y", *header_args(headers), "-i", str(source), "-vn", "-ac", "1", "-ar", "16000",
         "-c:a", "pcm_s16le", str(dest)],
        settings=settings,
    )
    return dest
