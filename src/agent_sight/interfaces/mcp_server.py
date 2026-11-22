"""MCP server exposing agent-sight tools, returning contact sheets as image content blocks.

Works with the MCP Python SDK v2 (``MCPServer``) and v1 (``FastMCP``).
"""

from __future__ import annotations

import functools
import json
import logging
from collections.abc import Callable
from typing import Annotated, Any, Literal, TypeVar

import anyio
from pydantic import BaseModel, Field

from agent_sight import __version__
from agent_sight.errors import AgentError, ErrorCode
from agent_sight.models import ContactSheet, DownloadRequest, Frame, parse_timestamp

try:  # MCP Python SDK >= 2
    from mcp.server.mcpserver import Image
    from mcp.server.mcpserver import MCPServer as _Server
    from mcp.server.mcpserver.exceptions import ToolError

    _SDK_V2 = True
except ImportError:  # MCP Python SDK 1.x
    from mcp.server.fastmcp import FastMCP as _Server  # type: ignore[no-redef,attr-defined]
    from mcp.server.fastmcp import Image  # type: ignore[no-redef,attr-defined]
    from mcp.server.fastmcp.exceptions import ToolError  # type: ignore[no-redef]

    _SDK_V2 = False

from mcp.types import ToolAnnotations

log = logging.getLogger("agent_sight.mcp")
T = TypeVar("T")

INSTRUCTIONS = """\
agent-sight gives you eyes and ears for public online videos. Work cheapest first:
1. probe_video: metadata, duration, chapters, caption availability (no media download).
2. digest_video: <=1500-token text overview + a timestamped contact-sheet image.
3. watch_segment: dense frames + transcript for one time range you care about.
4. get_transcript: exact quotes, paginated (follow next_start when truncated).
5. download_media: only when the user wants a file; prefer audio, clips or lower quality.
Each contact-sheet tile has its timestamp burned in. Cite timestamps in answers.
Never attempt to bypass DRM, paywalls, logins or geo-blocks: on such errors, report and stop.
Errors are JSON with error_code and agent_recovery_hint; follow the hint."""

TimeArg = Annotated[
    str | float | None,
    Field(description="Seconds (90) or a timestamp string ('1:30', '01:02:03.5', '1m30s')."),
]

# model_validate with wire (camelCase) names works with both SDK v1 and v2 field naming.
_READ = ToolAnnotations.model_validate(
    {"readOnlyHint": True, "openWorldHint": True, "idempotentHint": True})
_WRITE = ToolAnnotations.model_validate(
    {"readOnlyHint": False, "destructiveHint": False, "openWorldHint": True})


@functools.lru_cache(maxsize=1)
def _engine() -> Any:
    from agent_sight.pipeline.engine import AgentSight

    return AgentSight()


async def _call(fn: Callable[[], T]) -> T:
    try:
        return await anyio.to_thread.run_sync(fn)
    except AgentError as err:
        raise ToolError(json.dumps({"ok": False, **err.to_dict()})) from err
    except ValueError as exc:  # pydantic validation errors subclass ValueError
        raise ToolError(json.dumps({"ok": False, **AgentError(
            ErrorCode.INVALID_ARGUMENT, str(exc).splitlines()[0]).to_dict()})) from exc


def _json(model: BaseModel | dict[str, Any]) -> str:
    if isinstance(model, BaseModel):
        return model.model_dump_json(exclude_none=True)
    return json.dumps(model, default=str)


def _images(sheets: list[ContactSheet], frames: list[Frame], include: bool) -> list[Any]:
    if not include:
        return []
    if sheets:
        return [Image(path=s.path) for s in sheets]
    return [Image(path=f.path) for f in frames]


def _t(v: str | float | None) -> float | None:
    return parse_timestamp(v)


def build_server() -> Any:
    server = _Server(name="agent-sight", instructions=INSTRUCTIONS, version=__version__) if _SDK_V2 \
        else _Server(name="agent-sight", instructions=INSTRUCTIONS)

    def tool(annotations: ToolAnnotations) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return server.tool(annotations=annotations, structured_output=False)  # type: ignore[no-any-return]

    @tool(_READ)
    async def probe_video(
        url: Annotated[str, Field(description="Video page URL or direct media URL.")],
        refresh: bool = False,
    ) -> str:
        """Stage 1. Metadata only (title, duration, chapters, captions, formats, DRM/live flags)."""
        return _json(await _call(lambda: _engine().probe(url, refresh=refresh)))

    @tool(_READ)
    async def digest_video(
        url: str,
        token_budget: Annotated[int, Field(ge=200, le=50000, description="Text token budget.")] = 1500,
        max_frames: Annotated[int, Field(ge=0, le=36, description="Keyframes on the contact sheet.")] = 9,
        language: Annotated[str, Field(description="Preferred caption languages, comma separated.")] = "en",
        include_images: Annotated[bool, Field(description="Attach contact-sheet images.")] = True,
        allow_whisper: Annotated[bool | None, Field(description="Local speech-to-text if no captions.")] = None,
    ) -> list[Any]:
        """Stage 2. Skim a whole video: merged timeline of chapters, scene cuts and transcript
        excerpts within a token budget, plus a timestamped keyframe contact sheet image."""
        d = await _call(lambda: _engine().digest(
            url, token_budget=token_budget, max_frames=max_frames if include_images else 0,
            language=language, include_frames=include_images and max_frames > 0,
            allow_whisper=allow_whisper))
        return [_json(d), *_images(d.contact_sheets, [], include_images)]

    @tool(_READ)
    async def watch_segment(
        url: str,
        start: TimeArg,
        end: TimeArg,
        frames: Annotated[int, Field(ge=1, le=48)] = 6,
        include_transcript: bool = True,
        language: str = "en",
        as_contact_sheet: Annotated[bool, Field(description="One grid image instead of separate frames.")] = True,
        width: Annotated[int, Field(ge=160, le=1920)] = 640,
    ) -> list[Any]:
        """Stage 3. Look closely at one time range: scene-aware frames (as images) plus the
        transcript for that range."""
        def go() -> Any:
            s, e = _t(start), _t(end)
            if s is None or e is None:
                raise AgentError(ErrorCode.INVALID_TIME_RANGE, "start and end are required.")
            return _engine().watch_segment(url, start=s, end=e, frames=frames,
                                           include_transcript=include_transcript, language=language,
                                           width=width, contact_sheet=as_contact_sheet)
        r = await _call(go)
        return [_json(r), *_images(r.contact_sheets, r.frames, True)]

    @tool(_READ)
    async def get_transcript(
        url: str,
        language: str = "en",
        start: TimeArg = None,
        end: TimeArg = None,
        format: Literal["timestamped", "plain", "segments"] = "timestamped",
        max_tokens: Annotated[int, Field(ge=100, le=200000)] = 4000,
        allow_whisper: bool | None = None,
    ) -> str:
        """Timestamped transcript (native captions, auto captions, or local Whisper), paginated.
        If the result has truncated=true, call again with start=next_start."""
        return _json(await _call(lambda: _engine().transcript(
            url, language=language, start=_t(start), end=_t(end), fmt=format,
            max_tokens=max_tokens, allow_whisper=allow_whisper)))

    @tool(_READ)
    async def get_frames(
        url: str,
        timestamps: Annotated[list[float] | None, Field(description="Exact seconds to capture.")] = None,
        start: TimeArg = None,
        end: TimeArg = None,
        count: Annotated[int, Field(ge=1, le=48)] = 9,
        width: Annotated[int, Field(ge=160, le=1920)] = 640,
        as_contact_sheet: bool = True,
    ) -> list[Any]:
        """Frames at exact timestamps, or `count` evenly spaced frames over [start, end]."""
        r = await _call(lambda: _engine().frames(
            url, timestamps=timestamps, start=_t(start), end=_t(end), count=count, width=width,
            contact_sheet=as_contact_sheet))
        return [_json(r), *_images(r.contact_sheets, r.frames, True)]

    @tool(_READ)
    async def detect_scenes(
        url: str,
        threshold: Annotated[float, Field(gt=0, lt=1, description="Higher = only hard cuts.")] = 0.3,
        max_scenes: Annotated[int, Field(ge=1, le=1000)] = 60,
        method: Literal["auto", "ffmpeg", "pyscenedetect", "uniform"] = "auto",
        start: TimeArg = None,
        end: TimeArg = None,
    ) -> str:
        """Timestamps of visual scene changes (useful to pick ranges for watch_segment)."""
        return _json(await _call(lambda: _engine().scenes(
            url, threshold=threshold, max_scenes=max_scenes, method=method, start=_t(start), end=_t(end))))

    @tool(_WRITE)
    async def download_media(
        url: str,
        mode: Literal["video", "audio", "subtitles", "thumbnail"] = "video",
        quality: Literal["best", "1080p", "720p", "480p", "360p", "worst"] | None = None,
        start: TimeArg = None,
        end: TimeArg = None,
        audio_format: Literal["m4a", "mp3", "opus", "wav", "best"] = "m4a",
        container: Literal["mp4", "webm", "mkv"] = "mp4",
        subtitle_languages: list[str] | None = None,
        subdirectory: Annotated[str | None, Field(
            description="Optional relative folder inside the configured download directory.")] = None,
        filename: Annotated[str | None, Field(description="File stem without extension or path.")] = None,
    ) -> str:
        """Stage 4. Save video, audio, subtitles, a thumbnail or an exact clip (start/end) to the
        configured download directory. Returns the file paths."""
        def go() -> Any:
            req = DownloadRequest(
                url=url, mode=mode, quality=quality, start=_t(start), end=_t(end),
                audio_format=audio_format, container=container,
                subtitle_languages=subtitle_languages or ["en"], filename=filename,
            )
            return _engine().download(req, output_dir=subdirectory, confine=True)
        return _json(await _call(go))

    @tool(_READ)
    async def doctor() -> str:
        """Check ffmpeg, yt-dlp, JS runtime, Whisper and cache health with fix commands."""
        from agent_sight.doctor import run_doctor

        return _json(await _call(run_doctor))

    return server


def run_server(transport: str = "stdio", *, host: str = "127.0.0.1", port: int = 8765) -> None:
    server = build_server()
    if transport == "stdio":
        server.run("stdio")
    elif transport == "streamable-http":
        if _SDK_V2:
            server.run("streamable-http", host=host, port=port)
        else:
            server.settings.host = host
            server.settings.port = port
            server.run("streamable-http")
    else:
        raise AgentError(ErrorCode.INVALID_ARGUMENT, f"Unknown transport {transport!r}.")
