"""Agent-first CLI. Every command accepts ``--json`` and then prints exactly one JSON object.

Envelope: ``{"ok": true, "command": "...", "data": {...}}`` or
``{"ok": false, "command": "...", "error": {"error_code", "message", "agent_recovery_hint", ...}}``.
Exit codes are documented in :mod:`agent_sight.errors`.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections.abc import Callable
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

import typer
from pydantic import BaseModel, ValidationError
from rich.console import Console
from rich.markup import escape as esc
from rich.table import Table

from agent_sight import __version__
from agent_sight.errors import EXIT_DEPENDENCY, EXIT_USAGE, AgentError, ErrorCode
from agent_sight.models import DownloadRequest, format_timestamp, parse_timestamp


def _exc_types(*paths: str) -> tuple[type[BaseException], ...]:
    """Resolve exception classes from click and/or typer's vendored click (typer >= 0.26)."""
    found: list[type[BaseException]] = []
    for path in paths:
        mod, _, name = path.rpartition(".")
        try:
            cls = getattr(__import__(mod, fromlist=[name]), name)
        except (ImportError, AttributeError):
            continue
        if cls not in found:
            found.append(cls)
    return tuple(found)


_USAGE_ERRORS = _exc_types("click.exceptions.UsageError", "typer._click.exceptions.UsageError")
_ABORTS = _exc_types("click.exceptions.Abort", "typer.exceptions.Abort")
_EXITS = _exc_types("click.exceptions.Exit", "typer.exceptions.Exit")

app = typer.Typer(
    name="agent-sight",
    help="Multimodal video watching, transcripts and downloads for AI agents.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)
cache_app = typer.Typer(help="Inspect or clear the local cache.", no_args_is_help=True)
app.add_typer(cache_app, name="cache")

out = Console()
err_console = Console(stderr=True)


class _State:
    json: bool = False
    command: str = "agent-sight"


state = _State()

JsonOpt = Annotated[bool, typer.Option("--json", help="Print one machine-readable JSON object.")]
RefreshOpt = Annotated[bool, typer.Option("--refresh", help="Ignore cached results.")]


class Mode(StrEnum):
    video = "video"
    audio = "audio"
    subtitles = "subtitles"
    thumbnail = "thumbnail"


class Quality(StrEnum):
    best = "best"
    p1080 = "1080p"
    p720 = "720p"
    p480 = "480p"
    p360 = "360p"
    worst = "worst"


class AudioFormat(StrEnum):
    m4a = "m4a"
    mp3 = "mp3"
    opus = "opus"
    wav = "wav"
    best = "best"


class Container(StrEnum):
    mp4 = "mp4"
    webm = "webm"
    mkv = "mkv"


class TranscriptFormat(StrEnum):
    timestamped = "timestamped"
    plain = "plain"
    segments = "segments"


class SceneMethod(StrEnum):
    auto = "auto"
    ffmpeg = "ffmpeg"
    pyscenedetect = "pyscenedetect"
    uniform = "uniform"


class Transport(StrEnum):
    stdio = "stdio"
    http = "streamable-http"


class HostChoice(StrEnum):
    all = "all"
    claude_code = "claude-code"
    opencode = "opencode"
    cursor = "cursor"
    windsurf = "windsurf"
    codex = "codex"


class ScopeChoice(StrEnum):
    project = "project"
    global_ = "global"


@lru_cache(maxsize=1)
def _engine() -> Any:
    from agent_sight.pipeline.engine import AgentSight

    return AgentSight()


def _json_mode(flag: bool) -> bool:
    return flag or state.json or os.environ.get("AGENT_SIGHT_JSON", "").lower() in ("1", "true", "yes")


def _to_data(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, list):
        return [_to_data(v) for v in value]
    return value


def _print_json(obj: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
    sys.stdout.flush()


def _fail(command: str, error: AgentError, as_json: bool) -> None:
    if as_json:
        _print_json({"ok": False, "command": command, "error": error.to_dict()})
    else:
        err_console.print(f"[bold red]{error.code.value}[/]: {esc(error.message)}")
        if error.hint:
            err_console.print(f"[yellow]hint:[/] {esc(error.hint)}")
    raise typer.Exit(error.exit_code)


def _run(command: str, as_json: bool, fn: Callable[[], Any], human: Callable[[Any], None] | None = None) -> None:
    as_json = _json_mode(as_json)
    state.command = command
    try:
        result = fn()
    except AgentError as error:
        _fail(command, error, as_json)
    except ValidationError as exc:
        first = exc.errors()[0]
        _fail(command, AgentError(ErrorCode.INVALID_ARGUMENT,
                                  f"{'.'.join(map(str, first['loc'])) or 'input'}: {first['msg']}"), as_json)
    except KeyboardInterrupt:
        _fail(command, AgentError(ErrorCode.INTERNAL_ERROR, "Interrupted.", retryable=True), as_json)
    except typer.Exit:
        raise
    except Exception as exc:
        logging.getLogger("agent_sight").debug("unhandled error", exc_info=True)
        _fail(command, AgentError(ErrorCode.INTERNAL_ERROR, f"{type(exc).__name__}: {exc}"), as_json)
    data = _to_data(result)
    if as_json:
        _print_json({"ok": True, "command": command, "data": data})
    elif human is not None:
        human(result)
    else:
        out.print_json(data=data)


def _ts(value: str | None) -> float | None:
    return parse_timestamp(value) if value is not None else None


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=level, stream=sys.stderr, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING if not verbose else logging.DEBUG)


def _version_cb(value: bool) -> None:
    if value:
        sys.stdout.write(f"agent-sight {__version__}\n")
        raise typer.Exit()


@app.callback()
def _root(
    json_: Annotated[bool, typer.Option("--json", help="JSON output for every command.")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Debug logs on stderr.")] = False,
    version: Annotated[bool | None, typer.Option("--version", callback=_version_cb, is_eager=True,
                                                 help="Show the version and exit.")] = None,
) -> None:
    state.json = json_
    _setup_logging(verbose)


# ============================================================================ human renderers


def _human_probe(r: Any) -> None:
    m = r.metadata
    t = Table(show_header=False, box=None, pad_edge=False)
    for k, v in [
        ("Title", m.title), ("Platform", f"{m.platform} ({m.extractor})"), ("Uploader", m.uploader),
        ("Duration", format_timestamp(m.duration)), ("Uploaded", m.upload_date),
        ("Views", f"{m.view_count:,}" if m.view_count else None),
        ("Resolution", f"{m.width}x{m.height}" if m.width else None),
        ("Captions", ", ".join(m.subtitle_languages) or None),
        ("Auto captions", ", ".join(m.auto_caption_languages[:8]) or None),
        ("Transcript", r.transcript_source), ("Chapters", str(len(m.chapters)) if m.chapters else None),
        ("DRM", "yes" if m.has_drm else None), ("Live", "yes" if m.is_live else None),
    ]:
        if v:
            t.add_row(f"[bold]{k}[/]", esc(str(v)))
    out.print(t)
    out.print("\n[bold]Next:[/]")
    for a in r.next_actions:
        out.print(f"  [cyan]{esc(a.cli)}[/]  [dim]# {esc(a.reason)}[/]")


def _human_digest(d: Any) -> None:
    out.print(f"[bold]{esc(str(d.title))}[/]  ({d.duration_label}, {d.platform})")
    if d.description_excerpt:
        out.print(f"[dim]{esc(d.description_excerpt)}[/]")
    out.print(f"transcript: {d.transcript_source} (coverage {d.transcript_coverage:.0%}), "
              f"scenes: {d.scene_count} via {d.scene_method}")
    for e in d.timeline:
        color = {"chapter": "magenta", "scene": "cyan", "transcript": "white"}[e.kind]
        out.print(f"[{color}]{e.label:>8} {e.kind:<10}[/] {esc(e.text)}")
    for s in d.contact_sheets:
        out.print(f"[green]contact sheet[/] {esc(s.path)}  ({s.width}x{s.height}, ~{s.estimated_tokens} tokens)")
    for w in d.warnings:
        out.print(f"[yellow]warning:[/] {esc(w)}")
    out.print(f"[dim]~{d.estimated_text_tokens} text tokens + ~{d.estimated_image_tokens} image tokens[/]")


def _human_segment(r: Any) -> None:
    out.print(f"[bold]{r.label}[/]  {len(r.frames)} frames from {r.source}")
    for s in r.contact_sheets:
        out.print(f"[green]contact sheet[/] {esc(s.path)}")
    if not r.contact_sheets:
        for f in r.frames:
            out.print(f"  {f.label:>8}  {esc(f.path)}")
    if r.transcript:
        out.print(r.transcript, markup=False, highlight=False)


def _human_transcript(r: Any) -> None:
    if r.text:
        out.print(r.text, markup=False, highlight=False)
    else:
        out.print_json(data=_to_data(r))
    if r.truncated:
        err_console.print(f"[yellow]truncated[/]: continue with --start {r.next_start}")


def _human_download(r: Any) -> None:
    for f in r.files:
        out.print(f"[green]saved[/] {esc(f)}")
    out.print(f"[dim]{r.size_bytes / 1e6:.1f} MB{' (cached)' if r.cached else ''}[/]")


def _human_doctor(r: Any) -> None:
    icons = {"ok": "[green]✔[/]", "warn": "[yellow]![/]", "fail": "[red]✘[/]"}
    for c in r.checks:
        out.print(f"{icons[c.status]} [bold]{c.name:<12}[/] {esc(c.detail)}")
        if c.fix and c.status != "ok":
            out.print(f"    [cyan]fix:[/] {esc(c.fix)}")
    out.print(f"\n{r.summary}")


# ============================================================================ commands


@app.command()
def probe(
    url: Annotated[str, typer.Argument(help="Video page or media URL.")],
    refresh: RefreshOpt = False,
    json_: JsonOpt = False,
) -> None:
    """Stage 1: metadata, formats, captions and chapters. Downloads no media."""
    _run("probe", json_, lambda: _engine().probe(url, refresh=refresh), _human_probe)


@app.command()
def digest(
    url: Annotated[str, typer.Argument(help="Video page or media URL.")],
    budget: Annotated[int, typer.Option("--budget", min=200, max=50000, help="Text token budget.")] = 1500,
    frames: Annotated[int, typer.Option("--frames", min=0, max=36, help="Keyframes on the contact sheet(s).")] = 9,
    lang: Annotated[str, typer.Option("--lang", help="Preferred caption languages, comma separated.")] = "en",
    no_frames: Annotated[bool, typer.Option("--no-frames", help="Skip visual analysis.")] = False,
    whisper: Annotated[bool | None, typer.Option("--whisper/--no-whisper",
                                                 help="Allow local transcription if captions are missing.")] = None,
    refresh: RefreshOpt = False,
    json_: JsonOpt = False,
) -> None:
    """Stage 2: budgeted overview (timeline + transcript excerpts + contact sheet)."""
    _run("digest", json_, lambda: _engine().digest(
        url, token_budget=budget, max_frames=frames, language=lang, include_frames=not no_frames,
        allow_whisper=whisper, refresh=refresh), _human_digest)


@app.command()
def watch(
    url: Annotated[str, typer.Argument(help="Video page or media URL.")],
    start: Annotated[str, typer.Option("--start", "-s", help="Start time (90, 1:30, 1m30s).")],
    end: Annotated[str, typer.Option("--end", "-e", help="End time.")],
    frames: Annotated[int, typer.Option("--frames", "-n", min=1, max=48)] = 6,
    lang: Annotated[str, typer.Option("--lang")] = "en",
    width: Annotated[int, typer.Option("--width", min=160, max=1920, help="Frame width in px.")] = 640,
    no_transcript: Annotated[bool, typer.Option("--no-transcript")] = False,
    no_sheet: Annotated[bool, typer.Option("--no-sheet", help="Individual frames instead of a grid.")] = False,
    json_: JsonOpt = False,
) -> None:
    """Stage 3: dense frames + transcript for one time range."""
    def go() -> Any:
        s, e = _ts(start), _ts(end)
        assert s is not None and e is not None
        return _engine().watch_segment(
            url, start=s, end=e, frames=frames, include_transcript=not no_transcript,
            language=lang, width=width, contact_sheet=not no_sheet)
    _run("watch", json_, go, _human_segment)


@app.command()
def transcript(
    url: Annotated[str, typer.Argument(help="Video page or media URL.")],
    lang: Annotated[str, typer.Option("--lang")] = "en",
    start: Annotated[str | None, typer.Option("--start", "-s")] = None,
    end: Annotated[str | None, typer.Option("--end", "-e")] = None,
    fmt: Annotated[TranscriptFormat, typer.Option("--format", "-f")] = TranscriptFormat.timestamped,
    max_tokens: Annotated[int, typer.Option("--max-tokens", min=100, max=200000)] = 4000,
    whisper: Annotated[bool | None, typer.Option("--whisper/--no-whisper")] = None,
    refresh: RefreshOpt = False,
    json_: JsonOpt = False,
) -> None:
    """Captions (native, auto or local Whisper), paginated by token budget."""
    _run("transcript", json_, lambda: _engine().transcript(
        url, language=lang, start=_ts(start), end=_ts(end), fmt=fmt.value, max_tokens=max_tokens,
        allow_whisper=whisper, refresh=refresh), _human_transcript)


@app.command()
def frames(
    url: Annotated[str, typer.Argument(help="Video page or media URL.")],
    at: Annotated[str | None, typer.Option("--at", help="Exact timestamps, comma separated.")] = None,
    start: Annotated[str | None, typer.Option("--start", "-s")] = None,
    end: Annotated[str | None, typer.Option("--end", "-e")] = None,
    count: Annotated[int, typer.Option("--count", "-n", min=1, max=48)] = 9,
    width: Annotated[int, typer.Option("--width", min=160, max=1920)] = 640,
    no_sheet: Annotated[bool, typer.Option("--no-sheet")] = False,
    json_: JsonOpt = False,
) -> None:
    """Frames at exact timestamps, or evenly spaced over a range."""
    def go() -> Any:
        ts = [t for t in (_ts(x) for x in at.split(",")) if t is not None] if at else None
        return _engine().frames(url, timestamps=ts, start=_ts(start), end=_ts(end), count=count,
                                width=width, contact_sheet=not no_sheet)
    _run("frames", json_, go)


@app.command()
def scenes(
    url: Annotated[str, typer.Argument(help="Video page or media URL.")],
    threshold: Annotated[float, typer.Option("--threshold", min=0.01, max=0.99)] = 0.3,
    max_scenes: Annotated[int, typer.Option("--max", min=1, max=1000)] = 60,
    method: Annotated[SceneMethod, typer.Option("--method")] = SceneMethod.auto,
    start: Annotated[str | None, typer.Option("--start", "-s")] = None,
    end: Annotated[str | None, typer.Option("--end", "-e")] = None,
    refresh: RefreshOpt = False,
    json_: JsonOpt = False,
) -> None:
    """Scene-change timestamps (visual cuts)."""
    _run("scenes", json_, lambda: _engine().scenes(
        url, threshold=threshold, max_scenes=max_scenes, method=method.value, start=_ts(start),
        end=_ts(end), refresh=refresh))


@app.command()
def download(
    url: Annotated[str, typer.Argument(help="Video page or media URL.")],
    mode: Annotated[Mode, typer.Option("--mode", "-m")] = Mode.video,
    quality: Annotated[Quality | None, typer.Option("--quality", "-q")] = None,
    start: Annotated[str | None, typer.Option("--start", "-s", help="Clip start.")] = None,
    end: Annotated[str | None, typer.Option("--end", "-e", help="Clip end.")] = None,
    audio_format: Annotated[AudioFormat, typer.Option("--audio-format")] = AudioFormat.m4a,
    container: Annotated[Container, typer.Option("--container")] = Container.mp4,
    lang: Annotated[str, typer.Option("--lang", help="Subtitle languages for --mode subtitles.")] = "en",
    output_dir: Annotated[Path | None, typer.Option("--output-dir", "-o")] = None,
    filename: Annotated[str | None, typer.Option("--filename", help="File stem without extension.")] = None,
    refresh: RefreshOpt = False,
    json_: JsonOpt = False,
) -> None:
    """Stage 4: download video, audio, subtitles, thumbnail or an exact clip."""
    def go() -> Any:
        req = DownloadRequest(
            url=url, mode=mode.value, quality=quality.value if quality else None,
            start=_ts(start), end=_ts(end), audio_format=audio_format.value,
            container=container.value, subtitle_languages=[x.strip() for x in lang.split(",") if x.strip()],
            filename=filename,
        )
        return _engine().download(req, output_dir=output_dir, refresh=refresh)
    _run("download", json_, go, _human_download)


@app.command()
def doctor(
    network: Annotated[bool, typer.Option("--network", help="Also test connectivity.")] = False,
    json_: JsonOpt = False,
) -> None:
    """Check dependencies and print exact fix commands."""
    from agent_sight.doctor import run_doctor

    report = run_doctor(network=network)
    _run("doctor", json_, lambda: report, _human_doctor)
    if not report.ok:
        raise typer.Exit(EXIT_DEPENDENCY)


@cache_app.command("stats")
def cache_stats(json_: JsonOpt = False) -> None:
    """Show cache location, entry counts and size."""
    _run("cache.stats", json_, lambda: _engine().cache.stats())


@cache_app.command("clear")
def cache_clear(
    url: Annotated[str | None, typer.Option("--url", help="Only clear entries for this video.")] = None,
    json_: JsonOpt = False,
) -> None:
    """Delete cached entries (all, or for one URL)."""
    def go() -> Any:
        eng = _engine()
        target = eng.resolve(url).normalized if url else None
        return {"removed_entries": eng.cache.clear(target), "url": target}
    _run("cache.clear", json_, go)


@app.command()
def mcp(
    transport: Annotated[Transport, typer.Option("--transport", "-t")] = Transport.stdio,
    host: Annotated[str, typer.Option("--host", help="Bind address for streamable-http.")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", min=1, max=65535)] = 8765,
) -> None:
    """Run the MCP server (stdio for local agents, streamable-http for remote clients)."""
    from agent_sight.interfaces.mcp_server import run_server

    if host not in ("127.0.0.1", "localhost", "::1"):
        err_console.print(
            f"[yellow]warning:[/] binding to {host} exposes download tools to the network; "
            "put it behind authentication."
        )
    run_server(transport.value, host=host, port=port)


@app.command("install-skill")
def install_skill(
    host: Annotated[list[HostChoice] | None, typer.Option("--host", help="Target host(s); repeatable.")] = None,
    scope: Annotated[ScopeChoice, typer.Option("--scope")] = ScopeChoice.project,
    directory: Annotated[Path | None, typer.Option("--dir", help="Project directory (default: cwd).")] = None,
    with_mcp: Annotated[bool, typer.Option("--mcp", help="Also register the MCP server in host configs.")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    force: Annotated[bool, typer.Option("--force", help="Overwrite files not created by agent-sight.")] = False,
    print_mcp: Annotated[bool, typer.Option("--print-mcp", help="Only print MCP config snippets.")] = False,
    json_: JsonOpt = False,
) -> None:
    """Install SKILL.md / rules for Claude Code, OpenCode, Cursor, Windsurf and Codex."""
    from agent_sight.interfaces import skill_installer as si

    def go() -> Any:
        if print_mcp:
            return si.mcp_snippets()
        chosen = host or [HostChoice.all]
        hosts = list(si.HOSTS) if HostChoice.all in chosen else [h.value for h in chosen]
        actions = si.install(hosts, scope=scope.value, project_dir=directory, with_mcp=with_mcp,  # type: ignore[arg-type]
                             dry_run=dry_run, force=force)
        return {"actions": [a.to_dict() for a in actions]}

    def human(r: Any) -> None:
        if "actions" not in r:
            out.print_json(data=r)
            return
        for a in r["actions"]:
            out.print(f"[bold]{a['action']:<11}[/] {a['host']:<12} {a['kind']:<5} {esc(a['path'])}"
                      + (f"  [dim]({esc(a['note'])})[/]" if a["note"] else ""))

    _run("install-skill", json_, go, human)


@app.command()
def version(json_: JsonOpt = False) -> None:
    """Print the agent-sight version."""
    _run("version", json_, lambda: {"version": __version__},
         lambda r: out.print(f"agent-sight {r['version']}"))


# ============================================================================ entry point


def main(argv: list[str] | None = None) -> None:
    """Console entry point that turns usage errors into JSON when ``--json`` is present."""
    args = sys.argv[1:] if argv is None else argv
    wants_json = "--json" in args or os.environ.get("AGENT_SIGHT_JSON", "").lower() in ("1", "true", "yes")
    try:
        rv = app(args=args, prog_name="agent-sight", standalone_mode=False)
    except _USAGE_ERRORS as exc:
        if wants_json:
            command = next((a for a in args if not a.startswith("-")), "agent-sight")
            _print_json({"ok": False, "command": command, "error": AgentError(
                ErrorCode.INVALID_ARGUMENT, exc.format_message(),  # type: ignore[attr-defined]
                "Run `agent-sight <command> --help` for valid options.").to_dict()})
        else:
            exc.show()  # type: ignore[attr-defined]
        sys.exit(EXIT_USAGE)
    except _ABORTS:
        sys.exit(130)
    except _EXITS as exc:
        sys.exit(getattr(exc, "exit_code", 1))
    # In non-standalone mode click returns the exit code of typer.Exit instead of raising.
    sys.exit(rv if isinstance(rv, int) else 0)


if __name__ == "__main__":
    main()
