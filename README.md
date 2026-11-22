# agent-sight

Eyes, ears and a download button for AI agents working with online video.

`agent-sight` lets Claude Code, Cursor, OpenCode, Windsurf, Codex, LangGraph, CrewAI or any
custom LLM loop **watch** public videos (YouTube, Instagram, TikTok, X/Twitter, Reddit, Vimeo,
Twitch clips, Facebook and direct MP4/WebM/HLS links) without pushing hours of footage or
50k-token transcripts into context.

- **Visual watching, not just transcripts.** Scene-change detection picks keyframes, which are
  packed into timestamped contact-sheet grids: nine frames cost about 1,030 vision tokens
  instead of about 2,770 for nine separate 640x360 images. Silent reels, UI demos, gameplay
  and b-roll become readable.
- **Progressive disclosure.** `probe` → `digest` → `watch` → `download`. Skim a video in about
  1,500 text tokens plus one contact sheet, then zoom into exact timestamps.
- **Three interfaces.** An MCP server (stdio and streamable HTTP) that returns image content
  blocks, a JSON CLI with stable exit codes and recovery hints, and one-command skill
  installation for agent hosts.
- **Doctor and cache.** `agent-sight doctor` checks ffmpeg, yt-dlp, the JavaScript runtime
  YouTube needs, Whisper, disk space and cookie-file permissions, and prints fix commands. A
  SQLite index plus a file cache means nothing is downloaded or transcribed twice.

## Quickstart

```bash
# Recommended: isolated install with the bundled ffmpeg and a JS runtime for YouTube
pipx install "agent-sight[ffmpeg,js]"        # or: uv tool install "agent-sight[ffmpeg,js]"
agent-sight doctor

# Optional extras
pipx inject agent-sight faster-whisper       # local speech-to-text when captions are missing
```

From a clone:

```bash
uv venv && uv pip install -e ".[dev]"
.venv/bin/agent-sight doctor
.venv/bin/pytest
```

## The four stages

```bash
agent-sight probe    "https://youtu.be/VIDEO_ID" --json                       # metadata only
agent-sight digest   "https://youtu.be/VIDEO_ID" --json                       # timeline + contact sheet
agent-sight watch    "https://youtu.be/VIDEO_ID" --start 1:30 --end 2:10 --json
agent-sight download "https://youtu.be/VIDEO_ID" --start 1:30 --end 2:10 --quality 720p --json
```

Other commands:

| Command | Purpose |
|---|---|
| `transcript URL [--start --end --format timestamped\|plain\|segments --max-tokens N]` | Paginated captions (native, auto, or local Whisper) |
| `frames URL --at 12,45,1:30` or `--start --end --count N` | Frames at exact timestamps |
| `scenes URL [--threshold 0.3]` | Scene-cut timestamps with scores |
| `download URL --mode audio\|subtitles\|thumbnail` | Audio-only, captions, thumbnail |
| `doctor [--network]` | Dependency checks with fix commands |
| `cache stats` / `cache clear [--url URL]` | Inspect or clear the cache |
| `mcp [--transport streamable-http --port 8765]` | Run the MCP server |
| `install-skill [--host ...] [--scope project\|global] [--mcp] [--dry-run]` | Install the agent skill |

Timestamps accept `90`, `1:30`, `01:02:03.5` and `1m30s`.

### What a digest looks like

```json
{"ok": true, "command": "digest", "data": {
  "title": "How the UI works", "duration": 16.0, "duration_label": "0:16",
  "transcript_source": "native", "transcript_coverage": 1.0,
  "scene_method": "ffmpeg-scene", "scene_count": 3,
  "contact_sheets": [{"path": ".../sheet_01.jpg", "width": 1168, "height": 692,
                      "labels": ["0:00", "0:02", "..."], "estimated_tokens": 1078}],
  "timeline": [
    {"start": 0.0, "label": "0:00", "kind": "transcript", "text": "Welcome to ..."},
    {"start": 5.0, "label": "0:05", "kind": "scene", "text": "sheet 1 tile 4, scene cut 0.89"}],
  "estimated_text_tokens": 533, "estimated_image_tokens": 1078, "token_budget": 1500,
  "next_actions": [{"tool": "watch_segment", "cli": "agent-sight watch ... --start 0 --end 4 --json"}]}}
```

`timeline` merges chapters, scene cuts and transcript excerpts in time order. Each contact
sheet tile has its timestamp burned in at the bottom-left. When the transcript doesn't fit
the budget, excerpts are spread evenly across the whole video and `transcript_coverage`
reports the fraction included.

## JSON contract and exit codes

With `--json` (or `AGENT_SIGHT_JSON=1`), every command prints exactly one JSON object, usage
errors included:

```json
{"ok": false, "command": "probe", "error": {
  "error_code": "LOGIN_REQUIRED", "message": "...",
  "agent_recovery_hint": "The platform requires sign-in ...", "retryable": false, "details": {}}}
```

| Exit | Meaning | Error codes |
|---|---|---|
| 0 | success | |
| 2 | bad input | `INVALID_URL`, `INVALID_ARGUMENT`, `INVALID_TIME_RANGE`, `UNSUPPORTED_URL`, `LIVE_STREAM` |
| 3 | missing dependency | `DEPENDENCY_MISSING` |
| 4 | access restricted | `LOGIN_REQUIRED`, `ACCESS_DENIED`, `PRIVATE_CONTENT`, `AGE_RESTRICTED`, `GEO_RESTRICTED`, `DRM_PROTECTED`, `PAYWALLED` |
| 5 | nothing there | `NOT_FOUND`, `NO_TRANSCRIPT`, `NO_VIDEO_STREAM` |
| 6 | transient, retry later | `NETWORK_ERROR`, `RATE_LIMITED` |
| 7 | processing failure | `EXTRACTION_FAILED`, `FFMPEG_FAILED`, `TRANSCRIPTION_FAILED` |
| 8 | size limit | `FILE_TOO_LARGE` |
| 1 | internal error | `INTERNAL_ERROR` |

## MCP server

```bash
agent-sight mcp                                          # stdio
agent-sight mcp --transport streamable-http --port 8765  # http://127.0.0.1:8765/mcp
agent-sight install-skill --print-mcp                    # config snippets for every host
```

Tools: `probe_video`, `digest_video`, `watch_segment`, `get_transcript`, `get_frames`,
`detect_scenes`, `download_media`, `doctor`. `digest_video`, `watch_segment` and `get_frames`
return the JSON result plus JPEG image blocks, so vision-capable models see the frames
directly. Errors are raised as tool errors whose text is the same JSON error object as the CLI.
The server works with MCP Python SDK v2 (`MCPServer`) and v1 (`FastMCP`).

Example for Claude Code:

```bash
claude mcp add agent-sight -- agent-sight mcp
```

Hardening for MCP use: only `http(s)` URLs are accepted, hosts that resolve to private,
loopback or link-local addresses are refused (SSRF protection), and `download_media` can only
write inside `AGENT_SIGHT_OUTPUT_DIR`. The HTTP transport binds to `127.0.0.1` by default.

## Agent skills

```bash
agent-sight install-skill                       # all hosts, into the current project
agent-sight install-skill --host claude-code --scope global
agent-sight install-skill --host cursor --mcp   # also registers the MCP server
agent-sight install-skill --dry-run --json      # show what would be written
```

| Host | Skill / rule | MCP config (`--mcp`) |
|---|---|---|
| Claude Code | `.claude/skills/agent-sight/SKILL.md` | `.mcp.json` (project) or the `claude mcp add` command (global) |
| OpenCode | `.opencode/skills/agent-sight/SKILL.md`, `~/.config/opencode/skills/` | `opencode.json` |
| Cursor | `.cursor/rules/agent-sight.mdc` | `.cursor/mcp.json`, `~/.cursor/mcp.json` |
| Windsurf | `.windsurf/rules/agent-sight.md` | `~/.codeium/windsurf/mcp_config.json` |
| Codex | `.agents/skills/agent-sight/SKILL.md` | `~/.codex/config.toml` |

Existing config files are merged: other MCP servers are preserved and a `.bak` copy is written
first. Files that agent-sight didn't create are not overwritten without `--force`. The skill
text lives in [`skills/SKILL.md`](skills/SKILL.md).

## Architecture

```text
             CLI (Typer, --json)      MCP server (stdio / streamable-http)
                        \                 /
                     pipeline/engine.py  AgentSight facade
          probe ─ digest (pipeline/digest.py) ─ watch_segment ─ transcript ─ download
             |            |                        |               |           |
   adapters/router.py ── URL normalisation, platform detection, SSRF guard
             |
   ┌─────────┴──────────┐
   ytdlp_adapter.py     generic_html.py      (fallback: <video>, og:video, JSON-LD, HLS/MP4)
   metadata, captions,  rate limiting + retries with exponential backoff (adapters/base.py)
   streams, downloads
             |
   vision/  scene_detector (ffmpeg scene filter | PySceneDetect) → frame_extractor (ffmpeg seek)
            → contact_sheet (Pillow grid, burned-in timestamps)
   audio/   transcripts (VTT/SRT/json3, rolling-caption dedupe, budget fitting) → transcriber (Whisper)
   downloader/ media_downloader (size limits, clips, proxies) · ffprobe_utils
   cache.py SQLite TTL index + content-addressed blobs
```

How the visual pipeline chooses its source:

- **Frames at timestamps** (`watch`, `frames`) seek directly into the remote stream with
  ffmpeg, so only the needed bytes are fetched. If the stream refuses seeking, a low-resolution
  proxy is downloaded instead.
- **Scene detection** (`digest`, `scenes`) needs to decode every frame, so videos up to
  `AGENT_SIGHT_PROXY_MAX_DURATION` (30 minutes by default) get a cached 360p proxy. Longer videos
  use chapters, or evenly spaced frames, to keep time and bandwidth bounded.
- The proxy is reused for Whisper transcription, so audio is never downloaded twice.

## Configuration

All settings use the `AGENT_SIGHT_` prefix and can live in `.env`. See
[`.env.example`](.env.example) for the full list. The main ones:

| Variable | Default | |
|---|---|---|
| `AGENT_SIGHT_CACHE_DIR` | `~/.cache/agent-sight` | metadata, transcripts, proxies, frames, sheets |
| `AGENT_SIGHT_OUTPUT_DIR` | `./downloads` | where downloads go (MCP downloads are confined here) |
| `AGENT_SIGHT_MAX_DOWNLOAD_MB` | `2048` | refuse larger downloads (checked before and after) |
| `AGENT_SIGHT_DEFAULT_QUALITY` | `720p` | `best`, `1080p`, `720p`, `480p`, `360p`, `worst` |
| `AGENT_SIGHT_WHISPER_MODEL` | `small` | faster-whisper model name |
| `AGENT_SIGHT_COOKIES_FILE` | unset | cookies.txt for content you are permitted to access |
| `AGENT_SIGHT_MIN_REQUEST_INTERVAL` | `1.0` | seconds between requests to the same host |
| `AGENT_SIGHT_ALLOW_PRIVATE_HOSTS` | `false` | allow localhost/LAN media servers |

## Scope and legality

agent-sight is for publicly accessible, non-DRM media and for content you own or are
authorised to access.

- It does **not** circumvent DRM, paywalls, logins, geo-blocks or CAPTCHAs. DRM-only formats
  are reported as `DRM_PROTECTED`, encrypted HLS is refused, and bot checks are reported
  instead of worked around.
- Cookies are only read from a file the machine owner configures (`AGENT_SIGHT_COOKIES_FILE`).
  Their contents and path are never logged, and `doctor` warns when the file is readable by
  other users.
- Requests to each host are rate-limited, and only transient failures (network errors, HTTP
  429/5xx) are retried, with exponential backoff and jitter.

Respect each platform's terms of service and copyright when downloading.

## Development

```bash
uv pip install -e ".[dev]"
pytest              # unit tests plus a local end-to-end pipeline run (needs ffmpeg; no internet)
ruff check src tests
mypy src
```

## License

MIT
