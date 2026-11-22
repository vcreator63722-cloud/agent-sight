---
name: agent-sight
description: Watch, read and download public online videos (YouTube, Instagram, TikTok, X/Twitter, Reddit, Vimeo, Twitch clips, direct MP4/WebM/HLS links). Use when the user shares a video URL, asks what happens in a video, needs a transcript or quote with timestamps, wants to see specific moments, or wants a video, audio track or clip saved locally.
---

# agent-sight: video eyes, ears and downloads

`agent-sight` lets you inspect a video without dumping it into context. Always work
**cheapest first** and stop as soon as you can answer.

| Stage | Command | Typical cost |
|---|---|---|
| 1. probe | `agent-sight probe URL --json` | ~300-800 text tokens, no media downloaded |
| 2. digest | `agent-sight digest URL --json` | <=1,500 text tokens + one ~1,000-token contact sheet |
| 3. watch | `agent-sight watch URL --start 1:30 --end 2:10 --json` | 6 frames on one sheet + that range's transcript |
| 4. download | `agent-sight download URL [--mode audio] [--start .. --end ..] --json` | writes files to disk |

If the MCP server is connected, use the matching tools instead of the CLI:
`probe_video`, `digest_video`, `watch_segment`, `get_transcript`, `get_frames`,
`detect_scenes`, `download_media`, `doctor`.

## Workflow

1. **probe** to learn duration, chapters, caption availability and whether the media is
   DRM-protected, live or restricted.
2. **digest** to skim. Read `timeline` (chapters, scene cuts, transcript excerpts, sorted by
   time). Open the image files in `contact_sheets[].path`: each tile has its timestamp burned
   in at the bottom-left, and `timeline` entries of kind `scene` tell you the sheet and tile.
   `transcript_coverage < 1` means only excerpts are included.
3. **watch** a range when you need detail: what is on screen, UI steps, text in frames, or
   who is speaking. Use `--frames 4..12`. Narrow ranges give denser frames.
4. **transcript** for exact quotes: `agent-sight transcript URL --start 5:00 --end 9:00 --json`.
   If `truncated` is true, call again with `--start <next_start>`.
5. **download** only when the user wants a file. Prefer the smallest useful artifact:
   `--mode audio`, `--mode subtitles`, `--quality 480p`, or a clip with `--start/--end`.

For visual-only videos (no speech, b-roll, screen recordings, gameplay) skip transcripts
and go straight to digest + watch.

Timestamps accept `90`, `1:30`, `01:01:30.5` or `1m30s`.

## Reading results

Every CLI command with `--json` prints exactly one JSON object to stdout:

```json
{"ok": true, "command": "digest", "data": {...}}
{"ok": false, "command": "digest", "error": {"error_code": "LOGIN_REQUIRED", "message": "...", "agent_recovery_hint": "...", "retryable": false}}
```

Exit codes: `0` ok, `2` bad input, `3` missing dependency, `4` access restricted
(login/private/DRM/geo/paywall), `5` not found / no transcript, `6` transient (network,
rate limit: retry later), `7` processing failure, `8` size limit.

Always follow `agent_recovery_hint`. Results are cached, so repeating a call is cheap.

## Rules

- Only public content, or content the user owns or is authorised to access.
- Never try to bypass DRM, paywalls, logins, geo-blocks or CAPTCHAs. On `DRM_PROTECTED`,
  `PAYWALLED`, `GEO_RESTRICTED`, `PRIVATE_CONTENT` or `LOGIN_REQUIRED`, report it to the user
  and stop. Cookies are configured only by the machine owner (`AGENT_SIGHT_COOKIES_FILE`).
- Don't download whole videos just to look at them; digest and watch stream only what they need.
- On `DEPENDENCY_MISSING`, run `agent-sight doctor --json` and show the user the `fix` commands.
- Cite timestamps (e.g. "at 3:12") when describing video content.
