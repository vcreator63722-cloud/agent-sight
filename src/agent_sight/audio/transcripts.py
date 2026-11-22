"""Caption parsing (WebVTT, SRT, YouTube json3), cleaning and token-budgeted condensing."""

from __future__ import annotations

import html
import json
import re

from agent_sight.errors import AgentError, ErrorCode
from agent_sight.models import (
    Transcript,
    TranscriptExcerpt,
    TranscriptSegment,
    estimate_text_tokens,
    format_timestamp,
)

_CUE_TIME = re.compile(
    r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{1,3})\s*-->\s*(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{1,3})"
)
_TAG = re.compile(r"<[^>]+>")
_SPACE = re.compile(r"\s+")
_NOISE = re.compile(r"^\s*[\[(](music|applause|laughter|silence|musique|música)[\])]\s*$", re.I)


def _secs(h: str | None, m: str, s: str, ms: str) -> float:
    return int(h or 0) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000


def clean_text(text: str) -> str:
    text = _TAG.sub("", text)
    text = html.unescape(text).replace("\u200b", "")
    return _SPACE.sub(" ", text).strip()


def parse_vtt(content: str) -> list[TranscriptSegment]:
    """Parse WebVTT and SRT (same cue structure), removing YouTube rolling-caption repeats."""
    segments: list[TranscriptSegment] = []
    blocks = re.split(r"\r?\n\s*\r?\n", content.replace("\ufeff", ""))
    last_line = ""
    for block in blocks:
        lines = block.strip().splitlines()
        idx = next((i for i, ln in enumerate(lines) if "-->" in ln), None)
        if idx is None:
            continue
        m = _CUE_TIME.search(lines[idx])
        if not m:
            continue
        g = m.groups()
        start, end = _secs(g[0], g[1], g[2], g[3]), _secs(g[4], g[5], g[6], g[7])
        # YouTube auto-captions insert ~10 ms "transition" cues duplicating the previous line.
        if end - start < 0.05:
            continue
        new_lines: list[str] = []
        for raw in lines[idx + 1 :]:
            line = clean_text(raw)
            if not line or line == last_line or line in new_lines:
                continue
            new_lines.append(line)
        if not new_lines:
            continue
        last_line = new_lines[-1]
        text = " ".join(new_lines)
        if segments and segments[-1].text == text:
            segments[-1].end = max(segments[-1].end, end)
            continue
        segments.append(TranscriptSegment(start=start, end=end, text=text))
    return segments


def parse_json3(content: str) -> list[TranscriptSegment]:
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise AgentError(ErrorCode.EXTRACTION_FAILED, f"Malformed json3 captions: {exc}") from exc
    segments: list[TranscriptSegment] = []
    for ev in data.get("events", []):
        segs = ev.get("segs")
        if not segs:
            continue
        text = clean_text("".join(s.get("utf8", "") for s in segs))
        if not text:
            continue
        start = ev.get("tStartMs", 0) / 1000
        end = start + ev.get("dDurationMs", 0) / 1000
        segments.append(TranscriptSegment(start=start, end=end, text=text))
    return segments


def parse_captions(content: str, ext: str) -> list[TranscriptSegment]:
    ext = ext.lower().lstrip(".")
    if ext == "json3" or content.lstrip().startswith("{"):
        segs = parse_json3(content)
    elif ext in ("vtt", "srt", "webvtt") or "-->" in content:
        segs = parse_vtt(content)
    else:
        raise AgentError(ErrorCode.EXTRACTION_FAILED, f"Unsupported caption format: {ext}")
    return [s for s in segs if not _NOISE.match(s.text)] or segs


def merge_segments(
    segments: list[TranscriptSegment], *, window: float = 30.0, max_chars: int = 600
) -> list[TranscriptSegment]:
    """Merge short caption cues into paragraph-sized blocks of up to ``window`` seconds."""
    merged: list[TranscriptSegment] = []
    for seg in segments:
        if merged:
            cur = merged[-1]
            if seg.start - cur.start < window and len(cur.text) + len(seg.text) < max_chars:
                cur.text = f"{cur.text} {seg.text}".strip()
                cur.end = max(cur.end, seg.end)
                continue
        merged.append(TranscriptSegment(start=seg.start, end=seg.end, text=seg.text))
    return merged


def render_timestamped(segments: list[TranscriptSegment]) -> str:
    return "\n".join(f"[{format_timestamp(s.start)}] {s.text}" for s in segments)


def fit_to_budget(
    transcript: Transcript, max_tokens: int, *, window: float = 30.0
) -> tuple[list[TranscriptExcerpt], float]:
    """Choose evenly spread paragraphs that fit ``max_tokens``.

    Returns the excerpts and the fraction of transcript characters included. Long
    paragraphs are trimmed at word boundaries so coverage of the whole video wins over
    detail at the start.
    """
    blocks = merge_segments(transcript.segments, window=window)
    if not blocks or max_tokens <= 0:
        return [], 0.0
    total_chars = sum(len(b.text) for b in blocks)
    overhead = 4  # "[m:ss] " prefix tokens per block
    full_cost = sum(estimate_text_tokens(b.text) + overhead for b in blocks)
    if full_cost <= max_tokens:
        chosen = blocks
    else:
        # Keep every k-th block, then trim each to an equal share of the budget.
        n = max(1, min(len(blocks), max_tokens // 40))
        step = len(blocks) / n
        chosen = [blocks[int(i * step)] for i in range(n)]
        share_chars = max(40, (max_tokens // n - overhead) * 4)
        chosen = [
            TranscriptSegment(start=b.start, end=b.end, text=_trim(b.text, share_chars))
            for b in chosen
        ]
    excerpts = [
        TranscriptExcerpt(start=b.start, end=b.end, label=format_timestamp(b.start), text=b.text)
        for b in chosen
    ]
    included = sum(len(e.text.rstrip("…")) for e in excerpts)
    return excerpts, round(min(1.0, included / total_chars), 3) if total_chars else 0.0


def paginate(
    segments: list[TranscriptSegment], max_tokens: int
) -> tuple[list[TranscriptSegment], float | None]:
    """Take leading segments up to ``max_tokens``; return them and the next start time."""
    out: list[TranscriptSegment] = []
    used = 0
    for i, seg in enumerate(segments):
        cost = estimate_text_tokens(seg.text) + 4
        if out and used + cost > max_tokens:
            return out, segments[i].start
        out.append(seg)
        used += cost
    return out, None


def _trim(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rsplit(" ", 1)[0]
    return cut + "…"
