"""Agent Digest: metadata + budgeted transcript + scene timeline + contact sheet(s).

The text part is kept under ``token_budget`` (measured on the JSON the agent receives).
Contact sheets are reported separately in ``estimated_image_tokens`` because they are
delivered as image blocks, and their cost is controlled with ``max_frames``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from agent_sight.audio import transcriber
from agent_sight.audio.transcripts import fit_to_budget
from agent_sight.errors import AgentError, ErrorCode
from agent_sight.models import (
    ContactSheet,
    Digest,
    Frame,
    NextAction,
    NormalizedURL,
    TimelineEntry,
    Transcript,
    TranscriptExcerpt,
    VideoMetadata,
    estimate_text_tokens,
    format_timestamp,
)
from agent_sight.vision.scene_detector import select_keyframes

if TYPE_CHECKING:
    from agent_sight.pipeline.engine import AgentSight

DIGEST_FRAME_WIDTH = 480


def build_digest(
    engine: AgentSight,
    nurl: NormalizedURL,
    *,
    token_budget: int = 1500,
    max_frames: int = 9,
    language: str = "en",
    include_frames: bool = True,
    allow_whisper: bool | None = None,
    refresh: bool = False,
) -> Digest:
    from agent_sight.pipeline.engine import _header, _languages, cli_hint

    s = engine.settings
    meta, _ = engine.metadata(nurl, refresh=refresh)
    warnings: list[str] = []
    if meta.is_live:
        warnings.append("Video is live; transcript and frames are unavailable until it ends.")
        include_frames = False

    # ------------------------------------------------------------- transcript
    if allow_whisper is None:
        allow_whisper = bool(
            s.whisper_enabled and transcriber.available_backend()
            and meta.duration and meta.duration <= s.whisper_max_duration
        )
    transcript: Transcript | None = None
    if not meta.is_live:
        try:
            transcript, _ = engine.get_transcript_obj(
                nurl, languages=_languages(language), allow_whisper=allow_whisper, refresh=refresh
            )
        except AgentError as err:
            warnings.append(f"Transcript unavailable ({err.code}): {err.message}")
    if transcript is None and not any(w.startswith("Transcript") for w in warnings):
        warnings.append(
            "No captions found; rely on the contact sheet(s) for content. "
            "Install agent-sight[whisper] for local speech-to-text."
        )

    # ------------------------------------------------------------- visuals
    frames: list[Frame] = []
    sheets: list[ContactSheet] = []
    scene_method: str | None = None
    scene_count = 0
    if include_frames and max_frames > 0:
        try:
            engine._guard_visual(meta)
            if not meta.duration:
                raise AgentError(ErrorCode.INVALID_TIME_RANGE, "Duration unknown; cannot plan keyframes.")
            scenes = engine._scenes(nurl, meta, threshold=0.3, start=None, end=None, method="auto")
            scene_method = scenes.method
            scene_count = len(scenes.scenes)
            cuts = [(c.timestamp, c.score or 0.0) for c in scenes.scenes]
            if not cuts and meta.chapters:
                cuts = [(c.start, 0.5) for c in meta.chapters if c.start > 0]
            end = max(0.2, meta.duration - 0.1)
            picks = select_keyframes(
                cuts, start=0.0, end=end, max_frames=max_frames,
                min_gap=max(0.5, min(5.0, meta.duration / (max_frames * 4))),
            )
            scores = {t: sc for t, sc in picks}
            frames, _ = engine._extract(nurl, meta, [t for t, _ in picks], DIGEST_FRAME_WIDTH, scores)
            sheets = engine._sheets(nurl, frames, header=_header(meta), columns=3, rows=3)
        except AgentError as err:
            warnings.append(f"Visual analysis skipped ({err.code}): {err.message}")

    # ------------------------------------------------------------- assemble within budget
    base_timeline = _visual_timeline(meta, frames, sheets)
    next_actions = _next_actions(meta, transcript, frames, cli_hint)

    def assemble(excerpt_budget: int) -> Digest:
        excerpts: list[TranscriptExcerpt] = []
        coverage = 0.0
        if transcript and excerpt_budget > 0:
            excerpts, coverage = fit_to_budget(transcript, excerpt_budget)
        timeline = base_timeline + [
            TimelineEntry(start=e.start, end=round(e.end, 2), label=e.label, kind="transcript", text=e.text)
            for e in excerpts
        ]
        timeline.sort(key=lambda t: (t.start, t.kind != "chapter"))
        desc = (meta.description or "").strip().replace("\n", " ")
        return Digest(
            url=meta.url, title=meta.title, platform=meta.platform, uploader=meta.uploader,
            duration=meta.duration, duration_label=format_timestamp(meta.duration),
            upload_date=meta.upload_date, view_count=meta.view_count,
            description_excerpt=(desc[:280] + "…") if len(desc) > 280 else (desc or None),
            transcript_source=transcript.source if transcript else "none",
            transcript_language=transcript.language if transcript else None,
            transcript_coverage=coverage, scene_method=scene_method, scene_count=scene_count,
            contact_sheets=sheets, timeline=timeline,
            estimated_text_tokens=0,
            estimated_image_tokens=sum(sh.estimated_tokens for sh in sheets),
            token_budget=token_budget, warnings=warnings, next_actions=next_actions,
        )

    fixed = _measure(assemble(0))
    budget = max(0, token_budget - fixed)
    digest = assemble(budget)
    for _ in range(4):
        used = _measure(digest)
        if used <= token_budget or budget == 0:
            break
        budget = max(0, budget - (used - token_budget) - 20)
        digest = assemble(budget)
    digest.estimated_text_tokens = _measure(digest)
    if digest.estimated_text_tokens > token_budget:
        digest.warnings.append(
            f"Metadata alone needs ~{digest.estimated_text_tokens} tokens; raise --budget for transcript excerpts."
        )
    return digest


def _measure(d: Digest) -> int:
    return estimate_text_tokens(d.model_dump_json(exclude_none=True))


def _visual_timeline(
    meta: VideoMetadata, frames: list[Frame], sheets: list[ContactSheet]
) -> list[TimelineEntry]:
    entries = [
        TimelineEntry(start=c.start, end=c.end, label=format_timestamp(c.start), kind="chapter", text=c.title)
        for c in meta.chapters[:30]
    ]
    sheet_of: dict[float, tuple[int, int]] = {}
    for n, sh in enumerate(sheets, 1):
        for i, ts in enumerate(sh.timestamps, 1):
            sheet_of[ts] = (n, i)
    for f in frames:
        where = sheet_of.get(f.timestamp)
        text = f"sheet {where[0]} tile {where[1]}" if where else "frame"
        if f.scene_score:
            text += f", scene cut {f.scene_score:.2f}"
        entries.append(TimelineEntry(start=f.timestamp, label=f.label, kind="scene", text=text))
    return entries


def _next_actions(
    meta: VideoMetadata,
    transcript: Transcript | None,
    frames: list[Frame],
    cli_hint: Callable[..., str],
) -> list[NextAction]:
    u = meta.url
    acts: list[NextAction] = []
    if meta.duration:
        if len(frames) >= 2:
            a, b = frames[0].timestamp, frames[1].timestamp
            lo, hi = max(0.0, a - 2), min(meta.duration, b + 2)
        else:
            lo, hi = 0.0, min(meta.duration, 30.0)
        acts.append(NextAction(
            tool="watch_segment",
            reason="Zoom into a time range with denser frames and its transcript.",
            cli=cli_hint("watch", u, "--start", f"{lo:.0f}", "--end", f"{hi:.0f}", "--json"),
        ))
    if transcript:
        acts.append(NextAction(
            tool="get_transcript", reason="Read the full timestamped transcript (paginated).",
            cli=cli_hint("transcript", u, "--json"),
        ))
    acts.append(NextAction(
        tool="download_media", reason="Download the video, audio or an exact clip.",
        cli=cli_hint("download", u, "--start", "0", "--end", "30", "--json"),
    ))
    return acts
