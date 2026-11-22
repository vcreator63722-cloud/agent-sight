"""Exact-timestamp frame sampling via ffmpeg input seeking (works on local files and remote streams)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from agent_sight.config import Settings, get_settings
from agent_sight.downloader.ffprobe_utils import header_args, run_ffmpeg
from agent_sight.errors import AgentError, ErrorCode
from agent_sight.models import Frame, estimate_image_tokens, format_timestamp

MAX_FRAMES = 48


def plan_timestamps(start: float, end: float, count: int) -> list[float]:
    """``count`` timestamps at the centres of equal sub-intervals of [start, end]."""
    if end <= start:
        raise AgentError(ErrorCode.INVALID_TIME_RANGE, f"end ({end}) must be greater than start ({start}).")
    count = max(1, min(count, MAX_FRAMES))
    step = (end - start) / count
    return [round(start + step * (i + 0.5), 3) for i in range(count)]


def frames_for_budget(
    image_token_budget: int, *, tile_width: int = 384, aspect: float = 16 / 9, per_sheet: int = 9
) -> int:
    """How many frames fit a vision-token budget when packed into contact sheets."""
    tile_h = tile_width / aspect if aspect >= 1 else tile_width
    tile_w = tile_width if aspect >= 1 else tile_width * aspect
    per_frame = estimate_image_tokens(int(tile_w), int(tile_h))
    return max(1, min(MAX_FRAMES, math.floor(image_token_budget / per_frame), per_sheet * 4))


def extract_frame(
    source: str | Path,
    timestamp: float,
    out_path: Path,
    *,
    width: int = 640,
    headers: Mapping[str, str] | None = None,
    settings: Settings | None = None,
) -> Path | None:
    """Grab the frame at ``timestamp``; returns None if the timestamp is past the end."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(
        [
            "-y", *header_args(headers), "-ss", f"{timestamp:.3f}", "-i", str(source),
            "-frames:v", "1", "-an", "-sn", "-dn",
            "-vf", f"scale='min({width},iw)':-2", "-q:v", "3", "-update", "1", str(out_path),
        ],
        settings=settings,
        timeout=120,
    )
    return out_path if out_path.exists() and out_path.stat().st_size > 0 else None


def extract_frames(
    source: str | Path,
    timestamps: Sequence[float],
    out_dir: Path,
    *,
    width: int = 640,
    headers: Mapping[str, str] | None = None,
    settings: Settings | None = None,
    scores: Mapping[float, float | None] | None = None,
    max_workers: int = 4,
) -> list[Frame]:
    s = settings or get_settings()
    if not timestamps:
        return []
    if len(timestamps) > MAX_FRAMES:
        raise AgentError(
            ErrorCode.INVALID_ARGUMENT, f"At most {MAX_FRAMES} frames per call (got {len(timestamps)})."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    def grab(ts: float) -> Frame | None:
        path = out_dir / f"frame_{int(round(ts * 1000)):010d}_w{width}.jpg"
        if not path.exists():
            if extract_frame(source, ts, path, width=width, headers=headers, settings=s) is None:
                return None
        with Image.open(path) as im:
            w, h = im.size
        return Frame(
            timestamp=ts, label=format_timestamp(ts), path=str(path), width=w, height=h,
            scene_score=(scores or {}).get(ts),
        )

    workers = 1 if str(source).startswith("http") and len(timestamps) <= 2 else max_workers
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(grab, sorted(set(timestamps))))
    frames = [f for f in results if f is not None]
    if not frames:
        raise AgentError(
            ErrorCode.NO_VIDEO_STREAM,
            "No frames could be decoded at the requested timestamps.",
            "Check the timestamps against the duration from probe; the media may be audio-only.",
        )
    return frames
