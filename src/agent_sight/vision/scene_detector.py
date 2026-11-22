"""Scene-change detection and keyframe selection.

Default backend: ffmpeg's ``select='gt(scene,T)'`` filter on a downscaled, frame-rate
limited decode, which needs no extra Python dependencies. PySceneDetect is used instead
when installed and requested (``method="pyscenedetect"``).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

from agent_sight.config import Settings
from agent_sight.downloader.ffprobe_utils import header_args, run_ffmpeg
from agent_sight.errors import AgentError, ErrorCode

_PTS = re.compile(r"pts_time:([\d.]+)")
_SCORE = re.compile(r"lavfi\.scene_score=([\d.]+)")


def detect_scenes_ffmpeg(
    source: str | Path,
    *,
    threshold: float = 0.3,
    start: float | None = None,
    end: float | None = None,
    headers: Mapping[str, str] | None = None,
    analysis_width: int = 192,
    analysis_fps: float = 4.0,
    settings: Settings | None = None,
    timeout: float | None = None,
) -> list[tuple[float, float]]:
    """Return ``[(timestamp, score)]`` for cuts whose scene score exceeds ``threshold``."""
    if not 0 < threshold < 1:
        raise AgentError(ErrorCode.INVALID_ARGUMENT, "threshold must be between 0 and 1.")
    args = ["-y", *header_args(headers)]
    if start:
        args += ["-ss", f"{start:.3f}"]
    args += ["-i", str(source)]
    if end is not None:
        args += ["-t", f"{end - (start or 0):.3f}"]
    vf = (
        f"fps={analysis_fps},scale={analysis_width}:-2,"
        f"select='gt(scene\\,{threshold})',metadata=print"
    )
    args += ["-an", "-sn", "-dn", "-vf", vf, "-f", "null", "-"]
    proc = run_ffmpeg(args, settings=settings, timeout=timeout)
    cuts: list[tuple[float, float]] = []
    pending: float | None = None
    for line in proc.stderr.splitlines():
        if m := _PTS.search(line):
            pending = float(m.group(1))
        elif (m := _SCORE.search(line)) and pending is not None:
            cuts.append((round(pending + (start or 0), 3), round(float(m.group(1)), 3)))
            pending = None
    return cuts


def pyscenedetect_available() -> bool:
    try:
        import scenedetect  # type: ignore[import-not-found]  # noqa: F401

        return True
    except ImportError:
        return False


def detect_scenes_pyscenedetect(path: Path, *, threshold: float = 27.0) -> list[tuple[float, float]]:
    try:
        from scenedetect import ContentDetector, detect  # type: ignore[import-not-found]
    except ImportError as exc:
        raise AgentError(
            ErrorCode.DEPENDENCY_MISSING, "PySceneDetect is not installed.",
            "pip install 'agent-sight[scenes]', or use the default ffmpeg method.",
        ) from exc
    scenes = detect(str(path), ContentDetector(threshold=threshold))
    return [(round(s[0].get_seconds(), 3), 1.0) for s in scenes[1:]]


def select_keyframes(
    cuts: list[tuple[float, float]],
    *,
    start: float,
    end: float,
    max_frames: int,
    min_gap: float = 1.0,
) -> list[tuple[float, float | None]]:
    """Pick up to ``max_frames`` representative timestamps.

    Each detected scene is represented by its midpoint. When there are more scenes than
    frames, the strongest cuts win. When there are fewer (static or silent footage), the
    longest scenes are subdivided so the whole range stays covered.
    """
    if end <= start or max_frames <= 0:
        return []
    inside = sorted((t, sc) for t, sc in cuts if start + min_gap <= t <= end - min_gap)
    # Drop cuts closer than min_gap to a stronger neighbour.
    kept: list[tuple[float, float]] = []
    for t, sc in sorted(inside, key=lambda c: -c[1]):
        if all(abs(t - k) >= min_gap for k, _ in kept):
            kept.append((t, sc))
    kept = sorted(kept, key=lambda c: -c[1])[: max_frames - 1]
    kept.sort()

    bounds = [start, *[t for t, _ in kept], end]
    score_at = {t: sc for t, sc in kept}
    segments: list[tuple[float, float, float | None, int]] = [
        (bounds[i], bounds[i + 1], score_at.get(bounds[i]), 1) for i in range(len(bounds) - 1)
    ]

    # Subdivide the longest segments until the frame budget is used.
    total = sum(n for *_, n in segments)
    while total < max_frames:
        idx = max(range(len(segments)), key=lambda i: (segments[i][1] - segments[i][0]) / segments[i][3])
        a, b, seg_score, n = segments[idx]
        if (b - a) / (n + 1) < min_gap:
            break
        segments[idx] = (a, b, seg_score, n + 1)
        total += 1

    picks: list[tuple[float, float | None]] = []
    for a, b, seg_score, n in segments:
        step = (b - a) / n
        for j in range(n):
            ts = round(a + step * (j + 0.5), 3)
            picks.append((ts, seg_score if j == 0 else None))
    return picks
