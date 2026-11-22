"""End-to-end pipeline against a local HTTP server (no internet needed, requires ffmpeg)."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_sight.config import get_settings, reset_settings
from agent_sight.errors import AgentError, ErrorCode
from agent_sight.models import DownloadRequest
from agent_sight.pipeline.engine import AgentSight

pytestmark = pytest.mark.ffmpeg


@pytest.fixture
def engine(monkeypatch: pytest.MonkeyPatch, media_server: str) -> AgentSight:
    monkeypatch.setenv("AGENT_SIGHT_ALLOW_PRIVATE_HOSTS", "true")
    monkeypatch.setenv("AGENT_SIGHT_WHISPER_ENABLED", "false")
    reset_settings()
    return AgentSight(get_settings())


def test_probe_direct(engine: AgentSight, media_server: str) -> None:
    r = engine.probe(f"{media_server}/demo.mp4")
    assert r.metadata.platform == "direct"
    assert r.metadata.duration == pytest.approx(12, abs=0.2)
    assert r.metadata.width == 320
    assert engine.probe(f"{media_server}/demo.mp4").cached


def test_digest_page(engine: AgentSight, media_server: str) -> None:
    d = engine.digest(f"{media_server}/page.html", max_frames=6)
    assert d.transcript_source == "native"
    assert d.transcript_coverage == 1.0
    assert d.scene_method == "ffmpeg-scene" and d.scene_count == 2
    assert len(d.contact_sheets) == 1 and Path(d.contact_sheets[0].path).exists()
    kinds = {e.kind for e in d.timeline}
    assert {"scene", "transcript"} <= kinds
    assert d.estimated_text_tokens <= d.token_budget


def test_watch_segment(engine: AgentSight, media_server: str) -> None:
    r = engine.watch_segment(f"{media_server}/page.html", start=3, end=9, frames=4)
    assert len(r.frames) == 4
    assert all(3 <= f.timestamp <= 9 for f in r.frames)
    assert r.transcript and "color bars" in r.transcript


def test_transcript_pagination(engine: AgentSight, media_server: str) -> None:
    r = engine.transcript(f"{media_server}/page.html", fmt="segments", max_tokens=100)
    assert r.source == "native" and r.segment_count >= 1
    r2 = engine.transcript(f"{media_server}/page.html", start=7, fmt="plain")
    assert "red screen" in (r2.text or "") and "Welcome" not in (r2.text or "")


def test_download_clip_and_audio(engine: AgentSight, media_server: str) -> None:
    clip = engine.download(DownloadRequest(url=f"{media_server}/demo.mp4", start=2, end=6, filename="clip"))
    assert clip.media and clip.media.duration == pytest.approx(4, abs=0.3)
    audio = engine.download(DownloadRequest(url=f"{media_server}/demo.mp4", mode="audio"))
    assert audio.files[0].endswith(".m4a") and audio.media and audio.media.video is None
    again = engine.download(DownloadRequest(url=f"{media_server}/demo.mp4", mode="audio"))
    assert again.cached


def test_time_range_validation(engine: AgentSight, media_server: str) -> None:
    with pytest.raises(AgentError) as exc:
        engine.watch_segment(f"{media_server}/demo.mp4", start=30, end=40)
    assert exc.value.code == ErrorCode.INVALID_TIME_RANGE


def test_mcp_download_confined(engine: AgentSight, media_server: str) -> None:
    with pytest.raises(AgentError) as exc:
        engine.download(DownloadRequest(url=f"{media_server}/demo.mp4"), output_dir="../../etc", confine=True)
    assert exc.value.code == ErrorCode.INVALID_ARGUMENT
