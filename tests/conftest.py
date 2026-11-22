from __future__ import annotations

import functools
import http.server
import shutil
import subprocess
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from agent_sight.config import reset_settings
from agent_sight.downloader.ffprobe_utils import find_ffmpeg


@pytest.fixture(autouse=True)
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Every test gets its own cache/output directory and fresh settings."""
    monkeypatch.setenv("AGENT_SIGHT_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("AGENT_SIGHT_OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("AGENT_SIGHT_MIN_REQUEST_INTERVAL", "0")
    monkeypatch.setenv("AGENT_SIGHT_MAX_RETRIES", "0")
    monkeypatch.delenv("AGENT_SIGHT_JSON", raising=False)
    reset_settings()
    from agent_sight.interfaces import cli

    cli._engine.cache_clear()
    yield tmp_path
    reset_settings()
    cli._engine.cache_clear()


VTT = """WEBVTT

00:00:00.500 --> 00:00:03.500
Welcome to the demo.

00:00:04.000 --> 00:00:07.500
These are color bars.

00:00:08.000 --> 00:00:11.500
Finally a red screen.
"""


@pytest.fixture(scope="session")
def media_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """Serve a generated 12 s, 3-scene MP4 + VTT + HTML page on 127.0.0.1."""
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        pytest.skip("ffmpeg not available")
    root = tmp_path_factory.mktemp("media")
    subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
         "-t", "4", "-f", "lavfi", "-i", "testsrc=size=320x180:rate=25",
         "-t", "4", "-f", "lavfi", "-i", "smptebars=size=320x180:rate=25",
         "-t", "4", "-f", "lavfi", "-i", "color=c=red:size=320x180:rate=25",
         "-t", "12", "-f", "lavfi", "-i", "sine=frequency=440",
         "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]",
         "-map", "[v]", "-map", "3:a", "-c:v", "libx264", "-preset", "ultrafast",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", "-movflags", "+faststart",
         str(root / "demo.mp4")],
        check=True, timeout=120,
    )
    (root / "demo.vtt").write_text(VTT)
    (root / "page.html").write_text(
        '<html><head><title>Demo</title><meta property="og:title" content="Local Demo"></head>'
        '<body><video controls><source src="/demo.mp4" type="video/mp4">'
        '<track kind="captions" srclang="en" src="/demo.vtt"></video></body></html>'
    )
    handler = functools.partial(_QuietHandler, directory=str(root))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    shutil.rmtree(root, ignore_errors=True)


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        pass
