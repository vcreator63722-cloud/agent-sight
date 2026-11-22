"""Environment health checks with exact fix commands."""

from __future__ import annotations

import importlib
import importlib.metadata as md
import os
import platform
import shutil
import stat
import subprocess
import sys
from datetime import date
from typing import Literal

from agent_sight.config import Settings, get_settings
from agent_sight.downloader.ffprobe_utils import find_ffmpeg, find_ffprobe
from agent_sight.models import CheckResult, DoctorReport


def _pm_install(pkg: str) -> str:
    system = platform.system()
    if system == "Darwin":
        return f"brew install {pkg}"
    if system == "Windows":
        return {"ffmpeg": "winget install Gyan.FFmpeg", "deno": "winget install DenoLand.Deno"}.get(pkg, pkg)
    if shutil.which("apt-get"):
        return f"sudo apt-get install -y {pkg}"
    if shutil.which("dnf"):
        return f"sudo dnf install -y {pkg}"
    if shutil.which("pacman"):
        return f"sudo pacman -S {pkg}"
    return f"install {pkg} with your package manager"


def _version(dist: str) -> str | None:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None


def _binary_version(exe: str) -> str:
    try:
        out = subprocess.run([exe, "-version"], capture_output=True, text=True, timeout=10).stdout
        return out.splitlines()[0][:80] if out else "unknown version"
    except (OSError, subprocess.SubprocessError):
        return "unknown version"


def check_python() -> CheckResult:
    v = sys.version_info
    ok = v >= (3, 11)
    return CheckResult(
        name="python", status="ok" if ok else "fail",
        detail=f"Python {platform.python_version()} ({sys.executable})",
        fix=None if ok else "Install Python 3.11+ (e.g. `uv python install 3.12`).",
    )


def check_ytdlp() -> CheckResult:
    ver = _version("yt-dlp")
    if not ver:
        return CheckResult(name="yt-dlp", status="fail", detail="yt-dlp is not installed.",
                           fix="pip install -U 'yt-dlp[default]'")
    try:
        y, m, d = (int(p) for p in ver.split(".")[:3])
        age = (date.today() - date(y, m, d)).days
    except ValueError:
        age = 0
    if age > 60:
        return CheckResult(
            name="yt-dlp", status="warn",
            detail=f"yt-dlp {ver} is {age} days old; site extractors break often.",
            fix="pip install -U 'yt-dlp[default]'",
        )
    return CheckResult(name="yt-dlp", status="ok", detail=f"yt-dlp {ver}")


def check_js_runtime() -> CheckResult:
    from agent_sight.adapters.ytdlp_adapter import detect_js_runtimes

    found = detect_js_runtimes()
    ejs = _version("yt-dlp-ejs")
    if found and ejs:
        return CheckResult(name="js-runtime", status="ok", required=False,
                           detail=f"{', '.join(found)} available; yt-dlp-ejs {ejs}")
    missing = []
    if not found:
        missing.append("no JavaScript runtime (deno/node/bun)")
    if not ejs:
        missing.append("yt-dlp-ejs not installed")
    return CheckResult(
        name="js-runtime", status="warn", required=False,
        detail="YouTube needs a JS runtime for full format access: " + "; ".join(missing) + ".",
        fix="pip install 'agent-sight[js]'  (installs Deno), or " + _pm_install("deno")
        + ("; pip install yt-dlp-ejs" if not ejs else ""),
    )


def check_ffmpeg(s: Settings) -> CheckResult:
    exe = find_ffmpeg(s)
    if not exe:
        return CheckResult(
            name="ffmpeg", status="fail", detail="ffmpeg not found (needed for frames, clips, merging).",
            fix=f"{_pm_install('ffmpeg')}  or  pip install 'agent-sight[ffmpeg]'",
        )
    bundled = "imageio_ffmpeg" in exe
    return CheckResult(name="ffmpeg", status="ok",
                       detail=f"{_binary_version(exe)} ({'bundled' if bundled else exe})")


def check_ffprobe(s: Settings) -> CheckResult:
    exe = find_ffprobe(s)
    if not exe:
        return CheckResult(
            name="ffprobe", status="warn", required=False,
            detail="ffprobe not found; falling back to parsing `ffmpeg -i` (less detail).",
            fix=_pm_install("ffmpeg"),
        )
    return CheckResult(name="ffprobe", status="ok", detail=_binary_version(exe))


def check_module(name: str, dist: str, purpose: str, fix: str, required: bool) -> CheckResult:
    try:
        importlib.import_module(name)
        return CheckResult(name=dist, status="ok", detail=f"{dist} {_version(dist) or ''}".strip(),
                           required=required)
    except ImportError:
        return CheckResult(name=dist, status="fail" if required else "warn",
                           detail=f"{dist} not installed ({purpose}).", fix=fix, required=required)


def check_whisper() -> CheckResult:
    from agent_sight.audio.transcriber import available_backend

    backend = available_backend()
    if backend:
        return CheckResult(name="whisper", status="ok", detail=f"{backend} available", required=False)
    return CheckResult(
        name="whisper", status="warn", required=False,
        detail="No local Whisper backend; videos without captions get visual analysis only.",
        fix="pip install 'agent-sight[whisper]'",
    )


def check_cache(s: Settings) -> list[CheckResult]:
    results: list[CheckResult] = []
    try:
        s.cache_dir.mkdir(parents=True, exist_ok=True)
        probe = s.cache_dir / ".write_test"
        probe.write_text("ok")
        probe.unlink()
        results.append(CheckResult(name="cache-dir", status="ok", detail=f"writable: {s.cache_dir}"))
    except OSError as exc:
        results.append(CheckResult(
            name="cache-dir", status="fail", detail=f"cannot write {s.cache_dir}: {exc}",
            fix="Set AGENT_SIGHT_CACHE_DIR to a writable directory.",
        ))
        return results
    free = shutil.disk_usage(s.cache_dir).free / 1024**3
    status: Literal["ok", "warn", "fail"] = "ok" if free >= 2 else "warn" if free >= 0.5 else "fail"
    results.append(CheckResult(
        name="disk-space", status=status, detail=f"{free:.1f} GB free at {s.cache_dir}",
        fix=None if status == "ok" else "Free disk space or run `agent-sight cache clear`.",
    ))
    return results


def check_cookies(s: Settings) -> CheckResult | None:
    if not s.cookies_file:
        return None
    path = s.cookies_file
    if not path.is_file():
        return CheckResult(name="cookies", status="fail", detail="AGENT_SIGHT_COOKIES_FILE does not exist.",
                           fix="Fix the path or unset AGENT_SIGHT_COOKIES_FILE.")
    mode = path.stat().st_mode
    if os.name != "nt" and mode & (stat.S_IRGRP | stat.S_IROTH):
        return CheckResult(
            name="cookies", status="warn",
            detail="Cookies file is readable by other users. Only use cookies for content you are permitted to access.",
            fix=f"chmod 600 {path}",
        )
    return CheckResult(name="cookies", status="ok",
                       detail="Cookies file configured (contents never logged).", required=False)


def check_network(timeout: float = 8.0) -> list[CheckResult]:
    import httpx

    results = []
    for name, url in (("net-youtube", "https://www.youtube.com/generate_204"),
                      ("net-pypi", "https://pypi.org/simple/yt-dlp/")):
        try:
            r = httpx.head(url, timeout=timeout, follow_redirects=True)
            results.append(CheckResult(name=name, status="ok" if r.status_code < 500 else "warn",
                                       detail=f"HTTP {r.status_code} from {url}", required=False))
        except httpx.HTTPError as exc:
            results.append(CheckResult(name=name, status="warn", detail=f"{url} unreachable: {exc}",
                                       fix="Check proxy/firewall settings.", required=False))
    return results


def run_doctor(settings: Settings | None = None, *, network: bool = False) -> DoctorReport:
    s = settings or get_settings()
    checks = [
        check_python(),
        check_ytdlp(),
        check_js_runtime(),
        check_ffmpeg(s),
        check_ffprobe(s),
        check_module("PIL", "pillow", "contact sheets", "pip install pillow", True),
        check_module("mcp", "mcp", "MCP server", "pip install mcp", True),
        check_whisper(),
        check_module("scenedetect", "scenedetect", "optional PySceneDetect backend",
                     "pip install 'agent-sight[scenes]'", False),
        *check_cache(s),
    ]
    if (c := check_cookies(s)) is not None:
        checks.append(c)
    if network:
        checks += check_network()
    failed = [c for c in checks if c.status == "fail"]
    warned = [c for c in checks if c.status == "warn"]
    summary = (
        f"{len(failed)} failing, {len(warned)} warnings, {len(checks) - len(failed) - len(warned)} ok"
    )
    return DoctorReport(ok=not failed, checks=checks, summary=summary)
