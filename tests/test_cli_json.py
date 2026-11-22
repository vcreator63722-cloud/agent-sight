"""The CLI contract agents rely on: one JSON object on stdout and stable exit codes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_sight.interfaces.cli import main


def run(capsys: pytest.CaptureFixture[str], *args: str) -> tuple[int, dict]:  # type: ignore[type-arg]
    with pytest.raises(SystemExit) as exc:
        main(list(args))
    out = capsys.readouterr().out.strip()
    lines = out.splitlines()
    assert len(lines) == 1, f"expected exactly one JSON line, got: {out!r}"
    return int(exc.value.code or 0), json.loads(lines[0])


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    code, payload = run(capsys, "version", "--json")
    assert code == 0
    assert payload == {"ok": True, "command": "version", "data": {"version": "0.1.0"}}


def test_global_json_flag(capsys: pytest.CaptureFixture[str]) -> None:
    code, payload = run(capsys, "--json", "version")
    assert code == 0 and payload["ok"]


def test_invalid_url(capsys: pytest.CaptureFixture[str]) -> None:
    code, payload = run(capsys, "probe", "ftp://example.com/x.mp4", "--json")
    assert code == 2
    assert payload["ok"] is False
    assert payload["error"]["error_code"] == "INVALID_URL"
    assert payload["error"]["agent_recovery_hint"]


def test_private_host_blocked(capsys: pytest.CaptureFixture[str]) -> None:
    code, payload = run(capsys, "probe", "http://127.0.0.1:9/video.mp4", "--json")
    assert code == 2
    assert payload["error"]["error_code"] == "INVALID_URL"


def test_usage_error_is_json(capsys: pytest.CaptureFixture[str]) -> None:
    code, payload = run(capsys, "watch", "https://example.com/v.mp4", "--json")
    assert code == 2
    assert payload["error"]["error_code"] == "INVALID_ARGUMENT"
    assert "--start" in payload["error"]["message"]


def test_bad_timestamp(capsys: pytest.CaptureFixture[str]) -> None:
    code, payload = run(capsys, "transcript", "https://example.com/v.mp4", "--start", "soon", "--json")
    assert code == 2
    assert payload["error"]["error_code"] == "INVALID_TIME_RANGE"


def test_env_json_mode(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_SIGHT_JSON", "1")
    code, payload = run(capsys, "version")
    assert code == 0 and payload["command"] == "version"


def test_doctor_json(capsys: pytest.CaptureFixture[str]) -> None:
    code, payload = run(capsys, "doctor", "--json")
    data = payload["data"]
    assert code in (0, 3)
    assert code == (0 if data["ok"] else 3)
    names = {c["name"] for c in data["checks"]}
    assert {"python", "yt-dlp", "ffmpeg", "cache-dir"} <= names
    for c in data["checks"]:
        assert c["status"] in ("ok", "warn", "fail")
        if c["status"] == "fail":
            assert c["fix"]


def test_install_skill_dry_run_and_real(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    proj = tmp_path / "proj"
    proj.mkdir()
    code, payload = run(capsys, "install-skill", "--dir", str(proj), "--dry-run", "--json")
    assert code == 0
    assert {a["action"] for a in payload["data"]["actions"]} == {"would-write"}
    assert not (proj / ".claude").exists()

    code, payload = run(capsys, "install-skill", "--host", "claude-code", "--host", "cursor",
                        "--dir", str(proj), "--mcp", "--json")
    assert code == 0
    skill = proj / ".claude/skills/agent-sight/SKILL.md"
    assert skill.read_text().startswith("---\nname: agent-sight")
    assert (proj / ".cursor/rules/agent-sight.mdc").exists()
    mcp = json.loads((proj / ".mcp.json").read_text())
    assert mcp["mcpServers"]["agent-sight"]["args"][-1] == "mcp"


def test_install_skill_preserves_existing_mcp_servers(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    proj = tmp_path / "p2"
    (proj / ".cursor").mkdir(parents=True)
    (proj / ".cursor/mcp.json").write_text(json.dumps({"mcpServers": {"other": {"command": "x"}}}))
    run(capsys, "install-skill", "--host", "cursor", "--dir", str(proj), "--mcp", "--json")
    cfg = json.loads((proj / ".cursor/mcp.json").read_text())
    assert set(cfg["mcpServers"]) == {"other", "agent-sight"}
    assert (proj / ".cursor/mcp.json.bak").exists()


def test_cache_stats(capsys: pytest.CaptureFixture[str]) -> None:
    code, payload = run(capsys, "cache", "stats", "--json")
    assert code == 0 and payload["data"]["entries"] == 0
