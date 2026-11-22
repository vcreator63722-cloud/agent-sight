"""Install the agent-sight skill/rules and (optionally) MCP registration for agent hosts.

Existing config files are merged, never overwritten wholesale: other servers are kept and a
``.bak`` copy is written before any modification. Invalid JSON is left untouched.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import tomllib
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Literal

from agent_sight.errors import AgentError, ErrorCode

Host = Literal["claude-code", "opencode", "cursor", "windsurf", "codex"]
HOSTS: tuple[Host, ...] = ("claude-code", "opencode", "cursor", "windsurf", "codex")
Scope = Literal["project", "global"]
SERVER_NAME = "agent-sight"


@dataclass
class InstallAction:
    host: str
    kind: Literal["skill", "rule", "mcp"]
    path: str
    action: Literal["created", "updated", "unchanged", "skipped", "would-write"]
    note: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {"host": self.host, "kind": self.kind, "path": self.path,
                "action": self.action, "note": self.note}


def skill_text() -> str:
    try:
        return (resources.files("agent_sight") / "_skill" / "SKILL.md").read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError, OSError):
        repo = Path(__file__).resolve().parents[3] / "skills" / "SKILL.md"
        if repo.is_file():
            return repo.read_text(encoding="utf-8")
    raise AgentError(ErrorCode.INTERNAL_ERROR, "Bundled SKILL.md not found; reinstall agent-sight.")


def _body_and_description(text: str) -> tuple[str, str]:
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.S)
    if not m:
        return "", text
    desc = next((ln.split(":", 1)[1].strip() for ln in m.group(1).splitlines()
                 if ln.startswith("description:")), "")
    return desc, m.group(2).lstrip()


def mcp_command() -> tuple[str, list[str]]:
    """Absolute path to the installed CLI when available (hosts often have a minimal PATH)."""
    exe = shutil.which("agent-sight")
    if exe:
        return exe, ["mcp"]
    candidate = Path(sys.prefix) / "bin" / "agent-sight"
    if candidate.exists():
        return str(candidate), ["mcp"]
    return "uvx", ["agent-sight", "mcp"]


def mcp_snippets() -> dict[str, object]:
    cmd, args = mcp_command()
    return {
        "claude-code": f"claude mcp add {SERVER_NAME} -- {cmd} {' '.join(args)}",
        "cursor (.cursor/mcp.json)": {"mcpServers": {SERVER_NAME: {"command": cmd, "args": args}}},
        "windsurf (~/.codeium/windsurf/mcp_config.json)": {
            "mcpServers": {SERVER_NAME: {"command": cmd, "args": args}}},
        "opencode (opencode.json)": {
            "mcp": {SERVER_NAME: {"type": "local", "command": [cmd, *args], "enabled": True}}},
        "codex (~/.codex/config.toml)": _codex_block(cmd, args).strip(),
        "streamable-http": f"{cmd} mcp --transport streamable-http --port 8765  ->  http://127.0.0.1:8765/mcp",
    }


# ------------------------------------------------------------------------------ writers


def _write_text(
    path: Path, content: str, *, host: str, kind: Literal["skill", "rule", "mcp"],
    dry_run: bool, force: bool,
) -> InstallAction:
    if path.exists():
        current = path.read_text(encoding="utf-8")
        if current == content:
            return InstallAction(host, kind, str(path), "unchanged")
        if not force and "agent-sight" not in current:
            return InstallAction(host, kind, str(path), "skipped",
                                 "file exists and was not written by agent-sight; use --force")
    if dry_run:
        return InstallAction(host, kind, str(path), "would-write")
    existed = path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return InstallAction(host, kind, str(path), "updated" if existed else "created")


def _merge_json(path: Path, key_path: list[str], entry: dict[str, object], *, host: str,
                dry_run: bool) -> InstallAction:
    data: dict[str, object] = {}
    if path.exists():
        raw = path.read_text(encoding="utf-8").strip()
        if raw:
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                return InstallAction(host, "mcp", str(path), "skipped",
                                     "existing file is not plain JSON (comments?); add the entry manually")
    node: dict[str, object] = data
    for k in key_path:
        child = node.setdefault(k, {})
        if not isinstance(child, dict):
            return InstallAction(host, "mcp", str(path), "skipped", f"'{k}' is not an object")
        node = child
    if node.get(SERVER_NAME) == entry:
        return InstallAction(host, "mcp", str(path), "unchanged")
    if dry_run:
        return InstallAction(host, "mcp", str(path), "would-write")
    existed = path.exists()
    if existed:
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
    node[SERVER_NAME] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return InstallAction(host, "mcp", str(path), "updated" if existed else "created")


def _codex_block(cmd: str, args: list[str]) -> str:
    return f'\n[mcp_servers.{SERVER_NAME}]\ncommand = {json.dumps(cmd)}\nargs = {json.dumps(args)}\n'


def _merge_codex_toml(path: Path, *, dry_run: bool) -> InstallAction:
    cmd, args = mcp_command()
    if path.exists():
        text = path.read_text(encoding="utf-8")
        try:
            parsed = tomllib.loads(text)
        except tomllib.TOMLDecodeError:
            return InstallAction("codex", "mcp", str(path), "skipped", "config.toml is not valid TOML")
        if SERVER_NAME in (parsed.get("mcp_servers") or {}):
            return InstallAction("codex", "mcp", str(path), "unchanged", "entry already present")
    else:
        text = ""
    if dry_run:
        return InstallAction("codex", "mcp", str(path), "would-write")
    if path.exists():
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n" + _codex_block(cmd, args), encoding="utf-8")
    return InstallAction("codex", "mcp", str(path), "updated" if text else "created")


# ------------------------------------------------------------------------------ main entry


def install(
    hosts: list[Host],
    *,
    scope: Scope = "project",
    project_dir: Path | None = None,
    with_mcp: bool = False,
    dry_run: bool = False,
    force: bool = False,
    home: Path | None = None,
) -> list[InstallAction]:
    root = (project_dir or Path.cwd()).resolve()
    home = home or Path.home()
    text = skill_text()
    desc, body = _body_and_description(text)
    cmd, args = mcp_command()
    std_entry: dict[str, object] = {"command": cmd, "args": args}
    actions: list[InstallAction] = []
    base = root if scope == "project" else home

    for host in hosts:
        if host == "claude-code":
            actions.append(_write_text(base / ".claude/skills/agent-sight/SKILL.md", text,
                                       host=host, kind="skill", dry_run=dry_run, force=force))
            if with_mcp:
                if scope == "project":
                    actions.append(_merge_json(root / ".mcp.json", ["mcpServers"], std_entry,
                                               host=host, dry_run=dry_run))
                else:
                    actions.append(InstallAction(host, "mcp", "~/.claude.json", "skipped",
                                                 f"run: claude mcp add --scope user {SERVER_NAME} -- {cmd} {' '.join(args)}"))
        elif host == "opencode":
            skill_dir = (root / ".opencode/skills") if scope == "project" else (home / ".config/opencode/skills")
            actions.append(_write_text(skill_dir / "agent-sight/SKILL.md", text,
                                       host=host, kind="skill", dry_run=dry_run, force=force))
            if with_mcp:
                cfg = root / "opencode.json" if scope == "project" else home / ".config/opencode/opencode.json"
                actions.append(_merge_json(cfg, ["mcp"], {"type": "local", "command": [cmd, *args],
                                                         "enabled": True}, host=host, dry_run=dry_run))
        elif host == "cursor":
            if scope == "project":
                rule = f"---\ndescription: {desc}\nglobs:\nalwaysApply: false\n---\n\n{body}"
                actions.append(_write_text(root / ".cursor/rules/agent-sight.mdc", rule,
                                           host=host, kind="rule", dry_run=dry_run, force=force))
            else:
                actions.append(InstallAction(host, "rule", "Cursor Settings > Rules", "skipped",
                                             "Cursor user rules are not file-based; use --scope project"))
            if with_mcp:
                cfg = (root if scope == "project" else home) / ".cursor/mcp.json"
                actions.append(_merge_json(cfg, ["mcpServers"], std_entry, host=host, dry_run=dry_run))
        elif host == "windsurf":
            if scope == "project":
                rule = f"---\ntrigger: model_decision\ndescription: {desc}\n---\n\n{body}"
                actions.append(_write_text(root / ".windsurf/rules/agent-sight.md", rule,
                                           host=host, kind="rule", dry_run=dry_run, force=force))
            if with_mcp:
                actions.append(_merge_json(home / ".codeium/windsurf/mcp_config.json", ["mcpServers"],
                                           std_entry, host=host, dry_run=dry_run))
        elif host == "codex":
            actions.append(_write_text(base / ".agents/skills/agent-sight/SKILL.md", text,
                                       host=host, kind="skill", dry_run=dry_run, force=force))
            if with_mcp:
                actions.append(_merge_codex_toml(home / ".codex/config.toml", dry_run=dry_run))
        else:  # pragma: no cover - guarded by the CLI
            raise AgentError(ErrorCode.INVALID_ARGUMENT, f"Unknown host {host!r}.")
    return actions
