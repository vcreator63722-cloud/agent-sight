"""SQLite index + content-addressed blob store.

* JSON payloads (metadata, transcripts, scene lists) live in SQLite with a TTL.
* Files (proxies, frames, contact sheets) live under ``blobs/`` named by SHA-256 so the
  same bytes are never stored twice, plus per-key working directories under ``work/``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    key        TEXT PRIMARY KEY,
    url        TEXT NOT NULL,
    operation  TEXT NOT NULL,
    created_at REAL NOT NULL,
    expires_at REAL,
    payload    TEXT,
    file_path  TEXT
);
CREATE INDEX IF NOT EXISTS idx_entries_url ON entries(url);
CREATE INDEX IF NOT EXISTS idx_entries_expires ON entries(expires_at);
"""


def make_key(url: str, operation: str, params: dict[str, Any] | None = None) -> str:
    """Deterministic key from a normalized URL, an operation name and its parameters."""
    blob = json.dumps(
        {"u": url, "op": operation, "p": params or {}}, sort_keys=True, separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while data := fh.read(chunk):
            h.update(data)
    return h.hexdigest()


class Cache:
    def __init__(self, root: Path, default_ttl: int = 7 * 24 * 3600) -> None:
        self.root = Path(root)
        self.default_ttl = default_ttl
        self.blobs_dir = self.root / "blobs"
        self.work_root = self.root / "work"
        for d in (self.root, self.blobs_dir, self.work_root):
            d.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "cache.sqlite3"
        self._lock = threading.Lock()
        with self._connect() as con:
            con.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
        try:
            con.execute("PRAGMA journal_mode=WAL")
            con.execute("PRAGMA busy_timeout=30000")
            yield con
        finally:
            con.close()

    # ------------------------------------------------------------------ JSON payloads

    def get(self, key: str) -> Any | None:
        with self._connect() as con:
            row = con.execute(
                "SELECT payload, expires_at, file_path FROM entries WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        payload, expires_at, file_path = row
        if expires_at is not None and expires_at < time.time():
            self.delete(key)
            return None
        if file_path and not Path(file_path).exists():
            self.delete(key)
            return None
        return json.loads(payload) if payload is not None else None

    def set(
        self,
        key: str,
        url: str,
        operation: str,
        payload: Any,
        ttl: int | None = None,
        file_path: Path | None = None,
    ) -> None:
        ttl = self.default_ttl if ttl is None else ttl
        now = time.time()
        expires = now + ttl if ttl > 0 else None
        with self._lock, self._connect() as con:
            con.execute(
                "INSERT OR REPLACE INTO entries"
                "(key, url, operation, created_at, expires_at, payload, file_path)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    key,
                    url,
                    operation,
                    now,
                    expires,
                    json.dumps(payload, default=str),
                    str(file_path) if file_path else None,
                ),
            )

    def delete(self, key: str) -> None:
        with self._lock, self._connect() as con:
            con.execute("DELETE FROM entries WHERE key = ?", (key,))

    # ------------------------------------------------------------------ files

    def store_file(self, src: Path, *, move: bool = True) -> Path:
        """Store ``src`` under its SHA-256 and return the blob path (deduplicated)."""
        src = Path(src)
        digest = sha256_file(src)
        dest = self.blobs_dir / digest[:2] / f"{digest}{src.suffix.lower()}"
        if dest.exists():
            if move and src.resolve() != dest.resolve():
                src.unlink(missing_ok=True)
            return dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + f".tmp{os.getpid()}")
        if move:
            shutil.move(str(src), tmp)
        else:
            shutil.copy2(src, tmp)
        os.replace(tmp, dest)
        return dest

    def work_dir(self, key: str) -> Path:
        d = self.work_root / key[:2] / key
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ------------------------------------------------------------------ maintenance

    def purge_expired(self) -> int:
        now = time.time()
        with self._lock, self._connect() as con:
            cur = con.execute(
                "DELETE FROM entries WHERE expires_at IS NOT NULL AND expires_at < ?", (now,)
            )
            return cur.rowcount

    def clear(self, url: str | None = None) -> int:
        """Remove entries (all, or only those for one normalized URL) and orphaned files."""
        with self._lock, self._connect() as con:
            if url:
                cur = con.execute("DELETE FROM entries WHERE url = ?", (url,))
            else:
                cur = con.execute("DELETE FROM entries")
            removed = cur.rowcount
        if not url:
            shutil.rmtree(self.blobs_dir, ignore_errors=True)
            shutil.rmtree(self.work_root, ignore_errors=True)
            self.blobs_dir.mkdir(parents=True, exist_ok=True)
            self.work_root.mkdir(parents=True, exist_ok=True)
        return removed

    def stats(self) -> dict[str, Any]:
        with self._connect() as con:
            total = con.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
            by_op = dict(
                con.execute(
                    "SELECT operation, COUNT(*) FROM entries GROUP BY operation"
                ).fetchall()
            )
        size = sum(f.stat().st_size for f in self.root.rglob("*") if f.is_file())
        return {
            "root": str(self.root),
            "entries": total,
            "by_operation": by_op,
            "size_bytes": size,
            "size_mb": round(size / (1024 * 1024), 2),
        }
