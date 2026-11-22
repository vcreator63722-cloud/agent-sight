from __future__ import annotations

import time
from pathlib import Path

from agent_sight.cache import Cache, make_key


def test_make_key_is_stable_and_param_sensitive() -> None:
    a = make_key("u", "probe", {"x": 1, "y": 2})
    assert a == make_key("u", "probe", {"y": 2, "x": 1})
    assert a != make_key("u", "probe", {"x": 2, "y": 2})
    assert a != make_key("u", "transcript", {"x": 1, "y": 2})


def test_set_get_and_ttl(tmp_path: Path) -> None:
    c = Cache(tmp_path / "c", default_ttl=60)
    c.set("k", "u", "probe", {"title": "hi"})
    assert c.get("k") == {"title": "hi"}
    c.set("short", "u", "probe", {"v": 1}, ttl=1)
    time.sleep(1.1)
    assert c.get("short") is None
    c.set("forever", "u", "whisper", [1, 2], ttl=0)
    assert c.get("forever") == [1, 2]


def test_persists_across_instances(tmp_path: Path) -> None:
    Cache(tmp_path / "c").set("k", "u", "op", {"a": 1})
    assert Cache(tmp_path / "c").get("k") == {"a": 1}


def test_entry_dropped_when_file_missing(tmp_path: Path) -> None:
    c = Cache(tmp_path / "c")
    f = tmp_path / "media.mp4"
    f.write_bytes(b"x")
    c.set("k", "u", "download", {"path": str(f)}, file_path=f)
    assert c.get("k") is not None
    f.unlink()
    assert c.get("k") is None


def test_content_addressed_dedup(tmp_path: Path) -> None:
    c = Cache(tmp_path / "c")
    a, b = tmp_path / "a.JPG", tmp_path / "b.jpg"
    a.write_bytes(b"same-bytes")
    b.write_bytes(b"same-bytes")
    pa, pb = c.store_file(a), c.store_file(b)
    assert pa == pb and pa.exists() and pa.suffix == ".jpg"
    assert not a.exists() and not b.exists()


def test_clear_and_stats(tmp_path: Path) -> None:
    c = Cache(tmp_path / "c")
    c.set("1", "u1", "probe", {})
    c.set("2", "u2", "probe", {})
    c.set("3", "u2", "scenes", {})
    s = c.stats()
    assert s["entries"] == 3 and s["by_operation"] == {"probe": 2, "scenes": 1}
    assert c.clear("u2") == 2
    assert c.get("1") == {}
    assert c.clear() == 1
    assert c.stats()["entries"] == 0


def test_purge_expired(tmp_path: Path) -> None:
    c = Cache(tmp_path / "c")
    c.set("old", "u", "probe", {}, ttl=1)
    c.set("new", "u", "probe", {}, ttl=100)
    time.sleep(1.1)
    assert c.purge_expired() == 1
