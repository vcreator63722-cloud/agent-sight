from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from agent_sight.errors import AgentError
from agent_sight.models import Frame
from agent_sight.vision.contact_sheet import build_contact_sheet, build_contact_sheets, tile_size
from agent_sight.vision.frame_extractor import frames_for_budget, plan_timestamps
from agent_sight.vision.scene_detector import select_keyframes


def _frames(tmp: Path, n: int, size: tuple[int, int] = (640, 360)) -> list[Frame]:
    out = []
    for i in range(n):
        p = tmp / f"f{i}.jpg"
        Image.new("RGB", size, ((i * 40) % 255, 100, 200)).save(p)
        out.append(Frame(timestamp=i * 10.5, label="", path=str(p), width=size[0], height=size[1]))
    return out


def test_grid_dimensions_and_tokens(tmp_path: Path) -> None:
    frames = _frames(tmp_path, 9)
    sheet = build_contact_sheet([(f.path, f.timestamp) for f in frames], tmp_path / "s.jpg", header="Title")
    assert (sheet.columns, sheet.rows) == (3, 3)
    assert sheet.width == 3 * 384 + 4 * 4
    with Image.open(sheet.path) as im:
        assert im.size == (sheet.width, sheet.height)
        assert im.format == "JPEG"
    assert sheet.labels[1] == "0:10"
    assert 900 < sheet.estimated_tokens < 1300  # nine frames for ~1k vision tokens


def test_portrait_frames_use_portrait_tiles(tmp_path: Path) -> None:
    frames = _frames(tmp_path, 4, size=(360, 640))
    sheet = build_contact_sheet([(f.path, f.timestamp) for f in frames], tmp_path / "p.jpg", columns=4)
    assert tile_size(360 / 640, 384) == (216, 384)
    assert sheet.height > sheet.width / 4


def test_timestamp_is_burned_in(tmp_path: Path) -> None:
    p = tmp_path / "black.jpg"
    Image.new("RGB", (640, 360), (0, 0, 0)).save(p)
    sheet = build_contact_sheet([(p, 61)], tmp_path / "one.jpg")
    with Image.open(sheet.path) as im:
        # The label sits in the bottom-left corner; white text pixels must exist there.
        region = im.crop((4, sheet.height - 50, 120, sheet.height - 4)).convert("L")
        assert max(region.tobytes()) > 200
    assert sheet.labels == ["1:01"]


def test_multiple_sheets(tmp_path: Path) -> None:
    sheets = build_contact_sheets(_frames(tmp_path, 11), tmp_path / "out", header="H")
    assert [len(s.timestamps) for s in sheets] == [9, 2]
    assert all(Path(s.path).exists() for s in sheets)


def test_empty_rejected(tmp_path: Path) -> None:
    with pytest.raises(AgentError):
        build_contact_sheet([], tmp_path / "x.jpg")


def test_plan_timestamps() -> None:
    assert plan_timestamps(0, 10, 5) == [1.0, 3.0, 5.0, 7.0, 9.0]
    with pytest.raises(AgentError):
        plan_timestamps(5, 5, 3)


def test_frames_for_budget() -> None:
    assert frames_for_budget(1100) == 9
    assert frames_for_budget(50) == 1


def test_select_keyframes_prefers_strong_cuts_and_covers_range() -> None:
    cuts = [(10.0, 0.9), (10.3, 0.4), (30.0, 0.5), (50.0, 0.8)]
    picks = select_keyframes(cuts, start=0, end=60, max_frames=4)
    ts = [t for t, _ in picks]
    assert len(ts) == 4
    assert ts == sorted(ts)
    assert ts[0] < 10 < ts[1] < 30 < ts[2] < 50 < ts[3]  # one frame per scene


def test_select_keyframes_fills_static_video() -> None:
    picks = select_keyframes([], start=0, end=90, max_frames=9)
    assert [round(t) for t, _ in picks] == [5, 15, 25, 35, 45, 55, 65, 75, 85]


def test_select_keyframes_limits_to_budget() -> None:
    cuts = [(float(t), t / 100) for t in range(2, 100, 2)]
    assert len(select_keyframes(cuts, start=0, end=100, max_frames=6)) == 6
