"""Timestamped contact-sheet grids sized for vision LLMs.

A 3x3 sheet of 16:9 frames with a 384 px long edge is 1168x664 px including gutters,
about 1,030 vision tokens for nine frames, versus roughly 2,770 tokens for nine separate
640x360 images.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from agent_sight.errors import AgentError, ErrorCode
from agent_sight.models import ContactSheet, Frame, estimate_image_tokens, format_timestamp

BACKGROUND = (18, 18, 18)
LABEL_BG = (0, 0, 0, 190)
LABEL_FG = (255, 255, 255, 255)
HEADER_FG = (230, 230, 230)
GUTTER = 4


@lru_cache(maxsize=8)
def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "Arial.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def tile_size(aspect: float, long_edge: int) -> tuple[int, int]:
    if aspect >= 1:
        return long_edge, max(1, round(long_edge / aspect))
    return max(1, round(long_edge * aspect)), long_edge


def build_contact_sheet(
    frames: Sequence[tuple[Path | str, float]],
    out_path: Path,
    *,
    columns: int = 3,
    tile_long_edge: int = 384,
    header: str | None = None,
    quality: int = 80,
) -> ContactSheet:
    """Lay ``(image_path, timestamp)`` pairs on a grid with burned-in timestamps."""
    if not frames:
        raise AgentError(ErrorCode.INVALID_ARGUMENT, "Cannot build a contact sheet from zero frames.")
    if columns < 1 or tile_long_edge < 32:
        raise AgentError(ErrorCode.INVALID_ARGUMENT, "columns must be >= 1 and tile size >= 32.")

    images: list[Image.Image] = []
    try:
        for path, _ in frames:
            with Image.open(path) as im:
                images.append(im.convert("RGB"))
    except OSError as exc:
        raise AgentError(ErrorCode.INTERNAL_ERROR, f"Could not read frame image: {exc}") from exc

    first_w, first_h = images[0].size
    tw, th = tile_size(first_w / first_h, tile_long_edge)
    cols = min(columns, len(frames))
    rows = -(-len(frames) // cols)
    header_h = 0
    header_font = _font(max(12, tile_long_edge // 24))
    if header:
        header_h = int(getattr(header_font, "size", 12)) + 12

    width = cols * tw + (cols + 1) * GUTTER
    height = header_h + rows * th + (rows + 1) * GUTTER
    sheet = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(sheet, "RGBA")
    if header:
        draw.text((GUTTER + 4, 6), _ellipsize(draw, header, header_font, width - 16), fill=HEADER_FG, font=header_font)

    label_font = _font(max(11, tile_long_edge // 20))
    labels: list[str] = []
    for i, (img, (_, ts)) in enumerate(zip(images, frames, strict=True)):
        r, c = divmod(i, cols)
        x0 = GUTTER + c * (tw + GUTTER)
        y0 = header_h + GUTTER + r * (th + GUTTER)
        fitted = img.copy()
        fitted.thumbnail((tw, th), Image.Resampling.LANCZOS)
        ox = x0 + (tw - fitted.width) // 2
        oy = y0 + (th - fitted.height) // 2
        sheet.paste(fitted, (ox, oy))

        label = format_timestamp(ts)
        labels.append(label)
        bbox = draw.textbbox((0, 0), label, font=label_font)
        lw, lh = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad = 4
        bx, by = x0 + 4, y0 + th - lh - 2 * pad - 4
        draw.rounded_rectangle((bx, by, bx + lw + 2 * pad, by + lh + 2 * pad), radius=4, fill=LABEL_BG)
        draw.text((bx + pad - bbox[0], by + pad - bbox[1]), label, fill=LABEL_FG, font=label_font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, "JPEG", quality=quality, optimize=True)
    return ContactSheet(
        path=str(out_path),
        width=width,
        height=height,
        columns=cols,
        rows=rows,
        timestamps=[float(ts) for _, ts in frames],
        labels=labels,
        estimated_tokens=estimate_image_tokens(width, height),
    )


def build_contact_sheets(
    frames: Sequence[Frame],
    out_dir: Path,
    *,
    prefix: str = "sheet",
    columns: int = 3,
    rows: int = 3,
    tile_long_edge: int = 384,
    header: str | None = None,
) -> list[ContactSheet]:
    """Split frames into as many ``columns x rows`` sheets as needed."""
    per_sheet = max(1, columns * rows)
    chunks = [list(frames[i : i + per_sheet]) for i in range(0, len(frames), per_sheet)]
    sheets: list[ContactSheet] = []
    for n, chunk in enumerate(chunks, 1):
        title = header
        if title and len(chunks) > 1:
            title = f"{header}  [{n}/{len(chunks)}]"
        sheets.append(
            build_contact_sheet(
                [(f.path, f.timestamp) for f in chunk],
                out_dir / f"{prefix}_{n:02d}.jpg",
                columns=columns,
                tile_long_edge=tile_long_edge,
                header=title,
            )
        )
    return sheets


def _ellipsize(draw: ImageDraw.ImageDraw, text: str, font: object, max_w: int) -> str:
    if draw.textlength(text, font=font) <= max_w:  # type: ignore[arg-type]
        return text
    while text and draw.textlength(text + "…", font=font) > max_w:  # type: ignore[arg-type]
        text = text[:-1]
    return text + "…"
