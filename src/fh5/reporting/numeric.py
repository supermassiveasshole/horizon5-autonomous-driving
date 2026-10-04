"""Display-only previews and an offline viewer; never used as model inputs."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fh5.artifacts.io import asset
from fh5.artifacts.json_view import JsonArray, write_json


def preview_png(pixels: bytes, size: tuple[int, int]) -> bytes:
    """Lossless RGB preview; callers run this in preparation or the archive worker."""
    width, height = size
    stride = width * 3

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    scanlines = b"".join(b"\0" + pixels[y * stride : (y + 1) * stride] for y in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(scanlines, level=1))
        + chunk(b"IEND", b"")
    )


def write_numeric_report(path: Path, summary: dict[str, Any], root: Path) -> None:
    def preview_url(relative: str) -> str:
        target = asset(root, relative)
        if target.is_relative_to(path.parent.resolve()):
            return quote(target.relative_to(path.parent.resolve()).as_posix())
        return target.as_uri()

    def display_sample(original: dict[str, Any]) -> dict[str, Any]:
        sample = dict(original)
        for key in ("preview", "model_preview"):
            try:
                sample[key + "_url"] = preview_url(sample[key])
            except ValueError:
                sample[key + "_url"] = None
        return sample

    def display_decision(original: dict[str, Any]) -> dict[str, Any]:
        row = dict(original)
        references = (row.get("archive") or {}).get("previews", row.get("previews", []))
        urls: list[str | None] = []
        for relative in references:
            if relative is None:
                urls.append(None)
                continue
            try:
                urls.append(preview_url(relative))
            except ValueError:
                urls.append(None)
        row["preview_urls"] = urls
        return row

    display = dict(summary, decisions=JsonArray(map(display_decision, summary["decisions"])))
    if summary.get("raw_samples"):
        samples = summary["raw_samples"]
        display["raw_samples"] = dict(
            samples, records=JsonArray(map(display_sample, samples.get("records", [])))
        )
    template = Path(__file__).with_name("numeric-report.html").read_text(encoding="utf-8")
    before, after = template.split("/*NUMERIC_DATA*/null")
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(before)
        write_json(stream, display, script_safe=True)
        stream.write(after)
