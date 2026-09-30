"""Display-only previews and an offline viewer; never used as model inputs."""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path
from typing import Any

from fh5.numeric_images import asset


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
    display = json.loads(json.dumps(summary))
    for row in display["decisions"]:
        references = (row.get("archive") or {}).get("previews", row.get("previews", []))
        urls: list[str | None] = []
        for relative in references:
            if relative is None:
                urls.append(None)
                continue
            try:
                urls.append(asset(root, relative).as_uri())
            except ValueError:
                urls.append(None)
        row["preview_urls"] = urls
    data = json.dumps(display, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    template = Path(__file__).with_name("numeric-report.html").read_text(encoding="utf-8")
    path.write_text(template.replace("/*NUMERIC_DATA*/null", data), encoding="utf-8")
