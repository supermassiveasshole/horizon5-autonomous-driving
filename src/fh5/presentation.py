"""Best-effort HTML presentation of already sealed experiment results."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def optional_report(
    path: Path,
    title: str,
    summary: dict[str, Any],
    *,
    fallback: Path,
    diagnostic: Path | None = None,
) -> Path:
    """Stream display data; resource failure returns the existing durable artifact."""
    if diagnostic is not None:
        try:
            diagnostic.parent.mkdir(parents=True, exist_ok=True)
            with diagnostic.open("x", encoding="utf-8") as stream:
                json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
            fallback = diagnostic
        except (OSError, MemoryError) as error:
            summary["diagnostic_export"] = {
                "status": "unavailable",
                "path": str(diagnostic),
                "error": f"{type(error).__name__}: {error}",
            }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as stream:
            stream.write('<!doctype html><meta charset="utf-8"><h1>' + html.escape(title))
            stream.write("</h1><pre>")
            for chunk in json.JSONEncoder(ensure_ascii=False, indent=2, allow_nan=False).iterencode(
                summary
            ):
                stream.write(html.escape(chunk))
            stream.write("</pre>")
    except (OSError, MemoryError) as error:
        summary["presentation"] = {
            "status": "unavailable",
            "path": str(path),
            "error": f"{type(error).__name__}: {error}",
            "retained_result": str(fallback),
        }
        return fallback
    return path
