"""Standalone, offline viewing of a recording; data is never interpolated as HTML."""

import json
from pathlib import Path
from typing import Any


def write_report(path: Path, payload: dict[str, Any]) -> None:
    if path.suffix.lower() != ".html":
        raise ValueError("Report output must have an .html extension")
    json_path = path.with_suffix(".json")
    if path.exists() or json_path.exists():
        raise FileExistsError(f"Report outputs already exist: {path} / {json_path}")
    template = Path(__file__).with_name("report.html").read_text(encoding="utf-8")
    data = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    # Prevent closing the inert data script, including from user-supplied snapshot text.
    data = data.replace("<", "\\u003c").replace("&", "\\u0026")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        output.write(template.replace("<!--REPORT_DATA-->", data))
    with json_path.open("x", encoding="utf-8") as output:
        output.write(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
