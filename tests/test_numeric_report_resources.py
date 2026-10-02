"""Report generation grows on disk without cloning the complete prediction corpus."""

import json
import tracemalloc
from dataclasses import replace
from pathlib import Path

from test_numeric_images import NumericalProbe, decisions

from fh5.experiment import run_experiment
from fh5.numeric_images import NumericInfer, PixelContract


def test_numeric_html_does_not_retain_a_second_complete_report(tmp_path, monkeypatch):
    root = tmp_path / "numerical"
    original_open = Path.open
    observed = []

    def observe_html_publication(path, mode="r", *args, **kwargs):
        if path == root / "report.html" and "w" in mode:
            observed.append(tracemalloc.get_traced_memory()[0])
        return original_open(path, mode, *args, **kwargs)

    def observations():
        base = next(decisions())
        frames = tuple(
            replace(frame, time_quality="synthetic</script>" + "x" * 262144)
            for frame in base.frames
        )
        for i in range(12):
            yield replace(base, decision_id=f"large-{i}", frames=frames)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", observe_html_publication)
        tracemalloc.start()
        try:
            result = run_experiment(
                NumericInfer(root, PixelContract(size=(2, 1)), archive_capacity=12),
                numeric_actor=NumericalProbe(),
                numeric_inputs=observations(),
            )
        finally:
            tracemalloc.stop()
    evidence = root / "report.json"
    assert observed and max(observed) < evidence.stat().st_size, (
        "HTML publication retains another complete serialized prediction report",
        observed,
        evidence.stat().st_size,
    )
    saved = json.loads(evidence.read_bytes())
    assert saved == result.summary["numeric"]
    html = result.report_path.read_text(encoding="utf-8")
    display, _ = json.JSONDecoder().raw_decode(html.split("const data=", 1)[1])
    assert "synthetic</script>" not in html
    assert len(display["decisions"]) == 12
    for row, expected in zip(display["decisions"], saved["decisions"]):
        assert row.pop("preview_urls")
        assert row == expected
        assert row["prediction"] == [0.2, -0.1]
