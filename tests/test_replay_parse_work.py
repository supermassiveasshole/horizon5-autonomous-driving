"""Growing individual JSON records must not cause quadratic re-parsing."""

import hashlib
import json

from test_critic_resume import warm_inputs

from fh5.experiment import run_experiment
from fh5.sac import SACCriticWarmup


def test_large_nested_record_has_linear_incomplete_decode_work(tmp_path, monkeypatch):
    model, path, _ = warm_inputs(tmp_path)
    replay = json.loads(path.read_bytes())
    marker = "nested-parse-work-probe:"
    payload = marker + "x" * (2 * 1024**2)
    replay["notes"] = [{"diagnostic": payload}]
    path.write_text(json.dumps(replay), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    decode = json.JSONDecoder.raw_decode
    incomplete_work = []

    def raw_decode(decoder, text, idx=0):
        try:
            return decode(decoder, text, idx)
        except json.JSONDecodeError:
            if marker in text:
                incomplete_work.append(len(text) - idx)
            raise

    with monkeypatch.context() as patch:
        patch.setattr(json.JSONDecoder, "raw_decode", raw_decode)
        result = run_experiment(
            SACCriticWarmup(model, path, digest, tmp_path / "warm", steps=1)
        ).summary["sac"]
    assert result["steps_completed"] == 1
    # Geometric retries visit at most a small linear multiple of a growing
    # value; fixed-size retries instead revisit a quadratic amount of text.
    assert sum(incomplete_work) < 4 * len(payload), (len(incomplete_work), sum(incomplete_work))
