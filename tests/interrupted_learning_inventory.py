"""Exit a public learning run at an external filesystem publication boundary."""

import json
import os
import sys
from pathlib import Path

from recovery_inventory_cases import FailedOriginals

from fh5.experiment import run_experiment
from fh5.learning_loop import LearningLoop


def main():
    config, root, source = (Path(value).resolve() for value in sys.argv[1:4])
    boundary = sys.argv[4]
    index = root / "round-000/learning-sources.sqlite3"
    descriptor = root / "round-000/learning-originals.json"
    state = root / "state.json"
    replace = os.replace

    def crash(source, target):
        (root / "inventory-interruption.json").write_text(
            json.dumps({"source": str(source), "target": str(target), "boundary": boundary}),
            encoding="utf-8",
        )
        os._exit(73)

    def publish(source, target, *args, **kwargs):
        destination = Path(target).resolve()
        if destination == index and boundary == "before_index":
            crash(source, target)
        if destination == state and boundary == "before_parent":
            proposed = json.loads(Path(source).read_bytes())
            if proposed["phase"] == "retrying_sampling":
                crash(source, target)
        result = replace(source, target, *args, **kwargs)
        if destination == index and boundary == "after_index":
            crash(source, target)
        if destination == descriptor and boundary == "after_descriptor":
            crash(source, target)
        return result

    os.replace = publish
    run_experiment(LearningLoop(config, root), learning_environment=FailedOriginals(source))
    raise SystemExit("Expected inventory publication boundary was not reached")


if __name__ == "__main__":
    main()
