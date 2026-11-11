"""Optional bounded pre-native phases, then resume the accepted Dojo training."""

import os
import subprocess
import time
from pathlib import Path

from .common import write_json
from .pipeline import cleanup


def run(directory, plan, launch, status, read):
    d = Path(directory)
    for phase in plan.get("pre_native_phases", []):
        started = time.time()
        status(phase["name"], started=started)
        env = dict(os.environ, **read(d / "training-environment.json"))
        p = launch(phase["name"], phase["command"], env)
        try:
            code = p.wait(timeout=phase["timeout"])
        except subprocess.TimeoutExpired:
            # Clean exact tagged descendants before the formal native process starts.
            for identifier in phase.get("stage_identifiers", []):
                cleanup(identifier)
            p.terminate()
            try:
                code = p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                p.kill()
                code = p.wait(timeout=15)
            code = 124
        except BaseException:
            # Stop the dispatcher first, so it cannot start another tagged stage.
            p.terminate()
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(timeout=15)
            raise
        finally:
            for identifier in phase.get("stage_identifiers", []):
                cleanup(identifier)
        write_json(
            d / (phase["name"] + "-finished.json"),
            dict(
                returncode=code,
                started=started,
                ended=time.time(),
                continue_to_formal=bool(phase.get("continue_on_failure")),
            ),
        )
        if code and not phase.get("continue_on_failure"):
            raise RuntimeError(f"{phase['name']} exited {code}")
