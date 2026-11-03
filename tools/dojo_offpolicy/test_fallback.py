# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Real CPU-only exit cleanup; never stop a GPU task to test its fallback."""

import json
import os
import subprocess
import sys

from tools.dojo_offpolicy.owner import proc
from tools.dojo_offpolicy.recover_queue import abnormal_empty, release_missing


def test_dead_owner_cleans_only_its_tagged_child(tmp_path):
    command = [sys.executable, "-c", "import time; time.sleep(60)"]
    owner = subprocess.Popen(command)
    identity = proc(owner.pid)
    owner.kill()
    owner.wait(timeout=5)
    (tmp_path / "identity.json").write_text(json.dumps(identity))
    child = subprocess.Popen(
        command, env=dict(os.environ, DOJO_GRPO_OWNER=str(tmp_path))
    )
    neighbor = subprocess.Popen(command)
    try:
        assert release_missing(tmp_path)
        assert child.wait(timeout=5) != 0
        assert neighbor.poll() is None
        receipt = json.loads((tmp_path / "released.json").read_text())
        assert [x["pid"] for x in receipt["cleaned"]] == [child.pid]
    finally:
        for process in (child, neighbor):
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=5)


def test_live_owner_is_never_released(tmp_path):
    (tmp_path / "identity.json").write_text(json.dumps(proc(os.getpid())))
    assert not release_missing(tmp_path)
    assert not (tmp_path / "released.json").exists()


def test_startup_and_fresh_progress_are_not_abnormal_idle():
    assert not abnormal_empty(500, 0, 0, 100)  # Within startup allowance.
    assert not abnormal_empty(800, 0, 750, 600)  # CPU work still progressing.
    assert not abnormal_empty(800, 0, 0, None)  # GPU has a context.
    assert abnormal_empty(800, 0, 600, 600)
