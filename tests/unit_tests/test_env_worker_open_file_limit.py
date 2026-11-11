# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Check file-limit setup inside the actual environment worker entry point."""

import os
import subprocess
import sys

import pytest


@pytest.mark.skipif(sys.platform != "linux", reason="Linux worker resource limits")
@pytest.mark.parametrize("requested, expected", [(0, 64), (32, 64), (8192, 8192)])
def test_env_worker_file_limit_in_child(requested, expected):
    code = r"""
import resource
import sys
from types import SimpleNamespace
from rlinf.workers.env.env_worker import EnvWorker

class SetupReached(Exception):
    pass

def stop_before_simulator(*args, **kwargs):
    raise SetupReached

worker = object.__new__(EnvWorker)
worker.cfg = SimpleNamespace(env={"max_open_files": int(sys.argv[1])})
worker._group_name = "test_env"
worker._world_size = 1
worker.log_info = lambda message: None
worker.broadcast = stop_before_simulator
_, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (64, hard))
try:
    worker.init_worker()
except SetupReached:
    pass
actual, after_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
assert actual == min(int(sys.argv[2]), hard)
assert after_hard == hard
if actual >= 256:
    handles = [open("/dev/null", "rb") for _ in range(128)]
    for handle in handles:
        handle.close()
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(requested), str(expected)],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
