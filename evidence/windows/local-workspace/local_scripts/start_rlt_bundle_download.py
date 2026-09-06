"""Start the bounded RLT Git-bundle SFTP download without a visible window."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STDOUT = ROOT / ".tmp" / "rlt_bundle_get.stdout.log"
STDERR = ROOT / ".tmp" / "rlt_bundle_get.stderr.log"

command = [
    sys.executable,
    str(ROOT / "local_scripts" / "remote_exec_autodl.py"),
    "get",
    "/root/autodl-tmp/experiment_exports/rlt_base_48a_20260729.bundle",
    str(ROOT / ".tmp" / "rlt_base_48a_20260729.bundle"),
]

creationflags = 0
if os.name == "nt":
    creationflags = subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS

with STDOUT.open("wb") as stdout, STDERR.open("wb") as stderr:
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=os.environ.copy(),
        stdin=subprocess.DEVNULL,
        stdout=stdout,
        stderr=stderr,
        close_fds=True,
        creationflags=creationflags,
    )

print(process.pid)
