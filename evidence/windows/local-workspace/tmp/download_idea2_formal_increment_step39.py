from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

workspace = Path(r"C:\Users\86136\Documents\rl")
sys.path.insert(0, str(workspace))

from local_scripts.remote_exec_autodl import (
    DEFAULT_HOST,
    DEFAULT_HOST_KEY_SHA256,
    DEFAULT_PORT,
    DEFAULT_USER,
    connect,
)


target = workspace / "docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/formal_live_step39_20260821"
remote_run = "/root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_apply_formal_100step_2gpu16env_20260821"
remote_runtime = "/root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_apply_formal_100step_2gpu16env_20260821"


def download(sftp, remote: str, relative: str) -> int:
    local = target / relative
    local.parent.mkdir(parents=True, exist_ok=True)
    sftp.get(remote, str(local))
    return local.stat().st_size


args = argparse.Namespace(
    host=DEFAULT_HOST,
    port=DEFAULT_PORT,
    user=DEFAULT_USER,
    host_key_sha256=DEFAULT_HOST_KEY_SHA256,
    timeout=20.0,
)
client = connect(args)
files: list[tuple[str, str]] = [
    (f"{remote_run}/metrics.log", "run/metrics.log"),
    (f"{remote_runtime}/resource_monitor/resources.csv", "runtime/resources.csv"),
]
for rank in (0, 1):
    remote_rank = f"{remote_run}/dvac_train/actor_rank{rank:02d}"
    local_rank = f"run/dvac_train/actor_rank{rank:02d}"
    files.extend(
        [
            (f"{remote_rank}/runner_step_metrics.csv", f"{local_rank}/runner_step_metrics.csv"),
            (f"{remote_rank}/rolling_stats_state.json", f"{local_rank}/rolling_stats_state.json"),
            (f"{remote_rank}/run_manifest.json", f"{local_rank}/run_manifest.json"),
        ]
    )
    for runner_step in range(23, 39):
        name = f"rollout_step{runner_step:04d}.npz"
        files.append((f"{remote_rank}/{name}", f"{local_rank}/{name}"))

total = 0
try:
    with client.open_sftp() as sftp:
        for remote, relative in files:
            total += download(sftp, remote, relative)
finally:
    client.close()

print(f"FILES={len(files)}")
print(f"BYTES={total}")
print(f"TARGET={target}")
