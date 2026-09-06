from __future__ import annotations

import argparse
import getpass
import gzip
import os
import sys
from pathlib import Path


WORKSPACE = Path(r"C:\Users\86136\Documents\rl")
sys.path.insert(0, str(WORKSPACE))

from local_scripts.remote_exec_autodl import (  # noqa: E402
    DEFAULT_HOST,
    DEFAULT_HOST_KEY_SHA256,
    DEFAULT_PORT,
    DEFAULT_USER,
    connect,
)


TARGET = (
    WORKSPACE
    / "docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/"
    "r_only_formal_live_g50_20260822"
)
REMOTE_RUN = (
    "/root/autodl-tmp/idea2_dvac_train_runs/"
    "idea2_dvac_r_only_downweight_formal_100step_2gpu16env_20260822"
)
REMOTE_RUNTIME = (
    "/root/autodl-tmp/idea2_dvac_train_runtime/"
    "idea2_dvac_r_only_downweight_formal_100step_2gpu16env_20260822"
)
LATEST_GLOBAL_STEP = 50
LATEST_RUNNER_STEP = LATEST_GLOBAL_STEP - 1


def get_file(sftp, remote: str, relative: str) -> int:
    local = TARGET / relative
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

files = [
    (f"{REMOTE_RUN}/metrics.log", "raw/run/metrics.log"),
    (f"{REMOTE_RUNTIME}/launch_started_at.txt", "raw/runtime/launch_started_at.txt"),
    (f"{REMOTE_RUNTIME}/resolved_config.yaml", "raw/runtime/resolved_config.yaml"),
]
for rank in (0, 1):
    remote_rank = f"{REMOTE_RUN}/dvac_train/actor_rank{rank:02d}"
    local_rank = f"raw/run/dvac_train/actor_rank{rank:02d}"
    files.extend(
        [
            (
                f"{remote_rank}/runner_step_metrics.csv",
                f"{local_rank}/runner_step_metrics.csv",
            ),
            (
                f"{remote_rank}/rolling_stats_state.json",
                f"{local_rank}/rolling_stats_state.json",
            ),
            (
                f"{remote_rank}/run_manifest.json",
                f"{local_rank}/run_manifest.json",
            ),
            (
                f"{remote_rank}/rollout_step{LATEST_RUNNER_STEP:04d}.npz",
                f"{local_rank}/rollout_step{LATEST_RUNNER_STEP:04d}.npz",
            ),
        ]
    )

password = [REDACTED]"SSH password: ")
os.environ["SEETA_SSH_PASSWORD"] = password
client = connect(args)
try:
    total = 0
    _, stdout, stderr = client.exec_command(
        f"gzip -c -- {REMOTE_RUNTIME}/resource_monitor/resources.csv"
    )
    compressed_resources = stdout.read()
    resource_stderr = stderr.read()
    resource_rc = stdout.channel.recv_exit_status()
    if resource_rc != 0:
        raise RuntimeError(
            f"remote gzip failed with exit {resource_rc}: "
            + resource_stderr.decode("utf-8", errors="replace")
        )
    resource_bytes = gzip.decompress(compressed_resources)
    resource_local = TARGET / "raw/runtime/resource_monitor/resources.csv"
    resource_local.parent.mkdir(parents=True, exist_ok=True)
    resource_local.write_bytes(resource_bytes)
    total += len(resource_bytes)
    with client.open_sftp() as sftp:
        for remote, relative in files:
            total += get_file(sftp, remote, relative)
finally:
    client.close()
    password = ""
    os.environ.pop("SEETA_SSH_PASSWORD", None)

print(f"GLOBAL_STEP={LATEST_GLOBAL_STEP}")
print(f"DOWNLOADED_FILES={len(files)}")
print(f"TOTAL_LOCAL_BYTES={total}")
print(f"TARGET={TARGET}")
