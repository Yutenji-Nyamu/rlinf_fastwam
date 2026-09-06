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


target = workspace / "docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/formal_live_step22_20260821"
remote_run = "/root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_apply_formal_100step_2gpu16env_20260821"
remote_runtime = "/root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_apply_formal_100step_2gpu16env_20260821"


def download_file(sftp, remote: str, relative: str) -> int:
    local = target / relative
    local.parent.mkdir(parents=True, exist_ok=True)
    sftp.get(remote, str(local))
    return local.stat().st_size


def download_tree(sftp, remote: str, relative: str) -> tuple[int, int]:
    files = 0
    size = 0
    for item in sftp.listdir_attr(remote):
        child_remote = f"{remote}/{item.filename}"
        child_relative = f"{relative}/{item.filename}"
        if item.st_mode & 0o170000 == 0o040000:
            child_files, child_size = download_tree(sftp, child_remote, child_relative)
            files += child_files
            size += child_size
        elif item.st_mode & 0o170000 == 0o100000:
            size += download_file(sftp, child_remote, child_relative)
            files += 1
    return files, size


args = argparse.Namespace(
    host=DEFAULT_HOST,
    port=DEFAULT_PORT,
    user=DEFAULT_USER,
    host_key_sha256=DEFAULT_HOST_KEY_SHA256,
    timeout=20.0,
)
client = connect(args)
count = 0
total = 0
try:
    with client.open_sftp() as sftp:
        fixed = [
            (f"{remote_run}/metrics.log", "run/metrics.log"),
            (f"{remote_run}/tensorboard/config.yaml", "run/resolved_config.yaml"),
            (f"{remote_runtime}/driver.log", "runtime/driver.log"),
            (f"{remote_runtime}/launch_command.txt", "runtime/launch_command.txt"),
            (f"{remote_runtime}/launch_started_at.txt", "runtime/launch_started_at.txt"),
            (f"{remote_runtime}/resource_monitor/resources.csv", "runtime/resources.csv"),
            (f"{remote_runtime}/resource_monitor/process_rss.tsv", "runtime/process_rss.tsv"),
        ]
        for remote, relative in fixed:
            total += download_file(sftp, remote, relative)
            count += 1
        for remote, relative in [
            (f"{remote_run}/dvac_train", "run/dvac_train"),
            (f"{remote_run}/control_trace", "run/control_trace"),
        ]:
            tree_count, tree_size = download_tree(sftp, remote, relative)
            count += tree_count
            total += tree_size
finally:
    client.close()

print(f"SNAPSHOT_FILES={count}")
print(f"SNAPSHOT_BYTES={total}")
print(f"SNAPSHOT_TARGET={target}")
