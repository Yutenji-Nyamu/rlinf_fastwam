from __future__ import annotations

import argparse
import getpass
import os
import stat
import sys
from pathlib import Path


WORKSPACE = Path(r"C:\Users\86136\Documents\rl")
LOCAL = (
    WORKSPACE
    / "docs"
    / "rlinf-robotwin-pi0-dvac-telemetry"
    / "evidence"
    / "formal_stop_g54_20260821"
)
RUN = "/root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_apply_formal_100step_2gpu16env_20260821"
RUNTIME = "/root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_apply_formal_100step_2gpu16env_20260821"

FILES = {
    f"{RUN}/metrics.log": "run/metrics.log",
    f"{RUNTIME}/launch_command.txt": "runtime/launch_command.txt",
    f"{RUNTIME}/launch_started_at.txt": "runtime/launch_started_at.txt",
    f"{RUNTIME}/stop_requested_at.txt": "runtime/stop_requested_at.txt",
    f"{RUNTIME}/terminate_requested_at.txt": "runtime/terminate_requested_at.txt",
    f"{RUNTIME}/launch_finished_at.txt": "runtime/launch_finished_at.txt",
    f"{RUNTIME}/driver.exitcode": "runtime/driver.exitcode",
    f"{RUNTIME}/driver.log": "runtime/driver.log",
    f"{RUNTIME}/wrapper.log": "runtime/wrapper.log",
    f"{RUNTIME}/observer.log": "runtime/observer.log",
    f"{RUNTIME}/resource_monitor/observer_exit.txt": "runtime/observer_exit.txt",
}

for rank in ("actor_rank00", "actor_rank01"):
    for name in (
        "run_manifest.json",
        "runner_step_metrics.csv",
        "rolling_stats_state.json",
        "rollout_step0000.npz",
        "rollout_step0001.npz",
        "rollout_step0053.npz",
    ):
        FILES[f"{RUN}/dvac_train/{rank}/{name}"] = (
            f"run/dvac_train/{rank}/{name}"
        )

sys.path.insert(0, str(WORKSPACE))
from local_scripts.remote_exec_autodl import (  # noqa: E402
    DEFAULT_HOST,
    DEFAULT_HOST_KEY_SHA256,
    DEFAULT_PORT,
    DEFAULT_USER,
    connect,
)


def download_tree(sftp, remote: str, local: Path) -> int:
    total = 0
    local.mkdir(parents=True, exist_ok=True)
    for entry in sftp.listdir_attr(remote):
        remote_child = f"{remote}/{entry.filename}"
        local_child = local / entry.filename
        if stat.S_ISDIR(entry.st_mode):
            total += download_tree(sftp, remote_child, local_child)
        elif stat.S_ISREG(entry.st_mode):
            local_child.parent.mkdir(parents=True, exist_ok=True)
            sftp.get(remote_child, str(local_child))
            total += entry.st_size
    return total


def main() -> None:
    password = [REDACTED]"SSH password: ")
    os.environ["SEETA_SSH_PASSWORD"] = password
    args = argparse.Namespace(
        host=DEFAULT_HOST,
        port=DEFAULT_PORT,
        user=DEFAULT_USER,
        host_key_sha256=DEFAULT_HOST_KEY_SHA256,
        timeout=20.0,
    )
    client = connect(args)
    total = 0
    try:
        with client.open_sftp() as sftp:
            for remote, relative in FILES.items():
                local = LOCAL / relative
                local.parent.mkdir(parents=True, exist_ok=True)
                sftp.get(remote, str(local))
                total += local.stat().st_size
            total += download_tree(
                sftp,
                f"{RUN}/control_trace",
                LOCAL / "run" / "control_trace",
            )
    finally:
        client.close()
        password = ""
        os.environ.pop("SEETA_SSH_PASSWORD", None)
    print(f"FILES={len(FILES)}+control_trace")
    print(f"BYTES={total}")
    print(f"LOCAL={LOCAL}")


if __name__ == "__main__":
    main()
