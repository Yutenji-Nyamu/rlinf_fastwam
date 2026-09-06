from __future__ import annotations

import argparse
import getpass
import gzip
import os
import sys
from pathlib import Path


WORKSPACE = Path(r"C:\Users\86136\Documents\rl")
TARGET = (
    WORKSPACE
    / "docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/"
    "r_only_formal_stop_g51_20260822"
)
REMOTE_RUN = (
    "/root/autodl-tmp/idea2_dvac_train_runs/"
    "idea2_dvac_r_only_downweight_formal_100step_2gpu16env_20260822"
)
REMOTE_RUNTIME = (
    "/root/autodl-tmp/idea2_dvac_train_runtime/"
    "idea2_dvac_r_only_downweight_formal_100step_2gpu16env_20260822"
)

sys.path.insert(0, str(WORKSPACE / "local_scripts"))
from remote_exec_autodl import (  # noqa: E402
    DEFAULT_HOST,
    DEFAULT_HOST_KEY_SHA256,
    DEFAULT_PORT,
    DEFAULT_USER,
    connect,
)


def get_file(sftp, remote: str, relative: str) -> int:
    local = TARGET / relative
    local.parent.mkdir(parents=True, exist_ok=True)
    sftp.get(remote, str(local))
    return local.stat().st_size


def capture_command(client, command: str, stem: str) -> int:
    _, stdout, stderr = client.exec_command(command)
    out = stdout.read()
    err = stderr.read()
    rc = stdout.channel.recv_exit_status()
    (TARGET / f"{stem}.stdout.txt").write_bytes(out)
    (TARGET / f"{stem}.stderr.txt").write_bytes(err)
    (TARGET / f"{stem}.exitcode.txt").write_text(str(rc) + "\n", encoding="utf-8")
    if rc != 0:
        raise RuntimeError(f"remote command {stem} failed with exit {rc}")
    return len(out) + len(err)


FILES: list[tuple[str, str]] = [
    (f"{REMOTE_RUN}/metrics.log", "raw/run/metrics.log"),
    (f"{REMOTE_RUNTIME}/resolved_config.yaml", "raw/runtime/resolved_config.yaml"),
    (f"{REMOTE_RUNTIME}/launch_command.txt", "raw/runtime/launch_command.txt"),
    (f"{REMOTE_RUNTIME}/launch_started_at.txt", "raw/runtime/launch_started_at.txt"),
    (f"{REMOTE_RUNTIME}/launch_finished_at.txt", "raw/runtime/launch_finished_at.txt"),
    (f"{REMOTE_RUNTIME}/driver.exitcode", "raw/runtime/driver.exitcode"),
    (f"{REMOTE_RUNTIME}/observer.exitcode", "raw/runtime/observer.exitcode"),
    (f"{REMOTE_RUNTIME}/driver.log", "raw/runtime/driver.log"),
    (f"{REMOTE_RUNTIME}/observer.log", "raw/runtime/observer.log"),
    (f"{REMOTE_RUNTIME}/wrapper.log", "raw/runtime/wrapper.log"),
    (f"{REMOTE_RUNTIME}/launch_formal.sh", "raw/runtime/launch_formal.sh"),
    (f"{REMOTE_RUNTIME}/observe_resources.sh", "raw/runtime/observe_resources.sh"),
]

for rank in (0, 1):
    remote_rank = f"{REMOTE_RUN}/dvac_train/actor_rank{rank:02d}"
    local_rank = f"raw/run/dvac_train/actor_rank{rank:02d}"
    for name in (
        "run_manifest.json",
        "runner_step_metrics.csv",
        "rolling_stats_state.json",
        "rollout_step0000.npz",
        "rollout_step0001.npz",
        "rollout_step0024.npz",
        "rollout_step0050.npz",
    ):
        FILES.append((f"{remote_rank}/{name}", f"{local_rank}/{name}"))

trace = (
    f"{REMOTE_RUN}/control_trace/adjust_bottle/worker_000/env_slot_000/"
    "recording_0000_episode_0000_reset_57"
)
for name in ("frames.csv", "head_camera.mp4", "metadata.json"):
    FILES.append((f"{trace}/{name}", f"raw/run/control_trace/reset_57/{name}"))


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
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
    skipped: list[str] = []
    try:
        inventory_command = (WORKSPACE / "tmp/idea2_r_only_post_stop_inventory_20260822.sh").read_text(
            encoding="utf-8"
        )
        total += capture_command(client, inventory_command, "FINAL_READONLY_AUDIT")

        _, stdout, stderr = client.exec_command(
            f"gzip -c -- {REMOTE_RUNTIME}/resource_monitor/resources.csv"
        )
        compressed = stdout.read()
        error = stderr.read()
        rc = stdout.channel.recv_exit_status()
        if rc != 0:
            raise RuntimeError(
                f"resource gzip failed with exit {rc}: "
                + error.decode("utf-8", errors="replace")
            )
        resource = gzip.decompress(compressed)
        resource_local = TARGET / "raw/runtime/resource_monitor/resources.csv"
        resource_local.parent.mkdir(parents=True, exist_ok=True)
        resource_local.write_bytes(resource)
        total += len(resource)

        with client.open_sftp() as sftp:
            for remote, relative in FILES:
                try:
                    sftp.stat(remote)
                except OSError:
                    skipped.append(remote)
                    continue
                total += get_file(sftp, remote, relative)
    finally:
        client.close()
        password = ""
        os.environ.pop("SEETA_SSH_PASSWORD", None)

    (TARGET / "DOWNLOAD_SKIPPED.txt").write_text(
        "\n".join(skipped) + ("\n" if skipped else ""), encoding="utf-8"
    )
    print(f"REQUESTED_FILES={len(FILES)}")
    print(f"SKIPPED_FILES={len(skipped)}")
    print(f"TOTAL_LOCAL_BYTES={total}")
    print(f"TARGET={TARGET}")


if __name__ == "__main__":
    main()
