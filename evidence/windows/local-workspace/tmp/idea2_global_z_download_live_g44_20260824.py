from __future__ import annotations

import os
from pathlib import Path
import sys
from types import SimpleNamespace

WORKSPACE = Path(r"C:\Users\86136\Documents\rl")
sys.path.insert(0, str(WORKSPACE / "local_scripts"))
import remote_exec_autodl  # noqa: E402

REMOTE_RUN = "/root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_global_z_w0to2_formal_100step_2gpu16env_20260823"
REMOTE_RUNTIME = "/root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_global_z_w0to2_formal_100step_2gpu16env_20260823"
DEST = (
    WORKSPACE
    / "docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/"
    "global_z_w0to2_live_g44_20260824/raw"
)

FILES = [
    (f"{REMOTE_RUN}/metrics.log", "run/metrics.log"),
    (f"{REMOTE_RUN}/tensorboard/config.yaml", "run/tensorboard/config.yaml"),
    (
        f"{REMOTE_RUN}/control_trace/adjust_bottle/worker_000/env_slot_000/recording_0000_episode_0000_reset_57/frames.csv",
        "run/control_trace/reset_57/frames.csv",
    ),
    (
        f"{REMOTE_RUN}/control_trace/adjust_bottle/worker_000/env_slot_000/recording_0000_episode_0000_reset_57/metadata.json",
        "run/control_trace/reset_57/metadata.json",
    ),
    (
        f"{REMOTE_RUN}/control_trace/adjust_bottle/worker_000/env_slot_000/recording_0000_episode_0000_reset_57/head_camera.mp4",
        "run/control_trace/reset_57/head_camera.mp4",
    ),
    (f"{REMOTE_RUN}/dvac_train/actor_rank00/runner_step_metrics.csv", "run/dvac_train/actor_rank00/runner_step_metrics.csv"),
    (f"{REMOTE_RUN}/dvac_train/actor_rank01/runner_step_metrics.csv", "run/dvac_train/actor_rank01/runner_step_metrics.csv"),
    (f"{REMOTE_RUN}/dvac_train/actor_rank00/rolling_stats_state.json", "run/dvac_train/actor_rank00/rolling_stats_state.json"),
    (f"{REMOTE_RUN}/dvac_train/actor_rank01/rolling_stats_state.json", "run/dvac_train/actor_rank01/rolling_stats_state.json"),
    (f"{REMOTE_RUN}/dvac_train/actor_rank00/run_manifest.json", "run/dvac_train/actor_rank00/run_manifest.json"),
    (f"{REMOTE_RUN}/dvac_train/actor_rank01/run_manifest.json", "run/dvac_train/actor_rank01/run_manifest.json"),
    (f"{REMOTE_RUN}/dvac_train/actor_rank00/rollout_step0043.npz", "run/dvac_train/actor_rank00/rollout_step0043.npz"),
    (f"{REMOTE_RUN}/dvac_train/actor_rank01/rollout_step0043.npz", "run/dvac_train/actor_rank01/rollout_step0043.npz"),
    (f"{REMOTE_RUNTIME}/driver.log", "runtime/driver.log"),
    (f"{REMOTE_RUNTIME}/wrapper.log", "runtime/wrapper.log"),
    (f"{REMOTE_RUNTIME}/observer.log", "runtime/observer.log"),
    (f"{REMOTE_RUNTIME}/resolved_config.yaml", "runtime/resolved_config.yaml"),
    (f"{REMOTE_RUNTIME}/launch_command.txt", "runtime/launch_command.txt"),
    (f"{REMOTE_RUNTIME}/resource_monitor/resources.csv", "runtime/resource_monitor/resources.csv"),
]

GROWING = {
    "run/metrics.log",
    "run/dvac_train/actor_rank00/runner_step_metrics.csv",
    "run/dvac_train/actor_rank01/runner_step_metrics.csv",
    "runtime/driver.log",
    "runtime/wrapper.log",
    "runtime/observer.log",
    "runtime/resource_monitor/resources.csv",
}


def main() -> None:
    if not os.environ.get("SEETA_SSH_PASSWORD"):
        raise SystemExit("SEETA_SSH_PASSWORD is required in the current process")
    args = SimpleNamespace(
        host=remote_exec_autodl.DEFAULT_HOST,
        port=remote_exec_autodl.DEFAULT_PORT,
        user=remote_exec_autodl.DEFAULT_USER,
        host_key_sha256=remote_exec_autodl.DEFAULT_HOST_KEY_SHA256,
        timeout=20.0,
    )
    client = remote_exec_autodl.connect(args)
    try:
        with client.open_sftp() as sftp:
            DEST.mkdir(parents=True, exist_ok=True)
            total = 0
            for remote, rel in FILES:
                attrs = sftp.stat(remote)
                local = DEST / Path(rel)
                local.parent.mkdir(parents=True, exist_ok=True)
                sftp.get(remote, str(local))
                actual = local.stat().st_size
                if rel in GROWING:
                    if actual < attrs.st_size:
                        raise RuntimeError(f"truncated growing file {rel}: {actual} < {attrs.st_size}")
                elif actual != attrs.st_size:
                    raise RuntimeError(f"size mismatch for {rel}: {actual} != {attrs.st_size}")
                print(f"GET {rel} bytes={actual}")
                total += actual
            event_dir = f"{REMOTE_RUN}/tensorboard"
            for attrs in sftp.listdir_attr(event_dir):
                if not attrs.filename.startswith("events.out.tfevents"):
                    continue
                rel = f"run/tensorboard/{attrs.filename}"
                local = DEST / rel
                local.parent.mkdir(parents=True, exist_ok=True)
                sftp.get(f"{event_dir}/{attrs.filename}", str(local))
                actual = local.stat().st_size
                if actual < attrs.st_size:
                    raise RuntimeError(f"truncated growing file {rel}: {actual} < {attrs.st_size}")
                print(f"GET {rel} bytes={actual}")
                total += actual
            print(f"SNAPSHOT_BYTES={total}")
            print(f"DEST={DEST}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
