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
    "global_z_w0to2_stop_g49_20260824/raw"
)

FILES = [
    (f"{REMOTE_RUN}/metrics.log", "run/metrics.log"),
    (f"{REMOTE_RUN}/dvac_train/actor_rank00/runner_step_metrics.csv", "run/dvac_train/actor_rank00/runner_step_metrics.csv"),
    (f"{REMOTE_RUN}/dvac_train/actor_rank01/runner_step_metrics.csv", "run/dvac_train/actor_rank01/runner_step_metrics.csv"),
    (f"{REMOTE_RUN}/dvac_train/actor_rank00/rolling_stats_state.json", "run/dvac_train/actor_rank00/rolling_stats_state.json"),
    (f"{REMOTE_RUN}/dvac_train/actor_rank01/rolling_stats_state.json", "run/dvac_train/actor_rank01/rolling_stats_state.json"),
    (f"{REMOTE_RUN}/dvac_train/actor_rank00/run_manifest.json", "run/dvac_train/actor_rank00/run_manifest.json"),
    (f"{REMOTE_RUN}/dvac_train/actor_rank01/run_manifest.json", "run/dvac_train/actor_rank01/run_manifest.json"),
    (f"{REMOTE_RUN}/dvac_train/actor_rank00/rollout_step0048.npz", "run/dvac_train/actor_rank00/rollout_step0048.npz"),
    (f"{REMOTE_RUN}/dvac_train/actor_rank01/rollout_step0048.npz", "run/dvac_train/actor_rank01/rollout_step0048.npz"),
    (f"{REMOTE_RUNTIME}/driver.log", "runtime/driver.log"),
    (f"{REMOTE_RUNTIME}/resolved_config.yaml", "runtime/resolved_config.yaml"),
    (f"{REMOTE_RUNTIME}/launch_command.txt", "runtime/launch_command.txt"),
    (f"{REMOTE_RUNTIME}/resource_monitor/resources.csv", "runtime/resource_monitor/resources.csv"),
]

GROWING = {
    "run/metrics.log",
    "run/dvac_train/actor_rank00/runner_step_metrics.csv",
    "run/dvac_train/actor_rank01/runner_step_metrics.csv",
    "runtime/driver.log",
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
            if DEST.exists():
                raise RuntimeError(f"refusing to reuse snapshot directory: {DEST}")
            DEST.mkdir(parents=True)
            total = 0
            for remote, rel in FILES:
                attrs = sftp.stat(remote)
                local = DEST / Path(rel)
                local.parent.mkdir(parents=True, exist_ok=True)
                sftp.get(remote, str(local))
                actual = local.stat().st_size
                if rel in GROWING:
                    if actual < attrs.st_size:
                        raise RuntimeError(f"truncated file {rel}: {actual} < {attrs.st_size}")
                elif actual != attrs.st_size:
                    raise RuntimeError(f"size mismatch for {rel}: {actual} != {attrs.st_size}")
                print(f"GET {rel} bytes={actual}")
                total += actual
            print(f"SNAPSHOT_BYTES={total}")
            print(f"DEST={DEST}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
