"""Download a bounded, high-information snapshot of RLT formal250.

The AutoDL password is read only from SEETA_SSH_PASSWORD by
remote_exec_autodl.connect and is never persisted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import remote_exec_autodl


RUNTIME = (
    "/root/autodl-tmp/experiment_exports/"
    "rlt_stage2_formal_8env_250c_20260730_v1/runtime"
)
RUN_ROOT = "/root/autodl-tmp/experiments/rlt_stage2_formal_8env_250c_20260730_v1"
EXPERIMENT = "robotwin_adjust_bottle_rlt_stage2_formal_8env_250c_v1"
CHECKPOINT_ROOT = f"{RUN_ROOT}/{EXPERIMENT}/checkpoints"


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    args = parser.parse_args()

    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    remote_args = argparse.Namespace(
        host=remote_exec_autodl.DEFAULT_HOST,
        port=remote_exec_autodl.DEFAULT_PORT,
        user=remote_exec_autodl.DEFAULT_USER,
        host_key_sha256=remote_exec_autodl.DEFAULT_HOST_KEY_SHA256,
        timeout=20.0,
    )

    manifest: list[dict[str, str | int]] = []
    client = remote_exec_autodl.connect(remote_args)
    try:
        with client.open_sftp() as sftp:
            files = {
                f"{RUNTIME}/driver.log": "driver.log",
                f"{RUNTIME}/resources.csv": "resources.csv",
                f"{RUNTIME}/started_at.txt": "started_at.txt",
                f"{RUNTIME}/driver_pid.txt": "driver_pid.txt",
                f"{RUNTIME}/monitor_pid.txt": "monitor_pid.txt",
                f"{RUNTIME}/exact_command.txt": "exact_command.txt",
                f"{RUNTIME}/resolved.yaml": "resolved.yaml",
                f"{RUNTIME}/run_provenance.tsv": "run_provenance.tsv",
                f"{RUNTIME}/budget.json": "budget.json",
                f"{RUNTIME}/stop_conditions.txt": "stop_conditions.txt",
                f"{RUN_ROOT}/metrics.log": "metrics.log",
                f"{RUN_ROOT}/tensorboard/config.yaml": "tensorboard_config.yaml",
            }
            tensorboard_names = sftp.listdir(f"{RUN_ROOT}/tensorboard")
            event_names = sorted(
                name for name in tensorboard_names if name.startswith("events.out.tfevents.")
            )
            if len(event_names) != 1:
                raise RuntimeError(f"expected one TensorBoard event file, got {event_names}")
            event_name = event_names[0]
            files[f"{RUN_ROOT}/tensorboard/{event_name}"] = event_name

            checkpoint_names = sorted(
                (
                    name
                    for name in sftp.listdir(CHECKPOINT_ROOT)
                    if name.startswith("global_step_")
                ),
                key=lambda name: int(name.rsplit("_", 1)[1]),
            )
            for checkpoint_name in checkpoint_names:
                step = int(checkpoint_name.rsplit("_", 1)[1])
                base = (
                    f"{CHECKPOINT_ROOT}/{checkpoint_name}/actor/sac_components"
                )
                files[
                    f"{base}/rlt_trainer_state/rlt_trainer_state_complete.json"
                ] = f"rlt_completion_step{step}.json"
                for rank in (0, 1):
                    files[
                        f"{base}/replay_buffer/rank_{rank}/metadata.json"
                    ] = f"replay_rank_{rank}_metadata_step{step}.json"
                    files[
                        f"{base}/rlt_trainer_state/checkpoint_rank_{rank}.pt"
                    ] = f"rlt_state_rank_{rank}_step{step}.pt"

            for remote_path, local_name in files.items():
                local_path = destination / local_name
                sftp.get(remote_path, str(local_path))
                payload = local_path.read_bytes()
                manifest.append(
                    {
                        "remote_path": remote_path,
                        "local_name": local_name,
                        "bytes": len(payload),
                        "sha256": sha256(payload),
                    }
                )
    finally:
        client.close()

    manifest_path = destination / "download_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "files": len(manifest),
                "bytes": sum(int(item["bytes"]) for item in manifest),
                "manifest": str(manifest_path),
                "event_file": event_name,
                "checkpoints": checkpoint_names,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
