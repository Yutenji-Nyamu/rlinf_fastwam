"""Download a bounded, high-information snapshot of RLT resume250-to-480.

The AutoDL password is read only from ``SEETA_SSH_PASSWORD`` by
``remote_exec_autodl.connect`` and is never persisted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import remote_exec_autodl


RUNTIME = (
    "/root/autodl-tmp/experiment_exports/"
    "rlt_stage2_formal_resume250_to480_20260730_v1/runtime"
)
RUN_ROOT = "/root/autodl-tmp/experiments/rlt_stage2_formal_resume250_to480_20260730_v1"
EXPERIMENT = "robotwin_adjust_bottle_rlt_stage2_formal_resume250_to480_v1"
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
        timeout=30.0,
    )

    manifest: list[dict[str, str | int]] = []
    client = remote_exec_autodl.connect(remote_args)
    try:
        with client.open_sftp() as sftp:
            runtime_names = (
                "driver.log",
                "resources.csv",
                "started_at.txt",
                "finished_at.txt",
                "exit_code.txt",
                "exact_command.txt",
                "resolved.yaml",
                "source_config.yaml",
                "source_resolved.yaml",
                "run_provenance.tsv",
                "budget.json",
                "stop_conditions.txt",
                "config_parity.json",
                "source_checkpoint_preflight.json",
                "source_completion.json",
                "eval_seed_bank.json",
                "resources_before.txt",
                "resources_after.txt",
            )
            files = {
                f"{RUNTIME}/{name}": name
                for name in runtime_names
            }
            tensorboard_names = sftp.listdir(f"{RUN_ROOT}/tensorboard")
            event_names = sorted(
                name for name in tensorboard_names if name.startswith("events.out.tfevents.")
            )
            if len(event_names) != 1:
                raise RuntimeError(f"expected one TensorBoard event file, got {event_names}")
            event_name = event_names[0]
            files[f"{RUN_ROOT}/tensorboard/{event_name}"] = event_name
            files[f"{RUN_ROOT}/tensorboard/config.yaml"] = "tensorboard_config.yaml"

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
                base = f"{CHECKPOINT_ROOT}/{checkpoint_name}/actor/sac_components"
                files[
                    f"{base}/rlt_trainer_state/rlt_trainer_state_complete.json"
                ] = f"rlt_completion_step{step}.json"
                for rank in (0, 1):
                    files[
                        f"{base}/replay_buffer/rank_{rank}/metadata.json"
                    ] = f"replay_rank_{rank}_metadata_step{step}.json"

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
