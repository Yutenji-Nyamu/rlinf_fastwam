"""Download the bounded, high-information RLT formal status evidence.

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
    "rlt_stage2_formal_100c_20260730_v1/runtime"
)
RUN_ROOT = "/root/autodl-tmp/experiments/rlt_stage2_formal_100c_20260730_v1"
CHECKPOINT_ROOT = (
    f"{RUN_ROOT}/robotwin_adjust_bottle_rlt_stage2_formal_100c_v1/checkpoints"
)
EVENT_NAME = (
    "events.out.tfevents.1785345371."
    "autodl-container-nekaqbwt43-6ce5babb.744889.0"
)


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

    files = {
        f"{RUNTIME}/driver.log": "driver.log",
        f"{RUNTIME}/resources.csv": "resources.csv",
        f"{RUNTIME}/finished_at.txt": "finished_at.txt",
        f"{RUNTIME}/exit_code.txt": "exit_code.txt",
        f"{RUN_ROOT}/metrics.log": "metrics.log",
        f"{RUN_ROOT}/tensorboard/{EVENT_NAME}": EVENT_NAME,
        f"{RUN_ROOT}/tensorboard/config.yaml": "tensorboard_config.yaml",
        (
            f"{CHECKPOINT_ROOT}/global_step_100/actor/sac_components/"
            "replay_buffer/rank_0/metadata.json"
        ): "replay_rank_0_metadata_step100.json",
        (
            f"{CHECKPOINT_ROOT}/global_step_100/actor/sac_components/"
            "replay_buffer/rank_1/metadata.json"
        ): "replay_rank_1_metadata_step100.json",
        (
            f"{CHECKPOINT_ROOT}/global_step_100/actor/sac_components/"
            "rlt_trainer_state/checkpoint_rank_0.pt"
        ): "rlt_state_rank_0_step100.pt",
        (
            f"{CHECKPOINT_ROOT}/global_step_100/actor/sac_components/"
            "rlt_trainer_state/checkpoint_rank_1.pt"
        ): "rlt_state_rank_1_step100.pt",
    }
    for step in range(10, 101, 10):
        files[
            (
                f"{CHECKPOINT_ROOT}/global_step_{step}/actor/sac_components/"
                "rlt_trainer_state/rlt_trainer_state_complete.json"
            )
        ] = f"rlt_completion_step{step}.json"

    manifest: list[dict[str, str | int]] = []
    client = remote_exec_autodl.connect(remote_args)
    try:
        with client.open_sftp() as sftp:
            for remote_path, local_name in files.items():
                local_path = destination / local_name
                sftp.get(remote_path, str(local_path))
                payload = local_path.read_bytes()
                manifest.append(
                    {
                        "remote_path": remote_path,
                        "local_name": local_name,
                        "bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                )
    finally:
        client.close()

    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
