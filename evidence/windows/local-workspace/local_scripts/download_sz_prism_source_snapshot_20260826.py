"""Download the exact small source set for the SZ Prism implementation.

The password is prompted and remains process-only. Host-key verification and
password authentication reuse remote_exec_autodl.connect.
"""

from __future__ import annotations

import argparse
import getpass
import os
from pathlib import Path

import remote_exec_autodl


FILES = (
    "rlinf/algorithms/__init__.py",
    "rlinf/algorithms/advantages.py",
    "rlinf/algorithms/registry.py",
    "rlinf/algorithms/utils.py",
    "rlinf/algorithms/dvac_train_weighting.py",
    "rlinf/config.py",
    "rlinf/utils/metric_utils.py",
    "rlinf/utils/nested_dict_process.py",
    "rlinf/workers/actor/embodied_fsdp_actor_worker.py",
    "rlinf/workers/rollout/hf/huggingface_worker.py",
    "examples/embodiment/config/robotwin_adjust_bottle_grpo_openpi.yaml",
    "tests/unit_tests/test_dvac_train_weighting.py",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-root", required=True)
    args = parser.parse_args()

    password = [REDACTED]"SSH password: ")
    os.environ["SEETA_SSH_PASSWORD"] = password
    conn_args = argparse.Namespace(
        host="120.241.223.9",
        port=22,
        user="chenyiteng",
        host_key_sha256="qw92OXne52y6NsQMQ6+PdjOu0Hosc4ZX78XmkYxcuEY",
        timeout=15.0,
    )
    remote_root = "/data/chenyiteng/projects/rlinf-shenzhen/worktrees/prism-dvac-rank-rloo"
    local_root = Path(args.local_root)
    client = None
    try:
        client = remote_exec_autodl.connect(conn_args)
        with client.open_sftp() as sftp:
            for relative in FILES:
                local = local_root / relative
                local.parent.mkdir(parents=True, exist_ok=True)
                sftp.get(f"{remote_root}/{relative}", str(local))
                print(relative)
    finally:
        password = ""
        os.environ.pop("SEETA_SSH_PASSWORD", None)
        if client is not None:
            client.close()


if __name__ == "__main__":
    main()
