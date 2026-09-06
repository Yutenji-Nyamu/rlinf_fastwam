"""Fetch the small exact Fast-WAM 7faa source subset needed for the current port."""

from __future__ import annotations

import argparse
import getpass
import os
from pathlib import Path

import remote_exec_autodl


ROOT = Path(r"C:\Users\86136\Documents\rl")
DEST = ROOT / "references" / "FastWAM-7faa711-subset"
REMOTE = Path("/data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711")
FILES = (
    "src/fastwam/models/wan22/fastwam.py",
    "src/fastwam/models/wan22/mot.py",
    "src/fastwam/models/wan22/action_dit.py",
    "src/fastwam/models/wan22/schedulers/scheduler_continuous.py",
    "src/fastwam/datasets/lerobot/utils/normalizer.py",
    "configs/sim_robotwin.yaml",
    "configs/task/robotwin_uncond_3cam_384_1e-4.yaml",
    "experiments/robotwin/fastwam_policy/deploy_policy.py",
)


def main() -> None:
    password = [REDACTED]"SSH password: ")
    os.environ["SEETA_SSH_PASSWORD"] = password
    args = argparse.Namespace(
        host="120.241.223.9",
        port=22,
        user="chenyiteng",
        host_key_sha256="qw92OXne52y6NsQMQ6+PdjOu0Hosc4ZX78XmkYxcuEY",
        timeout=20.0,
    )
    client = None
    try:
        client = remote_exec_autodl.connect(args)
        with client.open_sftp() as sftp:
            for relative in FILES:
                remote = str(REMOTE / relative).replace("\\", "/")
                local = DEST / relative
                local.parent.mkdir(parents=True, exist_ok=True)
                sftp.get(remote, str(local))
                print(f"{relative}\t{local.stat().st_size}")
    finally:
        password = ""
        os.environ.pop("SEETA_SSH_PASSWORD", None)
        if client is not None:
            client.close()


if __name__ == "__main__":
    main()
