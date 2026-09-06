"""Fetch the exact small source/config subset for the Sidney pi0.5 adapter."""

from __future__ import annotations

import argparse
import getpass
import os
from pathlib import Path

import remote_exec_autodl


ROOT = Path(r"C:\Users\86136\Documents\rl")
DEST = ROOT / "references" / "sz_sidney_pi05_adapter_20260903" / "work"
REMOTE = Path(
    "/data/chenyiteng/projects/rlinf-shenzhen/worktrees/"
    "sidney-pi05-current-rlinf"
)
FILES = (
    "rlinf/utils/ckpt_convertor/openpi/_core.py",
    "rlinf/utils/ckpt_convertor/openpi/convert.py",
    "rlinf/utils/ckpt_convertor/openpi/openpi_pytorch_to_openpi_rlinf.py",
    "rlinf/utils/ckpt_convertor/openpi/README.md",
    "rlinf/models/embodiment/openpi/__init__.py",
    "rlinf/models/embodiment/openpi/dataconfig/__init__.py",
    "rlinf/models/embodiment/openpi/dataconfig/robotwin_aloha_dataconfig.py",
    "examples/embodiment/config/robotwin_adjust_bottle_grpo_openpi_pi05.yaml",
    "examples/embodiment/config/robotwin_adjust_bottle_ppo_openpi_pi05.yaml",
    "toolkits/lerobot/sidney_pi05_parity.py",
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
