"""Upload the focused Sidney adapter implementation through fixed-host-key SFTP."""

from __future__ import annotations

import argparse
import getpass
import os
from pathlib import Path, PurePosixPath

import remote_exec_autodl


LOCAL_ROOT = Path(
    r"C:\Users\86136\Documents\rl\references\sz_sidney_pi05_adapter_20260903\work"
)
REMOTE_ROOT = PurePosixPath(
    "/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf"
)
FILES = [
    "rlinf/utils/ckpt_convertor/openpi/lerobot_pi05_to_openpi_rlinf.py",
    "rlinf/utils/ckpt_convertor/openpi/convert.py",
    "rlinf/models/embodiment/openpi/dataconfig/robotwin_aloha_dataconfig.py",
    "rlinf/models/embodiment/openpi/dataconfig/__init__.py",
    "examples/embodiment/config/robotwin_move_stapler_pad_grpo_openpi_pi05_sidney.yaml",
    "tests/unit_tests/test_lerobot_pi05_importer.py",
    "toolkits/lerobot/sidney_pi05_parity.py",
]


def ensure_dir(sftp, path: PurePosixPath) -> None:
    current = PurePosixPath("/")
    for part in path.parts[1:]:
        current /= part
        try:
            sftp.stat(str(current))
        except FileNotFoundError:
            sftp.mkdir(str(current))


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
                local = LOCAL_ROOT / relative
                remote = REMOTE_ROOT / relative
                ensure_dir(sftp, remote.parent)
                sftp.put(str(local), str(remote))
                print(f"{relative}: {local.stat().st_size} bytes")
    finally:
        password = ""
        os.environ.pop("SEETA_SSH_PASSWORD", None)
        if client is not None:
            client.close()


if __name__ == "__main__":
    main()
