from __future__ import annotations

import argparse
import getpass
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "local_scripts"))
import remote_exec_autodl


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
        command = Path("tmp_checkpoint_file_targets.sh").read_text(encoding="utf-8")
        stdin, stdout, stderr = client.exec_command(command)
        stdin.close()
        data = stdout.read().decode("utf-8")
        error = stderr.read().decode("utf-8")
        rc = stdout.channel.recv_exit_status()
        if rc != 0:
            raise RuntimeError(f"remote rc={rc}: {error}")
        Path("tmp_grpo_ppo_checkpoint_target_inventory.tsv").write_text(data, encoding="utf-8")
    finally:
        password = ""
        os.environ.pop("SEETA_SSH_PASSWORD", None)
        if client is not None:
            client.close()


if __name__ == "__main__":
    main()
