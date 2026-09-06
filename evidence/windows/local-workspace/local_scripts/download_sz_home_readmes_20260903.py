from __future__ import annotations

import argparse
import getpass
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
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
    target = Path(__file__).parents[1] / "docs" / "server-admin" / "home-readmes-20260903"
    target.mkdir(parents=True, exist_ok=True)
    client = None
    try:
        client = remote_exec_autodl.connect(args)
        with client.open_sftp() as sftp:
            for name in (
                "readme_to_codex.md",
                "readme_storage_to_codex.md",
                "readme_network_to_codex.md",
            ):
                sftp.get(f"/home/{name}", str(target / name))
                print(target / name)
    finally:
        password = ""
        os.environ.pop("SEETA_SSH_PASSWORD", None)
        if client is not None:
            client.close()


if __name__ == "__main__":
    main()
