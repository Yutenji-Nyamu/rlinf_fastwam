from __future__ import annotations

import argparse
import getpass
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "local_scripts"))
import remote_exec_autodl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--command-file", required=True)
    parser.add_argument("--stdout-file", required=True)
    parser.add_argument("--stderr-file", required=True)
    cli = parser.parse_args()
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
        command = Path(cli.command_file).read_text(encoding="utf-8")
        stdin, stdout, stderr = client.exec_command(command)
        stdin.close()
        stdout_data = stdout.read().decode("utf-8")
        stderr_data = stderr.read().decode("utf-8")
        rc = stdout.channel.recv_exit_status()
        Path(cli.stdout_file).write_text(stdout_data, encoding="utf-8")
        Path(cli.stderr_file).write_text(stderr_data, encoding="utf-8")
        print(stdout_data, end="")
        if stderr_data:
            print(stderr_data, end="", file=sys.stderr)
        if rc != 0:
            raise SystemExit(rc)
    finally:
        password = ""
        os.environ.pop("SEETA_SSH_PASSWORD", None)
        if client is not None:
            client.close()


if __name__ == "__main__":
    main()
