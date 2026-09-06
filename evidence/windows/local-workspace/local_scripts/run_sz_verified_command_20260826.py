"""Run one command file on SZ-H100 through the fixed host-key route."""

from __future__ import annotations

import argparse
import getpass
import os
from pathlib import Path

import remote_exec_autodl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--command-file", required=True)
    parser.add_argument("--pty", action="store_true")
    args = parser.parse_args()

    password = [REDACTED]"SSH password: ")
    os.environ["SEETA_SSH_PASSWORD"] = password
    conn_args = argparse.Namespace(
        host="120.241.223.9",
        port=22,
        user="chenyiteng",
        host_key_sha256="qw92OXne52y6NsQMQ6+PdjOu0Hosc4ZX78XmkYxcuEY",
        timeout=20.0,
    )
    client = None
    try:
        client = remote_exec_autodl.connect(conn_args)
        command = Path(args.command_file).read_text(encoding="utf-8")
        stdin, stdout, stderr = client.exec_command(command, get_pty=args.pty)
        stdin.close()
        for line in iter(stdout.readline, ""):
            print(line, end="")
        error = stderr.read().decode()
        if error:
            print(error, end="", file=__import__("sys").stderr)
        raise SystemExit(stdout.channel.recv_exit_status())
    finally:
        password = ""
        os.environ.pop("SEETA_SSH_PASSWORD", None)
        if client is not None:
            client.close()


if __name__ == "__main__":
    main()
