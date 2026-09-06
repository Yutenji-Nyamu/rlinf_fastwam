"""Run a fixed-host SSH command that changes one account password via sudo.

Both the SSH/sudo password and the replacement password are prompted without
echo, passed only through the remote process stdin, and never written to disk.
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
import time

import remote_exec_autodl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--user", required=True)
    parser.add_argument("--host-key-sha256", required=True)
    parser.add_argument("--target-user", required=True)
    parser.add_argument("--command-file", required=True)
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()

    ssh_password = [REDACTED]"SSH/sudo password: ")
    new_password = [REDACTED]"New account password: ")
    os.environ["SEETA_SSH_PASSWORD"] = ssh_password
    client = None
    try:
        client = remote_exec_autodl.connect(args)
        with open(args.command_file, encoding="utf-8") as handle:
            command = handle.read()
        stdin, stdout, stderr = client.exec_command(command)
        stdin.write(ssh_password + "\n")
        stdin.write(f"{args.target_user}:{new_password}\n")
        stdin.flush()
        stdin.channel.shutdown_write()

        channel = stdout.channel
        while True:
            wrote = False
            if channel.recv_ready():
                sys.stdout.buffer.write(channel.recv(65536))
                sys.stdout.buffer.flush()
                wrote = True
            if channel.recv_stderr_ready():
                sys.stderr.buffer.write(channel.recv_stderr(65536))
                sys.stderr.buffer.flush()
                wrote = True
            if channel.exit_status_ready() and not channel.recv_ready() and not channel.recv_stderr_ready():
                break
            if not wrote:
                time.sleep(0.05)
        raise SystemExit(channel.recv_exit_status())
    finally:
        ssh_password = ""
        new_password = ""
        os.environ.pop("SEETA_SSH_PASSWORD", None)
        if client is not None:
            client.close()


if __name__ == "__main__":
    main()
