"""Verify password-backed sudo privileges over the pinned Paramiko route.

The SSH/sudo password is entered once through a no-echo prompt and remains only
in this process. No credential is accepted on the command line or written out.
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import remote_exec_autodl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--user", required=True)
    parser.add_argument("--host-key-sha256", required=True)
    parser.add_argument("--timeout", type=float, default=25.0)
    args = parser.parse_args()

    password = [REDACTED]"SSH/sudo password: ")
    os.environ["SEETA_SSH_PASSWORD"] = password
    client = remote_exec_autodl.connect(args)
    try:
        stdin, stdout, stderr = client.exec_command(
            "env PAGER=cat SUDO_PAGER=cat sudo -S -k -p '' -l",
            get_pty=False,
        )
        stdin.write(password + "\n")
        stdin.flush()
        stdin.channel.shutdown_write()

        channel = stdout.channel
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            wrote = False
            if channel.recv_ready():
                sys.stdout.buffer.write(channel.recv(65536))
                sys.stdout.buffer.flush()
                wrote = True
            if channel.recv_stderr_ready():
                sys.stderr.buffer.write(channel.recv_stderr(65536))
                sys.stderr.buffer.flush()
                wrote = True
            if (
                channel.exit_status_ready()
                and not channel.recv_ready()
                and not channel.recv_stderr_ready()
            ):
                return channel.recv_exit_status()
            if not wrote:
                time.sleep(0.05)
        channel.close()
        print("sudo privilege probe timed out", file=sys.stderr)
        return 124
    finally:
        password = ""
        os.environ.pop("SEETA_SSH_PASSWORD", None)
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
