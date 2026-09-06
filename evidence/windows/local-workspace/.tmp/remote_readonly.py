import argparse
import os
import sys

import paramiko


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command_file")
    args = parser.parse_args()

    password = [REDACTED]"SEETA_SSH_PASSWORD")
    if not password:
        [REDACTED] SystemExit("SEETA_SSH_PASSWORD is required")

    with open(args.command_file, "r", encoding="utf-8") as handle:
        command = handle.read()

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        "connect.bjb1.seetacloud.com",
        port=36406,
        username="root",
        [REDACTED]=[REDACTED],
        look_for_keys=False,
        allow_agent=False,
        timeout=20,
        banner_timeout=20,
        auth_timeout=20,
    )
    try:
        stdin, stdout, stderr = client.exec_command(command, timeout=300)
        stdin.close()
        sys.stdout.buffer.write(stdout.read())
        sys.stderr.buffer.write(stderr.read())
        raise SystemExit(stdout.channel.recv_exit_status())
    finally:
        client.close()


if __name__ == "__main__":
    main()
