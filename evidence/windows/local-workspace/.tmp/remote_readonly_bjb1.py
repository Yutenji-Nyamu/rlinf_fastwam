import argparse
import os
import sys
import time

import paramiko


HOST = "connect.bjb1.seetacloud.com"
PORT = 36406
USER = "root"


def connect():
    password = [REDACTED]"SEETA_SSH_PASSWORD")
    if not password:
        [REDACTED] SystemExit("SEETA_SSH_PASSWORD is required")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        HOST,
        port=PORT,
        username=USER,
        [REDACTED]=[REDACTED],
        look_for_keys=False,
        allow_agent=False,
        timeout=20,
        auth_timeout=20,
        banner_timeout=20,
    )
    return client


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("command", nargs="?")
    run.add_argument("--command-file")
    get = subparsers.add_parser("get")
    get.add_argument("remote")
    get.add_argument("local")
    args = parser.parse_args()

    client = connect()
    try:
        if args.action == "get":
            os.makedirs(os.path.dirname(os.path.abspath(args.local)), exist_ok=True)
            with client.open_sftp() as sftp:
                sftp.get(args.remote, args.local)
            return

        command = args.command
        if args.command_file:
            with open(args.command_file, "r", encoding="utf-8") as handle:
                command = handle.read()
        if not command:
            raise SystemExit("command or --command-file is required")
        _, stdout, stderr = client.exec_command(command)
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
        client.close()


if __name__ == "__main__":
    main()
