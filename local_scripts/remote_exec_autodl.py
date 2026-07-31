"""Run commands or transfer files over the verified AutoDL SSH endpoint.

The password is read only from SEETA_SSH_PASSWORD in the current process.
This helper intentionally contains no credential and performs an explicit
SHA256 host-key check before password authentication.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import os
import socket
import subprocess
import sys
import tempfile
import time

try:
    import paramiko
except ModuleNotFoundError as exc:
    if exc.name != "paramiko":
        raise
    paramiko_target = os.environ.get(
        "CODEX_AUTODL_SSH_DEPENDENCIES",
        r"E:\Codex\home\tools\autodl-ssh-py312",
    )
    if not os.path.isdir(paramiko_target):
        raise
    sys.path.insert(0, paramiko_target)
    import paramiko


DEFAULT_HOST = "connect.bjb1.seetacloud.com"
DEFAULT_PORT = 36406
DEFAULT_USER = "root"
DEFAULT_HOST_KEY_SHA256 = "liZ36vNCsNcNdXeWs4f+g5ZIhPM/ZihP834vxs8Ulqc"
CONNECT_ATTEMPTS = 3
CONNECT_BACKOFF_SECONDS = (1.0, 3.0)
KEEPALIVE_SECONDS = 30


def connect(args: argparse.Namespace) -> paramiko.SSHClient:
    password = os.environ.get("SEETA_SSH_PASSWORD")
    if not password:
        raise SystemExit("SEETA_SSH_PASSWORD is required in the current process")

    transport = None
    retryable_errors = (
        EOFError,
        socket.timeout,
        TimeoutError,
        ConnectionAbortedError,
        ConnectionResetError,
        BrokenPipeError,
        OSError,
        paramiko.SSHException,
    )
    for attempt in range(1, CONNECT_ATTEMPTS + 1):
        try:
            transport = paramiko.Transport((args.host, args.port))
            transport.start_client(timeout=args.timeout)
            server_key = transport.get_remote_server_key()
            fingerprint = (
                base64.b64encode(hashlib.sha256(server_key.asbytes()).digest())
                .decode("ascii")
                .rstrip("=")
            )
            if fingerprint != args.host_key_sha256:
                transport.close()
                raise SystemExit(
                    "SSH host-key mismatch: "
                    f"expected SHA256:{args.host_key_sha256}, "
                    f"got SHA256:{fingerprint}"
                )
            break
        except retryable_errors as exc:
            if transport is not None:
                transport.close()
                transport = None
            if attempt == CONNECT_ATTEMPTS:
                raise
            delay = CONNECT_BACKOFF_SECONDS[attempt - 1]
            print(
                "SSH pre-auth handshake failed "
                f"(attempt {attempt}/{CONNECT_ATTEMPTS}: "
                f"{type(exc).__name__}); retrying in {delay:g}s",
                file=sys.stderr,
            )
            time.sleep(delay)

    assert transport is not None
    try:
        transport.auth_password(args.user, password)
    except Exception:
        transport.close()
        raise
    transport.set_keepalive(KEEPALIVE_SECONDS)
    client = paramiko.SSHClient()
    client._transport = transport
    return client


def run_command(client: paramiko.SSHClient, args: argparse.Namespace) -> int:
    command = args.command
    if args.command_file:
        with open(args.command_file, encoding="utf-8") as handle:
            command = handle.read()
    if not command:
        raise SystemExit("command or --command-file is required")

    stdin, stdout, stderr = client.exec_command(command, get_pty=args.pty)
    stdin_modes = [
        bool(args.stdin_file),
        bool(args.stdin_git_diff),
        bool(args.stdin_git_full_diff),
    ]
    if sum(stdin_modes) > 1:
        raise SystemExit(
            "--stdin-file, --stdin-git-diff, and --stdin-git-full-diff "
            "are mutually exclusive"
        )
    if args.stdin_file:
        with open(args.stdin_file, "rb") as handle:
            while chunk := handle.read(1024 * 1024):
                stdin.write(chunk)
    elif args.stdin_git_diff:
        result = subprocess.run(
            [
                "git",
                "-C",
                args.stdin_git_diff,
                "diff",
                "--binary",
                "--no-ext-diff",
            ],
            check=True,
            stdout=subprocess.PIPE,
        )
        stdin.write(result.stdout)
    elif args.stdin_git_full_diff:
        repo = os.path.abspath(args.stdin_git_full_diff)
        with tempfile.NamedTemporaryFile(delete=False) as temp_index:
            index_path = temp_index.name
        os.unlink(index_path)
        env = os.environ.copy()
        env["GIT_INDEX_FILE"] = index_path
        try:
            subprocess.run(
                ["git", "-C", repo, "read-tree", "HEAD"],
                check=True,
                env=env,
            )
            subprocess.run(
                ["git", "-C", repo, "add", "-A"],
                check=True,
                env=env,
            )
            result = subprocess.run(
                [
                    "git",
                    "-C",
                    repo,
                    "diff",
                    "--cached",
                    "--binary",
                    "--no-ext-diff",
                    "HEAD",
                ],
                check=True,
                env=env,
                stdout=subprocess.PIPE,
            )
            stdin.write(result.stdout)
        finally:
            if os.path.exists(index_path):
                os.unlink(index_path)
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
        if (
            channel.exit_status_ready()
            and not channel.recv_ready()
            and not channel.recv_stderr_ready()
        ):
            break
        if not wrote:
            time.sleep(0.05)
    return channel.recv_exit_status()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--user", default=DEFAULT_USER)
    parser.add_argument("--host-key-sha256", default=DEFAULT_HOST_KEY_SHA256)
    parser.add_argument("--timeout", type=float, default=20.0)

    subparsers = parser.add_subparsers(dest="action", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("command", nargs="?")
    run.add_argument("--command-file")
    run.add_argument("--stdin-file")
    run.add_argument(
        "--stdin-git-diff",
        help="Stream the unstaged Git diff from this local worktree to stdin",
    )
    run.add_argument(
        "--stdin-git-full-diff",
        help=(
            "Stream tracked changes and untracked files through a temporary "
            "alternate Git index without changing the worktree index"
        ),
    )
    run.add_argument("--pty", action="store_true")

    put = subparsers.add_parser("put")
    put.add_argument("local")
    put.add_argument("remote")

    get = subparsers.add_parser("get")
    get.add_argument("remote")
    get.add_argument("local")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    client = connect(args)
    try:
        if args.action == "run":
            raise SystemExit(run_command(client, args))
        with client.open_sftp() as sftp:
            if args.action == "put":
                sftp.put(args.local, args.remote)
            else:
                local_dir = os.path.dirname(os.path.abspath(args.local))
                os.makedirs(local_dir, exist_ok=True)
                sftp.get(args.remote, args.local)
    finally:
        client.close()


if __name__ == "__main__":
    main()
