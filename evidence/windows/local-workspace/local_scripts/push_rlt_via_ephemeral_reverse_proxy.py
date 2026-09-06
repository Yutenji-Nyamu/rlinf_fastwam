"""Push the verified server RLT branch through a one-shot loopback proxy.

The AutoDL SSH host key and password handling are inherited from
remote_exec_autodl.py.  The reverse listener exists only for this SSH
transport, binds server loopback, and accepts CONNECT github.com:443 only.
GitHub TLS and Git credentials remain end-to-end inside server Git.
"""

from __future__ import annotations

import argparse
import select
import socket
import sys
import threading
import time

import paramiko

from remote_exec_autodl import (
    DEFAULT_HOST,
    DEFAULT_HOST_KEY_SHA256,
    DEFAULT_PORT,
    DEFAULT_USER,
    connect,
)


REPO = "/root/autodl-tmp/RLinf_rlt_pi0_robotwin"
BRANCH = "codex/rlt-pi0-robotwin"
EXPECTED_HEAD = "2b8199d8ab2e7b110994fd3234bf7007196c3af9"
EXPECTED_REMOTE = "9bb2dd78feff7133780c3df6a88618d10168c4e4"
EXPECTED_AHEAD = 4
ALLOWED_CONNECT = "github.com:443"


def _relay_proxy(channel: paramiko.Channel) -> None:
    upstream: socket.socket | None = None
    try:
        channel.settimeout(10.0)
        request = bytearray()
        while b"\r\n\r\n" not in request:
            chunk = channel.recv(4096)
            if not chunk:
                raise ConnectionError("proxy client closed before headers")
            request.extend(chunk)
            if len(request) > 16384:
                raise ValueError("proxy request headers too large")

        header, remainder = bytes(request).split(b"\r\n\r\n", 1)
        first_line = header.split(b"\r\n", 1)[0].decode("ascii", "strict")
        method, authority, _version = first_line.split(" ", 2)
        if method != "CONNECT" or authority.lower() != ALLOWED_CONNECT:
            channel.sendall(b"HTTP/1.1 403 Forbidden\r\n\r\n")
            return

        upstream = socket.create_connection(("github.com", 443), timeout=10.0)
        channel.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
        if remainder:
            upstream.sendall(remainder)

        channel.settimeout(None)
        upstream.settimeout(None)
        peers = (channel, upstream)
        while True:
            readable, _, _ = select.select(peers, [], [], 1.0)
            if not readable:
                if channel.closed:
                    break
                continue
            for source in readable:
                data = source.recv(65536)
                if not data:
                    return
                destination = upstream if source is channel else channel
                destination.sendall(data)
    except Exception as exc:
        print(
            f"PROXY_CONNECTION_FAILED={type(exc).__name__}",
            file=sys.stderr,
            flush=True,
        )
    finally:
        if upstream is not None:
            upstream.close()
        channel.close()


def _proxy_handler(
    channel: paramiko.Channel,
    _origin: tuple[str, int],
    _server: tuple[str, int],
) -> None:
    threading.Thread(
        target=_relay_proxy,
        args=(channel,),
        daemon=True,
        name="rlt-github-connect-relay",
    ).start()


def _stream_command(client: paramiko.SSHClient, command: str) -> int:
    stdin, stdout, _stderr = client.exec_command(command, get_pty=False)
    stdin.channel.shutdown_write()
    channel = stdout.channel
    deadline = time.monotonic() + 55.0
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
            return channel.recv_exit_status()
        if time.monotonic() > deadline:
            channel.close()
            raise TimeoutError("remote push wrapper exceeded 55 seconds")
        if not wrote:
            time.sleep(0.05)


def main() -> None:
    args = argparse.Namespace(
        host=DEFAULT_HOST,
        port=DEFAULT_PORT,
        user=DEFAULT_USER,
        host_key_sha256=DEFAULT_HOST_KEY_SHA256,
        timeout=20.0,
    )
    client = connect(args)
    transport = client.get_transport()
    if transport is None:
        client.close()
        raise SystemExit("SSH transport unavailable")

    remote_port: int | None = None
    try:
        remote_port = transport.request_port_forward(
            "127.0.0.1",
            0,
            handler=_proxy_handler,
        )
        print(f"EPHEMERAL_REMOTE_LOOPBACK_PORT={remote_port}", flush=True)
        command = f"""set -euo pipefail
repo={REPO}
branch={BRANCH}
expected_head={EXPECTED_HEAD}
expected_remote={EXPECTED_REMOTE}
expected_ahead={EXPECTED_AHEAD}
cd "$repo"
test "$(git branch --show-current)" = "$branch"
test "$(git rev-parse HEAD)" = "$expected_head"
test -z "$(git status --porcelain)"
test "$(git rev-parse personal/$branch)" = "$expected_remote"
test "$(git rev-list --left-right --count personal/$branch...HEAD)" = $'0\\t'"$expected_ahead"
env -u no_proxy -u NO_PROXY GIT_TERMINAL_PROMPT=0 \
  timeout 45 git -c http.proxy=http://127.0.0.1:{remote_port} \
  push personal "refs/heads/$branch:refs/heads/$branch"
test "$(git rev-parse personal/$branch)" = "$expected_head"
test "$(git rev-list --left-right --count personal/$branch...HEAD)" = $'0\\t0'
test -z "$(git status --porcelain)"
printf 'PUSH_OK_HEAD=%s\\n' "$expected_head"
"""
        rc = _stream_command(client, command)
        print(f"REMOTE_PUSH_RC={rc}", flush=True)
        raise SystemExit(rc)
    finally:
        if remote_port is not None:
            try:
                transport.cancel_port_forward("127.0.0.1", remote_port)
            except Exception:
                pass
        client.close()


if __name__ == "__main__":
    main()
