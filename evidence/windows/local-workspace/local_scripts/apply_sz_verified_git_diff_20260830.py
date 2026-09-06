"""Apply one reviewed local worktree diff to the fixed SZ worktree.

The SSH password is prompted and kept only in this process.  The remote base
HEAD and clean-tree checks run before the streamed patch is applied.
"""

from __future__ import annotations

import argparse
import getpass
import os
import subprocess
import tempfile

import remote_exec_autodl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-worktree", required=True)
    parser.add_argument("--remote-worktree", required=True)
    parser.add_argument("--expected-head", required=True)
    args = parser.parse_args()

    local = os.path.abspath(args.local_worktree)
    with tempfile.NamedTemporaryFile(delete=False) as temp_index:
        index_path = temp_index.name
    os.unlink(index_path)
    env = os.environ.copy()
    env["GIT_INDEX_FILE"] = index_path
    git = ["git", "-c", f"safe.directory={local}", "-C", local]
    try:
        subprocess.run([*git, "read-tree", "HEAD"], check=True, env=env)
        subprocess.run([*git, "add", "-A"], check=True, env=env)
        patch = subprocess.run(
            [
                *git,
                "diff",
                "--cached",
                "--binary",
                "--no-ext-diff",
                "--no-color",
                "HEAD",
            ],
            check=True,
            env=env,
            stdout=subprocess.PIPE,
        ).stdout
    finally:
        if os.path.exists(index_path):
            os.unlink(index_path)
    if not patch:
        raise SystemExit("local worktree has no tracked diff")

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
        remote = args.remote_worktree
        expected = args.expected_head
        command = (
            "set -euo pipefail; "
            f'test "$(git -C {remote} rev-parse HEAD)" = "{expected}"; '
            f'test -z "$(git -C {remote} status --porcelain)"; '
            f"git -C {remote} apply -"
        )
        stdin, stdout, stderr = client.exec_command(command)
        stdin.write(patch)
        stdin.channel.shutdown_write()
        out = stdout.read().decode()
        err = stderr.read().decode()
        code = stdout.channel.recv_exit_status()
        if out:
            print(out, end="")
        if err:
            print(err, end="", file=__import__("sys").stderr)
        if code:
            raise SystemExit(code)
    finally:
        password = ""
        os.environ.pop("SEETA_SSH_PASSWORD", None)
        if client is not None:
            client.close()


if __name__ == "__main__":
    main()
