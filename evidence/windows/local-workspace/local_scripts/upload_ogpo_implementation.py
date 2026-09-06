"""Atomically upload the reviewed OGPO implementation into its isolated worktree.

The password is never accepted as an argument or stored here.  Connection and
host-key handling are delegated to ``remote_exec_autodl.connect``.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path, PurePosixPath

from remote_exec_autodl import (
    DEFAULT_HOST,
    DEFAULT_HOST_KEY_SHA256,
    DEFAULT_PORT,
    DEFAULT_USER,
    connect,
)


LOCAL_ROOT = Path(__file__).resolve().parents[1] / ".tmp" / "ogpo_server" / "base"
REMOTE_ROOT = PurePosixPath("/root/autodl-tmp/RLinf_ogpo_pi0_robotwin")
EXPECTED_HEAD = "6d0db56bf26f972cd27fa29535f5eb939e80e5bf"
EXPECTED_BRANCH = "codex/ogpo-pi0-robotwin"

FILES = (
    "examples/embodiment/config/robotwin_adjust_bottle_ogpo_openpi.yaml",
    "examples/embodiment/train_embodied_agent.py",
    "rlinf/algorithms/ogpo/__init__.py",
    "rlinf/algorithms/ogpo/core.py",
    "rlinf/config.py",
    "rlinf/data/ogpo_replay.py",
    "rlinf/models/embodiment/base_policy.py",
    "rlinf/models/embodiment/modules/ogpo_critic.py",
    "rlinf/models/embodiment/modules/ogpo_modules.py",
    "rlinf/models/embodiment/openpi/__init__.py",
    "rlinf/models/embodiment/openpi/openpi_action_model.py",
    "rlinf/models/embodiment/openpi/openpi_ogpo.py",
    "rlinf/runners/embodied_runner.py",
    "rlinf/workers/actor/fsdp_ogpo_policy_worker.py",
    "rlinf/workers/env/env_worker.py",
    "rlinf/workers/rollout/hf/huggingface_worker.py",
    "tests/algorithms/test_ogpo_core.py",
    "tests/data/test_ogpo_replay.py",
    "tests/embodiment/test_ogpo_critic.py",
    "tests/embodiment/test_openpi_ogpo_adapter.py",
)


def _exec_checked(client, command: str) -> str:
    stdin, stdout, stderr = client.exec_command(command)
    stdin.channel.shutdown_write()
    output = stdout.read().decode("utf-8", errors="replace")
    error = stderr.read().decode("utf-8", errors="replace")
    status = stdout.channel.recv_exit_status()
    if status:
        raise RuntimeError(
            f"remote command failed with exit {status}\nstdout:\n{output}\nstderr:\n{error}"
        )
    if error:
        print(error, end="")
    return output


def _mkdirs(sftp, directory: PurePosixPath) -> None:
    current = PurePosixPath("/")
    for part in directory.parts[1:]:
        current /= part
        try:
            sftp.stat(str(current))
        except OSError:
            sftp.mkdir(str(current))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_remote(sftp, path: PurePosixPath) -> str:
    digest = hashlib.sha256()
    with sftp.open(str(path), "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--user", default=DEFAULT_USER)
    parser.add_argument("--host-key-sha256", default=DEFAULT_HOST_KEY_SHA256)
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()

    missing = [name for name in FILES if not (LOCAL_ROOT / name).is_file()]
    if missing:
        raise SystemExit(f"local upload inputs are missing: {missing}")

    client = connect(args)
    try:
        preflight = _exec_checked(
            client,
            "set -euo pipefail\n"
            f"repo={REMOTE_ROOT}\n"
            'test "$(git -C "$repo" rev-parse HEAD)" = '
            f'"{EXPECTED_HEAD}"\n'
            'test "$(git -C "$repo" branch --show-current)" = '
            f'"{EXPECTED_BRANCH}"\n'
            'test -z "$(git -C "$repo" status --porcelain)"\n'
            'printf "PRECHECK head=%s branch=%s clean=yes\\n" '
            '"$(git -C "$repo" rev-parse HEAD)" '
            '"$(git -C "$repo" branch --show-current)"\n',
        )
        print(preflight, end="")

        with client.open_sftp() as sftp:
            for relative_name in FILES:
                local_path = LOCAL_ROOT / relative_name
                remote_path = REMOTE_ROOT / PurePosixPath(relative_name)
                _mkdirs(sftp, remote_path.parent)
                temporary = remote_path.with_name(
                    f".{remote_path.name}.codex-upload-{os.getpid()}"
                )
                try:
                    sftp.put(str(local_path), str(temporary))
                    sftp.posix_rename(str(temporary), str(remote_path))
                except Exception:
                    try:
                        sftp.remove(str(temporary))
                    except OSError:
                        pass
                    raise

                local_hash = _sha256_file(local_path)
                remote_hash = _sha256_remote(sftp, remote_path)
                if remote_hash != local_hash:
                    raise RuntimeError(f"post-upload hash mismatch: {relative_name}")
                print(
                    "SYNC "
                    f"{relative_name} bytes={local_path.stat().st_size} "
                    f"sha256={local_hash} status=match"
                )

        postflight = _exec_checked(
            client,
            "set -euo pipefail\n"
            f"repo={REMOTE_ROOT}\n"
            'git -C "$repo" diff --check\n'
            'git -C "$repo" status --short\n'
            'git -C "$repo" diff --stat\n',
        )
        print(postflight, end="")
    finally:
        client.close()


if __name__ == "__main__":
    main()
