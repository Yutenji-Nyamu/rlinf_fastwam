"""Atomically synchronize the first reviewed OGPO correction batch."""

from __future__ import annotations

import argparse
import os
from pathlib import PurePosixPath

from remote_exec_autodl import (
    DEFAULT_HOST,
    DEFAULT_HOST_KEY_SHA256,
    DEFAULT_PORT,
    DEFAULT_USER,
    connect,
)
from upload_ogpo_implementation import (
    EXPECTED_BRANCH,
    EXPECTED_HEAD,
    LOCAL_ROOT,
    REMOTE_ROOT,
    _exec_checked,
    _sha256_file,
    _sha256_remote,
)


UPDATES = {
    "rlinf/models/embodiment/modules/ogpo_modules.py": (
        "810255f8904500b2c461dc8531e7ce17bca55e6e58a36a984472de0cbf9cedbf"
    ),
    "rlinf/workers/actor/fsdp_ogpo_policy_worker.py": (
        "ca746cfe4ec00c1b767393f4a4b2322f8912c0f453b7255a5eaa8b62171215ae"
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--user", default=DEFAULT_USER)
    parser.add_argument("--host-key-sha256", default=DEFAULT_HOST_KEY_SHA256)
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()

    client = connect(args)
    staged: list[tuple[PurePosixPath, PurePosixPath, str]] = []
    try:
        print(
            _exec_checked(
                client,
                "set -euo pipefail\n"
                f"repo={REMOTE_ROOT}\n"
                'test "$(git -C "$repo" rev-parse HEAD)" = '
                f'"{EXPECTED_HEAD}"\n'
                'test "$(git -C "$repo" branch --show-current)" = '
                f'"{EXPECTED_BRANCH}"\n'
                'printf "INCREMENTAL_PRECHECK head=%s branch=%s\\n" '
                '"$(git -C "$repo" rev-parse HEAD)" '
                '"$(git -C "$repo" branch --show-current)"\n',
            ),
            end="",
        )
        with client.open_sftp() as sftp:
            for name, expected_remote_hash in UPDATES.items():
                local = LOCAL_ROOT / name
                remote = REMOTE_ROOT / PurePosixPath(name)
                if expected_remote_hash == "ABSENT":
                    try:
                        sftp.stat(str(remote))
                    except OSError:
                        pass
                    else:
                        raise RuntimeError(f"unexpected existing remote file: {name}")
                else:
                    actual_remote_hash = _sha256_remote(sftp, remote)
                    if actual_remote_hash != expected_remote_hash:
                        raise RuntimeError(
                            f"unexpected remote baseline for {name}: "
                            f"{actual_remote_hash}"
                        )
                temporary = remote.with_name(
                    f".{remote.name}.codex-correction-{os.getpid()}"
                )
                sftp.put(str(local), str(temporary))
                local_hash = _sha256_file(local)
                if _sha256_remote(sftp, temporary) != local_hash:
                    raise RuntimeError(f"temporary upload hash mismatch: {name}")
                staged.append((temporary, remote, local_hash))

            for temporary, remote, local_hash in staged:
                sftp.posix_rename(str(temporary), str(remote))
                if _sha256_remote(sftp, remote) != local_hash:
                    raise RuntimeError(f"post-rename hash mismatch: {remote}")
                print(f"INCREMENTAL_SYNC {remote.relative_to(REMOTE_ROOT)} sha256={local_hash}")
        staged.clear()
        print(
            _exec_checked(
                client,
                "set -euo pipefail\n"
                f"repo={REMOTE_ROOT}\n"
                'git -C "$repo" diff --check\n'
                'git -C "$repo" diff --stat\n',
            ),
            end="",
        )
    finally:
        if staged:
            try:
                with client.open_sftp() as sftp:
                    for temporary, _, _ in staged:
                        try:
                            sftp.remove(str(temporary))
                        except OSError:
                            pass
            except Exception:
                pass
        client.close()


if __name__ == "__main__":
    main()
