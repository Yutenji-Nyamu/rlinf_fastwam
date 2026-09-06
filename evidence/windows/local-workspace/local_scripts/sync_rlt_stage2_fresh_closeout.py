"""Upload the exact RLT Stage 2 fresh-smoke closeout to server staging.

The SSH password is read only from SEETA_SSH_PASSWORD by the shared helper.
This script creates an isolated staging directory and does not touch Git.
"""

from __future__ import annotations

import argparse
import hashlib
import posixpath
from pathlib import Path
from types import SimpleNamespace

from remote_exec_autodl import (
    DEFAULT_HOST,
    DEFAULT_HOST_KEY_SHA256,
    DEFAULT_PORT,
    DEFAULT_USER,
    connect,
)


LOCAL_ROOT = Path(__file__).resolve().parents[1]
RLT_ROOT = Path("docs/rlinf-robotwin-pi0-rltoken")
FRESH_EVIDENCE = RLT_ROOT / "evidence/stage2_fresh_smoke_20260730"
EXACT_FILES = (
    Path("HANDOFF.md"),
    RLT_ROOT / "00_INDEX_AND_IMPLEMENTATION_PLAN.md",
    RLT_ROOT / "04_STAGE2_PRE_SMOKE_PACKET_20260729.md",
    RLT_ROOT / "05_STAGE2_FRESH_SMOKE_RESULT_20260730.md",
    RLT_ROOT / "06_AUTODL_NETWORK_PLAYBOOK.md",
    RLT_ROOT / "evidence/IMPLEMENTATION_LOG.md",
    RLT_ROOT / "evidence/stage2_pre_smoke_20260729/README.md",
)


def execute_checked(client, command: str) -> str:
    _, stdout, stderr = client.exec_command(command)
    status = stdout.channel.recv_exit_status()
    out = stdout.read().decode("utf-8", errors="strict")
    err = stderr.read().decode("utf-8", errors="strict")
    if status:
        raise RuntimeError(
            f"remote command failed with exit {status}: {command}\n{err}"
        )
    return out


def ensure_remote_dir(sftp, path: str) -> None:
    current = ""
    for part in path.strip("/").split("/"):
        current += f"/{part}"
        try:
            sftp.stat(current)
        except FileNotFoundError:
            sftp.mkdir(current)


def collect_files() -> list[Path]:
    files = list(EXACT_FILES)
    files.extend(
        path.relative_to(LOCAL_ROOT)
        for path in sorted((LOCAL_ROOT / FRESH_EVIDENCE).rglob("*"))
        if path.is_file()
    )
    if len(files) != len(set(files)):
        raise RuntimeError("duplicate upload path")
    for relative in files:
        path = LOCAL_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        path.read_bytes().decode("utf-8", errors="strict")
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--remote-stage",
        default="/root/autodl-tmp/tmp/rlt_stage2_fresh_closeout_20260730_v1",
    )
    args = parser.parse_args()
    files = collect_files()
    connection_args = SimpleNamespace(
        host=DEFAULT_HOST,
        port=DEFAULT_PORT,
        user=DEFAULT_USER,
        host_key_sha256=DEFAULT_HOST_KEY_SHA256,
        timeout=20.0,
    )
    client = connect(connection_args)
    try:
        execute_checked(
            client,
            f"test ! -e {args.remote_stage} && mkdir -p {args.remote_stage}",
        )
        checksums: list[str] = []
        with client.open_sftp() as sftp:
            for relative in files:
                local = LOCAL_ROOT / relative
                remote = posixpath.join(
                    args.remote_stage, relative.as_posix()
                )
                ensure_remote_dir(sftp, posixpath.dirname(remote))
                temporary = f"{remote}.part"
                sftp.put(str(local), temporary)
                sftp.posix_rename(temporary, remote)
                checksums.append(
                    f"{hashlib.sha256(local.read_bytes()).hexdigest()}"
                    f"  {relative.as_posix()}"
                )
            manifest = ("\n".join(checksums) + "\n").encode("utf-8")
            with sftp.open(
                posixpath.join(args.remote_stage, "UPLOAD_SHA256SUMS"), "wb"
            ) as handle:
                handle.write(manifest)
        verify = execute_checked(
            client,
            "set -e; "
            f"cd {args.remote_stage}; "
            "sha256sum -c UPLOAD_SHA256SUMS >/dev/null; "
            "find . -type f -printf '%P\\n' | LC_ALL=C sort",
        )
        actual = [line for line in verify.splitlines() if line]
        if len(actual) != len(files) + 1:
            raise RuntimeError(
                f"remote staging file count {len(actual)} != {len(files) + 1}"
            )
        print(
            "RLT_STAGE2_FRESH_CLOSEOUT_STAGED_OK "
            f"files={len(files)} "
            f"manifest_sha256={hashlib.sha256(manifest).hexdigest()}"
        )
        print(f"REMOTE_STAGE={args.remote_stage}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
