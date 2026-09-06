"""Upload the exact RLT Stage 2 documentation packet to an isolated server staging dir.

The SSH password is read by remote_exec_autodl from SEETA_SSH_PASSWORD and is
never written to disk. This script does not touch the server Git worktree.
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
EVIDENCE_REL = Path(
    "docs/rlinf-robotwin-pi0-rltoken/evidence/stage2_pre_smoke_20260729"
)
TOP_LEVEL_FILES = (
    Path("HANDOFF.md"),
    Path("docs/rlinf-robotwin-pi0-rltoken/00_INDEX_AND_IMPLEMENTATION_PLAN.md"),
    Path(
        "docs/rlinf-robotwin-pi0-rltoken/"
        "01_CONFIG_PROVENANCE_AND_PRE_SMOKE_PACKET.md"
    ),
    Path(
        "docs/rlinf-robotwin-pi0-rltoken/"
        "03_STAGE1_FORMAL_TRAINING_20260729.md"
    ),
    Path(
        "docs/rlinf-robotwin-pi0-rltoken/"
        "04_STAGE2_PRE_SMOKE_PACKET_20260729.md"
    ),
    Path(
        "docs/rlinf-robotwin-pi0-rltoken/evidence/"
        "IMPLEMENTATION_LOG.md"
    ),
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
    parts = path.strip("/").split("/")
    current = ""
    for part in parts:
        current += f"/{part}"
        try:
            sftp.stat(current)
        except FileNotFoundError:
            sftp.mkdir(current)


def collect_files() -> list[Path]:
    evidence = LOCAL_ROOT / EVIDENCE_REL
    files = list(TOP_LEVEL_FILES)
    files.extend(
        path.relative_to(LOCAL_ROOT)
        for path in sorted(evidence.rglob("*"))
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
        default="/root/autodl-tmp/tmp/rlt_stage2_docs_upload_20260729_v2",
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
            f"test ! -e {args.remote_stage!s} && mkdir -p {args.remote_stage!s}",
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
            manifest_path = posixpath.join(
                args.remote_stage, "UPLOAD_SHA256SUMS"
            )
            with sftp.open(manifest_path, "wb") as handle:
                handle.write(manifest)

        verify = execute_checked(
            client,
            "set -e; "
            f"cd {args.remote_stage!s}; "
            "sha256sum -c UPLOAD_SHA256SUMS >/dev/null; "
            "find . -type f -printf '%P\\n' | LC_ALL=C sort",
        )
        expected_count = len(files) + 1
        actual = [line for line in verify.splitlines() if line]
        if len(actual) != expected_count:
            raise RuntimeError(
                f"remote staging file count {len(actual)} != {expected_count}"
            )
        manifest_sha = hashlib.sha256(manifest).hexdigest()
        print(
            "RLT_STAGE2_DOCS_STAGED_OK "
            f"files={len(files)} manifest_sha256={manifest_sha}"
        )
        print(f"REMOTE_STAGE={args.remote_stage}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
