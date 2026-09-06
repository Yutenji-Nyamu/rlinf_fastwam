"""Download the two live SZ-H100 two-GPU GRPO driver logs for comparison plots."""

from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path

import remote_exec_autodl


ROOT = Path(r"C:\Users\86136\Documents\rl")
SNAPSHOT = os.environ.get(
    "SZ_GRPO_SNAPSHOT", "six-2gpu-grpo-comparison-live-20260829"
)
DEST = (
    ROOT
    / "docs"
    / "rlinf-shenzhen-grpo-dvac-action-adv"
    / "evidence"
    / SNAPSHOT
    / "raw"
)
REMOTE_ROOT = "/data/chenyiteng/results/rlinf-shenzhen/grpo/runs"
RUNS = {
    "action_adv_fix": "dvac-action-adv-fix-w0to2-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v1",
    "st_half": "dvac-st-global-z-w0p5to1p5-formal100-2gpu64x4-b1024-fixed32-eval5-phys67-v1",
}


def copy_file(sftp, remote: str, local: Path, size: int) -> None:
    local.parent.mkdir(parents=True, exist_ok=True)
    with sftp.open(remote, "rb") as source, local.open("wb") as target:
        remaining = size
        while remaining:
            data = source.read(min(1024 * 1024, remaining))
            if not data:
                break
            target.write(data)
            remaining -= len(data)


def main() -> None:
    if DEST.exists():
        raise FileExistsError(f"refusing to overwrite {DEST}")
    password = [REDACTED]"SSH password: ")
    os.environ["SEETA_SSH_PASSWORD"] = password
    args = argparse.Namespace(
        host="120.241.223.9",
        port=22,
        user="chenyiteng",
        host_key_sha256="qw92OXne52y6NsQMQ6+PdjOu0Hosc4ZX78XmkYxcuEY",
        timeout=20.0,
    )
    client = None
    manifest: list[dict[str, object]] = []
    try:
        client = remote_exec_autodl.connect(args)
        with client.open_sftp() as sftp:
            for label, run in RUNS.items():
                remote = f"{REMOTE_ROOT}/{run}/runtime/driver.log"
                stat = sftp.stat(remote)
                local = DEST / label / "runtime" / "driver.log"
                copy_file(sftp, remote, local, stat.st_size)
                manifest.append(
                    {
                        "label": label,
                        "run": run,
                        "remote": remote,
                        "bytes": stat.st_size,
                        "mtime": stat.st_mtime,
                    }
                )
    finally:
        password = ""
        os.environ.pop("SEETA_SSH_PASSWORD", None)
        if client is not None:
            client.close()
    DEST.parent.mkdir(parents=True, exist_ok=True)
    (DEST.parent / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "files": len(manifest),
                "bytes": sum(int(row["bytes"]) for row in manifest),
                "dest": str(DEST),
            }
        )
    )


if __name__ == "__main__":
    main()
