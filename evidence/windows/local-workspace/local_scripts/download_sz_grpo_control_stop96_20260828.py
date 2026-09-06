"""Download the small, high-information files from the stopped two-GPU Control."""

from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path

import remote_exec_autodl


ROOT = Path(r"C:\Users\86136\Documents\rl")
DEST = ROOT / "docs" / "rlinf-shenzhen-grpo-dvac-action-adv" / "evidence" / "grpo-control-2gpu-stopped-step96-20260828" / "raw"
REMOTE = "/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/grpo-control-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v2"
FILES = (
    "runtime/resolved.yaml",
    "runtime/pair_parity.json",
    "runtime/contract.json",
    "runtime/command.txt",
    "runtime/launch_manifest.txt",
    "runtime/resource.csv",
    "runtime/driver.log",
    "runtime/started_at.txt",
    "runtime/stopped_by_user_for_action_adv.txt",
    "tensorboard/config.yaml",
)


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
            for relative in FILES:
                remote = f"{REMOTE}/{relative}"
                try:
                    stat = sftp.stat(remote)
                except FileNotFoundError:
                    continue
                local = DEST / relative
                copy_file(sftp, remote, local, stat.st_size)
                manifest.append({"remote": remote, "relative": relative, "bytes": stat.st_size})
            events = sorted(
                (item for item in sftp.listdir_attr(f"{REMOTE}/tensorboard") if item.filename.startswith("events.out.tfevents.")),
                key=lambda item: item.st_mtime,
            )
            if not events:
                raise RuntimeError("Control TensorBoard event is missing")
            event = events[-1]
            relative = f"tensorboard/{event.filename}"
            copy_file(sftp, f"{REMOTE}/{relative}", DEST / relative, event.st_size)
            manifest.append({"remote": f"{REMOTE}/{relative}", "relative": relative, "bytes": event.st_size})
    finally:
        password = ""
        os.environ.pop("SEETA_SSH_PASSWORD", None)
        if client is not None:
            client.close()
    (DEST.parent / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"files": len(manifest), "bytes": sum(int(row["bytes"]) for row in manifest), "dest": str(DEST)}))


if __name__ == "__main__":
    main()
