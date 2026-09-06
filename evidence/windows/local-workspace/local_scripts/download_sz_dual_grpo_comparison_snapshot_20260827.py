"""Download a compact read-only snapshot of the live paired Shenzhen GRPO runs."""

from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path

import remote_exec_autodl


ROOT = Path(r"C:\Users\86136\Documents\rl")
DEST = (
    ROOT
    / "docs"
    / "rlinf-shenzhen-pi0-ppo-rlt"
    / "evidence"
    / "dual-2gpu-comparison-live-20260827"
    / "raw"
)
REMOTE_ROOT = "/data/chenyiteng/results/rlinf-shenzhen/grpo/runs"
RUNS = {
    "control": "grpo-control-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v2",
    "dvac_w0to2": "dvac-global-z-w0to2-formal100-2gpu64x4-b1024-fixed32-eval5-phys67-v2",
}
SMALL_FILES = (
    "runtime/resolved.yaml",
    "runtime/pair_parity.json",
    "runtime/contract.json",
    "runtime/command.txt",
    "runtime/launch_manifest.txt",
    "runtime/resource.csv",
    "runtime/driver.log",
    "runtime/started_at.txt",
    "runtime/finished_at.txt",
    "runtime/exit_code.txt",
    "runtime/stopped_by_user_for_prism.txt",
    "tensorboard/config.yaml",
)


def copy_snapshot(sftp, remote: str, local: Path, size: int) -> None:
    local.parent.mkdir(parents=True, exist_ok=True)
    remaining = size
    with sftp.open(remote, "rb") as source, local.open("wb") as target:
        while remaining:
            chunk = source.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            target.write(chunk)
            remaining -= len(chunk)


def main() -> None:
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
    manifest: list[dict[str, object]] = []
    try:
        client = remote_exec_autodl.connect(conn_args)
        with client.open_sftp() as sftp:
            for label, run_name in RUNS.items():
                remote_run = f"{REMOTE_ROOT}/{run_name}"
                local_run = DEST / label
                for relative in SMALL_FILES:
                    remote = f"{remote_run}/{relative}"
                    try:
                        stat = sftp.stat(remote)
                    except FileNotFoundError:
                        continue
                    local = local_run / relative
                    copy_snapshot(sftp, remote, local, stat.st_size)
                    manifest.append(
                        {
                            "label": label,
                            "remote": remote,
                            "local": str(local.relative_to(ROOT)),
                            "bytes": stat.st_size,
                            "mtime": stat.st_mtime,
                        }
                    )

                tb_dir = f"{remote_run}/tensorboard"
                events = sorted(
                    (
                        item
                        for item in sftp.listdir_attr(tb_dir)
                        if item.filename.startswith("events.out.tfevents.")
                    ),
                    key=lambda item: item.st_mtime,
                )
                if not events:
                    raise RuntimeError(f"no TensorBoard event file under {tb_dir}")
                event = events[-1]
                remote = f"{tb_dir}/{event.filename}"
                local = local_run / "tensorboard" / event.filename
                copy_snapshot(sftp, remote, local, event.st_size)
                manifest.append(
                    {
                        "label": label,
                        "remote": remote,
                        "local": str(local.relative_to(ROOT)),
                        "bytes": event.st_size,
                        "mtime": event.st_mtime,
                    }
                )
    finally:
        password = ""
        os.environ.pop("SEETA_SSH_PASSWORD", None)
        if client is not None:
            client.close()

    DEST.parent.mkdir(parents=True, exist_ok=True)
    (DEST.parent / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"downloaded_files={len(manifest)}")
    print(f"downloaded_bytes={sum(int(row['bytes']) for row in manifest)}")
    print(DEST.parent)


if __name__ == "__main__":
    main()
