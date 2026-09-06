"""Download compact TensorBoard/resource snapshots for live Control and Prism runs."""

from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path
import shlex

import remote_exec_autodl


ROOT = Path(r"C:\Users\86136\Documents\rl")
DEST = (
    ROOT
    / "docs"
    / "rlinf-shenzhen-prism-dvac-grpo"
    / "evidence"
    / "control-prism-live-20260827"
    / "raw"
)
REMOTE_ROOT = "/data/chenyiteng/results/rlinf-shenzhen/grpo/runs"
RUNS = {
    "control": "grpo-control-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v2",
    "prism": "prism-dvac-rank-rloo-formal100-2gpu64x4-b1024-fixed32-eval5-phys67-v1",
}
SMALL_FILES = (
    "runtime/resolved.yaml",
    "runtime/resource.csv",
    "runtime/started_at.txt",
    "runtime/exit_code.txt",
)


def copy_snapshot(sftp, remote: str, local: Path, size: int) -> None:
    local.parent.mkdir(parents=True, exist_ok=True)
    with sftp.open(remote, "rb") as source, local.open("wb") as target:
        remaining = size
        while remaining:
            chunk = source.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            target.write(chunk)
            remaining -= len(chunk)


def main() -> None:
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
                    manifest.append({"label": label, "remote": remote, "bytes": stat.st_size})

                tb_dir = f"{remote_run}/tensorboard"
                events = sorted(
                    (item for item in sftp.listdir_attr(tb_dir) if item.filename.startswith("events.out.tfevents.")),
                    key=lambda item: item.st_mtime,
                )
                event = events[-1]
                remote = f"{tb_dir}/{event.filename}"
                local = local_run / "tensorboard" / event.filename
                copy_snapshot(sftp, remote, local, event.st_size)
                manifest.append({"label": label, "remote": remote, "bytes": event.st_size})

        extractor = r'''
from pathlib import Path
import json, sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

out = {}
for label, root_s in zip(("control", "prism"), sys.argv[1:]):
    root = Path(root_s)
    event = sorted((root / "tensorboard").glob("events.out.tfevents.*"), key=lambda p: p.stat().st_mtime)[-1]
    ea = EventAccumulator(str(event), size_guidance={"scalars": 0})
    ea.Reload()
    tags = set(ea.Tags().get("scalars", []))
    def values(tag):
        return [
            {"step": x.step + 1, "value": x.value, "wall_time": x.wall_time}
            for x in (ea.Scalars(tag) if tag in tags else [])
        ]
    out[label] = {
        "train_success": values("env/success_once"),
        "eval_success": values("eval/success_once"),
        "time_step": values("time/step"),
    }
print(json.dumps(out, separators=(",", ":")))
'''
        roots = [f"{REMOTE_ROOT}/{name}" for name in RUNS.values()]
        command = " ".join(
            [
                "/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python",
                "-c",
                shlex.quote(extractor),
                *(shlex.quote(root) for root in roots),
            ]
        )
        _stdin, stdout, stderr = client.exec_command(command, timeout=60)
        payload = stdout.read().decode("utf-8")
        error = stderr.read().decode("utf-8", errors="replace")
        if stdout.channel.recv_exit_status() != 0:
            raise RuntimeError(error)
        series = json.loads(payload)
        DEST.parent.mkdir(parents=True, exist_ok=True)
        (DEST.parent / "series.json").write_text(
            json.dumps(series, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    finally:
        password = ""
        os.environ.pop("SEETA_SSH_PASSWORD", None)
        if client is not None:
            client.close()

    DEST.parent.mkdir(parents=True, exist_ok=True)
    (DEST.parent / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"downloaded_files={len(manifest)}")
    print(f"downloaded_bytes={sum(int(row['bytes']) for row in manifest)}")
    print(DEST.parent)


if __name__ == "__main__":
    main()
