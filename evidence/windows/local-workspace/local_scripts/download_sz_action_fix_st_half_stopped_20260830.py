"""Download small post-stop evidence for Action-Adv Fix [0,2] and ST-DVAC [0.5,1.5]."""

from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path

import remote_exec_autodl


ROOT = Path(r"C:\Users\86136\Documents\rl")
DEFAULT_DEST = (
    ROOT
    / "docs"
    / "rlinf-shenzhen-grpo-dvac-action-adv"
    / "evidence"
    / "action-fix-st-half-stopped-20260830"
)
REMOTE_ROOT = "/data/chenyiteng/results/rlinf-shenzhen/grpo/runs"
RUNS = {
    "action_fix": "dvac-action-adv-fix-w0to2-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v1",
    "st_half": "dvac-st-global-z-w0p5to1p5-formal100-2gpu64x4-b1024-fixed32-eval5-phys67-v1",
}
SMALL_SUFFIXES = {".csv", ".json", ".log", ".txt", ".yaml", ".yml"}
MAX_SMALL_FILE_BYTES = 64 * 1024 * 1024


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    cli = parser.parse_args()
    raw = cli.dest / "raw"
    if cli.dest.exists():
        raise FileExistsError(f"refusing to overwrite {cli.dest}")

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
                remote_run = f"{REMOTE_ROOT}/{run}"
                runtime = f"{remote_run}/runtime"
                selected = []
                for item in sftp.listdir_attr(runtime):
                    suffix = Path(item.filename).suffix.lower()
                    if suffix in SMALL_SUFFIXES and item.st_size <= MAX_SMALL_FILE_BYTES:
                        selected.append((f"runtime/{item.filename}", item.st_size))
                required = {"runtime/driver.log", "runtime/resource.csv", "runtime/resolved.yaml"}
                present = {relative for relative, _ in selected}
                missing = sorted(required - present)
                if missing:
                    raise RuntimeError(f"{label} is missing required evidence: {missing}")

                tensorboard = f"{remote_run}/tensorboard"
                events = sorted(
                    (
                        item
                        for item in sftp.listdir_attr(tensorboard)
                        if item.filename.startswith("events.out.tfevents.")
                    ),
                    key=lambda item: item.st_mtime,
                )
                if not events:
                    raise RuntimeError(f"{label} TensorBoard event is missing")
                event = events[-1]
                if event.st_size > MAX_SMALL_FILE_BYTES:
                    raise RuntimeError(f"{label} TensorBoard event exceeds 64 MiB")
                selected.append((f"tensorboard/{event.filename}", event.st_size))
                try:
                    config = sftp.stat(f"{tensorboard}/config.yaml")
                    if config.st_size <= MAX_SMALL_FILE_BYTES:
                        selected.append(("tensorboard/config.yaml", config.st_size))
                except FileNotFoundError:
                    pass

                for relative, size in sorted(set(selected)):
                    remote = f"{remote_run}/{relative}"
                    local = raw / label / relative
                    copy_file(sftp, remote, local, size)
                    manifest.append(
                        {
                            "label": label,
                            "run": run,
                            "relative": relative,
                            "bytes": size,
                        }
                    )
    finally:
        password = ""
        os.environ.pop("SEETA_SSH_PASSWORD", None)
        if client is not None:
            client.close()

    cli.dest.mkdir(parents=True, exist_ok=True)
    (cli.dest / "download_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "files": len(manifest),
                "bytes": sum(int(row["bytes"]) for row in manifest),
                "dest": str(cli.dest),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()


