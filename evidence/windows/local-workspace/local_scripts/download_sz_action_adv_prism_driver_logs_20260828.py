"""Fetch only the two current driver logs for the live comparison snapshot."""

from __future__ import annotations

import argparse
import getpass
import os
from pathlib import Path

import remote_exec_autodl


ROOT = Path(r"C:\Users\86136\Documents\rl")
DEST = ROOT / "docs" / "rlinf-shenzhen-grpo-dvac-action-adv" / "evidence" / "action-adv-prism-live-20260828-1424" / "raw"
REMOTE_ROOT = "/data/chenyiteng/results/rlinf-shenzhen/grpo/runs"
RUNS = {
    "action_adv": "dvac-action-adv-w0to2-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v1",
    "prism": "prism-dvac-rank-rloo-formal100-2gpu64x4-b1024-fixed32-eval5-localshard-phys67-v2",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", type=Path, default=DEST)
    cli = parser.parse_args()
    targets = {label: cli.dest / label / "runtime" / "driver.log" for label in RUNS}
    if any(path.exists() for path in targets.values()):
        raise FileExistsError("refusing to overwrite an existing driver log")
    password = [REDACTED]"SSH password: ")
    os.environ["SEETA_SSH_PASSWORD"] = password
    args = argparse.Namespace(host="120.241.223.9", port=22, user="chenyiteng", host_key_sha256="qw92OXne52y6NsQMQ6+PdjOu0Hosc4ZX78XmkYxcuEY", timeout=20.0)
    client = None
    try:
        client = remote_exec_autodl.connect(args)
        with client.open_sftp() as sftp:
            for label, run in RUNS.items():
                remote = f"{REMOTE_ROOT}/{run}/runtime/driver.log"
                stat = sftp.stat(remote)
                target = targets[label]
                target.parent.mkdir(parents=True, exist_ok=True)
                with sftp.open(remote, "rb") as source, target.open("wb") as output:
                    remaining = stat.st_size
                    while remaining:
                        data = source.read(min(1024 * 1024, remaining))
                        if not data:
                            break
                        output.write(data)
                        remaining -= len(data)
                print(f"{label} bytes={stat.st_size}")
    finally:
        password = ""
        os.environ.pop("SEETA_SSH_PASSWORD", None)
        if client is not None:
            client.close()


if __name__ == "__main__":
    main()
