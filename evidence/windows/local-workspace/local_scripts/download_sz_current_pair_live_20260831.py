"""Download the current SZ two-GPU GRPO pair through the fixed host-key route."""

from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path

import remote_exec_autodl


ROOT = Path(r"C:\Users\86136\Documents\rl")
SNAPSHOT = "current-action-mid-st-narrow-live-20260831-1043"
DEST = ROOT / "docs/rlinf-shenzhen-grpo-dvac-action-adv/evidence" / SNAPSHOT
REMOTE_ROOT = "/data/chenyiteng/results/rlinf-shenzhen/grpo/runs"
RUNS = {
    "action_mid": "dvac-action-adv-fix-w0p5to1p5-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v1",
    "st_narrow": "dvac-st-global-z-w0p8to1p2-formal100-2gpu64x4-b1024-fixed32-eval5-phys67-localshard-v2",
}


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
                for name in ("driver.log", "resource.csv"):
                    remote = f"{REMOTE_ROOT}/{run}/runtime/{name}"
                    stat = sftp.stat(remote)
                    local = DEST / "raw" / label / "runtime" / name
                    local.parent.mkdir(parents=True, exist_ok=True)
                    with sftp.open(remote, "rb") as source, local.open("wb") as target:
                        target.write(source.read(stat.st_size))
                    manifest.append(
                        {
                            "label": label,
                            "run": run,
                            "name": name,
                            "bytes": stat.st_size,
                            "mtime": stat.st_mtime,
                        }
                    )
    finally:
        password = ""
        os.environ.pop("SEETA_SSH_PASSWORD", None)
        if client is not None:
            client.close()
    (DEST / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"files": len(manifest), "bytes": sum(int(x["bytes"]) for x in manifest)}))


if __name__ == "__main__":
    main()
