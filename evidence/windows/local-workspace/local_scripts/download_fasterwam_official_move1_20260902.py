"""Download the small evidence bundle for the official Faster-WAM one-episode oracle."""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
from pathlib import Path

import remote_exec_autodl


ROOT = Path(r"C:\Users\86136\Documents\rl")
DEST = ROOT / "docs" / "fasterwam-official-standalone" / "evidence" / "official-move1-20260902"
RUN = "/data/chenyiteng/results/fasterwam-standalone/official-move-stapler-pad-random1-20260902-v1"
OFFICIAL = (
    "/data/chenyiteng/projects/fasterwam-standalone/FasterWAM-hustvl-official/"
    "evaluate_results/robotwin/step_029355/official-move-stapler-pad-random1-20260902-v1"
)
FILES = {
    "driver.log": f"{RUN}/driver.log",
    "eval.log": f"{OFFICIAL}/eval_move_stapler_pad_20260902_071828.log",
    "eval_config.yaml": f"{OFFICIAL}/eval_config_move_stapler_pad.yaml",
    "result.txt": f"{OFFICIAL}/move_stapler_pad/_result_random.txt",
    "episode0_success.mp4": (
        f"{OFFICIAL}/move_stapler_pad/"
        "episode0_randomized-true_success-true.mp4"
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", type=Path, default=DEST)
    args = parser.parse_args()
    if args.dest.exists():
        raise FileExistsError(f"refusing to overwrite {args.dest}")

    password = [REDACTED]"SSH password: ")
    os.environ["SEETA_SSH_PASSWORD"] = password
    ssh_args = argparse.Namespace(
        host="120.241.223.9",
        port=22,
        user="chenyiteng",
        host_key_sha256="qw92OXne52y6NsQMQ6+PdjOu0Hosc4ZX78XmkYxcuEY",
        timeout=20.0,
    )
    client = None
    manifest = []
    try:
        client = remote_exec_autodl.connect(ssh_args)
        args.dest.mkdir(parents=True)
        with client.open_sftp() as sftp:
            for name, remote in FILES.items():
                stat = sftp.stat(remote)
                if stat.st_size > 16 * 1024 * 1024:
                    raise RuntimeError(f"unexpectedly large evidence file: {remote}")
                local = args.dest / name
                sftp.get(remote, str(local))
                manifest.append(
                    {
                        "name": name,
                        "remote": remote,
                        "bytes": local.stat().st_size,
                        "sha256": hashlib.sha256(local.read_bytes()).hexdigest(),
                    }
                )
        (args.dest / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"dest": str(args.dest), "files": len(manifest)}, ensure_ascii=False))
    finally:
        password = ""
        os.environ.pop("SEETA_SSH_PASSWORD", None)
        if client is not None:
            client.close()


if __name__ == "__main__":
    main()
