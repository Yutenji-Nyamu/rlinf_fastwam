"""Download a compact live snapshot of the current SZ PPO pair."""

from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path

import remote_exec_autodl


ROOT = Path(r"C:\Users\86136\Documents\rl")
DEST = ROOT / "docs/rlinf-shenzhen-experiment-expansion/evidence/ppo-control-dvac-live-20260831"
REMOTE_ROOT = "/data/chenyiteng/results/rlinf-shenzhen/ppo/runs"
RUNS = {
    "control": "ppo-control-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-localshard-v1",
    "dvac": "ppo-dvac-action-adv-fix-w0p5to1p5-formal100-2gpu64x4-b1024-fixed32-eval5-phys67-localshard-v1",
}


def drain(channel) -> tuple[int, str, str]:
    out: list[bytes] = []
    err: list[bytes] = []
    while True:
        if channel.recv_ready():
            out.append(channel.recv(65536))
        if channel.recv_stderr_ready():
            err.append(channel.recv_stderr(65536))
        if channel.exit_status_ready() and not channel.recv_ready() and not channel.recv_stderr_ready():
            break
    return (
        channel.recv_exit_status(),
        b"".join(out).decode("utf-8", errors="replace"),
        b"".join(err).decode("utf-8", errors="replace"),
    )


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
        command = """
set -eu
date -Is
echo '== gpu =='
nvidia-smi --query-gpu=index,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader,nounits
echo '== memory =='
grep -E 'MemTotal|MemAvailable' /proc/meminfo
echo '== wrappers =='
ps -eo pid,user,etimes,args --sort=pid | grep -E 'ppo-control-formal100-2gpu64x4|ppo-dvac-action-adv-fix-w0p5to1p5-formal100' | grep -v grep || true
echo '== ray =='
pgrep -af 'raylet|gcs_server' || true
echo '== checkpoint directories =='
for run in \
  ppo-control-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-localshard-v1 \
  ppo-dvac-action-adv-fix-w0p5to1p5-formal100-2gpu64x4-b1024-fixed32-eval5-phys67-localshard-v1
do
  root='/data/chenyiteng/results/rlinf-shenzhen/ppo/runs/'"$run"
  echo "-- $run"
  find "$root/checkpoints" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | sort -V | tail -n 3 || true
done
"""
        _, stdout, stderr = client.exec_command(command)
        rc, out, err = drain(stdout.channel)
        DEST.mkdir(parents=True, exist_ok=True)
        (DEST / "live_status.txt").write_text(out, encoding="utf-8")
        if stderr:
            (DEST / "live_status.stderr.txt").write_text(err, encoding="utf-8")
        if rc != 0:
            raise RuntimeError(f"remote status failed rc={rc}: {err}")
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
