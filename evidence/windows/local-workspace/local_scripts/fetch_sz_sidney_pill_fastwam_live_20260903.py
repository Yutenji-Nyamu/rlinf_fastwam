"""Fetch compact live TensorBoard and runtime evidence for current Sidney/Fast-WAM runs."""

from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path
import shlex

import remote_exec_autodl


ROOT = Path(r"C:\Users\86136\Documents\rl")
DEFAULT_DEST = ROOT / "docs" / "rlinf-shenzhen-experiment-expansion" / "evidence" / "current-pair-live-20260903-2035"
RUNS = {
    "sidney_pill": [
        "/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/runs/"
        "move-pillbottle-pad-grpo-formal100-2gpu64x4-g8-b1024-u2-m10-noise0p5-h200-fixed32-eval5-phys45-localshard-v1"
    ],
    "fastwam": [
        "/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/"
        "fastwam-grpo-control-formal100-2gpu32x4-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-v1",
        "/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/"
        "fastwam-grpo-control-resume10-to100-2gpu32x4-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-renderlife-v2",
    ],
}
ACTIVE = {key: values[-1] for key, values in RUNS.items()}
REMOTE_PYTHON = "/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python"


def remote_program() -> str:
    return f'''\
import json
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

runs = {RUNS!r}
tags = [
    "env/success_once", "eval/success_once",
    "train/actor/approx_kl", "train/actor/clip_fraction",
    "train/actor/grad_norm", "time/step",
]
out = {{}}
for label, run_list in runs.items():
    merged = {{tag: {{}} for tag in tags}}
    used = []
    for run_text in run_list:
        run = Path(run_text)
        events = sorted((run / "tensorboard").glob("events.out.tfevents.*"), key=lambda p: p.stat().st_mtime)
        if not events:
            continue
        event = events[-1]
        used.append({{"run": str(run), "event": str(event), "bytes": event.stat().st_size}})
        ea = EventAccumulator(str(event), size_guidance={{"scalars": 0}})
        ea.Reload()
        available = set(ea.Tags().get("scalars", []))
        for tag in tags:
            if tag not in available:
                continue
            for value in ea.Scalars(tag):
                merged[tag][int(value.step) + 1] = {{
                    "step": int(value.step) + 1,
                    "value": float(value.value),
                    "wall_time": float(value.wall_time),
                }}
    out[label] = {{
        "sources": used,
        "series": {{tag: [values[k] for k in sorted(values)] for tag, values in merged.items() if values}},
    }}
print(json.dumps(out, sort_keys=True))
'''


def download_small(sftp, remote: str, local: Path) -> None:
    stat = sftp.stat(remote)
    if stat.st_size > 16 * 1024 * 1024:
        raise RuntimeError(f"refusing large live file: {remote} ({stat.st_size})")
    local.parent.mkdir(parents=True, exist_ok=True)
    sftp.get(remote, str(local))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    args = parser.parse_args()
    if args.dest.exists():
        raise FileExistsError(args.dest)
    password = [REDACTED]"SSH password: ")
    os.environ["SEETA_SSH_PASSWORD"] = password
    ssh_args = argparse.Namespace(
        host="120.241.223.9", port=22, user="chenyiteng",
        host_key_sha256="qw92OXne52y6NsQMQ6+PdjOu0Hosc4ZX78XmkYxcuEY", timeout=20.0,
    )
    client = None
    try:
        client = remote_exec_autodl.connect(ssh_args)
        command = f"{REMOTE_PYTHON} -c {shlex.quote(remote_program())}"
        _stdin, stdout, stderr = client.exec_command(command, timeout=90)
        raw, error = stdout.read().decode(), stderr.read().decode()
        status = stdout.channel.recv_exit_status()
        if status:
            raise RuntimeError(f"metric export failed ({status}): {error}")
        metrics = json.loads(raw)
        args.dest.mkdir(parents=True)
        (args.dest / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        with client.open_sftp() as sftp:
            for label, run in ACTIVE.items():
                for name in ("driver.log", "resource.csv", "resolved.yaml", "launch_manifest.txt", "exit_code.txt"):
                    remote = f"{run}/runtime/{name}"
                    try:
                        download_small(sftp, remote, args.dest / "raw" / label / name)
                    except FileNotFoundError:
                        pass
        print(json.dumps({"dest": str(args.dest), "series": {k: {t: len(v) for t, v in x["series"].items()} for k, x in metrics.items()}}, ensure_ascii=False))
    finally:
        password = ""
        os.environ.pop("SEETA_SSH_PASSWORD", None)
        if client is not None:
            client.close()


if __name__ == "__main__":
    main()
