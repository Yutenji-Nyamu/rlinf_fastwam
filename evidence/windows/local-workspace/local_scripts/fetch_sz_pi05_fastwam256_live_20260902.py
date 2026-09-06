"""Read-only compact live snapshot for SZ pi0.5 Control and Fast-WAM256."""

from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path
import shlex

import remote_exec_autodl


ROOT = Path(r"C:\Users\86136\Documents\rl")
DEFAULT_DEST = (
    ROOT
    / "docs"
    / "rlinf-shenzhen-experiment-expansion"
    / "evidence"
    / "current-pair-live-20260902-2046"
)
RUNS = {
    "pi05_control": (
        "/data/chenyiteng/results/rlinf-shenzhen/pi05/runs/"
        "pi05-grpo-control-formal100-2gpu64x4-g8-b1024-u2-m5-fixed32-eval5-phys45-localshard-v2"
    ),
    "fastwam256": (
        "/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/"
        "fastwam-grpo-control-formal100-2gpu32x8-g8-b2048-u2-m10-fixed32-eval5-phys67-dcp-v2"
    ),
}
REMOTE_PYTHON = "/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python"


def metric_program() -> str:
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
for label, run_text in runs.items():
    run = Path(run_text)
    events = sorted((run / "tensorboard").glob("events.out.tfevents.*"), key=lambda p: p.stat().st_mtime)
    ea = EventAccumulator(str(events[-1]), size_guidance={{"scalars": 0}})
    ea.Reload()
    available = set(ea.Tags().get("scalars", []))
    series = {{}}
    for tag in tags:
        if tag not in available:
            continue
        series[tag] = [
            {{"step": int(v.step) + 1, "value": float(v.value), "wall_time": float(v.wall_time)}}
            for v in ea.Scalars(tag)
        ]
    out[label] = {{
        "run": str(run),
        "event": str(events[-1]),
        "event_bytes": events[-1].stat().st_size,
        "series": series,
    }}
print(json.dumps(out, sort_keys=True))
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
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
    try:
        client = remote_exec_autodl.connect(ssh_args)
        command = f"{REMOTE_PYTHON} -c {shlex.quote(metric_program())}"
        _stdin, stdout, stderr = client.exec_command(command, timeout=90)
        raw = stdout.read().decode("utf-8")
        error = stderr.read().decode("utf-8")
        rc = stdout.channel.recv_exit_status()
        if rc != 0:
            raise RuntimeError(f"metric export failed ({rc}): {error}")
        metrics = json.loads(raw)

        quoted_runs = " ".join(shlex.quote(v) for v in RUNS.values())
        status_cmd = f'''\
date '+now=%F %T %Z'
for run in {quoted_runs}; do
  echo RUN=$run
  pid=$(cat "$run/runtime/wrapper.pid" 2>/dev/null || true)
  test -n "$pid" && kill -0 "$pid" 2>/dev/null && echo wrapper=alive || echo wrapper=dead
  grep -aoE 'Global Step:[[:space:]]+[0-9]+/100' "$run/runtime/driver.log" | tail -n1 || true
  printf fatal_count=; grep -aEc 'Traceback|OutOfMemory|WorkerCrashed|Vulkan.*Error|OIDN Error|Fatal Python error|nonfinite|CUDA error' "$run/runtime/driver.log" || true
  test ! -e "$run/runtime/exit_code.txt" && echo exit_code=running || {{ printf exit_code=; cat "$run/runtime/exit_code.txt"; }}
done
nvidia-smi -i 4,5,6,7 --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader,nounits
awk '/MemAvailable/ {{print "mem_available_kib=" $2}}' /proc/meminfo
'''
        _stdin, stdout, stderr = client.exec_command(status_cmd, timeout=30)
        status_text = stdout.read().decode("utf-8") + stderr.read().decode("utf-8")
        rc = stdout.channel.recv_exit_status()
        if rc != 0:
            raise RuntimeError(f"status export failed ({rc}): {status_text}")

        args.dest.mkdir(parents=True)
        (args.dest / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (args.dest / "live_status.txt").write_text(status_text, encoding="utf-8")
        print(json.dumps({"dest": str(args.dest), "runs": list(metrics)}, ensure_ascii=False))
    finally:
        password = ""
        os.environ.pop("SEETA_SSH_PASSWORD", None)
        if client is not None:
            client.close()


if __name__ == "__main__":
    main()
