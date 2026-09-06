"""Download a compact live snapshot of the matched-update pi0.5 GRPO pair."""

from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path

import remote_exec_autodl


ROOT = Path(r"C:\Users\86136\Documents\rl")
DEST = ROOT / "docs/rlinf-shenzhen-pi05-robotwin/evidence/pi05-grpo-matched-u2-live-20260902-brief"
REMOTE_ROOT = "/data/chenyiteng/results/rlinf-shenzhen/pi05/runs"
RUNS = {
    "control": "pi05-grpo-control-formal100-2gpu64x4-g8-b1024-u2-m5-fixed32-eval5-phys45-localshard-v2",
    "dvac": "pi05-grpo-dvac-action-adv-w0p5to1p5-formal100-2gpu64x4-g8-b1024-u2-m5-fixed32-eval5-phys67-localshard-v2",
}


def main() -> None:
    if DEST.exists():
        raise FileExistsError(f"refusing to overwrite {DEST}")
    password = [REDACTED]"SSH password: ")
    os.environ["SEETA_SSH_PASSWORD"] = password
    args = argparse.Namespace(
        host="120.241.223.9", port=22, user="chenyiteng",
        host_key_sha256="qw92OXne52y6NsQMQ6+PdjOu0Hosc4ZX78XmkYxcuEY", timeout=20.0,
    )
    client = None
    manifest: list[dict[str, object]] = []
    try:
        client = remote_exec_autodl.connect(args)
        with client.open_sftp() as sftp:
            for label, run in RUNS.items():
                for name in ("driver.log", "resource.csv", "resolved.yaml", "contract.json"):
                    remote = f"{REMOTE_ROOT}/{run}/runtime/{name}"
                    stat = sftp.stat(remote)
                    local = DEST / "raw" / label / "runtime" / name
                    local.parent.mkdir(parents=True, exist_ok=True)
                    with sftp.open(remote, "rb") as source, local.open("wb") as target:
                        target.write(source.read(stat.st_size))
                    manifest.append({"label": label, "run": run, "name": name, "bytes": stat.st_size, "mtime": stat.st_mtime})
        remote_runs = " ".join(f"'{REMOTE_ROOT}/{run}'" for run in RUNS.values())
        command = f"""
date '+now=%F %T %Z'
for run in {remote_runs}; do
  echo RUN=$run
  pid=$(cat "$run/runtime/wrapper.pid")
  kill -0 "$pid" 2>/dev/null && echo wrapper=alive || echo wrapper=dead
  grep -aoE 'Global Step:[[:space:]]+[0-9]+/100' "$run/runtime/driver.log" | tail -n1 || true
  printf fatal_count=; grep -aEc 'Traceback|OutOfMemory|WorkerCrashed|Vulkan.*Error|nonfinite|CUDA error' "$run/runtime/driver.log" || true
  test ! -e "$run/runtime/exit_code.txt" && echo exit_code=running || {{ printf exit_code=; cat "$run/runtime/exit_code.txt"; }}
done
RAY_ADDRESS=172.17.0.1:6389 /home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python - <<'PY'
import os,ray,collections
ray.init(address=os.environ['RAY_ADDRESS'],namespace='codex_pi05_u2_snapshot',logging_level='ERROR')
c=collections.Counter(r.get('namespace') for r in ray.util.list_named_actors(all_namespaces=True))
print('actor_namespaces='+repr({{k:c[k] for k in ('RLinf','RLinf_1')}}))
ray.shutdown()
PY
nvidia-smi -i 4,5,6,7 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
awk '/MemAvailable/ {{print "mem_available_kib=" $2}}' /proc/meminfo
"""
        _stdin, stdout, stderr = client.exec_command(command)
        status = stdout.read().decode() + stderr.read().decode()
        rc = stdout.channel.recv_exit_status()
        if rc != 0:
            raise RuntimeError(f"status command failed: {rc}\n{status}")
        (DEST / "live_status.txt").write_text(status, encoding="utf-8")
    finally:
        password = ""
        os.environ.pop("SEETA_SSH_PASSWORD", None)
        if client is not None:
            client.close()
    (DEST / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"files": len(manifest), "bytes": sum(int(x["bytes"]) for x in manifest)}))


if __name__ == "__main__":
    main()
