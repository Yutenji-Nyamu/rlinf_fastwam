from __future__ import annotations

import gzip
import importlib.util
import os
import shlex
import shutil
from pathlib import Path
from types import SimpleNamespace


workspace = Path(r"C:\Users\86136\Documents\rl")
helper_path = workspace / "local_scripts" / "remote_exec_autodl.py"
spec = importlib.util.spec_from_file_location("remote_exec_autodl", helper_path)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load {helper_path}")
remote = importlib.util.module_from_spec(spec)
spec.loader.exec_module(remote)

args = SimpleNamespace(
    host=remote.DEFAULT_HOST,
    port=remote.DEFAULT_PORT,
    user=remote.DEFAULT_USER,
    host_key_sha256=remote.DEFAULT_HOST_KEY_SHA256,
    timeout=20.0,
)
run = (
    "/root/autodl-tmp/RLinf_fastwam_rlinf/logs/"
    "20260728_dsrl_pi0_robotwin_n20_formal_v1"
)
downloads = {
    f"{run}/tensorboard/events.out.tfevents.1785236070.autodl-container-nekaqbwt43-6ce5babb.70062.0":
        workspace / ".tmp" / "dsrl_step198_events.tfevents",
    f"{run}/resource_monitor/resources.csv":
        workspace / ".tmp" / "dsrl_step198_resources.csv",
    f"{run}/resource_monitor/cgroup_detail.csv":
        workspace / ".tmp" / "dsrl_step198_cgroup_detail.csv",
    f"{run}/resource_monitor/peak.txt":
        workspace / ".tmp" / "dsrl_step198_peak.txt",
    f"{run}/formal_driver.log":
        workspace / ".tmp" / "dsrl_step198_formal_driver.log",
    f"{run}/metrics.log":
        workspace / ".tmp" / "dsrl_step198_metrics.log",
}

client = remote.connect(args)
try:
    for source, destination in downloads.items():
        compressed = destination.with_suffix(destination.suffix + ".gz.partial")
        temporary = destination.with_suffix(destination.suffix + ".partial")
        command = f"gzip -c -- {shlex.quote(source)}"
        stdin, stdout, stderr = client.exec_command(command)
        stdin.channel.shutdown_write()
        with compressed.open("wb") as handle:
            while chunk := stdout.read(1024 * 1024):
                handle.write(chunk)
        error_text = stderr.read().decode("utf-8", errors="replace")
        status = stdout.channel.recv_exit_status()
        if status != 0:
            raise RuntimeError(f"{command!r} exited {status}: {error_text}")
        with gzip.open(compressed, "rb") as source_handle, temporary.open("wb") as target:
            shutil.copyfileobj(source_handle, target)
        os.replace(temporary, destination)
        compressed.unlink()
        print(f"{source} -> {destination} ({destination.stat().st_size} bytes)")
finally:
    client.close()
