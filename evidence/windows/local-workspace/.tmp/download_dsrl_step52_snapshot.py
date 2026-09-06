from __future__ import annotations

import importlib.util
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
run = Path(
    "/root/autodl-tmp/RLinf_fastwam_rlinf/logs/"
    "20260728_dsrl_pi0_robotwin_n20_formal_v1"
)
downloads = {
    run / "tensorboard" / "events.out.tfevents.1785236070.autodl-container-nekaqbwt43-6ce5babb.70062.0":
        workspace / ".tmp" / "dsrl_step52_events.tfevents",
    run / "resource_monitor" / "resources.csv":
        workspace / ".tmp" / "dsrl_step52_resources.csv",
    run / "resource_monitor" / "cgroup_detail.csv":
        workspace / ".tmp" / "dsrl_step52_cgroup_detail.csv",
    run / "resource_monitor" / "peak.txt":
        workspace / ".tmp" / "dsrl_step52_peak.txt",
    run / "formal_driver.log":
        workspace / ".tmp" / "dsrl_step52_formal_driver.log",
    run / "metrics.log":
        workspace / ".tmp" / "dsrl_step52_metrics.log",
}

client = remote.connect(args)
try:
    with client.open_sftp() as sftp:
        for source, destination in downloads.items():
            temporary = destination.with_suffix(destination.suffix + ".partial")
            sftp.get(source.as_posix(), temporary.as_posix())
            temporary.replace(destination)
            print(f"{source} -> {destination} ({destination.stat().st_size} bytes)")
finally:
    client.close()
