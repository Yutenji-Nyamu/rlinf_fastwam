"""Fetch only the small, explicitly scoped OIDN trial evidence."""
import argparse
import getpass
import json
import os
from pathlib import Path
from remote_exec_autodl import connect

args = argparse.Namespace(host="120.241.223.9", port=22, user="chenyiteng", host_key_sha256="qw92OXne52y6NsQMQ6+PdjOu0Hosc4ZX78XmkYxcuEY", timeout=20)
os.environ["SEETA_SSH_PASSWORD"] = getpass.getpass("SSH password: ")
remote_root = "/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/diagnostics/oidn-toggle-20260904"
local_root = Path("docs/fastwam-robotwin-rlinf-grpo/evidence/oidn-toggle-20260904")
client = connect(args)
try:
    with client.open_sftp() as sftp:
        files = []
        for mode in ["oidn", "none"]:
            with sftp.open(f"{remote_root}/{mode}/summary.json") as handle:
                summary = json.load(handle)
            for name in ["summary.json", "trial.log", "exit_code.txt", "started_at.txt", "finished_at.txt", "resolved.yaml"]:
                files.append(f"{mode}/{name}")
            for episode in summary["episodes"]:
                prefix = f"{mode}/episode_{episode['episode']}_seed_{episode['requested_seed']}"
                files.append(f"{prefix}/result.json")
                for camera in ["head", "left", "right"]:
                    files.append(f"{prefix}/q00_{camera}.png")
                for query in range(len(episode["queries"]) + 1):
                    files.append(f"{prefix}/q{query:02d}_model_input.png")
        sizes = {name: sftp.stat(f"{remote_root}/{name}").st_size for name in files}
        total = sum(sizes.values())
        assert total < 40 * 1024**2, f"Unexpected evidence size: {total}"
        for name in files:
            target = local_root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            sftp.get(f"{remote_root}/{name}", str(target))
        print(json.dumps({"files": sizes, "bytes": total}, indent=2))
finally:
    client.close()
    os.environ.pop("SEETA_SSH_PASSWORD", None)
