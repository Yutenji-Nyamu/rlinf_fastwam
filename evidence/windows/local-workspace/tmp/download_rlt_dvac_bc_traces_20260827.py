from __future__ import annotations

import argparse
from pathlib import Path
import sys


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "local_scripts"))
import remote_exec_autodl as remote  # noqa: E402


REMOTE_DIR = (
    "/root/autodl-tmp/experiments/"
    "rlt_single_gpu_success_episode_bc_dvac_matched_width_formal480_20260826_v3/"
    "robotwin_adjust_bottle_rlt_single_gpu_success_episode_bc_dvac_"
    "matched_width_formal480_20260826_v3/rlt_dvac/actor_rank00"
)
LOCAL_DIR = WORKSPACE / "tmp" / "rlt_dvac_amplitude_audit_20260827" / "all_traces"


args = argparse.Namespace(
    host=remote.DEFAULT_HOST,
    port=remote.DEFAULT_PORT,
    user=remote.DEFAULT_USER,
    host_key_sha256=remote.DEFAULT_HOST_KEY_SHA256,
    timeout=20.0,
)
LOCAL_DIR.mkdir(parents=True, exist_ok=True)
client = remote.connect(args)
try:
    with client.open_sftp() as sftp:
        names = sorted(
            name
            for name in sftp.listdir(REMOTE_DIR)
            if name.startswith("update_") and name.endswith(".npz")
        )
        downloaded = 0
        for name in names:
            local_path = LOCAL_DIR / name
            if local_path.exists() and local_path.stat().st_size > 0:
                continue
            partial_path = local_path.with_suffix(".npz.partial")
            sftp.get(f"{REMOTE_DIR}/{name}", str(partial_path))
            partial_path.replace(local_path)
            downloaded += 1
        print(
            f"remote={len(names)} downloaded_now={downloaded} "
            f"local={len(list(LOCAL_DIR.glob('update_*.npz')))} local_dir={LOCAL_DIR}"
        )
finally:
    client.close()
