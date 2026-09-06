from __future__ import annotations

import argparse
import os
from pathlib import Path, PurePosixPath
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_scripts.remote_exec_autodl import connect


REMOTE_RUNTIME = PurePosixPath(
    "/root/autodl-tmp/experiment_exports/"
    "rlt_teacher_dvac_w0to2_smoke_8env1c_20260824_v1/runtime"
)
REMOTE_TRACE = PurePosixPath(
    "/root/autodl-tmp/experiments/"
    "rlt_teacher_dvac_w0to2_smoke_8env1c_20260824_v1/"
    "robotwin_adjust_bottle_rlt_teacher_dvac_w0to2_smoke_8env1c_v1/rlt_dvac"
)
LOCAL_ROOT = Path(
    "docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/"
    "rlt_dvac_real_smoke_8env1c_20260824"
)


def main() -> None:
    if not os.environ.get("SEETA_SSH_PASSWORD"):
        raise SystemExit("SEETA_SSH_PASSWORD is required")
    args = argparse.Namespace(
        host="connect.bjb1.seetacloud.com",
        port=36406,
        user="root",
        host_key_sha256="liZ36vNCsNcNdXeWs4f+g5ZIhPM/ZihP834vxs8Ulqc",
        timeout=20.0,
    )
    LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
    runtime_names = (
        "driver.log",
        "exact_command.txt",
        "exit_code.txt",
        "finished_at.txt",
        "resolved.yaml",
        "resources.csv",
        "resources_after.txt",
        "resources_before.txt",
        "run_provenance.tsv",
        "stage1_binding_preflight.json",
        "stage1_binding_preflight.stdout",
        "started_at.txt",
        "unit_tests.log",
    )
    ray_logs = {
        "/tmp/ray/session_latest/logs/"
        "worker-a1a4c73092f8cf39aa77bed695e56df2845b89ee1434e5c39516a2de-"
        "ffffffff-7876.err": "ray_actor_rank0.err",
        "/tmp/ray/session_latest/logs/"
        "worker-e3d56aa2c0a00b5a77e2dca229b6f9fd6676fd7d85c0864e52609796-"
        "ffffffff-7894.err": "ray_actor_rank1.err",
    }
    client = connect(args)
    try:
        with client.open_sftp() as sftp:
            for name in runtime_names:
                sftp.get(str(REMOTE_RUNTIME / name), str(LOCAL_ROOT / name))
            for remote, local_name in ray_logs.items():
                sftp.get(remote, str(LOCAL_ROOT / local_name))
            for rank in (0, 1):
                for update in (0, 2, 4, 6):
                    name = f"actor_rank{rank:02d}_update_{update:08d}.npz"
                    remote = (
                        REMOTE_TRACE
                        / f"actor_rank{rank:02d}"
                        / f"update_{update:08d}.npz"
                    )
                    sftp.get(str(remote), str(LOCAL_ROOT / name))
    finally:
        client.close()
    for path in sorted(LOCAL_ROOT.iterdir()):
        if path.is_file():
            print(f"{path.stat().st_size}\t{path.name}")


if __name__ == "__main__":
    main()
