from __future__ import annotations

import argparse
import os
from pathlib import Path, PurePosixPath
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_scripts.remote_exec_autodl import connect


REMOTE_RUNTIME = PurePosixPath(
    "/root/autodl-tmp/experiment_exports/"
    "rlt_teacher_dvac_w0to2_smoke_8env1c_20260824_v2/runtime"
)
REMOTE_RUN = PurePosixPath(
    "/root/autodl-tmp/experiments/"
    "rlt_teacher_dvac_w0to2_smoke_8env1c_20260824_v2"
)
REMOTE_EXPERIMENT = REMOTE_RUN / (
    "robotwin_adjust_bottle_rlt_teacher_dvac_w0to2_smoke_8env1c_v2"
)
REMOTE_TRACE = REMOTE_EXPERIMENT / "rlt_dvac"
REMOTE_TRAINER_STATE = (
    REMOTE_EXPERIMENT
    / "checkpoints/global_step_1/actor/sac_components/rlt_trainer_state"
)
LOCAL_ROOT = Path(
    "docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/"
    "rlt_dvac_real_smoke_8env1c_20260824_v2"
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
        "exact_command_v1_reference.txt",
        "exit_code.txt",
        "finished_at.txt",
        "launch.sh",
        "monitor.log",
        "resolved.yaml",
        "resources.csv",
        "resources_after.txt",
        "run_foreground.sh",
        "source_head.txt",
        "started_at.txt",
    )
    run_names = ("metrics.log",)
    state_names = (
        "checkpoint_rank_0.pt",
        "checkpoint_rank_1.pt",
        "rlt_trainer_state_complete.json",
    )
    client = connect(args)
    try:
        with client.open_sftp() as sftp:
            for name in runtime_names:
                sftp.get(str(REMOTE_RUNTIME / name), str(LOCAL_ROOT / name))
            for name in run_names:
                sftp.get(str(REMOTE_RUN / name), str(LOCAL_ROOT / name))
            for name in state_names:
                sftp.get(str(REMOTE_TRAINER_STATE / name), str(LOCAL_ROOT / name))
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
