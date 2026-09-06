from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_scripts import remote_exec_autodl


FILES = (
    "rlinf/models/embodiment/openpi/openpi_action_model.py",
    "rlinf/algorithms/rlt/__init__.py",
    "rlinf/algorithms/rlt/dvac_weighting.py",
    "rlinf/algorithms/rlt/rollout.py",
    "rlinf/algorithms/rlt/route.py",
    "rlinf/algorithms/rlt/transition.py",
    "rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py",
    "rlinf/models/embodiment/mlp_policy/rlt_mlp_policy.py",
    "rlinf/data/embodied_io_struct.py",
    "rlinf/data/replay_buffer.py",
    "examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp.yaml",
    "examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env250.yaml",
    "examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env250_teacher_dvac_w0to2.yaml",
    "tests/unit_tests/test_rlt_dvac_weighting.py",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("direction", choices=("get", "put"))
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--local-root", required=True)
    args = parser.parse_args()

    connection_args = argparse.Namespace(
        host=remote_exec_autodl.DEFAULT_HOST,
        port=remote_exec_autodl.DEFAULT_PORT,
        user=remote_exec_autodl.DEFAULT_USER,
        host_key_sha256=remote_exec_autodl.DEFAULT_HOST_KEY_SHA256,
        timeout=20.0,
    )
    client = remote_exec_autodl.connect(connection_args)
    try:
        with client.open_sftp() as sftp:
            for relative in FILES:
                remote_path = str(PurePosixPath(args.remote_root) / relative)
                local_path = Path(args.local_root, *PurePosixPath(relative).parts)
                if args.direction == "get":
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    sftp.get(remote_path, str(local_path))
                else:
                    sftp.put(str(local_path), remote_path)
                print(f"{args.direction.upper()} {relative}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
