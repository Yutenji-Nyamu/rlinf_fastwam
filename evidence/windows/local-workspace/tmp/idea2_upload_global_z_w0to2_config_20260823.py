from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


WORKSPACE = Path(r"C:\Users\86136\Documents\rl")
RUNTIME = (
    "/root/autodl-tmp/idea2_dvac_train_runtime/"
    "idea2_dvac_global_z_w0to2_formal_100step_2gpu16env_20260823"
)
FILES = (
    (
        WORKSPACE
        / "tmp/idea2_residual_downweight_impl_source/rlinf/examples/embodiment/config/"
        "robotwin_adjust_bottle_grpo_openpi_dvac_global_z_w0to2_100step_formal.yaml",
        "/root/autodl-tmp/RLinf_idea2_dvac_residual_downweight/examples/embodiment/config/"
        "robotwin_adjust_bottle_grpo_openpi_dvac_global_z_w0to2_100step_formal.yaml",
    ),
    (WORKSPACE / "tmp/idea2_global_z_w0to2_run_formal.sh", f"{RUNTIME}/launch_formal.sh"),
    (WORKSPACE / "tmp/idea2_global_z_w0to2_observe_resources_formal.sh", f"{RUNTIME}/observe_resources.sh"),
)

sys.path.insert(0, str(WORKSPACE / "local_scripts"))
import remote_exec_autodl  # noqa: E402


def main() -> None:
    args = argparse.Namespace(
        host=remote_exec_autodl.DEFAULT_HOST,
        port=remote_exec_autodl.DEFAULT_PORT,
        user=remote_exec_autodl.DEFAULT_USER,
        password_env="SEETA_SSH_PASSWORD",
        host_key_sha256=remote_exec_autodl.DEFAULT_HOST_KEY_SHA256,
        timeout=20.0,
    )
    if not os.environ.get(args.password_env):
        raise RuntimeError("SEETA_SSH_PASSWORD is required")
    client = remote_exec_autodl.connect(args)
    try:
        with client.open_sftp() as sftp:
            try:
                sftp.stat(RUNTIME)
            except FileNotFoundError:
                sftp.mkdir(RUNTIME)
            for local, remote in FILES:
                sftp.put(str(local), remote)
                attrs = sftp.stat(remote)
                if attrs.st_size != local.stat().st_size:
                    raise RuntimeError(f"size mismatch for {remote}: {attrs.st_size} != {local.stat().st_size}")
                print(f"UPLOADED={remote} BYTES={attrs.st_size}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
