"""Download or upload the narrow Idea2 residual-downweight source set.

The SSH password is prompted without echo and remains process-local.
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from pathlib import Path


WORKSPACE = Path(r"C:\Users\86136\Documents\rl")
LOCAL_ROOT = WORKSPACE / "tmp" / "idea2_residual_downweight_impl_source" / "rlinf"
REMOTE_ROOT = "/root/autodl-tmp/RLinf_idea2_dvac_residual_downweight"
FILES = (
    "rlinf/algorithms/dvac_train_weighting.py",
    "rlinf/workers/actor/fsdp_actor_worker.py",
    "tests/unit_tests/test_dvac_train_weighting.py",
    "examples/embodiment/config/robotwin_adjust_bottle_grpo_openpi_dvac_train_2step_smoke.yaml",
    "examples/embodiment/config/robotwin_adjust_bottle_grpo_openpi_dvac_train_100step_formal.yaml",
    "examples/embodiment/config/robotwin_adjust_bottle_grpo_openpi_dvac_r_only_downweight_2step_smoke.yaml",
)

sys.path.insert(0, str(WORKSPACE))
from local_scripts.remote_exec_autodl import (  # noqa: E402
    DEFAULT_HOST,
    DEFAULT_HOST_KEY_SHA256,
    DEFAULT_PORT,
    DEFAULT_USER,
    connect,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("direction", choices=("download", "upload"))
    args = parser.parse_args()

    password = [REDACTED]"SSH password: ")
    os.environ["SEETA_SSH_PASSWORD"] = password
    connect_args = argparse.Namespace(
        host=DEFAULT_HOST,
        port=DEFAULT_PORT,
        user=DEFAULT_USER,
        host_key_sha256=DEFAULT_HOST_KEY_SHA256,
        timeout=20.0,
    )
    client = connect(connect_args)
    total = 0
    try:
        with client.open_sftp() as sftp:
            for relative in FILES:
                local = LOCAL_ROOT / relative
                remote = f"{REMOTE_ROOT}/{relative}"
                if args.direction == "download":
                    local.parent.mkdir(parents=True, exist_ok=True)
                    sftp.get(remote, str(local))
                else:
                    sftp.put(str(local), remote)
                total += local.stat().st_size
    finally:
        client.close()
        password = ""
        os.environ.pop("SEETA_SSH_PASSWORD", None)

    print(f"DIRECTION={args.direction}")
    print(f"FILES={len(FILES)}")
    print(f"BYTES={total}")
    print(f"LOCAL_ROOT={LOCAL_ROOT}")
    print(f"REMOTE_ROOT={REMOTE_ROOT}")


if __name__ == "__main__":
    main()
