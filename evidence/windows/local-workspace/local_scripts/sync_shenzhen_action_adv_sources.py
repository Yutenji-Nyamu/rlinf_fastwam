"""Fetch or atomically upload the fixed small source set for Action-Adv.

The SSH password is prompted once and kept only in this process.  Host identity
is verified by the existing fixed-host-key Paramiko helper.
"""

from __future__ import annotations

import argparse
import getpass
import os
from pathlib import Path, PurePosixPath
import shutil

import remote_exec_autodl


REMOTE_ROOT = PurePosixPath(
    "/data/chenyiteng/projects/rlinf-shenzhen/worktrees/grpo-dvac-action-adv"
)
LOCAL_ROOT = Path(__file__).resolve().parents[1] / "references" / "action_adv_impl_20260827"
FILES = (
    "rlinf/algorithms/dvac_train_weighting.py",
    "rlinf/workers/actor/embodied_fsdp_actor_worker.py",
    "rlinf/algorithms/utils.py",
    "examples/embodiment/config/robotwin_adjust_bottle_grpo_openpi.yaml",
    "tests/unit_tests/test_dvac_train_weighting.py",
)
PUT_FILES = tuple(
    relative
    for relative in FILES
    if relative != "rlinf/algorithms/dvac_train_weighting.py"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("get", "put"))
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--host-key-sha256", required=True)
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()

    password = [REDACTED]"SSH password: ")
    os.environ["SEETA_SSH_PASSWORD"] = password
    client = None
    try:
        client = remote_exec_autodl.connect(args)
        with client.open_sftp() as sftp:
            if args.action == "get":
                for relative in FILES:
                    remote = str(REMOTE_ROOT / relative)
                    base = LOCAL_ROOT / "base" / relative
                    work = LOCAL_ROOT / "work" / relative
                    base.parent.mkdir(parents=True, exist_ok=True)
                    work.parent.mkdir(parents=True, exist_ok=True)
                    sftp.get(remote, str(base))
                    shutil.copyfile(base, work)
                    print(f"GET {relative}")
            else:
                for relative in PUT_FILES:
                    local = LOCAL_ROOT / "work" / relative
                    if not local.is_file():
                        raise SystemExit(f"missing local source: {local}")
                    remote = str(REMOTE_ROOT / relative)
                    temporary = remote + ".action_adv_upload_tmp"
                    mode = sftp.stat(remote).st_mode
                    sftp.put(str(local), temporary)
                    sftp.chmod(temporary, mode)
                    sftp.posix_rename(temporary, remote)
                    print(f"PUT {relative}")
    finally:
        password = ""
        os.environ.pop("SEETA_SSH_PASSWORD", None)
        if client is not None:
            client.close()


if __name__ == "__main__":
    main()
