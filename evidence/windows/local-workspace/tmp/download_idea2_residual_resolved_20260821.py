from __future__ import annotations

import argparse
import getpass
import os
import sys
from pathlib import Path


WORKSPACE = Path(r"C:\Users\86136\Documents\rl")
LOCAL = WORKSPACE / "tmp" / "idea2_residual_downweight_pretest"
REMOTE = "/root/autodl-tmp/idea2_dvac_residual_downweight_pretest"
FILES = (
    "resolved_r_only_downweight_2step.yaml",
    "resolved_old_dvac_2step.yaml",
    "resolved_default_off_baseline.yaml",
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
    password = [REDACTED]"SSH password: ")
    os.environ["SEETA_SSH_PASSWORD"] = password
    args = argparse.Namespace(
        host=DEFAULT_HOST,
        port=DEFAULT_PORT,
        user=DEFAULT_USER,
        host_key_sha256=DEFAULT_HOST_KEY_SHA256,
        timeout=20.0,
    )
    client = connect(args)
    try:
        LOCAL.mkdir(parents=True, exist_ok=True)
        with client.open_sftp() as sftp:
            for name in FILES:
                sftp.get(f"{REMOTE}/{name}", str(LOCAL / name))
    finally:
        client.close()
        password = ""
        os.environ.pop("SEETA_SSH_PASSWORD", None)
    print(f"DOWNLOADED={len(FILES)}")
    print(f"LOCAL={LOCAL}")


if __name__ == "__main__":
    main()
