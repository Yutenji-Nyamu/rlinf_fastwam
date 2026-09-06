import argparse
import os
from pathlib import Path

import paramiko


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("remote")
    parser.add_argument("local")
    args = parser.parse_args()

    password = [REDACTED]"SEETA_SSH_PASSWORD")
    if not password:
        [REDACTED] SystemExit("SEETA_SSH_PASSWORD is required")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        "connect.bjb1.seetacloud.com",
        port=36406,
        username="root",
        [REDACTED]=[REDACTED],
        look_for_keys=False,
        allow_agent=False,
        timeout=20,
        banner_timeout=20,
        auth_timeout=20,
    )
    try:
        destination = Path(args.local)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with client.open_sftp() as sftp:
            sftp.get(args.remote, str(destination))
    finally:
        client.close()


if __name__ == "__main__":
    main()
