from __future__ import annotations

import argparse
import getpass
import hashlib
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "local_scripts"))
import remote_exec_autodl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", required=True)
    parser.add_argument("--remote", required=True)
    parser.add_argument("--expected-remote-sha256", required=True)
    cli = parser.parse_args()
    local_path = Path(cli.local)
    local_data = local_path.read_bytes()
    password = [REDACTED]"SSH password: ")
    os.environ["SEETA_SSH_PASSWORD"] = password
    args = argparse.Namespace(
        host="120.241.223.9",
        port=22,
        user="chenyiteng",
        host_key_sha256="qw92OXne52y6NsQMQ6+PdjOu0Hosc4ZX78XmkYxcuEY",
        timeout=20.0,
    )
    client = None
    try:
        client = remote_exec_autodl.connect(args)
        sftp = client.open_sftp()
        try:
            with sftp.open(cli.remote, "rb") as remote_file:
                remote_data = remote_file.read()
            remote_sha = hashlib.sha256(remote_data).hexdigest()
            if remote_sha != cli.expected_remote_sha256.lower():
                raise RuntimeError(
                    f"remote preimage mismatch: {remote_sha} != {cli.expected_remote_sha256}"
                )
            with sftp.open(cli.remote, "wb") as remote_file:
                remote_file.write(local_data)
            with sftp.open(cli.remote, "rb") as remote_file:
                uploaded_data = remote_file.read()
            uploaded_sha = hashlib.sha256(uploaded_data).hexdigest()
            local_sha = hashlib.sha256(local_data).hexdigest()
            if uploaded_sha != local_sha:
                raise RuntimeError(
                    f"upload mismatch: remote={uploaded_sha} local={local_sha}"
                )
            print(f"uploaded_sha256={uploaded_sha}")
        finally:
            sftp.close()
    finally:
        password = ""
        os.environ.pop("SEETA_SSH_PASSWORD", None)
        if client is not None:
            client.close()


if __name__ == "__main__":
    main()
