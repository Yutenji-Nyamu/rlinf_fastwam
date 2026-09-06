from __future__ import annotations

import argparse
import gzip
import shlex
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "local_scripts"))
import remote_exec_autodl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("remote")
    parser.add_argument("local", type=Path)
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
        _, stdout, stderr = client.exec_command(
            f"gzip -c -- {shlex.quote(args.remote)}"
        )
        compressed = stdout.read()
        error = stderr.read()
        status = stdout.channel.recv_exit_status()
        if status:
            raise RuntimeError(
                f"remote gzip failed with {status}: "
                f"{error.decode('utf-8', errors='replace')}"
            )
    finally:
        client.close()
    payload = gzip.decompress(compressed)
    args.local.parent.mkdir(parents=True, exist_ok=True)
    args.local.write_bytes(payload)
    print(
        f"compressed_bytes={len(compressed)} "
        f"uncompressed_bytes={len(payload)} local={args.local}"
    )


if __name__ == "__main__":
    main()
