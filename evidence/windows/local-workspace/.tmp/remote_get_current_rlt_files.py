from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_scripts.remote_exec_autodl import (
    DEFAULT_HOST,
    DEFAULT_HOST_KEY_SHA256,
    DEFAULT_PORT,
    DEFAULT_USER,
    connect,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    parser.add_argument("remote_files", nargs="+")
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)

    connection_args = argparse.Namespace(
        host=DEFAULT_HOST,
        port=DEFAULT_PORT,
        user=DEFAULT_USER,
        host_key_sha256=DEFAULT_HOST_KEY_SHA256,
        timeout=30.0,
    )
    client = connect(connection_args)
    try:
        with client.open_sftp() as sftp:
            for remote in args.remote_files:
                local = args.destination / remote.lstrip("/").replace("/", "__")
                sftp.get(remote, str(local))
                print(f"{remote}\t{local}\t{local.stat().st_size}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
