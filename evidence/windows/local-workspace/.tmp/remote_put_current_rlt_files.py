from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_scripts.remote_exec_autodl import (  # noqa: E402
    DEFAULT_HOST,
    DEFAULT_HOST_KEY_SHA256,
    DEFAULT_PORT,
    DEFAULT_USER,
    connect,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "pairs",
        nargs="+",
        help="Alternating local and remote paths.",
    )
    args = parser.parse_args()
    if len(args.pairs) % 2:
        raise SystemExit("pairs must contain alternating local and remote paths")

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
            for index in range(0, len(args.pairs), 2):
                local = Path(args.pairs[index])
                remote = args.pairs[index + 1]
                sftp.put(str(local), remote)
                print(f"{local}\t{remote}\t{local.stat().st_size}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
