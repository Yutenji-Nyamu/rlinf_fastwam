"""Run the existing pinned-host Paramiko CLI with a no-echo password prompt.

The credential is injected only into this process, never accepted on the
command line, and removed from the environment before exit.
"""

from __future__ import annotations

import getpass
import os

import remote_exec_autodl


def main() -> None:
    password = [REDACTED]"SSH password: ")
    os.environ["SEETA_SSH_PASSWORD"] = password
    try:
        remote_exec_autodl.main()
    finally:
        password = ""
        os.environ.pop("SEETA_SSH_PASSWORD", None)


if __name__ == "__main__":
    main()
