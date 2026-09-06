"""Run one fixed-host-key SSH action with a process-only prompted password.

This is a thin interactive wrapper around ``remote_exec_autodl.py``.  It keeps
the password out of command lines, files, logs, and the remote command stdin.
"""

from __future__ import annotations

import getpass
import os
import runpy
import sys


def main() -> None:
    password = [REDACTED]"SSH password: ")
    os.environ["SEETA_SSH_PASSWORD"] = password
    try:
        helper = os.path.join(os.path.dirname(__file__), "remote_exec_autodl.py")
        sys.argv[0] = helper
        runpy.run_path(helper, run_name="__main__")
    finally:
        password = ""
        os.environ.pop("SEETA_SSH_PASSWORD", None)


if __name__ == "__main__":
    main()
