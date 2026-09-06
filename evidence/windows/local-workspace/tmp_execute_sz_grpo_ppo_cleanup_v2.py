from __future__ import annotations

import argparse
import getpass
import hashlib
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "local_scripts"))
import remote_exec_autodl


SCRIPT = Path("tmp_grpo_ppo_delete_top2_files_v2.sh")
EXPECTED_SHA256 = "37F536BAC79F89B413000170A9A6E007E09E0E7450135EB0F69709A8BB79F15B"
STDOUT_PATH = Path("tmp_grpo_ppo_cleanup_v2_execution_stdout.txt")
STDERR_PATH = Path("tmp_grpo_ppo_cleanup_v2_execution_stderr.txt")


def main() -> None:
    payload = SCRIPT.read_bytes()
    actual_sha256 = hashlib.sha256(payload).hexdigest().upper()
    if actual_sha256 != EXPECTED_SHA256:
        raise SystemExit(
            f"refusing execution: script SHA256 {actual_sha256} != {EXPECTED_SHA256}"
        )

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
        # Match the established command-file route: keep the hashed file unchanged,
        # but normalize Windows CRLF while decoding it for the remote POSIX shell.
        command = SCRIPT.read_text(encoding="utf-8")
        stdin, stdout, stderr = client.exec_command(command)
        stdin.close()
        stdout_data = stdout.read().decode("utf-8")
        stderr_data = stderr.read().decode("utf-8")
        rc = stdout.channel.recv_exit_status()
        STDOUT_PATH.write_text(stdout_data, encoding="utf-8")
        STDERR_PATH.write_text(stderr_data, encoding="utf-8")
        print(stdout_data, end="")
        if stderr_data:
            print(stderr_data, end="", file=sys.stderr)
        if rc != 0:
            raise SystemExit(rc)
    finally:
        password = ""
        os.environ.pop("SEETA_SSH_PASSWORD", None)
        if client is not None:
            client.close()


if __name__ == "__main__":
    main()
