"""Upload the reviewed Prism patch to its dedicated SZ worktree."""

from __future__ import annotations

import argparse
import getpass
import os
from pathlib import Path

import remote_exec_autodl


FILES = (
    "rlinf/algorithms/advantages.py",
    "rlinf/algorithms/dvac_rank_reward.py",
    "rlinf/config.py",
    "rlinf/workers/actor/embodied_fsdp_actor_worker.py",
    "rlinf/workers/rollout/hf/huggingface_worker.py",
    "examples/embodiment/config/robotwin_adjust_bottle_grpo_openpi.yaml",
    "tests/unit_tests/test_prism_dvac_rank_rloo.py",
)

REMOTE_ROOT = "/data/chenyiteng/projects/rlinf-shenzhen/worktrees/prism-dvac-rank-rloo"
EXPECTED_HEAD = "0e28ac6f09f821ea12e7d54eba7118ce0000ca86"


def run(client, command: str) -> tuple[int, str, str]:
    stdin, stdout, stderr = client.exec_command(command)
    stdin.close()
    return stdout.channel.recv_exit_status(), stdout.read().decode(), stderr.read().decode()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-root", required=True)
    parser.add_argument("--only", action="append", choices=FILES)
    args = parser.parse_args()

    password = [REDACTED]"SSH password: ")
    os.environ["SEETA_SSH_PASSWORD"] = password
    conn_args = argparse.Namespace(
        host="120.241.223.9",
        port=22,
        user="chenyiteng",
        host_key_sha256="qw92OXne52y6NsQMQ6+PdjOu0Hosc4ZX78XmkYxcuEY",
        timeout=15.0,
    )
    local_root = Path(args.local_root)
    client = None
    try:
        client = remote_exec_autodl.connect(conn_args)
        code, head, err = run(client, f"git -C {REMOTE_ROOT} rev-parse HEAD")
        if code or head.strip() != EXPECTED_HEAD:
            raise RuntimeError(f"unexpected remote HEAD: {head.strip()} {err.strip()}")
        code, status, err = run(client, f"git -C {REMOTE_ROOT} status --short")
        if code:
            raise RuntimeError(f"remote status failed: {err}")
        if status.strip() and args.only is None:
            raise RuntimeError(f"remote worktree is not clean: {status} {err}")
        if status.strip():
            changed = {line[3:] for line in status.splitlines() if len(line) >= 4}
            if not changed.issubset(set(FILES)):
                raise RuntimeError(f"unexpected remote changes: {sorted(changed)}")

        with client.open_sftp() as sftp:
            selected_files = tuple(args.only) if args.only else FILES
            for relative in selected_files:
                local = local_root / relative
                remote = f"{REMOTE_ROOT}/{relative}"
                temporary = f"{remote}.codex-upload"
                sftp.put(str(local), temporary)
                sftp.posix_rename(temporary, remote)
                print(relative)
    finally:
        password = ""
        os.environ.pop("SEETA_SSH_PASSWORD", None)
        if client is not None:
            client.close()


if __name__ == "__main__":
    main()
