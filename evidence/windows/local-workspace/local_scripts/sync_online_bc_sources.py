"""Upload only this task's source edits to its pinned, isolated server branch."""
import getpass
import os
from pathlib import Path

import remote_exec_autodl as ssh

FILES = (
    "rlinf/data/online_bc.py",
    "rlinf/workers/actor/fsdp_online_bc_policy_worker.py",
    "rlinf/models/embodiment/openpi/openpi_action_model.py",
    "examples/embodiment/train_embodied_agent.py",
    "rlinf/workers/rollout/hf/huggingface_worker.py",
    "rlinf/workers/env/env_worker.py",
    "examples/embodiment/config/robotwin_adjust_bottle_online_bc_openpi.yaml",
    "tests/unit_tests/test_online_bc.py",
    "docs/online_bc.md",
)
ROOT = "/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-online-bc"
PIN = "5ae809d5a8c8c293540b57d14e423050c2c1e8d2"


def main():
    args = ssh.build_parser().parse_args([
        "--host", "120.241.223.9", "--port", "22", "--user", "chenyiteng",
        "--host-key-sha256", "qw92OXne52y6NsQMQ6+PdjOu0Hosc4ZX78XmkYxcuEY",
        "run", "--command-file", "local_scripts/remote_commands/sz_online_bc_basic_tests_20260904.sh",
    ])
    os.environ["SEETA_SSH_PASSWORD"] = getpass.getpass("SSH password: ")
    client = ssh.connect(args)
    try:
        _, out, err = client.exec_command(f"git -C {ROOT} rev-parse HEAD; git -C {ROOT} branch --show-current")
        actual = out.read().decode().splitlines()
        if actual != [PIN, "codex/sz-pi0-online-bc"]:
            raise RuntimeError(f"Unexpected target branch: {actual}; {err.read().decode()}")
        _, out, err = client.exec_command(f"git -C {ROOT} status --porcelain")
        if out.read().decode().strip() or out.channel.recv_exit_status():
            raise RuntimeError("Refuse to overwrite a dirty server source tree")
        with client.open_sftp() as sftp:
            for rel in FILES:
                local = Path("worktrees/pi0-online-bc") / rel
                remote = f"{ROOT}/{rel}"
                parent = str(Path(rel).parent).replace("\\", "/")
                current = ROOT
                for part in parent.split("/"):
                    current += "/" + part
                    try:
                        sftp.stat(current)
                    except FileNotFoundError:
                        sftp.mkdir(current)
                sftp.put(str(local), remote)
                print("UPLOADED", rel, flush=True)
        result = ssh.run_command(client, args)
        if result == 0:
            with client.open_sftp() as sftp:
                for rel in FILES:
                    sftp.get(f"{ROOT}/{rel}", str(Path("worktrees/pi0-online-bc") / rel))
                sftp.get(
                    "/data/chenyiteng/results/rlinf-shenzhen/online-bc/implementation-20260905/smoke-resolved-compose.yaml",
                    "docs/rlinf-robotwin-pi0-online-bc/evidence/GPU6_SMOKE_RESOLVED_20260905.yaml",
                )
        raise SystemExit(result)
    finally:
        client.close()
        os.environ.pop("SEETA_SSH_PASSWORD", None)


if __name__ == "__main__":
    main()
