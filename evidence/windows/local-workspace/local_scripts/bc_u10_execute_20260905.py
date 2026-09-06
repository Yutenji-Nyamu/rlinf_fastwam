"""Scoped source deployment and reviewed commands for the authorized GPU6 run."""
import argparse
import getpass
import json
import os
from pathlib import Path

import remote_exec_autodl as ssh

ROOT = "/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-online-bc"
BASE = "/data/chenyiteng/results/rlinf-shenzhen/online-bc"
PACKET = BASE + "/implementation-u10-20260905"
LOCAL = Path("docs/rlinf-robotwin-pi0-online-bc/evidence")
FILES = (
    "rlinf/envs/robotwin/robotwin_env.py",
    "examples/embodiment/config/robotwin_adjust_bottle_online_bc_openpi.yaml",
    "tests/unit_tests/test_online_bc.py",
)
RUNS = {
    "smoke": BASE + "/pi0-adjust-bottle-smoke32x1-b1024-u10-eval16x2-gpu6-20260905-v7",
    "formal": BASE + "/pi0-adjust-bottle-bc32x1-b1024-u10-eval16x2-gpu6-formal100-20260905-v1",
}


def command(client, content):
    _, out, err = client.exec_command(content)
    data, error = out.read(), err.read()
    rc = out.channel.recv_exit_status()
    print(data.decode(errors="replace"), flush=True)
    if error:
        print(error.decode(errors="replace"), flush=True)
    if rc:
        raise RuntimeError(f"Remote command returned {rc}; not replayed")
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["deploy", "command", "launch", "fetch", "watch", "publish", "fetch-startup"])
    parser.add_argument("--command-file")
    parser.add_argument("--run", choices=RUNS, default="smoke")
    args = parser.parse_args()
    os.environ["SEETA_SSH_PASSWORD"] = getpass.getpass("SSH password: ")
    client = ssh.connect(argparse.Namespace(
        host="120.241.223.9", port=22, user="chenyiteng", timeout=20.0,
        host_key_sha256="qw92OXne52y6NsQMQ6+PdjOu0Hosc4ZX78XmkYxcuEY",
    ))
    try:
        if args.mode == "deploy":
            actual = command(client, f"git -C {ROOT} rev-parse HEAD; git -C {ROOT} branch --show-current; git -C {ROOT} status --porcelain").decode().splitlines()
            assert actual == ["700b6846dbc2fe02398de05c044c8097cc974774", "codex/sz-pi0-online-bc"]
            with client.open_sftp() as sftp:
                for rel in FILES:
                    sftp.put(str(Path("worktrees/pi0-online-bc") / rel), ROOT + "/" + rel)
                    print("UPLOADED", rel, flush=True)
        if args.mode == "watch":
            _, out, err = client.exec_command(Path(args.command_file).read_text(encoding="utf-8"))
            for line in out:
                print(line.rstrip(), flush=True)
            error = err.read().decode(errors="replace")
            if error:
                print(error, flush=True)
            rc = out.channel.recv_exit_status()
            if rc:
                raise RuntimeError(f"Read-only watch exited {rc}")
        elif args.command_file:
            command(client, Path(args.command_file).read_text(encoding="utf-8"))
        if args.mode == "fetch":
            with client.open_sftp() as sftp:
                for name in ("smoke", "formal"):
                    sftp.get(PACKET + f"/{name}-resolved.yaml", str(LOCAL / f"U10_{name.upper()}_RESOLVED_20260905.yaml"))
                sftp.get(PACKET + "/validation.json", str(LOCAL / "U10_VALIDATION_20260905.json"))
                for rel in FILES:
                    sftp.get(ROOT + "/" + rel, str(Path("worktrees/pi0-online-bc") / rel))
        if args.mode == "fetch-startup":
            with client.open_sftp() as sftp:
                sftp.get(RUNS["formal"] + "/runtime/startup-verification.json", str(LOCAL / "U10_FORMAL_STARTUP_20260905.json"))
        if args.mode == "publish":
            actual = command(client, f"git -C {ROOT} rev-parse HEAD; git -C {ROOT} status --porcelain").decode().splitlines()
            assert actual == ["cb01451fbbb01f509bde126029a0cec3d577aedb"]
            with client.open_sftp() as sftp:
                proof = json.loads(sftp.open(PACKET + "/smoke-verification.json").read())
                assert proof["passed"]
                sftp.get(PACKET + "/smoke-verification.json", str(LOCAL / "U10_SMOKE_VERIFICATION_20260905.json"))
                sftp.put("worktrees/pi0-online-bc/docs/online_bc.md", ROOT + "/docs/online_bc.md")
                try:
                    sftp.stat(ROOT + "/docs/evidence")
                except FileNotFoundError:
                    sftp.mkdir(ROOT + "/docs/evidence")
                target = ROOT + "/docs/evidence/online-bc-u10-20260905"
                sftp.mkdir(target)
                for name in ["GPU6_U10_RUN_CONTRACT_20260905.md", "U10_SMOKE_RESOLVED_20260905.yaml", "U10_FORMAL_RESOLVED_20260905.yaml", "U10_VALIDATION_20260905.json", "U10_SMOKE_VERIFICATION_20260905.json"]:
                    sftp.put(str(LOCAL / name), target + "/" + name)
                sftp.put("local_scripts/bc_u10_wrapper_20260905.sh", target + "/wrapper.sh")
            command(client, f"set -eu\ncd {ROOT}\ngit diff --check\ngit add -- docs/online_bc.md docs/evidence/online-bc-u10-20260905\ngit commit -m 'Record U10 GPU6 smoke validation and formal run contract'\nenv -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy timeout 90s git -c http.proxy= -c https.proxy= push personal HEAD:refs/heads/codex/sz-pi0-online-bc\ngit rev-parse HEAD\ngit status --porcelain")
        if args.mode == "launch":
            run = RUNS[args.run]
            state = command(client, f"git -C {ROOT} branch --show-current; git -C {ROOT} status --porcelain").decode().splitlines()
            assert state == ["codex/sz-pi0-online-bc"]
            command(client, "nvidia-smi -i 6 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits")
            with client.open_sftp() as sftp:
                for rel in FILES:
                    assert sftp.open(ROOT + "/" + rel).read() == (Path("worktrees/pi0-online-bc") / rel).read_bytes()
                proof = json.loads(sftp.open(PACKET + "/validation.json").read())
                assert proof["passed"] and proof["u"] == 10
                if args.run == "formal":
                    assert sftp.open(RUNS["smoke"] + "/exit_code.txt").read().strip() == b"0"
                    verification = json.loads(sftp.open(PACKET + "/smoke-verification.json").read())
                    assert verification["passed"] and verification["optimizer_updates"] == 20
                sftp.mkdir(run)
                sftp.mkdir(run + "/runtime")
                for local, dest in (
                    ("local_scripts/bc_u10_wrapper_20260905.sh", "wrapper.sh"),
                    ("local_scripts/bc_gpu6_resource_observer_20260905.py", "resource_observer.py"),
                    (str(LOCAL / f"U10_{args.run.upper()}_RESOLVED_20260905.yaml"), "resolved.yaml"),
                    (str(LOCAL / "GPU6_U10_RUN_CONTRACT_20260905.md"), "contract.md"),
                ):
                    sftp.put(local, run + "/runtime/" + dest)
            command(client, f"nohup setsid bash {run}/runtime/wrapper.sh {args.run} > {run}/wrapper.log 2>&1 < /dev/null & p=$!; printf '%s\\n' \"$p\" > {run}/wrapper.pid; nohup /usr/bin/python3 {run}/runtime/resource_observer.py \"$p\" {run} > {run}/observer.log 2>&1 < /dev/null & printf 'WRAPPER_PID=%s OBSERVER_PID=%s\\n' \"$p\" \"$!\"")
    finally:
        client.close()
        os.environ.pop("SEETA_SSH_PASSWORD", None)


if __name__ == "__main__":
    main()
