"""Task-scoped transfer plus one SSH command; password remains process-only."""
import argparse
import getpass
import os
from pathlib import Path
from remote_exec_autodl import connect, run_command

parser = argparse.ArgumentParser()
parser.add_argument("--command-file", required=True)
parser.add_argument("--put", nargs=2, action="append", default=[])
parser.add_argument("--get", nargs=2, action="append", default=[])
parser.add_argument("--replace-own-trial-script", action="store_true")
args = parser.parse_args()
args.host = "120.241.223.9"
args.port = 22
args.user = "chenyiteng"
args.host_key_sha256 = "qw92OXne52y6NsQMQ6+PdjOu0Hosc4ZX78XmkYxcuEY"
args.timeout = 20
args.command = None
args.pty = False
for name in ("stdin_file", "stdin_password", "stdin_git_diff", "stdin_git_full_diff", "stdin_git_tree_diff"):
    setattr(args, name, None)
os.environ["SEETA_SSH_PASSWORD"] = getpass.getpass("SSH password: ")
client = connect(args)
try:
    with client.open_sftp() as sftp:
        for local, remote in args.put:
            try:
                sftp.stat(remote)
            except FileNotFoundError:
                pass
            else:
                if not (args.replace_own_trial_script and remote == "/data/chenyiteng/projects/rlinf-shenzhen/worktrees/robotwin-oidn-toggle-20260904/oidn_toggle_trial.py"):
                    raise RuntimeError(f"Refusing to overwrite remote file: {remote}")
            sftp.put(local, remote)
    code = run_command(client, args)
    if code:
        raise SystemExit(code)
    with client.open_sftp() as sftp:
        for remote, local in args.get:
            Path(local).parent.mkdir(parents=True, exist_ok=True)
            sftp.get(remote, local)
    raise SystemExit(code)
finally:
    client.close()
    os.environ.pop("SEETA_SSH_PASSWORD", None)
