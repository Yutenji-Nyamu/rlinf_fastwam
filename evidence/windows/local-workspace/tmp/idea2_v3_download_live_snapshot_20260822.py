from __future__ import annotations

import getpass
import os
from pathlib import Path
import posixpath
import stat
import sys
from types import SimpleNamespace

WORKSPACE = Path(r"C:\Users\86136\Documents\rl")
sys.path.insert(0, str(WORKSPACE / "local_scripts"))
import remote_exec_autodl  # noqa: E402

REMOTE_RUN = "/root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_r_only_v3_w0to2_formal_100step_2gpu16env_20260822"
REMOTE_RUNTIME = "/root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_r_only_v3_w0to2_formal_100step_2gpu16env_20260822"
DEST = WORKSPACE / "docs" / "rlinf-robotwin-pi0-dvac-telemetry" / "evidence" / "v3_formal_live_g1_20260822" / "raw"


def walk_files(sftp, root: str):
    for entry in sftp.listdir_attr(root):
        path = posixpath.join(root, entry.filename)
        if stat.S_ISDIR(entry.st_mode):
            yield from walk_files(sftp, path)
        else:
            yield path, entry.st_size


def main() -> None:
    password = [REDACTED]"SSH password: ")
    os.environ["SEETA_SSH_PASSWORD"] = password
    args = SimpleNamespace(
        host=remote_exec_autodl.DEFAULT_HOST,
        port=remote_exec_autodl.DEFAULT_PORT,
        user=remote_exec_autodl.DEFAULT_USER,
        host_key_sha256=remote_exec_autodl.DEFAULT_HOST_KEY_SHA256,
        timeout=20.0,
    )
    client = remote_exec_autodl.connect(args)
    try:
        with client.open_sftp() as sftp:
            files: list[tuple[str, str, int]] = []
            for root, prefix in ((REMOTE_RUN, "run"), (REMOTE_RUNTIME, "runtime")):
                for remote, size in walk_files(sftp, root):
                    rel_remote = posixpath.relpath(remote, root)
                    keep = False
                    if prefix == "run":
                        keep = (
                            rel_remote == "metrics.log"
                            or rel_remote.startswith("tensorboard/")
                            or rel_remote.startswith("dvac_train/")
                            or rel_remote.startswith("control_trace/")
                        )
                    else:
                        keep = (
                            rel_remote in {
                                "driver.log",
                                "wrapper.log",
                                "observer.log",
                                "resolved_config.yaml",
                                "launch_command.txt",
                                "launch_started_at.txt",
                                "wrapper.pid",
                                "driver.pid",
                                "observer.pid",
                            }
                            or rel_remote.startswith("resource_monitor/")
                        )
                    if keep:
                        files.append((remote, f"{prefix}/{rel_remote}", size))

            DEST.mkdir(parents=True, exist_ok=True)
            total = 0
            for remote, rel, size in sorted(files, key=lambda item: item[1]):
                local = DEST / Path(rel)
                local.parent.mkdir(parents=True, exist_ok=True)
                sftp.get(remote, str(local))
                actual = local.stat().st_size
                print(f"GET {rel} remote={size} local={actual}")
                total += actual
            print(f"SNAPSHOT_FILES={len(files)}")
            print(f"SNAPSHOT_BYTES={total}")
            print(f"DEST={DEST}")
    finally:
        password = ""
        os.environ.pop("SEETA_SSH_PASSWORD", None)
        client.close()


if __name__ == "__main__":
    main()
