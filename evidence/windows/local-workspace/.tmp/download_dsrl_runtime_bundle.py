from __future__ import annotations

import hashlib
import importlib.util
import os
import shlex
from pathlib import Path
from types import SimpleNamespace


workspace = Path(r"C:\Users\86136\Documents\rl")
helper_path = workspace / "local_scripts" / "remote_exec_autodl.py"
spec = importlib.util.spec_from_file_location("remote_exec_autodl", helper_path)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load {helper_path}")
remote = importlib.util.module_from_spec(spec)
spec.loader.exec_module(remote)

args = SimpleNamespace(
    host=remote.DEFAULT_HOST,
    port=remote.DEFAULT_PORT,
    user=remote.DEFAULT_USER,
    host_key_sha256=remote.DEFAULT_HOST_KEY_SHA256,
    timeout=20.0,
)
remote_path = (
    "/root/autodl-tmp/experiment_exports/"
    "dsrl_pi0_robotwin_formal_v1_20260729/"
    "dsrl_pi0_robotwin_formal_v1_runtime_step198_20260729.tar.gz"
)
destination = (
    workspace
    / "exports"
    / "dsrl_pi0_robotwin_formal_v1_runtime_step198_20260729.tar.gz"
)
temporary = destination.with_suffix(destination.suffix + ".partial")
checksum_destination = destination.with_suffix(destination.suffix + ".sha256")
checksum_temporary = checksum_destination.with_suffix(
    checksum_destination.suffix + ".partial"
)

client = remote.connect(args)
try:
    quoted = shlex.quote(remote_path)
    stdin, stdout, stderr = client.exec_command(
        f"stat -c '%s' -- {quoted}; sha256sum -- {quoted}"
    )
    stdin.channel.shutdown_write()
    metadata_lines = stdout.read().decode("utf-8", errors="strict").splitlines()
    metadata_error = stderr.read().decode("utf-8", errors="replace")
    metadata_status = stdout.channel.recv_exit_status()
    if metadata_status != 0 or len(metadata_lines) != 2:
        raise RuntimeError(
            f"metadata query failed ({metadata_status}): {metadata_error}"
        )
    expected_size = int(metadata_lines[0])
    expected_sha256 = metadata_lines[1].split()[0]

    stdin, stdout, stderr = client.exec_command(f"cat -- {quoted}")
    stdin.channel.shutdown_write()
    digest = hashlib.sha256()
    received = 0
    with temporary.open("wb") as handle:
        while chunk := stdout.read(1024 * 1024):
            handle.write(chunk)
            digest.update(chunk)
            received += len(chunk)
            print(f"received={received}/{expected_size}", flush=True)
    download_error = stderr.read().decode("utf-8", errors="replace")
    download_status = stdout.channel.recv_exit_status()
    actual_sha256 = digest.hexdigest()
    if download_status != 0:
        raise RuntimeError(
            f"archive stream failed ({download_status}): {download_error}"
        )
    if received != expected_size:
        raise RuntimeError(
            f"size mismatch: received={received}, expected={expected_size}"
        )
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"sha256 mismatch: actual={actual_sha256}, expected={expected_sha256}"
        )

    checksum_temporary.write_text(
        f"{expected_sha256}  {destination.name}\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    os.replace(checksum_temporary, checksum_destination)
    print(f"DOWNLOAD=PASS bytes={received} sha256={actual_sha256}", flush=True)
finally:
    client.close()
