"""Small, opt-in safeguards for the existing serial experiment queue."""

import errno
import os
from pathlib import Path


def require_save_space(output, reserve_gib):
    """Avoid a long segment when its next complete save cannot fit.

    Reserves are measured per-method checkpoint+policy sizes plus headroom;
    no checkpoint or other user's files are deleted by this check.
    """
    path = Path(output)
    while not path.exists():
        path = path.parent
    disk = os.statvfs(path)
    available = disk.f_bavail * disk.f_frsize
    required = int(reserve_gib * 2**30)
    if available < required:
        raise OSError(
            errno.ENOSPC,
            f"Next complete save needs {reserve_gib:g} GiB; "
            f"only {available / 2**30:.2f} GiB available at {output}",
        )


def record_stage_failure(path, row, save, notify, cleanup, emergency_dir):
    """Disk-full diagnostics must not prevent cleanup and the next method."""
    try:
        try:
            save(path, row)
            notify()
        except OSError as exc:
            if exc.errno not in (errno.ENOSPC, errno.EDQUOT):
                raise
            emergency_dir = Path(emergency_dir)
            emergency_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            if emergency_dir.is_symlink() or emergency_dir.stat().st_uid != os.getuid():
                raise PermissionError("Emergency receipt directory is not owned")
            save(emergency_dir / Path(path).name, dict(row, receipt_write_error=str(exc)))
    finally:
        cleanup()
