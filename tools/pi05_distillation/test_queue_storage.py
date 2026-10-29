import errno
from types import SimpleNamespace

import pytest

from queue_storage import record_stage_failure, require_save_space
from single_formal_plan import run_stages


@pytest.mark.parametrize("failure_point", ["save", "notify"])
def test_disk_full_failure_advances_and_cleans(tmp_path, failure_point):
    events = []
    primary, emergency = tmp_path / "home", tmp_path / "root"
    primary.mkdir()

    def save(path, row):
        if failure_point == "save" and path.parent == primary:
            raise OSError(errno.ENOSPC, "full")
        events.append(("record", path.parent.name))

    def notify():
        if failure_point == "notify":
            raise OSError(errno.ENOSPC, "full")

    def run(method):
        events.append(("run", method))
        if method == "first":
            raise OSError(errno.ENOSPC, "checkpoint save full")
        return "ok"

    def failed(method, exc):
        record_stage_failure(primary / "failed.json", {"error": str(exc)}, save,
                             notify, lambda: events.append(("cleanup", method)), emergency)

    result = run_stages(["first", "next"], run, failed)
    assert result["first"]["status"] == "failed"
    assert result["next"]["status"] == "complete"
    assert events.index(("cleanup", "first")) < events.index(("run", "next"))
    assert ("record", "root") in events


def test_other_write_error_still_cleans(tmp_path):
    cleaned = []
    def save(*_):
        raise PermissionError(errno.EACCES, "not disk pressure")
    with pytest.raises(PermissionError):
        record_stage_failure(tmp_path / "f.json", {}, save, lambda: None,
                             lambda: cleaned.append(True), tmp_path / "root")
    assert cleaned == [True]


def test_capacity_uses_unprivileged_available_blocks(tmp_path, monkeypatch):
    monkeypatch.setattr("queue_storage.os.statvfs", lambda _: SimpleNamespace(f_bavail=1024, f_frsize=1024**2))
    require_save_space(tmp_path / "future", 1)
    with pytest.raises(OSError) as exc:
        require_save_space(tmp_path / "future", 2)
    assert exc.value.errno == errno.ENOSPC
