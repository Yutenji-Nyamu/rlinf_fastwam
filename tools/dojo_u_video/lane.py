"""Finite current-inference -> Dojo recording -> existing per-card RLT queue."""

import fcntl
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

from common import read, save
from run_task import ops


def main():
    d = Path(sys.argv[1])
    g = int(sys.argv[2])
    plan = read(d / f"lane-g{g}-plan.json")
    o = ops()
    assert os.getuid() == 20001 and g in (6, 7)
    me = o.proc(os.getpid())
    save(d / f"lane-g{g}-identity.json", me)
    state = d / f"lane-g{g}-status.json"
    child = None
    predecessor_done = False
    release_allowed = True
    predecessor = plan["predecessor"]

    def stop(*_):
        if child and child.poll() is None:
            o.exact_signal(o.proc(child.pid), signal.SIGTERM)
            child.wait(timeout=90)
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, stop)
    try:
        save(
            state,
            dict(
                time=time.time(),
                phase="WAIT_PREDECESSOR",
                predecessor=predecessor["external_release"],
            ),
        )
        while (
            o.same(predecessor["external_identity"])
            or not Path(predecessor["external_release"]).is_file()
        ):
            time.sleep(15)
        assert not o.gpus()[g]["processes"]
        predecessor_done = True
        assert read(d / "smoke-accepted.json")["passed"]
        for config in plan["jobs"]:
            c = read(config)
            assert c["gpu"] == g and not Path(c["output"]).exists()
            assert not o.gpus()[g]["processes"]
            with (d / f"{Path(c['output']).name}-driver.log").open("x") as log:
                child = subprocess.Popen(
                    [
                        "/usr/bin/python3",
                        "-u",
                        "-B",
                        str(Path(__file__).with_name("run_task.py")),
                        config,
                    ],
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                save(
                    state,
                    dict(
                        time=time.time(),
                        phase="RUNNING",
                        config=config,
                        child=o.proc(child.pid),
                    ),
                )
                rc = child.wait()
            if rc == 79:
                release_allowed = False
            assert not o.gpus()[g]["processes"], (
                "Cleanup failed; no automatic RLT launch"
            )
            result = Path(c["output"]) / "result.json"
            if rc or not result.is_file():
                raise RuntimeError(
                    f"Recording/cleanup/device fault for {config}, rc={rc}"
                )
            with (d / "index.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                rows = [
                    read(p)
                    for p in sorted(Path(plan["output"]).glob("tasks/*/result.json"))
                ]
                save(
                    Path(plan["output"]) / "summary.json",
                    dict(
                        time=time.time(),
                        tasks=len(rows),
                        episodes=sum(r["episodes"] for r in rows),
                        successes=sum(r["successes"] for r in rows),
                        rows=rows,
                    ),
                )
        save(state, dict(time=time.time(), phase="COMPLETE"))
    except BaseException:
        save(
            d / f"lane-g{g}-error.json",
            dict(time=time.time(), error=traceback.format_exc()),
        )
        raise
    finally:
        # Do not release a failed waiting bridge while predecessor still owns GPU.
        if predecessor_done and release_allowed and not o.gpus()[g]["processes"]:
            save(
                d / f"lane-g{g}-released.json",
                dict(
                    time=time.time(),
                    identity=me,
                    clear=True,
                    resume="original unchanged per-card RLT fallback",
                ),
            )


if __name__ == "__main__":
    main()
