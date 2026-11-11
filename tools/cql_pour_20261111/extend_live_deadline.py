"""Bounded bridge for an already-running owner's immutable Popen.wait timeout.

Only its waiting owner may be stopped; simulation/model children keep running.
Always continue that exact owner after the child exits or progress stalls.
Future launches use the corrected plan timeouts and need no such bridge.
"""

import json
import os
import signal
import sys
import time
from pathlib import Path


def identity(pid):
    try:
        p = Path("/proc") / str(pid)
        fields = (p / "stat").read_text().rsplit(") ", 1)[1].split()
        return {
            "pid": pid,
            "uid": p.stat().st_uid,
            "start": int(fields[19]),
            "state": fields[0],
        }
    except (FileNotFoundError, ProcessLookupError):
        return None


def same(who):
    now = identity(who["pid"])
    return (
        now
        and all(now[k] == who[k] for k in ("pid", "uid", "start"))
        and now["state"] not in ("Z", "X")
    )


def bridge(directory, limits, hard_limit=7200, margin=120, poll=2, freshness=300):
    d = Path(directory)
    owner = json.loads((d / "identity.json").read_text())
    assert owner["uid"] == os.getuid()
    paused = False

    def record(event, **extra):
        with (d / "deadline-bridge.jsonl").open("a") as f:
            f.write(json.dumps({"time": time.time(), "event": event, **extra}) + "\n")

    def resume():
        nonlocal paused
        if paused and same(owner):
            os.kill(owner["pid"], signal.SIGCONT)
            record("owner_resumed", owner=owner)
        paused = False

    def stop(signum, _):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    record(
        "watch_started", owner=owner, original_timeouts=limits, hard_limit=hard_limit
    )
    try:
        while same(owner):
            state = json.loads((d / "status.json").read_text())
            phase = state["phase"]
            if phase not in limits:
                break
            elapsed = time.time() - state["started"]
            if elapsed < limits[phase] - margin:
                time.sleep(poll)
                continue
            child = json.loads((d / (phase + "-identity.json")).read_text())
            heartbeat = d / "native-g6/progress.json"
            if not same(child):
                time.sleep(poll)
                continue
            if time.time() - heartbeat.stat().st_mtime > freshness:
                record("no_extension_stale_progress", phase=phase)
                break
            os.kill(owner["pid"], signal.SIGSTOP)
            paused = True
            record("waiting_owner_paused", phase=phase, child=child)
            while same(owner) and same(child):
                if (
                    time.time() - state["started"] >= hard_limit
                    or time.time() - heartbeat.stat().st_mtime > freshness
                ):
                    record("extension_limit_reached", phase=phase)
                    break
                time.sleep(poll)
            resume()
            time.sleep(poll)
        record("watch_finished")
    finally:
        resume()


if __name__ == "__main__":
    bridge(sys.argv[1], {"reference_eval": 3600, "smoke": 2400, "smoke_resume": 2400})
