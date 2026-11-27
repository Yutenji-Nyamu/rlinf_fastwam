"""Keep a running pair's queue position during GPU-OOM recovery."""
import argparse
import fcntl
import json
import os
from pathlib import Path
from datetime import datetime
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback

QUEUE_OPS = os.environ.get("QUEUE_OPS", "")
ACTIVE_INDEX = Path(os.environ.get("ACTIVE_RUN_INDEX", "active-runs.json"))
from oom_classify_1127 import driver_failure, find_gpu_oom
from oom_contention import retry_delay

FORMAL_LOGS = ("formal-driver.log", "formal-retry.log")
NATIVE_LOGS = ("native-g5.log", "native-g6.log")
UTC_STAMP = re.compile(r"(?<!\d)(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?(?:Z|[+-]\d\d:\d\d))")


def retry_transport(lines):
    _, oom, _, transport = driver_failure(lines)
    return transport and not oom


def log_positions(pair):
    positions = {}
    for name in (*FORMAL_LOGS, *NATIVE_LOGS):
        path = pair / name
        if path.is_file():
            st = path.stat()
            positions[name] = (st.st_dev, st.st_ino, st.st_size)
    return positions


def appended_lines(path, positions):
    st = path.stat()
    prior = positions.get(path.name)
    offset = prior[2] if prior and prior[:2] == (st.st_dev, st.st_ino) else 0
    with path.open("rb") as stream:
        stream.seek(offset if offset <= st.st_size else 0)
        for line in stream:
            yield line.decode("utf-8", errors="replace")


def native_lines(path, positions, step, checkpoint_root):
    lines = appended_lines(path, positions)
    if step is None:
        yield from lines
        return
    if checkpoint_root is None:
        return
    marker = checkpoint_root / f"global_step_{step}" / "dojo-complete.json"
    try:
        done = json.loads(marker.read_text())
    except (FileNotFoundError, ValueError):
        return
    if done.get("step") != step or not isinstance(done.get("time"), (int, float)):
        return
    # Only timestamps on the actual error can order native output against the completed round.
    for line in lines:
        stamp = UTC_STAMP.search(line)
        if stamp and datetime.fromisoformat(stamp[1].replace("Z","+00:00")).timestamp() > done["time"]:
            yield line


def oom_evidence(pair, positions, released, driver_started, kind=None, native_during_formal=False, checkpoint_root=None):
    terminal = released.get("result", released)
    error = str(terminal.get("error", ""))
    if terminal.get("phase") != "FAILED_RETURN_RLT" or any(
            token in error for token in ("SystemExit", "KeyboardInterrupt", "SIGTERM")):
        return None
    # A transport retry belongs to the original owner. Its final driver log
    # takes precedence, even if the first attempt reported a GPU allocation error.
    for name in reversed(FORMAL_LOGS):
        path = pair / name
        if not path.is_file():
            continue
        st = path.stat()
        if st.st_mtime + 1 < driver_started:
            continue
        step, match, rpc_failure, _ = driver_failure(appended_lines(path, positions))
        if match:
            retry_receipt = pair / "simulator-retry.json"
            first = pair / FORMAL_LOGS[0]
            if kind == "fastwam" and name == "formal-retry.log" and first.is_file() and retry_receipt.is_file():
                receipt = json.loads(retry_receipt.read_text())
                if receipt.get("failed_log") == first.name and driver_failure(appended_lines(first, positions))[1]:
                    # This already-running owner used its internal transport retry for the same GPU OOM.
                    match["internal_oom_retry"] = step is None
            return {"log": name, **match}
        if kind == "dojo" and rpc_failure and native_during_formal:
            for native in NATIVE_LOGS:
                path = pair / native
                if path.is_file() and path.stat().st_mtime + 1 >= driver_started:
                    match = find_gpu_oom(native_lines(path, positions, step, checkpoint_root))
                    if match:
                        return {"log": native, "driver_log": name, "driver_rpc_error": rpc_failure, **match}
        return None
    return None


def complete_checkpoint(root, kind):
    marked = [(int(cp.name[12:]), cp) for cp in root.glob("global_step_*")
              if cp.name[12:].isdigit() and (cp / "dojo-complete.json").is_file()]
    required = (["actor/dcp_checkpoint/.metadata",
                 *[f"actor/dcp_checkpoint/__{i}_0.distcp" for i in (0, 1)]]
                if kind == "fastwam" else
                [f"actor/local_shard_checkpoint/checkpoint_rank_{i}.pt" for i in (0, 1)])
    for step, cp in sorted(marked, reverse=True):
        try:
            if int(json.loads((cp / "dojo-complete.json").read_text())["step"]) == step and all(
                    (cp / p).is_file() and (cp / p).stat().st_size > 0 for p in required):
                return step, cp
        except (FileNotFoundError, ValueError, TypeError, KeyError):
            continue
    raise ValueError(f"No complete two-rank checkpoint in {root}")


def live_driver(pair, kind, read, same):
    identity = pair / ("driver-identity.json" if kind == "fastwam" else "formal-driver-identity.json")
    if identity.is_file() and same(read(identity)):
        return True
    # Handle an owner dying between spawning its driver and recording the PID.
    for comm in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            if comm.stat().st_uid != os.getuid():
                continue
            argv = comm.read_bytes().split(bytes([0]))
        except OSError:
            continue
        if kind == "fastwam" and os.fsencode(pair / "fastwam_driver.py") in argv:
            return True
        if kind == "dojo" and os.fsencode(pair / "config") in argv and b"formal" in argv and any(
                a.endswith(b"/tools/dojo_grpo/training_entry.py") for a in argv):
            return True
    return False


def original_result(pair, kind, origin, state, read, same, save):
    receipt = pair / "released.json"
    if receipt.is_file():
        return read(receipt)
    if live_driver(pair, kind, read, same):
        state("WAIT_LIVE_ORPHAN_DRIVER", owner=origin)
        deadline = time.monotonic() + 120
        while not receipt.is_file() and live_driver(pair, kind, read, same):
            if time.monotonic() >= deadline:
                result = {"phase": "LIVE_ORPHAN_DRIVER", "owner": origin}
                completed = {"time": time.time(), "result": result, "owner": origin}
                save(receipt, completed)
                return completed
            time.sleep(3)
    if receipt.is_file():
        return read(receipt)
    native = [read(path) for path in pair.glob("native-g*-*.json")
              if path.name.endswith(("-identity.json", "-child.json"))]
    native_alive = [who["pid"] for who in native if "uid" in who and same(who)]
    current = read(pair / "status.json")
    if native_alive:
        # The original admission code will wait for device cleanup. Do not start a retry alongside an orphan.
        current = dict(phase="UNFINISHED_NATIVE_CLEANUP", prior_phase=current["phase"], pids=native_alive)
    elif current.get("phase") in ("FORMAL_TRAINING", "INITIAL_EVALUATION"):
        current = dict(phase="FAILED_RETURN_RLT", error="Owner and formal driver exited without final receipt")
    completed = {"time": time.time(), "result": current, "owner": origin, "reconstructed_after_exit": True}
    save(receipt, completed)
    return completed


def replace_active_identity(pair, cards, old, new, read, save):
    with ACTIVE_INDEX.with_suffix(".continuation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        index = read(ACTIVE_INDEX)
        touched = []
        for gpu in cards:
            row = index["slots"][str(gpu)]
            if row.get("owner") == str(pair) and row.get("identity") == old:
                row["identity"] = new
                touched.append(gpu)
        if touched:
            index["updated"] = time.time()
            save(ACTIVE_INDEX, index)
        return touched


def run(gate, pair, kind):
    gate.mkdir(parents=True, exist_ok=True)
    with (gate / "owner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run_locked(gate, pair, kind)


def run_locked(gate, pair, kind):
    from common import proc, read, same, save
    from omegaconf import OmegaConf

    if (gate / "released.json").exists() or (gate / "attempt-1").exists():
        raise RuntimeError("Gate already used; inspect its recorded attempt")
    plan = read(pair / "plan.json")
    expected = [4, 7] if kind == "fastwam" else [5, 6]
    assert sorted(plan["gpus"]) == expected and os.getuid() == 1003
    origin = read(pair / "identity.json")
    assert same(origin) and origin["uid"] == os.getuid()
    assert read(pair / "registered.json")["identity"] == origin and not (pair / "released.json").exists()
    positions = log_positions(pair)
    started = next(int(s.split()[1]) for s in Path("/proc/stat").read_text().splitlines()
                   if s.startswith("btime ")) + origin["start"] / os.sysconf("SC_CLK_TCK")
    own = proc(os.getpid())
    def state(phase, **extra):
        save(gate / "status.json", dict(phase=phase, time=time.time(), **extra))
    def stop(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    registered = False
    original_release = None
    child = None
    attempt = None
    outcome = "INTERRUPTED"
    detail = {}
    try:
        state("ADOPTING", owner=origin, deployment=str(pair))
        save(gate / "identity.json", own)
        deadline = time.monotonic() + 30
        while not (gate / "registered.json").is_file():
            if time.monotonic() >= deadline:
                raise TimeoutError("Queue did not register this CPU observer")
            time.sleep(.25)
        assert read(gate / "registered.json")["identity"] == own
        registered = True
        state("MONITORING_ORIGINAL", owner=origin)
        formal_path = pair / "formal-driver.log"
        native_during_formal = "formal-driver.log" in positions and formal_path.stat().st_mtime + 1 >= started
        while same(origin):
            if kind == "dojo" and not native_during_formal and formal_path.is_file():
                # Start counting native output only after the formal driver exists.
                positions.update({n:pos for n,pos in log_positions(pair).items() if n in NATIVE_LOGS})
                native_during_formal = True
            time.sleep(2)
        attempts, same_step_retries, contention_streak, last_step = 0, 0, 0, None
        while True:
            assert read(pair / "identity.json") == origin
            original_release = original_result(pair, kind, origin, state, read, same, save)
            checkpoint_root = (Path(OmegaConf.load(pair / "config/formal.yaml").runner.resume_dir).parent
                               if kind == "dojo" else None)
            evidence = oom_evidence(pair, positions, original_release, started, kind, native_during_formal,
                                    checkpoint_root)
            if not evidence:
                if attempts:
                    outcome = "RETRY_FINISHED"
                    detail.update(result=original_release)
                else:
                    outcome = "ORIGINAL_RELEASED"
                    detail = {"original_phase": original_release.get("result", original_release).get("phase")}
                return
            cfg_file = pair / "config/formal.yaml"
            cfg = OmegaConf.load(cfg_file)
            step, cp = complete_checkpoint(Path(cfg.runner.resume_dir).parent, kind)
            if step >= int(cfg.runner.max_steps):
                outcome = "TARGET_REACHED"
                detail = {"step": step}
                return
            if step != last_step:
                same_step_retries, contention_streak = 0, 0
            last_step = step
            same_step_retries += int(bool(evidence.get("internal_oom_retry")))
            delay, attribution = retry_delay(evidence, expected, same_step_retries, contention_streak)
            if delay is None:
                outcome = "RETRY_FINISHED" if attempts else "ORIGINAL_RELEASED"
                detail.update(result=original_release, resume=step, reason=attribution)
                return
            attempts += 1
            same_step_retries += 1
            contention_streak = contention_streak + 1 if attribution.get("external") else 0
            attempt = gate / f"attempt-{attempts}"
            attempt.mkdir()
            save(attempt / "reason.json", dict(checkpoint=str(cp), evidence=evidence, original_owner=origin,
                                                attribution=attribution, delay_seconds=delay))
            if delay:
                state("OOM_RETRY_BACKOFF", attempt=attempts, seconds=delay, reason=attribution)
                time.sleep(delay)
            shutil.copy2(cfg_file, attempt / "formal.before.yaml")
            state("PREPARING_OOM_RETRY", checkpoint=str(cp), evidence=evidence)
            # All current jobs use launchers that create fresh logs exclusively.
            # Archive only completed control artifacts in the original directory
            # to preserve atomic renames even when the run and gate use different mounts.
            archive = pair / f"prior-oom-retry-{own['pid']}-{own['start']}-{attempts}"
            archive.mkdir()
            names = ["released.json", "identity.json", "registered.json", "status.json",
                     "formal-driver.log", "formal-retry.log", "driver-identity.json",
                     "formal-driver-identity.json", "first-resumed-update.json", "simulator-retry.json"]
            if kind == "dojo":
                names += ["native-g5.log", "native-g6.log"]
            for name in names:
                source = pair / name
                if source.is_file():
                    source.rename(archive / name)
            cfg.runner.resume_dir = str(cp)
            cfg_tmp = cfg_file.with_name(".formal.oom-retry.yaml")
            OmegaConf.save(cfg, cfg_tmp)
            os.replace(cfg_tmp, cfg_file)
            launcher = pair / "fastwam_pair.py" if kind == "fastwam" else Path(
                plan["training_source"]) / "tools/dojo_grpo/pair_owner.py"
            started = time.time()
            with (attempt / "owner.log").open("x") as stream:
                child = subprocess.Popen([plan["python"], "-uB", str(launcher), str(pair)],
                    cwd=plan["training_source"], env=dict(os.environ, CUDA_VISIBLE_DEVICES="",
                    PYTHONPATH=plan["training_source"], PAIR_OOM_RETRY_RULES=str(Path(__file__).resolve().parent), OOM_RETRY_OWNER=str(pair)),
                    stdin=subprocess.DEVNULL,
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + 30
            while not (pair / "identity.json").is_file():
                if time.monotonic() >= deadline:
                    raise TimeoutError("Retry owner did not publish its identity")
                if child.poll() is not None:
                    raise RuntimeError(f"Retry owner exited before registration: {child.returncode}")
                time.sleep(.2)
            retry = read(pair / "identity.json")
            assert retry["pid"] == child.pid and same(retry)
            save(pair / "registered.json", {"identity": retry})
            try:
                updated = replace_active_identity(pair, expected, origin, retry, read, save)
            except (OSError, KeyError, ValueError) as exc:
                updated = []
                print(f"Active index refresh: {exc}", file=sys.stderr, flush=True)
            state("RUNNING_OOM_RETRY", owner=retry, resume=step, archive=str(archive), indexed_cards=updated)
            detail = {"returncode": child.wait(), "resume": step, "attempts": attempts}
            origin = retry
            positions = {}  # Previous attempt files were archived before this launch.
            native_during_formal = (pair / "formal-driver.log").is_file()
    except SystemExit:
        outcome = "OPERATOR_STOP"
        raise
    except Exception:
        outcome = "GATE_ERROR"
        detail = {"error": traceback.format_exc()}
        raise
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                detail["cleanup"] = "Retry owner still holds its slot after SIGTERM"
        if original_release is not None and not (pair / "released.json").is_file():
            save(pair / "released.json", original_release)
        result = dict(phase=outcome, time=time.time(), **detail)
        save(gate / "status.json", result)
        if registered:
            save(gate / "released.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gate", type=Path)
    parser.add_argument("pair", type=Path)
    parser.add_argument("kind", choices=("fastwam", "dojo"))
    args = parser.parse_args()
    sys.path.insert(0, QUEUE_OPS)
    run(args.gate.resolve(), args.pair.resolve(), args.kind)
