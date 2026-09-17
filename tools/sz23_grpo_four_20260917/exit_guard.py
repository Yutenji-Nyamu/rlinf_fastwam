"""One-shot companion: preserve exit evidence and clean only this frozen Ray job.

This script never stops Ray, signals a process, changes training files or restarts
training. While the original driver lives it only reads its /proc identity.
"""
import argparse
from dataclasses import asdict, is_dataclass
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ST = Path("/data/chenyiteng/deployment-20260917/grpo-four")
UID = 20001


def now():
    return datetime.datetime.now().astimezone().isoformat()


def read(path):
    return Path(path).read_text(encoding="utf-8")


def load(path):
    return json.loads(read(path))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def proc(pid):
    path = Path("/proc") / str(pid)
    try:
        fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
        return {"pid": int(pid), "uid": path.stat().st_uid, "start": int(fields[19]),
                "state": fields[0], "ppid": int(fields[1])}
    except (FileNotFoundError, ProcessLookupError):
        return None


def same(identity):
    current = proc(identity["pid"])
    return bool(current and current["state"] != "Z"
                and all(current[key] == identity[key] for key in ("pid", "uid", "start")))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def row_dict(row):
    if is_dataclass(row):
        return asdict(row)
    if isinstance(row, dict):
        return dict(row)
    raise TypeError("Unexpected Ray state row type: " + type(row).__name__)


def environment(runtime):
    result = os.environ.copy()
    result.update(load(runtime / "environment.json"))
    result["CUDA_VISIBLE_DEVICES"] = ""
    result["OMP_NUM_THREADS"] = "1"
    result["PYTHONDONTWRITEBYTECODE"] = "1"
    for key in ("NO_PROXY", "no_proxy"):
        result[key] = ",".join(dict.fromkeys((result.get(key, "") + ",127.0.0.1,localhost," + socket.gethostname()).strip(",").split(",")))
    return result


def snapshot_subprocess(runtime, contract):
    script = '''import json,sys,ray
from dataclasses import asdict,is_dataclass
from ray.util.state import list_actors
address,namespace=sys.argv[1:]
try:
 ray.init(address=address,namespace="sz23_exit_guard_install_check",logging_level="ERROR",log_to_driver=False)
 rows=list_actors(filters=[("ray_namespace","=",namespace)],detail=True,limit=10000,timeout=20)
 rows=[asdict(row) if is_dataclass(row) else dict(row) for row in rows if row["state"]!="DEAD"]
 nodes=[n for n in ray.nodes() if n.get("Alive")]
 print("EXIT_GUARD_CHECK="+json.dumps({"actors":rows,"nodes":[{"node_id":n["NodeID"],"gpus":n.get("Resources",{}).get("GPU",0)} for n in nodes]}))
finally:
 if ray.is_initialized():ray.shutdown()
'''
    env = environment(runtime)
    python = str(Path(env["VIRTUAL_ENV"]) / "bin/python")
    output = subprocess.check_output([python, "-B", "-c", script, contract["ray_address"], contract["namespace"]],
                                     text=True, timeout=60, env=env)
    rows = [line.split("=", 1)[1] for line in output.splitlines() if line.startswith("EXIT_GUARD_CHECK=")]
    require(len(rows) == 1, "Incomplete guard installation Ray query")
    return json.loads(rows[0])


def source_check(runtime, contract):
    repo = Path(contract["repo"])
    require(contract["uid"] == os.getuid() == UID, "Expected chenyiteng UID20001")
    require(contract["hostname"] == socket.gethostname(), "Wrong host")
    require(contract["ray_address"] == "127.0.0.1:26379", "Wrong Ray endpoint")
    require(repo.resolve().is_relative_to(Path("/data/chenyiteng").resolve()), "Source escaped personal storage")
    require(runtime.resolve() == (Path(contract["run"]) / "runtime").resolve(), "Runtime escaped this run")
    require(Path(contract["run"]).resolve().is_relative_to(Path("/data/chenyiteng").resolve()), "Run escaped personal storage")
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    require(head == contract["head"], "Production HEAD changed")
    require(not subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip(), "Production worktree is dirty")
    for relative, expected in contract["source_sha256"].items():
        path = repo / relative
        require(path.resolve().is_relative_to(repo.resolve()), "Source manifest path escaped repo")
        require(sha(path) == expected, "Production source changed: " + relative)
    for name, expected in contract["prepared_sha256"].items():
        require((runtime / name).resolve().is_relative_to(runtime.resolve()), "Prepared path escaped runtime")
        require(sha(runtime / name) == expected, "Prepared file changed: " + name)
    return head


def actor_snapshot(ray, namespace):
    from ray.util.state import list_actors
    rows = [row_dict(row) for row in list_actors(filters=[("ray_namespace", "=", namespace)], detail=True, limit=10000, timeout=20)
            if row["state"] != "DEAD"]
    names = {row["name"] for row in ray.util.list_named_actors(all_namespaces=True) if row["namespace"] == namespace}
    return rows, names


def verify_targets(rows, names, guard):
    require(names == {row["name"] for row in rows if row.get("name")}, "Named actor and state views differ")
    require(all(row.get("name") and row["ray_namespace"] == guard["namespace"]
                and row["job_id"] == guard["job_id"] for row in rows), "Actor namespace/job ownership mismatch")
    identities = [proc(row["pid"]) if row.get("pid") else None for row in rows]
    require(all(identity and identity["uid"] == UID and identity["state"] != "Z" for identity in identities), "Actor process UID/liveness not proven")
    return identities


def capture_exit_evidence(runtime, guard):
    # Wrapper writes the immutable original return code shortly after driver exit.
    for _ in range(20):
        if (runtime / "exit_code.txt").exists():
            break
        time.sleep(1)
    log_path = runtime / "driver.log"
    content = log_path.read_text(errors="replace") if log_path.exists() else ""
    tokens = ("Traceback (most recent call last)", "TypeError:", "cleanup_owned", "attribute name must be string", "RuntimeError:", "AssertionError:")
    lines = [line for line in content.splitlines() if any(token in line for token in tokens)]
    original = read(runtime / "exit_code.txt").strip() if (runtime / "exit_code.txt").exists() else None
    return {"time": now(), "driver": guard["driver"], "driver_alive": same(guard["driver"]),
            "original_exit_code": original, "original_finished": load(runtime / "finished.json") if (runtime / "finished.json").exists() else None,
            "driver_log_sha256": sha(log_path) if log_path.exists() else None,
            "driver_log_bytes": log_path.stat().st_size if log_path.exists() else None,
            "cleanup_error_lines": lines[-50:], "driver_log_tail": content[-6000:],
            "original_files_unchanged": True}


def run_guard(directory):
    require(os.getuid() == UID, "Expected chenyiteng UID20001")
    guard = load(directory / "guard-contract.json")
    require(guard["uid"] == UID and guard["hostname"] == socket.gethostname(), "Guard host/UID mismatch")
    runtime = Path(guard["run"]) / "runtime"
    require(directory.resolve() == (runtime / "exit-guard-20260917").resolve(), "Guard directory escaped run")
    require(sha(runtime / "contract.json") == guard["runtime_contract_sha256"], "Frozen runtime contract changed")
    require(sha(__file__) == guard["guard_script_sha256"], "Guard source changed")
    require(guard["driver"]["uid"] == UID and guard["ray_address"] == "127.0.0.1:26379", "Guard target ownership mismatch")
    require(not (directory / "guard-ready.json").exists() and not (directory / "guard-result.json").exists(), "Guard already started")
    save(directory / "guard-ready.json", {"time": now(), "passed": True, "identity": proc(os.getpid()),
                                         "namespace": guard["namespace"], "job_id": guard["job_id"],
                                         "driver": guard["driver"], "guard_contract_sha256": sha(directory / "guard-contract.json")})
    print(json.dumps({"time": now(), "status": "waiting_for_frozen_driver", "pid": guard["driver"]["pid"]}), flush=True)
    while same(guard["driver"]) or same(guard["wrapper"]):
        time.sleep(20)
    try:
        require(sha(runtime / "contract.json") == guard["runtime_contract_sha256"], "Runtime contract changed while waiting")
        require(sha(__file__) == guard["guard_script_sha256"], "Guard source changed while waiting")
        evidence = capture_exit_evidence(runtime, guard)
        require(not evidence["driver_alive"], "Original driver still alive")
        save(directory / "original-exit-evidence.json", evidence)
        os.environ.update(environment(runtime))
        import ray
        ray.init(address=guard["ray_address"], namespace=guard["namespace"] + "_exit_guard", logging_level="ERROR", log_to_driver=False)
        try:
            for attempt in range(20):
                rows, names = actor_snapshot(ray, guard["namespace"])
                try:
                    identities = verify_targets(rows, names, guard)
                    break
                except AssertionError:
                    if attempt == 19:
                        raise
                    time.sleep(1)
            save(directory / "cleanup-targets.json", {"time": now(), "namespace": guard["namespace"],
                                                      "job_id": guard["job_id"], "actors": rows, "identities": identities})
            managers = {"NodeManager", "WorkerManager", "CollectiveManager", "DeviceLockManager", "PortLockManager"}
            by_name = {row["name"]: (row, identity) for row, identity in zip(rows, identities)}
            killed, vanished = [], []
            for name in sorted(names, key=lambda value: (value in managers, value)):
                row, identity = by_name[name]
                try:
                    actor = ray.get_actor(name, namespace=guard["namespace"])
                except ValueError:
                    vanished.append(name)
                    continue
                require(same(identity), "Actor PID changed immediately before cleanup: " + name)
                require(actor._actor_id.hex() == row["actor_id"], "Named actor identity changed before cleanup")
                ray.kill(actor, no_restart=True)
                killed.append(name)
            for _ in range(30):
                remaining, remaining_names = actor_snapshot(ray, guard["namespace"])
                if not remaining and not remaining_names:
                    break
                time.sleep(1)
            require(not remaining and not remaining_names, "Own namespace still has residual actors")
            result = {"time": now(), "passed": True, "status": "passed", "namespace": guard["namespace"],
                      "job_id": guard["job_id"], "natural_cleanup": not rows, "killed_names": killed,
                      "vanished_before_kill": vanished, "remaining_actors": 0,
                      "original_exit_code": evidence["original_exit_code"], "original_files_unchanged": True}
        finally:
            ray.shutdown()
    except Exception as error:
        result = {"time": now(), "passed": False, "status": "failed", "namespace": guard["namespace"],
                  "job_id": guard["job_id"], "error": type(error).__name__ + ": " + str(error),
                  "original_files_unchanged": True}
    save(directory / "guard-result.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guard-dir", type=Path, required=True)
    arguments = parser.parse_args()
    raise SystemExit(run_guard(arguments.guard_dir.resolve()))
