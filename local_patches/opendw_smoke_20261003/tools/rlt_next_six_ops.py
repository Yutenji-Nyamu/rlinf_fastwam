"""Task-specific Stage1, then exact old-run cutover to a one-GPU RLT pair.

The caller supplies a frozen plan.json under --stage. No shared Ray lifecycle
operations or broad process-name matching are used.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import resource
import runpy
import signal
import subprocess
import sys
import time
import urllib.request


ST = None
MASKS = ("CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES")


def read(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def proc(pid):
    try:
        path = Path("/proc") / str(int(pid))
        fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
        return {"pid": int(pid), "uid": path.stat().st_uid,
                "start": int(fields[19]), "state": fields[0], "ppid": int(fields[1])}
    except (FileNotFoundError, ProcessLookupError):
        return None


def normalized(identity):
    result = dict(identity)
    if "start" not in result:
        result["start"] = result["start_ticks"]
    return result


def same(identity):
    identity = normalized(identity)
    current = proc(identity["pid"])
    return bool(current and current["state"] != "Z" and
                all(current[key] == identity[key] for key in ("pid", "uid", "start")))


def checked():
    plan = read(ST / "plan.json")
    assert os.getuid() == plan["uid"], "Wrong account"
    assert subprocess.check_output(
        ["git", "-C", plan["repo"], "rev-parse", "HEAD"], text=True).strip() == plan["head"]
    assert not subprocess.check_output(
        ["git", "-C", plan["repo"], "status", "--porcelain"], text=True).strip()
    assert set(plan["runs"]) == {"stage1-full", "clean", "combo"}
    assert set(plan["runs"]["stage1-full"]["gpus"]).isdisjoint(
        gpu for old in plan["old_runs"] for gpu in old["gpus"])
    assert {gpu for old in plan["old_runs"] for gpu in old["gpus"]} == {
        gpu for key in ("clean", "combo") for gpu in plan["runs"][key]["gpus"]}
    assert len(plan["runs"]["clean"]["gpus"]) == len(plan["runs"]["combo"]["gpus"]) == 1
    return plan


def actors(plan):
    url = plan["ray_dashboard_url"].rstrip("/") + "/api/v0/actors?limit=10000&detail=1"
    with urllib.request.urlopen(url, timeout=25) as response:
        payload = json.load(response)
    result = payload["data"]["result"]
    assert result.get("num_after_truncation", result.get("total", 0)) < 10000, "Actor list truncated"
    return result["result"]


def active_actors(plan, namespace):
    return [row for row in actors(plan)
            if row.get("ray_namespace") == namespace and row.get("state") != "DEAD"]


def gpu_pids(gpus):
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
    ids = {line.split(",")[1].strip(): int(line.split(",")[0])
           for line in rows.splitlines() if line.strip()}
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True)
    return sorted({int(line.split(",")[1]) for line in rows.splitlines()
                   if line.strip() and ids[line.split(",")[0].strip()] in gpus})


def process_tree(roots, uid):
    table = {}
    for path in Path("/proc").iterdir():
        if path.name.isdigit():
            item = proc(int(path.name))
            if item:
                table[item["pid"]] = item
    selected = {pid for pid in roots if pid in table}
    for _ in range(64):
        additional = {pid for pid, item in table.items() if item["ppid"] in selected}
        if additional <= selected:
            break
        selected |= additional
    else:
        raise RuntimeError("Unbounded process ancestry")
    assert all(table[pid]["uid"] == uid for pid in selected), "Foreign UID in target tree"
    return {pid: table[pid] for pid in selected}


def validate_scoped_actors(plan, rows, jobs=None):
    result = []
    for row in rows:
        if jobs is not None:
            assert row.get("job_id") in jobs, "Unexpected job in target namespace"
        assert row.get("name"), "Unnamed live actor cannot be safely addressed"
        current = proc(row.get("pid", 0))
        if current and current["state"] != "Z":
            assert current["uid"] == plan["uid"], "Foreign actor UID"
        result.append(row)
    return result


def kill_named_actors(plan, rows):
    import ray
    for row in rows:
        try:
            handle = ray.get_actor(row["name"], namespace=row["ray_namespace"])
        except ValueError:
            continue
        assert handle._actor_id.hex() == row["actor_id"], "Actor identity changed"
        ray.kill(handle, no_restart=True)


def cleanup_driver(plan, row, runtime):
    import ray
    if not ray.is_initialized():
        return
    job = ray.get_runtime_context().get_job_id()
    job = job.hex() if hasattr(job, "hex") else str(job)
    targets = validate_scoped_actors(plan, active_actors(plan, row["namespace"]), {job})
    save(runtime / "cleanup-targets.json", {"time": now(), "job_id": job, "actors": targets})
    kill_named_actors(plan, targets)


def driver(key):
    plan = checked()
    row = plan["runs"][key]
    runtime = Path(row["run"]) / "runtime"
    env = read(runtime / "environment.json")
    assert not any(key in env for key in MASKS)
    os.environ.update(env)
    for value in reversed(env["PYTHONPATH"].split(":")):
        sys.path.insert(0, value)
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (max(soft, min(4096, hard)), hard))
    from rlinf.scheduler import Cluster
    Cluster.NAMESPACE = row["namespace"]
    save(runtime / "driver-identity.json", {**proc(os.getpid()), "namespace": row["namespace"], "time": now()})

    def stop(signum, _frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    sys.argv = [str(Path(plan["repo"]) / row["entry"]), "--config-path", str(runtime),
                "--config-name", "resolved", "hydra.run.dir=.", "hydra.output_subdir=null",
                "hydra.job.chdir=false", "hydra/job_logging=stdout"]
    try:
        runpy.run_path(sys.argv[0], run_name="__main__")
    finally:
        import ray
        signal.signal(signal.SIGUSR1, signal.SIG_IGN)
        try:
            cleanup_driver(plan, row, runtime)
        finally:
            ray.shutdown()


def launch(key):
    plan = checked()
    row = plan["runs"][key]
    runtime = Path(row["run"]) / "runtime"
    assert not gpu_pids(row["gpus"]), "Target GPU has another process"
    assert not active_actors(plan, row["namespace"]), "Target namespace is not empty"
    env = read(runtime / "environment.json")
    assert not any(key in env for key in MASKS)
    with (runtime / "driver.log").open("x") as stream:
        child = subprocess.Popen(
            [plan["python"], "-u", "-B", plan["ops"], "--stage", str(ST), "driver", key],
            cwd=plan["repo"], env=env, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
    save(runtime / "launch.json", {"time": now(), "identity": proc(child.pid), "key": key})
    return child


def finished(key, exit_code):
    runtime = Path(read(ST / "plan.json")["runs"][key]["run"]) / "runtime"
    save(runtime / "finished.json", {"time": now(), "exit_code": exit_code})
    with (runtime / "exit_code.txt").open("x") as stream:
        stream.write(str(exit_code))


def wait_stage1(child):
    exit_code = child.wait()
    finished("stage1-full", exit_code)
    assert exit_code == 0, ("Stage1 failed", exit_code)
    plan = read(ST / "plan.json")
    for _ in range(60):
        if not gpu_pids(plan["runs"]["stage1-full"]["gpus"]):
            break
        time.sleep(1)
    else:
        raise RuntimeError("Stage1 GPU cleanup incomplete")
    weights = Path(plan["stage1_full_weights"])
    assert weights.is_file() and weights.stat().st_size > 1024 ** 3, "Stage1 full weights missing/incomplete"
    save(ST / "stage1-complete.json", {"time": now(), "weights": str(weights), "bytes": weights.stat().st_size})


def stop_old(old_runs=None, receipt_prefix=""):
    plan = checked()
    if old_runs is None:
        old_runs = plan["old_runs"]
    else:
        assert len(old_runs) == 1 and receipt_prefix in ("clean-", "combo-")
        for row in old_runs:
            assert any(all(row[k] == target[k] for k in ("run", "namespace", "gpus")) for target in plan["old_runs"]), "not a frozen old target"
            assert row["gpus"] == plan["runs"][receipt_prefix[:-1]]["gpus"], "Wrong formal GPU"
    def receipt(name): return ST / (receipt_prefix + name)
    assert len({row["namespace"] for row in old_runs}) == len(old_runs)
    snapshots = []
    identities = {}
    all_gpus = {gpu for row in old_runs for gpu in row["gpus"]}
    for row in old_runs:
        identity = normalized(row["identity"])
        assert identity["uid"] == plan["uid"]
        targets = validate_scoped_actors(plan, active_actors(plan, row["namespace"]))
        jobs = {target["job_id"] for target in targets}
        assert len(jobs) <= 1, "Multiple jobs in old namespace"
        roots = {target["pid"] for target in targets if target.get("pid")}
        if same(identity):
            roots.add(identity["pid"])
        tree = process_tree(roots, plan["uid"])
        assert set(gpu_pids(row["gpus"])) <= set(tree), "Unrelated GPU process in cutover target"
        identities.update(tree)
        snapshots.append({"namespace": row["namespace"], "run": row["run"], "identity": identity,
                          "jobs": sorted(jobs), "actors": targets, "gpus": row["gpus"]})
    save(receipt("old-stop-attempt.json"), {"time": now(), "runs": snapshots, "processes": list(identities.values())})
    for row in snapshots:
        if same(row["identity"]):
            os.kill(row["identity"]["pid"], signal.SIGTERM)
    for _ in range(10):
        if not gpu_pids(all_gpus) and not any(same(row["identity"]) for row in snapshots):
            break
        time.sleep(1)
    remaining = []
    for row in snapshots:
        targets = validate_scoped_actors(plan, active_actors(plan, row["namespace"]), set(row["jobs"]))
        remaining += targets
        roots = {target["pid"] for target in targets if target.get("pid")}
        identities.update(process_tree(roots, plan["uid"]))
    if remaining:
        import ray
        assert not ray.is_initialized()
        ray.init(address=plan["ray_address"], namespace=plan["management_namespace"], log_to_driver=False)
        try:
            kill_named_actors(plan, remaining)
        finally:
            ray.shutdown()
        save(receipt("old-scoped-actor-cleanup.json"), {"time": now(), "actors": remaining})
    for _ in range(8):
        if not any(same(identity) for identity in identities.values()):
            break
        time.sleep(1)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        current_tree = process_tree({pid for pid, identity in identities.items() if same(identity)}, plan["uid"])
        identities.update(current_tree)
        targets = [identity for identity in identities.values() if same(identity)]
        if not targets:
            break
        target_pids = {identity["pid"] for identity in targets}
        scoped_jobs = {row["namespace"]: set(row["jobs"]) for row in snapshots}
        for actor in actors(plan):
            if actor.get("state") != "DEAD" and actor.get("pid") in target_pids:
                assert actor.get("ray_namespace") in scoped_jobs and actor.get("job_id") in scoped_jobs[actor["ray_namespace"]], "Process is serving an unrelated actor"
        save(receipt("old-signal-" + str(int(sig)) + ".json"), {"time": now(), "processes": targets})
        for identity in targets:
            if same(identity):
                os.kill(identity["pid"], sig)
        for _ in range(8):
            if not any(same(identity) for identity in identities.values()):
                break
            time.sleep(1)
    assert not gpu_pids(all_gpus), "Old GPU processes remain"
    assert not any(same(row["identity"]) for row in snapshots), "Old driver remains"
    assert not any(active_actors(plan, row["namespace"]) for row in snapshots), "Old actors remain"
    save(receipt("old-stopped.json"), {"time": now(), "runs": [row["run"] for row in snapshots]})


def pipeline():
    plan = checked()
    save(ST / "pipeline-identity.json", {**proc(os.getpid()), "time": now()})
    child = launch("stage1-full")
    wait_stage1(child)
    stop_old()
    children = {}
    for key in ("clean", "combo"):
        children[key] = launch(key)
    save(ST / "formal-dispatched.json", {"time": now(), "drivers": {key: proc(child.pid) for key, child in children.items()}})
    failures = {}
    while children:
        for key, child in list(children.items()):
            exit_code = child.poll()
            if exit_code is not None:
                finished(key, exit_code)
                if exit_code:
                    failures[key] = exit_code
                del children[key]
        if children:
            time.sleep(10)
    save(ST / "pipeline-finished.json", {"time": now(), "failures": failures})
    assert not failures, failures


def main():
    global ST
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    parser.add_argument("action", choices=("pipeline", "driver"))
    parser.add_argument("key", nargs="?", choices=("stage1-full", "clean", "combo"))
    args = parser.parse_args()
    ST = Path(args.stage).resolve()
    if args.action == "driver":
        assert args.key
        driver(args.key)
    else:
        assert args.key is None
        pipeline()


if __name__ == "__main__":
    main()
