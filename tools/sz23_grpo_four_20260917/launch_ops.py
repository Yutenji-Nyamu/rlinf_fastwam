"""Frozen, fresh two-GPU GRPO launches on SZ2/SZ3; never starts or stops Ray.

The unmasked personal Ray must expose eight physical GPUs. Each experiment has
its own namespace and can touch only the physical pair in its signed-off config.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import runpy
import shutil
import signal
import socket
import subprocess
import sys
import time

UID = 20001
MASKS = ("CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES")


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def command(*argv, **kwargs):
    return subprocess.check_output(argv, text=True, timeout=60, **kwargs).strip()


def flat(value, prefix=""):
    if not isinstance(value, dict):
        return {prefix: value}
    return {key: child for name, item in value.items()
            for key, child in flat(item, f"{prefix}.{name}" if prefix else name).items()}


def proc(pid):
    path = Path("/proc") / str(pid)
    try:
        values = (path / "stat").read_text().rsplit(")", 1)[1].split()
        result = {"pid": int(pid), "uid": path.stat().st_uid, "start": int(values[19]),
                  "state": values[0], "ppid": int(values[1])}
        if result["uid"] == UID:
            result["cmd"] = (path / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        return result
    except (FileNotFoundError, ProcessLookupError):
        return None


def same(identity):
    current = proc(identity["pid"])
    return bool(current and current["state"] != "Z"
                and all(current[key] == identity[key] for key in ("pid", "uid", "start")))


def gpu_state():
    gpu_rows = command("nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader").splitlines()
    physical = {row.split(",", 1)[1].strip(): int(row.split(",", 1)[0]) for row in gpu_rows}
    assert len(physical) == 8, "Expected eight physical GPUs"
    result = []
    for row in command("nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader").splitlines():
        fields = [part.strip() for part in row.split(",")]
        if len(fields) == 2 and fields[1].isdigit():
            result.append({"gpu": physical[fields[0]], "identity": proc(int(fields[1]))})
    return result


def inspect_raylet(address):
    rows = []
    port = address.rsplit(":", 1)[1]
    for path in Path("/proc").glob("[0-9]*"):
        try:
            argv = path.joinpath("cmdline").read_bytes().split(b"\0")
            if not argv or Path(os.fsdecode(argv[0])).name != "raylet":
                continue
            gcs = [os.fsdecode(arg) for arg in argv if arg.startswith(b"--gcs-address=")]
            if len(gcs) != 1 or not gcs[0].endswith(":" + port):
                continue
            identity = proc(int(path.name))
            assert identity and identity["uid"] == UID, "Selected Ray is not owned by chenyiteng"
            env = dict(item.split(b"=", 1) for item in path.joinpath("environ").read_bytes().split(b"\0") if b"=" in item)
            assert not any(key.encode() in env for key in MASKS), "Ray GPU order is masked"
            rows.append(identity)
        except (FileNotFoundError, ProcessLookupError):
            continue
    assert len(rows) == 1, "Exactly one owned unmasked raylet must exist for the endpoint"
    return rows[0]


def launch_environment(base):
    environment = read(base / "environment.json")
    assert not os.environ.get("SLURM_JOB_ID"), "Slurm local GPU mapping is not supported by this launcher"
    assert not any(key in os.environ or key in environment for key in MASKS), "Physical GPU masks must be unset"
    assert os.environ.get("NVIDIA_VISIBLE_DEVICES", "all") == "all"
    result = os.environ.copy()
    result.update(environment)
    result["PYTHONDONTWRITEBYTECODE"] = "1"
    for name in ("NO_PROXY", "no_proxy"):
        result[name] = ",".join(dict.fromkeys((result.get(name, "") + ",127.0.0.1,localhost," + socket.gethostname()).strip(",").split(",")))
    return result


def ray_query(python, environment, contract):
    # Never initialize Ray in the future training driver before Cluster.
    script = '''import json,sys,ray
from ray.util.state import list_actors
address,namespace=sys.argv[1:]
try:
 ray.init(address=address,namespace="sz23_launch_readonly_check",logging_level="ERROR",log_to_driver=False)
 nodes=[n for n in ray.nodes() if n.get("Alive")]
 rows=list_actors(filters=[("ray_namespace","=",namespace)],detail=True,limit=10000,timeout=20)
 actors=[__import__("dataclasses").asdict(a) for a in rows if a["state"]!="DEAD"]
 print("SZ23_RAY="+json.dumps({"nodes":[{"node_id":n["NodeID"],"address":n["NodeManagerAddress"],"gpus":n.get("Resources",{}).get("GPU",0)} for n in nodes],"actors":actors}))
finally:
 if ray.is_initialized():ray.shutdown()
'''
    output = command(python, "-B", "-c", script, contract["ray_address"], contract["namespace"], env=environment)
    values = [line.split("=", 1)[1] for line in output.splitlines() if line.startswith("SZ23_RAY=")]
    assert len(values) == 1, "Ray probe returned incomplete output"
    return json.loads(values[0])


def check(base, launching=False):
    assert os.getuid() == UID, "Run as chenyiteng UID20001"
    contract = read(base / "contract.json")
    config = read(base / "resolved.yaml")
    environment = launch_environment(base)
    repo, run = Path(contract["repo"]), Path(contract["run"])
    assert contract["schema_version"] == 1 and contract["uid"] == UID
    assert socket.gethostname() == contract["hostname"]
    assert repo.resolve().is_relative_to(Path("/home/chenyiteng").resolve()) or repo.resolve().is_relative_to(Path("/data/chenyiteng").resolve())
    assert run.resolve().is_relative_to(Path("/data/chenyiteng").resolve())
    assert contract["gpus"] in ([4, 5], [6, 7])
    assert contract["ray_address"] == "127.0.0.1:26379"
    assert environment["RAY_ADDRESS"] == contract["ray_address"]
    assert Path(environment["REPO_PATH"]).resolve() == repo.resolve()
    assert Path(environment["RLINF_CODE_WORKING_DIR"]).resolve() == repo.resolve()
    assert Path(environment["PYTHONPATH"].split(":")[0]).resolve() == repo.resolve()
    assert all(Path(environment[key]).exists() for key in ("VIRTUAL_ENV", "CUDA_HOME", "VK_DRIVER_FILES", "TMPDIR"))
    for filename, expected in contract["prepared_sha256"].items():
        assert (base / filename).resolve().is_relative_to(base.resolve())
        assert sha(base / filename) == expected, "Prepared file changed: " + filename
    assert set(contract["prepared_sha256"]) >= {"resolved.yaml", "environment.json"}
    assert command("git", "-C", str(repo), "rev-parse", "HEAD") == contract["head"]
    assert not command("git", "-C", str(repo), "status", "--porcelain"), "Source tree is not clean"
    for relative, expected in contract["source_sha256"].items():
        target = repo / relative
        assert target.resolve().is_relative_to(repo.resolve())
        assert sha(target) == expected, "Production source changed: " + relative
    own_relative = Path(__file__).resolve().relative_to(repo.resolve()).as_posix()
    assert contract["source_sha256"][own_relative] == sha(__file__)
    values = flat(config)
    for key, expected in {**contract["budget"], **contract["method"]}.items():
        assert values.get(key) == expected, "Frozen parameter changed: " + key
    required = {"env.train.total_num_envs": 64, "env.train.rollout_epoch": 4,
                "actor.global_batch_size": 512, "actor.micro_batch_size": 32,
                "algorithm.update_epoch": 2, "runner.max_steps": 200, "rollout.seed": 42}
    for key, expected in required.items():
        assert values.get(key) == expected, "GRPO256 budget mismatch: " + key
    prefix = "algorithm.dvac_gradient_weighting."
    mode = values[prefix + "mode"]
    assert mode in ("off", "observe", "apply")
    if mode == "apply":
        assert values[prefix + "mapping"] == "exp_mean" and values[prefix + "scope"] == "both"
        assert values[prefix + "alpha_local"] == values[prefix + "alpha_chunk"] == 1.0
        assert values[prefix + "temperature_local"] == values[prefix + "temperature_chunk"]
        assert values[prefix + "temperature_local"] in (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
        enabled = values[prefix + "chunk_dropout.enabled"]
        assert enabled == values[prefix + "alpha_schedule.enabled"]
        if enabled:
            assert values[prefix + "temperature_local"] == 1.5 and values[prefix + "chunk_dropout.probability"] == 0.2
            for level in ("local", "chunk"):
                for field, expected in (("enabled", True), ("start_step", 1), ("end_step", 200), ("end_alpha", 0.0)):
                    assert values[prefix + "alpha_schedule." + level + "." + field] == expected
    assert values["cluster.component_placement.actor, env, rollout"] == ",".join(map(str, contract["gpus"]))
    assert values.get("runner.ckpt_path") is None
    if contract.get("fresh", True):
        assert values.get("runner.resume_dir") is None
    else:
        assert values.get("runner.resume_dir") == contract["resume_checkpoint"]
        checkpoint = Path(contract["resume_checkpoint"])
        assert checkpoint.name == "global_step_" + str(contract["resume_step"])
        assert checkpoint.resolve().is_relative_to(Path("/data/chenyiteng").resolve())
        assert (checkpoint / "actor").is_dir()
    assert values["runner.logger.log_path"] == str(run)
    for split in ("train", "eval"):
        seeds = Path(config["env"][split]["seeds_path"])
        relative = seeds.resolve().relative_to(repo.resolve()).as_posix()
        assert relative in contract["source_sha256"] and sha(seeds) == contract["source_sha256"][relative]
    python = str(Path(environment["VIRTUAL_ENV"]) / "bin/python")
    raylet = inspect_raylet(contract["ray_address"])
    state = ray_query(python, environment, contract)
    assert len(state["nodes"]) == 1 and state["nodes"][0]["gpus"] == 8
    gpu = gpu_state()
    if launching:
        assert not state["actors"], "Target namespace is already occupied"
        assert not any(row["gpu"] in contract["gpus"] for row in gpu), "Target physical GPUs are occupied"
    return contract, config, environment, {"time": now(), "raylet": raylet, "ray": state, "gpu": gpu}


def dispatch(base):
    contract, _, environment, snapshot = check(base, launching=True)
    run = Path(contract["run"])
    assert not run.exists(), "Fresh launch requires a new run directory"
    save(base / "dispatch-attempt.json", {"time": now(), "contract_sha256": sha(base / "contract.json"), "snapshot": snapshot})
    runtime = run / "runtime"
    runtime.mkdir(parents=True, exist_ok=False)
    for filename in ("resolved.yaml", "environment.json", "contract.json"):
        shutil.copyfile(base / filename, runtime / filename)
    python = str(Path(environment["VIRTUAL_ENV"]) / "bin/python")
    argv = [python, "-u", "-B", str(Path(__file__).resolve()), "wrapper", "--config-dir", str(runtime)]
    with (runtime / "wrapper.log").open("x") as stream:
        process = subprocess.Popen(argv, cwd=contract["repo"], env=environment, stdin=subprocess.DEVNULL,
                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    result = {"time": now(), "run": str(run), "namespace": contract["namespace"],
              "wrapper": proc(process.pid), "argv": argv, "prepared_from": str(base),
              "contract_sha256": sha(runtime / "contract.json"), "snapshot": snapshot}
    save(runtime / "dispatch.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


def wrapper(base):
    contract = read(base / "contract.json")
    assert os.getuid() == UID and base.resolve() == (Path(contract["run"]) / "runtime").resolve()
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (max(soft, min(4096, hard)), hard))
    save(base / "wrapper-identity.json", {**proc(os.getpid()), "time": now()})
    argv = [sys.executable, "-u", "-B", str(Path(__file__).resolve()), "driver", "--config-dir", str(base)]
    with (base / "driver.log").open("x") as stream:
        child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
    save(base / "driver-spawn.json", {"time": now(), "identity": proc(child.pid), "argv": argv})
    code = child.wait()
    (base / "exit_code.txt").write_text(str(code) + "\n")
    save(base / "finished.json", {"time": now(), "exit_code": code})
    return code


def cleanup_owned(contract, runtime):
    import ray
    if not ray.is_initialized():
        return
    from ray.util.state import list_actors
    job = ray.get_runtime_context().get_job_id()
    job = job.hex() if hasattr(job, "hex") else str(job)
    namespace = contract["namespace"]
    for _ in range(10):
        rows = [dict(row) for row in list_actors(filters=[("ray_namespace", "=", namespace)], detail=True, limit=10000, timeout=15) if row["state"] != "DEAD"]
        names = {row["name"] for row in ray.util.list_named_actors(all_namespaces=True) if row["namespace"] == namespace}
        identities = [proc(row["pid"]) for row in rows]
        if names == {row["name"] for row in rows if row.get("name")} and all(identities):
            break
        time.sleep(0.5)
    else:
        raise RuntimeError("Scoped cleanup actor identities did not converge; no actors killed")
    assert all(row["job_id"] == job for row in rows), "Cleanup job ownership mismatch"
    assert all(identity["uid"] == UID for identity in identities), "Cleanup UID mismatch"
    save(runtime / "owned-cleanup-targets.json", {"time": now(), "namespace": namespace, "job": job,
                                                 "actors": rows, "identities": identities})
    managers = {"NodeManager", "WorkerManager", "CollectiveManager", "DeviceLockManager", "PortLockManager"}
    for name in sorted(names, key=lambda value: (value in managers, value)):
        try:
            ray.kill(ray.get_actor(name, namespace=namespace), no_restart=True)
        except ValueError:
            pass
    save(runtime / "owned-cleanup.json", {"time": now(), "namespace": namespace, "job": job, "killed_names": sorted(names)})


def driver(base):
    contract, _, environment, snapshot = check(base, launching=True)
    assert Path(sys.prefix).resolve() == Path(environment["VIRTUAL_ENV"]).resolve()
    os.environ.update(environment)
    for path in reversed(environment["PYTHONPATH"].split(":")):
        if path:
            sys.path.insert(0, path)
    repo = Path(contract["repo"])
    import rlinf
    from rlinf.scheduler import Cluster
    assert Path(rlinf.__file__).resolve().is_relative_to(repo.resolve())
    Cluster.NAMESPACE = contract["namespace"]
    save(base / "driver-identity.json", {**proc(os.getpid()), "time": now(), "namespace": contract["namespace"]})
    save(base / "launch-checks.json", snapshot)
    def terminate(sig, _frame):
        raise SystemExit(128 + sig)
    signal.signal(signal.SIGTERM, terminate)
    entry = repo / "examples/embodiment/train_embodied_agent.py"
    sys.argv = [str(entry), "--config-path", str(base), "--config-name", "resolved", "hydra.run.dir=.",
                "hydra.output_subdir=null", "hydra.job.chdir=false", "hydra/job_logging=stdout"]
    os.chdir(repo)
    try:
        runpy.run_path(str(entry), run_name="__main__")
    finally:
        import ray
        signal.signal(signal.SIGUSR1, signal.SIG_IGN)
        try:
            cleanup_owned(contract, base)
        finally:
            if ray.is_initialized():
                ray.shutdown()


def ancestor_in(pid, allowed):
    for _ in range(12):
        if pid in allowed:
            return True
        identity = proc(pid)
        if not identity or identity["uid"] != UID or identity["ppid"] <= 1:
            return False
        pid = identity["ppid"]
    return False


def status(base, require_round=False):
    contract, config, _, snapshot = check(base)
    runtime = Path(contract["run"]) / "runtime"
    identity_file = runtime / "driver-identity.json"
    identity = read(identity_file) if identity_file.exists() else None
    actors = snapshot["ray"]["actors"]
    identities = [proc(row["pid"]) if row.get("pid") else None for row in actors]
    alive = [row for row in actors if row["state"] == "ALIVE"]
    actual = Path(contract["run"]) / "tensorboard/config.yaml"
    actual_diff = None
    if actual.exists():
        from omegaconf import OmegaConf
        before, after = flat(config), flat(OmegaConf.to_container(OmegaConf.load(actual), resolve=True))
        actual_diff = {key: [before.get(key), after.get(key)] for key in before.keys() | after.keys() if before.get(key) != after.get(key)}
    log_path = runtime / "driver.log"
    log = log_path.read_text(errors="replace") if log_path.exists() else ""
    fatal = [line for line in log.splitlines() if any(token in line for token in ("Traceback (most recent call last)", "CUDA out of memory", "AssertionError:", "RuntimeError:"))]
    scalar_summary, nonfinite, successes, gradients = {}, [], [], []
    event_path = Path(contract["run"]) / "tensorboard"
    if list(event_path.glob("events.out.tfevents.*")):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        accumulator = EventAccumulator(str(event_path), size_guidance={"scalars": 0})
        accumulator.Reload()
        for tag in accumulator.Tags().get("scalars", []):
            values = accumulator.Scalars(tag)
            if values:
                scalar_summary[tag] = {"step": values[-1].step, "value": values[-1].value}
            nonfinite.extend({"tag": tag, "step": row.step} for row in values if not math.isfinite(row.value))
            if tag == "env/success_once":
                successes.extend({"tag": tag, "step": row.step, "value": row.value} for row in values)
            if "grad_norm" in tag.lower():
                gradients.extend(row.value for row in values)
    worker_pids = {row["pid"] for row in actors if row.get("pid")}
    assigned_gpu_rows = [row for row in snapshot["gpu"] if row["gpu"] in contract["gpus"]]
    own_gpu_rows = [row for row in snapshot["gpu"] if row["identity"] and ancestor_in(row["identity"]["pid"], worker_pids)]
    gpu_valid = (set(row["gpu"] for row in assigned_gpu_rows) == set(contract["gpus"])
                 and all(row["identity"] and row["identity"]["uid"] == UID and ancestor_in(row["identity"]["pid"], worker_pids) for row in assigned_gpu_rows)
                 and all(row["gpu"] in contract["gpus"] for row in own_gpu_rows))
    exit_file = runtime / "exit_code.txt"
    result = {"time": now(), "run": contract["run"], "namespace": contract["namespace"], "head": contract["head"],
              "driver_alive": bool(identity and same(identity)), "driver": identity,
              "alive_actors": len(alive), "actors": actors, "actor_identities": identities,
              "all_actor_uid_valid": bool(identities) and all(row and row["uid"] == UID for row in identities),
              "single_job": len({row["job_id"] for row in actors}) == 1,
              "actual_config_diff": actual_diff, "gpu_mapping_valid": gpu_valid, "gpu": snapshot["gpu"],
              "success_scalars": successes, "grad_norm_positive": any(math.isfinite(value) and value > 0 for value in gradients),
              "nonfinite_scalars": nonfinite, "latest_scalars": scalar_summary,
              "fatal_lines": fatal[-10:], "exit_code": exit_file.read_text().strip() if exit_file.exists() else None,
              "rollout_seeds_verified": all("Rollout RNG seed=" + str(seed) in log for seed in (42, 43)),
              "log_tail": log[-2500:]}
    result["passed"] = (result["driver_alive"] and len(alive) == 15 and result["all_actor_uid_valid"]
                         and result["single_job"] and actual_diff == {} and gpu_valid
                         and not fatal and not nonfinite and result["exit_code"] is None
                         and result["rollout_seeds_verified"] and any(row["step"] >= 0 for row in successes)
                         and result["grad_norm_positive"])
    if require_round and result["passed"] and not (runtime / "FIRST_ROUND_VERIFIED.json").exists():
        save(runtime / "FIRST_ROUND_VERIFIED.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if not require_round or result["passed"] else 3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("precheck", "dispatch", "wrapper", "driver", "status", "verify"))
    parser.add_argument("--config-dir", type=Path, required=True)
    args = parser.parse_args()
    base = args.config_dir.resolve()
    if args.action == "precheck":
        contract, _, _, snapshot = check(base, launching=True)
        print(json.dumps({"passed": True, "run": contract["run"], "snapshot": snapshot}, ensure_ascii=False))
        return 0
    if args.action == "dispatch":
        dispatch(base)
    elif args.action == "wrapper":
        return wrapper(base)
    elif args.action == "driver":
        driver(base)
    else:
        return status(base, require_round=args.action == "verify")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
