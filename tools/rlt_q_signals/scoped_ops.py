# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Namespace and process helpers for the one-GPU Q experiment runners.

The caller supplies a frozen plan.json under --stage. No shared Ray lifecycle
operations or broad process-name matching are used.
"""

import datetime
import json
import os
import resource
import runpy
import signal
import subprocess
import sys
import urllib.request
from pathlib import Path

ST = None
MASKS = ("CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES")


def read(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.datetime.now(
        datetime.timezone(datetime.timedelta(hours=8))
    ).isoformat()


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def proc(pid):
    try:
        path = Path("/proc") / str(int(pid))
        fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
        return {
            "pid": int(pid),
            "uid": path.stat().st_uid,
            "start": int(fields[19]),
            "state": fields[0],
            "ppid": int(fields[1]),
        }
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
    return bool(
        current
        and current["state"] != "Z"
        and all(current[key] == identity[key] for key in ("pid", "uid", "start"))
    )


def checked():
    raise RuntimeError("The task runner must provide its frozen-plan validator")


def actors(plan):
    url = plan["ray_dashboard_url"].rstrip("/") + "/api/v0/actors?limit=10000&detail=1"
    with urllib.request.urlopen(url, timeout=25) as response:
        payload = json.load(response)
    result = payload["data"]["result"]
    assert result.get("num_after_truncation", result.get("total", 0)) < 10000, (
        "Actor list truncated"
    )
    return result["result"]


def active_actors(plan, namespace):
    return [
        row
        for row in actors(plan)
        if row.get("ray_namespace") == namespace and row.get("state") != "DEAD"
    ]


def gpu_pids(gpus):
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True
    )
    ids = {
        line.split(",")[1].strip(): int(line.split(",")[0])
        for line in rows.splitlines()
        if line.strip()
    }
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
        text=True,
    )
    return sorted(
        {
            int(line.split(",")[1])
            for line in rows.splitlines()
            if line.strip() and ids[line.split(",")[0].strip()] in gpus
        }
    )


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
    assert all(table[pid]["uid"] == uid for pid in selected), (
        "Foreign UID in target tree"
    )
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
    save(
        runtime / "cleanup-targets.json",
        {"time": now(), "job_id": job, "actors": targets},
    )
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
    save(
        runtime / "driver-identity.json",
        {**proc(os.getpid()), "namespace": row["namespace"], "time": now()},
    )

    def stop(signum, _frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    sys.argv = [
        str(Path(plan["repo"]) / row["entry"]),
        "--config-path",
        str(runtime),
        "--config-name",
        "resolved",
        "hydra.run.dir=.",
        "hydra.output_subdir=null",
        "hydra.job.chdir=false",
        "hydra/job_logging=stdout",
    ]
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
            [
                plan["python"],
                "-u",
                "-B",
                plan["ops"],
                "--stage",
                str(ST),
                "driver",
                key,
            ],
            cwd=plan["repo"],
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    save(
        runtime / "launch.json",
        {"time": now(), "identity": proc(child.pid), "key": key},
    )
    return child


def finished(key, exit_code):
    runtime = Path(read(ST / "plan.json")["runs"][key]["run"]) / "runtime"
    save(runtime / "finished.json", {"time": now(), "exit_code": exit_code})
    with (runtime / "exit_code.txt").open("x") as stream:
        stream.write(str(exit_code))
