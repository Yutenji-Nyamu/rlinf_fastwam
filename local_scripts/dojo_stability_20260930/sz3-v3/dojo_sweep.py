"""Fixed official Pi05/Dojo sweep; Linux, same UID, no RLT operations.

Run from the existing simulator Python environment. --plan-only reads sources,
configuration and layouts without starting GPU work. No task wall-time limit.
Exit: 0 complete, 2 incomplete, 3 interrupted, 4 infrastructure, 5 cleanup.
"""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback

import yaml

from process_guard import ProcessGuard, atomic_json, error_context
from eval_recovery import run_client


GPUS = [4, 4, 5, 5, 6, 6, 7, 7]
SEEDS = [0, 1, 2]
POLICY = "Pi_05"
CKPT_NAME = "sim"
UPSTREAM_DOJO = "726e9aabfaa642203722eb126f5eaf0f37f3e1ad"
XPL_HEAD = "10ab2651a0b7cd5b8a948cb13f18ccdf818c6de4"
CAMERAS = {"head", "left_wrist", "right_wrist"}
SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,110}\Z")


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(repo: Path, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def official_partition(repo: Path, task_names):
    """Use upstream's exact weights/algorithm, without its shell side effects."""
    source = (repo / "scripts/internal/smoke_all_tasks.sh").read_text()
    section = source.split("build_parallel_assignment() {", 1)[1]
    embedded = section.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    parsed = ast.parse(embedded)
    selected = [node for node in parsed.body if
                isinstance(node, (ast.Import, ast.ImportFrom)) or
                isinstance(node, ast.FunctionDef) and node.name in {"score", "partition"} or
                isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == "RUNTIME_WEIGHTS"
                    for target in node.targets)]
    namespace = {}
    exec(compile(ast.Module(body=selected, type_ignores=[]),
                 "official-smoke-all-tasks-partition", "exec"), namespace)
    weights = namespace["RUNTIME_WEIGHTS"]
    tasks = [{"task": name, "key": f"{name}/arx_x5", "seconds": weights[f"{name}/arx_x5"]}
             for name in task_names]
    tasks.sort(key=lambda item: (-item["seconds"], item["task"]))
    groups = namespace["partition"](tasks, len(GPUS))
    return groups, hashlib.sha256(embedded.encode()).hexdigest()


def build_plan(cfg):
    repo = Path(cfg["repo"])
    if os.getuid() != int(cfg["uid"]):
        raise RuntimeError("Wrong UID")
    if not SAFE_NAME.fullmatch(cfg["run_id"]):
        raise ValueError("Invalid run_id")
    if git(repo, "rev-parse", "HEAD") != cfg["dojo_head"]:
        raise RuntimeError("Dojo HEAD differs from fixed successful local commit")
    if git(repo / "XPolicyLab", "rev-parse", "HEAD") != cfg.get("xpl_head", XPL_HEAD):
        raise RuntimeError("XPolicyLab HEAD differs")
    if cfg.get("workers_per_gpu") != 1:
        raise RuntimeError("Approved recovery requires one simultaneous worker per GPU")
    ports = cfg["ports"]
    if len(ports) != 8 or len(set(ports)) != 8 or any(not 1024 < int(p) < 65536 for p in ports):
        raise ValueError("Eight distinct unprivileged ports required")
    env_cfg = yaml.safe_load((repo / "env_cfg/arx_x5.yml").read_text())
    sim_path = repo / f"env_cfg/sim/{env_cfg['config']['sim']}.yml"
    sim_cfg = yaml.safe_load(sim_path.read_text())
    deploy = yaml.safe_load((repo / "XPolicyLab/policy/Pi_05/deploy.yml").read_text())
    if (env_cfg["config_name"] != "arx_x5" or sim_cfg["scene"]["num_envs"] != 4
            or float(sim_cfg["dt"]) != 0.004 or sim_cfg["decimation"] != 1
            or env_cfg["observation"]["collect_freq"] != 25 or deploy.get("eval_batch") is not True):
        raise RuntimeError("Expected canonical arx_x5, N4, 25Hz, dt=.004, decimation=1, eval_batch=true")
    inventory = subprocess.check_output(
        [sys.executable, str(repo / "scripts/internal/task_inventory.py"), "--only-runnable"],
        cwd=repo, text=True).splitlines()
    tasks = [name.strip() for name in inventory if name.strip()]
    if len(tasks) != 54 or len(set(tasks)) != 54 or any(not SAFE_NAME.fullmatch(t) for t in tasks):
        raise RuntimeError("Expected 54 distinct official runnable configurations")
    task_cfg = yaml.safe_load((repo / "task/RoboDojo/config/_task.yml").read_text())
    budgets = {name: int((task_cfg["tasks"].get(name) or {}).get(
        "eval_nums", task_cfg["common"]["eval_nums"])) for name in tasks}
    if set(budgets.values()) != {25, 50} or sum(budgets.values()) != 2100:
        raise RuntimeError("Native episode budget changed")
    layout_counts = {}
    for seed in SEEDS:
        for task in tasks:
            folder = repo / f"Assets/Eval_Layout/RoboDojo/arx_x5/{seed}"
            matching = [p for p in folder.glob(f"{task}_*.json")
                        if re.fullmatch(re.escape(task) + r"_\d+\.json", p.name)]
            if len(matching) < budgets[task]:
                raise RuntimeError(f"Insufficient official layouts: {task}/s{seed}: {len(matching)}")
            layout_counts[f"{seed}/{task}"] = len(matching)
    groups, algorithm_sha = official_partition(repo, tasks)
    source_paths = [repo / rel for rel in (
        "scripts/robodojo.sh", "scripts/eval_policy.sh", "scripts/internal/smoke_all_tasks.sh",
        "src/eval_client/main.py", "src/eval_client/eval_env.py", "env_cfg/arx_x5.yml",
        "task/RoboDojo/config/_task.yml", "XPolicyLab/policy/Pi_05/deploy.yml",
        "XPolicyLab/policy/Pi_05/model.py", "XPolicyLab/policy/Pi_05/deploy.py",
        "XPolicyLab/policy/Pi_05/setup_eval_policy_server.sh")]
    source_paths += [sim_path, Path(cfg["project_env"]), Path(cfg["compat_script"])]
    checkpoint_files = {}
    checkpoint_manifests = {}
    for seed in SEEDS:
        checkpoint = Path(cfg["checkpoint"]) / f"RoboDojo-sim-arx_x5-joint-{seed}/59999"
        manifest_path = checkpoint / ".download-pi05/manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest["revision"] != cfg["checkpoint_revision"] or manifest["repo"] != "RoboDojo-Benchmark/RoboDojo":
            raise RuntimeError("Pi05 fixed official manifest differs")
        if manifest["prefix"] != f"ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-{seed}/59999":
            raise RuntimeError("Pi05 checkpoint seed differs")
        ready_path = checkpoint / "pi05-inference-ready.json"
        if not ready_path.is_file():
            if seed == 0:
                raise RuntimeError("Pi05 seed0 must be verified before starting the sweep")
            checkpoint_files[str(seed)] = manifest["files"]
            checkpoint_manifests[str(seed)] = sha(manifest_path)
            continue
        ready = json.loads(ready_path.read_text())
        if ready["revision"] != cfg["checkpoint_revision"] or ready["status"] != "ready":
            raise RuntimeError("Pi05 checkpoint revision/readiness mismatch")
        rows = ready["files"]
        if ready["file_count"] != len(rows) or not all(x["verified"] for x in rows):
            raise RuntimeError("Pi05 incomplete inference manifest")
        for item in rows:
            path = checkpoint / item["path"]
            if not path.is_file() or path.stat().st_size != item["size"]:
                raise RuntimeError(f"Pi05 inference file missing or wrong size: {seed}/{item['path']}")
        if [{k:r[k] for k in ("path", "size", "sha256")} for r in rows] != manifest["files"]:
            raise RuntimeError("Pi05 ready manifest differs from fixed download manifest")
        checkpoint_files[str(seed)] = manifest["files"]
        checkpoint_manifests[str(seed)] = sha(manifest_path)
    for prefix in (cfg["policy_env"], cfg["sim_env"]):
        if not (Path(prefix) / "bin/python").is_file():
            raise RuntimeError(f"Environment Python missing: {prefix}")
    return {"run_id": cfg["run_id"], "created_at": time.time(), "config": cfg,
            "upstream_dojo": UPSTREAM_DOJO, "dojo_head": cfg["dojo_head"],
            "xpl_head": git(repo / "XPolicyLab", "rev-parse", "HEAD"),
            "num_envs": 4, "workers_per_gpu": 1, "gpus": GPUS, "seeds": SEEDS, "episode_total": 6300,
            "tasks": tasks, "budgets": budgets, "layout_counts": layout_counts,
            "source_sha256": {str(path): sha(path) for path in source_paths},
            "controller_sha256": {name: sha(Path(__file__).with_name(name))
                                  for name in ("dojo_sweep.py", "process_guard.py", "eval_recovery.py", "resume_results.py")},
            "partition_sha256": algorithm_sha,
            "groups": [{"worker": i, "gpu": GPUS[i], "port": ports[i], "tasks": group}
                       for i, group in enumerate(groups)],
            "checkpoint_revision": cfg["checkpoint_revision"],
            "asset_revision": cfg["asset_revision"],
            "checkpoint_manifest_sha256": checkpoint_manifests,
            "checkpoint_files": checkpoint_files}


def result_check(path: Path, expected: int):
    """Do not accept upstream sweep's weaker PASS (one or more episodes)."""
    report = {"path": str(path), "expected": expected, "complete": False}
    if not path.is_file():
        return {**report, "reason": "missing_result"}
    try:
        result = json.loads(path.read_text())
        details = result.get("details", {})
        if not isinstance(details, dict):
            raise ValueError("details is not a mapping")
        eval_time = int(result.get("eval_time", -1))
        if eval_time != expected or len(details) != expected:
            return {**report, "reason": "native_budget_incomplete", "eval_time": eval_time,
                    "detail_count": len(details)}
        layout_ids = [int(item["layout_id"]) for item in details.values()]
        if len(set(layout_ids)) != expected:
            raise ValueError("duplicate layout IDs")
        if any(not isinstance(item.get("success"), bool) or
               not math.isfinite(float(item.get("score", float("nan")))) for item in details.values()):
            raise ValueError("invalid per-episode success or score")
        cameras = {}
        for video in path.parent.rglob("*.mp4"):
            match = re.match(r"episode_(\d+)_cam_(head|left_wrist|right_wrist)(?:_|$)", video.name)
            if match and video.stat().st_size > 0:
                cameras.setdefault(int(match.group(1)), set()).add(match.group(2))
        missing = {str(key): sorted(CAMERAS - cameras.get(int(key), set()))
                   for key in details if CAMERAS - cameras.get(int(key), set())}
        if missing:
            return {**report, "reason": "missing_camera_videos", "missing": missing}
        return {**report, "complete": True, "eval_time": eval_time,
                "success_rate": result.get("success_rate"), "score": result.get("score"),
                "layout_ids": layout_ids, "result_sha256": sha(path)}
    except (ValueError, TypeError, KeyError, OSError) as error:
        return {**report, "reason": "invalid_result", "error": str(error)}


def gpu_snapshot():
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
                                   "--format=csv,noheader,nounits"], text=True, timeout=15)
    return [{"gpu": int(row[0]), "uuid": row[1].strip(), "memory_mib": int(row[2]),
             "util_percent": int(row[3])} for row in csv.reader(io.StringIO(raw))
            if int(row[0]) in set(GPUS)]


class Sweep:
    def __init__(self, cfg, plan):
        self.cfg, self.plan = cfg, plan
        self.repo = Path(cfg["repo"])
        self.run = Path(cfg["project"]) / "runs" / cfg["run_id"]
        self.run.mkdir(parents=True, exist_ok=True)
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.guard = ProcessGuard(cfg["run_id"], int(cfg["uid"]), self.run / "cleanup")
        self.summaries = {}
        self.attempt = str(time.time_ns())
        self.events = self.run / f"events-{self.attempt}.jsonl"
        self.telemetry_stop = threading.Event()
        self.gpu_slots = {gpu: threading.Lock() for gpu in set(GPUS)}

    def event(self, kind, **values):
        row = {"time": time.time(), "event": kind, **values}
        with self.lock:
            with self.events.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
        print(json.dumps(row, ensure_ascii=False), flush=True)

    def verify_sources(self):
        for path, expected in self.plan["source_sha256"].items():
            if sha(Path(path)) != expected:
                raise RuntimeError(f"Source/config changed during run: {path}")

    def task_id(self, seed, task):
        return f"{self.cfg['run_id']}_s{seed}_{task}"

    def wait_checkpoint(self, seed):
        """Later seeds may download while seed0 runs, but cannot launch unchecked."""
        checkpoint = Path(self.cfg["checkpoint"]) / f"RoboDojo-sim-arx_x5-joint-{seed}/59999"
        manifest = checkpoint / ".download-pi05/manifest.json"
        ready_path = checkpoint / "pi05-inference-ready.json"
        started = time.monotonic()
        while not ready_path.is_file():
            if self.stop.is_set():
                raise RuntimeError("Interrupted while waiting for checkpoint")
            download_exit = Path(self.cfg["project"]) / f"logs/download-pi05-seed{seed}.exit"
            if download_exit.exists() and download_exit.read_text().strip() != "0":
                raise RuntimeError(f"Pi05 seed{seed} download failed; see its log")
            if time.monotonic() - started > 6 * 3600:
                raise RuntimeError(f"Pi05 seed{seed} checkpoint wait exceeded six hours")
            self.stop.wait(15)
        if sha(manifest) != self.plan["checkpoint_manifest_sha256"][str(seed)]:
            raise RuntimeError("Pinned checkpoint manifest changed")
        ready = json.loads(ready_path.read_text())
        rows = ready["files"]
        expected = self.plan["checkpoint_files"][str(seed)]
        if (ready["status"] != "ready" or ready["revision"] != self.cfg["checkpoint_revision"]
                or not all(row.get("verified") for row in rows)
                or [{k:r[k] for k in ("path", "size", "sha256")} for r in rows] != expected):
            raise RuntimeError(f"Pi05 seed{seed} incomplete or mismatched ready receipt")
        if not all((checkpoint/row["path"]).is_file() and (checkpoint/row["path"]).stat().st_size == row["size"] for row in rows):
            raise RuntimeError(f"Pi05 seed{seed} inference file missing")
        self.event("checkpoint_verified_for_seed", seed=seed, files=len(rows), revision=ready["revision"])

    def result_path(self, seed, task):
        return self.repo / "eval_result/RoboDojo" / task / POLICY / "arx_x5" / (
            f"{seed}_ckpt_name={CKPT_NAME},action_type=joint") / self.task_id(seed, task) / "_result.json"

    def update(self, seed, task, **values):
        with self.lock:
            summary = self.summaries[seed]
            previous = summary["results"].get(task, {})
            summary["results"][task] = {**previous, "task": task, "seed": seed,
                "run_id": self.task_id(seed, task), "expected_episodes": self.plan["budgets"][task],
                **values, "updated_at": time.time()}
            summary["complete_tasks"] = sum(row.get("status") == "COMPLETE"
                                            for row in summary["results"].values())
            atomic_json(self.run / f"seed{seed}" / "summary.json", summary)

    def launch(self, argv, run_id, role, gpu, prefix, directory):
        directory.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update(DOJO_SWEEP_ID=self.cfg["run_id"], ROBODOJO_RUN_ID=run_id, DOJO_ROLE=role)
        exports = {
            "DOJO_SWEEP_ID": self.cfg["run_id"], "ROBODOJO_RUN_ID": run_id, "DOJO_ROLE": role,
            "EVAL_ENV_TYPE": "sim", "OPENPI_DATA_HOME": str(Path(self.cfg["project"]) / "cache/openpi"),
            "UV_PYTHON_INSTALL_DIR": str(Path(self.cfg["project"]) / "tools/uv-python"),
            "ROBODOJO_EGL_VENDOR_FILE": str(Path(self.cfg["project"]) / "cache/nvidia-egl/10_nvidia.json"),
            "ROBODOJO_NVIDIA_LIBRARY_DIR": str(Path(self.cfg["project"]) / "cache/nvidia-libs"),
            "ROBODOJO_RENDER_GPU": str(gpu),
            "ROBODOJO_VULKAN_COMPAT_SCRIPT": self.cfg["compat_script"],
            "ROBODOJO_MAX_BASH_RETRIES": "10", "ROBODOJO_FATAL_RESTART_COUNT": "0",
            "PYTHONUNBUFFERED": "1"}
        script = ("set -eo pipefail\nsource " + shlex.quote(self.cfg["project_env"]) +
                  "\nsource " + shlex.quote(self.cfg["conda_profile"]) +
                  "\nconda activate " + shlex.quote(prefix) + "\nunset EVAL_NUM CUDA_VISIBLE_DEVICES\n" +
                  "\n".join("export " + key + "=" + shlex.quote(value) for key, value in exports.items()) +
                  "\ncd " + shlex.quote(str(self.repo)) + "\nexec " + shlex.join(argv) + "\n")
        script_path = directory / f"command-{self.attempt}.sh"
        script_path.write_text(script, encoding="utf-8")
        log_path = directory / f"stdout-{self.attempt}.log"
        with log_path.open("wb") as stream:
            proc = subprocess.Popen(["bash", str(script_path)], cwd=self.repo, env=env,
                                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        identity = self.guard.identity(proc.pid)
        if identity is None:
            # Very fast failures still retain command/log and the Popen return code.
            identity = {"pid": proc.pid, "run_id": run_id, "uid": int(self.cfg["uid"])}
        atomic_json(directory / f"process-{self.attempt}.json", identity)
        self.event("process_started", role=role, run_id=run_id, pid=proc.pid, log=str(log_path))
        return proc

    def wait_server(self, proc, port):
        deadline = time.monotonic() + 600  # same startup allowance as official eval
        while not self.stop.is_set() and time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"Policy server exited before ready: {proc.returncode}")
            # Read LISTEN state; avoids probe-induced WebSocket handshake errors.
            for table in (Path("/proc/net/tcp"), Path("/proc/net/tcp6")):
                if table.exists():
                    for line in table.read_text().splitlines()[1:]:
                        fields = line.split()
                        if fields[3] == "0A" and int(fields[1].rsplit(":", 1)[1], 16) == port:
                            return
            self.stop.wait(1)
        raise RuntimeError("Policy server startup interrupted or exceeded 600 seconds")

    def gpu_lane(self, gpu):
        """Keep original groups/ports, but never wait on another GPU's seed."""
        groups = sorted((group for group in self.plan["groups"] if group["gpu"] == gpu),
                        key=lambda group: group["worker"])
        if len(groups) != 2 or len({group["worker"] for group in groups}) != 2:
            raise RuntimeError(f"Expected original two distinct groups for GPU{gpu}")
        for seed in SEEDS:
            if self.stop.is_set():
                return
            self.wait_checkpoint(seed)
            self.event("gpu_lane_seed_started", gpu=gpu, seed=seed,
                       workers=[group["worker"] for group in groups])
            for group in groups:
                if self.stop.is_set():
                    return
                self.worker(seed, group)
        self.event("gpu_lane_finished", gpu=gpu)

    def worker(self, seed, group):
        slot = self.gpu_slots[group["gpu"]]
        while not self.stop.is_set():
            if slot.acquire(timeout=1):
                try:
                    return self.worker_body(seed, group)
                finally:
                    slot.release()

    def worker_body(self, seed, group):
        worker, gpu, port = group["worker"], group["gpu"], int(group["port"])
        tasks = [row["task"] for row in group["tasks"]]
        if self.stop.is_set():
            return
        pending = []
        for task in tasks:
            check = result_check(self.result_path(seed, task), self.plan["budgets"][task])
            if check["complete"]:
                self.update(seed, task, status="COMPLETE", skipped_existing=True, result=check)
            else:
                pending.append(task)
        if not pending:
            return
        server_id = f"{self.cfg['run_id']}_s{seed}_w{worker}_server"
        work_dir = self.run / f"seed{seed}" / f"worker{worker}"
        server = None
        current_task = None
        try:
            self.guard.cleanup(server_id, "before_server_start")
            command = ["bash", "scripts/robodojo.sh", "server", "--policy-dir", "XPolicyLab/policy/Pi_05",
                       "--task", pending[0], "--ckpt", CKPT_NAME, "--env-cfg", "arx_x5",
                       "--action-type", "joint", "--seed", str(seed), "--policy-env", "uv",
                       "--policy-gpu", str(gpu), "--policy-port", str(port), "--bind-host", "127.0.0.1"]
            server_command = command
            server = self.launch(command, server_id, "server", gpu, self.cfg["sim_env"], work_dir / "server")
            self.wait_server(server, port)
            self.event("server_ready", seed=seed, worker=worker, gpu=gpu, port=port)
            for task in pending:
                if self.stop.is_set():
                    break
                self.verify_sources()
                current_task = task
                run_id = self.task_id(seed, task)
                self.guard.cleanup(run_id, "before_task_start")
                if server.poll() is not None:
                    raise RuntimeError(f"Worker {worker} policy server exited: {server.returncode}")
                command = ["bash", "scripts/robodojo.sh", "client", "--policy-dir", "XPolicyLab/policy/Pi_05",
                           "--task", task, "--ckpt", CKPT_NAME, "--env-cfg", "arx_x5", "--action-type", "joint",
                           "--seed", str(seed), "--eval-num", "native", "--policy-host", "127.0.0.1",
                           "--policy-port", str(port), "--env-gpu", str(gpu)]
                server = run_client(self, seed, task, worker, gpu, command, work_dir,
                                    server, server_id, server_command, self.cfg["sim_env"])
        except Exception as error:
            if current_task:
                self.update(seed, current_task, status="ERROR", error=repr(error), error_context=error_context(error))
            self.event("worker_error", worker=worker, seed=seed, error=repr(error), traceback=traceback.format_exc(), error_context=error_context(error))
            self.stop.set()
            raise
        finally:
            self.guard.cleanup(server_id, "worker_finished_or_interrupted")
            if server is not None:
                server.wait(timeout=10)

    def official_summary(self, checks):
        """Give upstream summarize only this sweep's exact selected result folders."""
        selected_root = self.run / "official_eval" / "RoboDojo"
        selected_root.mkdir(parents=True, exist_ok=True)
        selected = []
        for check in checks:
            if not check["complete"]:
                continue
            source = Path(check["path"]).parent
            relative = source.relative_to(self.repo / "eval_result/RoboDojo")
            target = selected_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_symlink():
                if target.resolve() != source.resolve():
                    raise RuntimeError(f"Wrong existing summary link: {target}")
            elif target.exists():
                raise RuntimeError(f"Summary path is not a managed symlink: {target}")
            else:
                target.symlink_to(source, target_is_directory=True)
            selected.append(check)
        atomic_json(self.run / "official_eval" / "selected-results.json", selected)
        env = os.environ.copy()
        env["ROBODOJO_EVAL_ROOT"] = str(selected_root)
        with (self.run / f"summarize-{self.attempt}.log").open("wb") as log:
            subprocess.run([sys.executable, str(self.repo / "scripts/internal/summarize_result.py")],
                           cwd=self.repo, env=env, stdout=log, stderr=subprocess.STDOUT,
                           timeout=120, check=True)

    def telemetry(self):
        while not self.telemetry_stop.is_set():
            try:
                owned = self.guard.scan()
                atomic_json(self.run / "processes-current.json", {"time": time.time(), "processes": owned})
                mem = {line.split(":")[0]: int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()
                       if line.startswith(("MemTotal:", "MemAvailable:"))}
                row = {"time": time.time(), "gpus": gpu_snapshot(), "owned_processes": len(owned), "memory_kib": mem}
                with (self.run / f"resources-{self.attempt}.jsonl").open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
            except Exception as error:
                self.event("telemetry_error", error=repr(error), error_context=error_context(error))
            self.telemetry_stop.wait(10)

    def release_status(self):
        """Allow driver bookkeeping to settle after the exact processes exited."""
        deadline = time.monotonic() + 30
        after, error = [], None
        while True:
            processes_clear = not self.guard.scan()
            try:
                after = gpu_snapshot()
                error = None
            except Exception as exc:
                self.event("gpu_status_error", error=repr(exc), error_context=error_context(exc))
                after, error = [], repr(exc)
            released = processes_clear and len(after) == 4 and all(
                row["memory_mib"] <= int(self.cfg.get("idle_gpu_memory_mib", 512)) for row in after)
            if released or time.monotonic() >= deadline:
                return processes_clear, released, after, error
            time.sleep(2)

    def cleanup_only(self):
        """Does not read HEAD/config/layouts; must not overlap a live controller."""
        lockfile = (self.run / "controller.lock").open("a+")
        fcntl.flock(lockfile.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt, error = None, None
        cleanup_error_context = None
        try:
            receipt = self.guard.cleanup(None, "outer_controller_cleanup_only")
            clear, released, after, error = self.release_status()
        except Exception as exc:
            clear, released, after, error = False, False, [], repr(exc)
            cleanup_error_context = error_context(exc)
        result = {"run_id": self.cfg["run_id"], "operation": "cleanup_only", "time": time.time(),
                  "processes_clear": clear, "gpus_released": released,
                  "gpu_after": after, "cleanup_receipt": receipt, "error": error,
                  "error_context": cleanup_error_context,
                  "exit_code": 0 if clear and released else 5}
        atomic_json(self.run / f"cleanup-only-{self.attempt}.json", result)
        atomic_json(self.run / "cleanup-only-latest.json", result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        fcntl.flock(lockfile.fileno(), fcntl.LOCK_UN)
        lockfile.close()
        return result["exit_code"]

    def run_all(self):
        lockfile = (self.run / "controller.lock").open("a+")
        fcntl.flock(lockfile.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        prior = self.run / "plan.json"
        if prior.exists():
            saved = json.loads(prior.read_text())
            keys = ("config", "source_sha256", "controller_sha256", "checkpoint_manifest_sha256",
                    "checkpoint_files", "tasks", "budgets", "workers_per_gpu")
            if any(saved[key] != self.plan[key] for key in keys):
                raise RuntimeError("Existing run identity differs; do not mix versions/configuration")
        else:
            atomic_json(prior, self.plan)
            archive = self.run / "controller-source"
            archive.mkdir(exist_ok=True)
            for name in ("dojo_sweep.py", "process_guard.py", "eval_recovery.py", "resume_results.py"):
                shutil.copy2(Path(__file__).with_name(name), archive / name)
        atomic_json(self.run / f"controller-{self.attempt}.json",
                    {"pid": os.getpid(), "uid": os.getuid(), "stat": Path("/proc/self/stat").read_text(),
                     "attempt": self.attempt, "started_at": time.time()})
        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, lambda _s, _f: self.stop.set())
        exit_code, state = 4, "INFRASTRUCTURE_ERROR"
        monitor = None
        error_text = None
        processes_clear = False
        try:
            self.guard.cleanup(None, "controller_start_same_sweep_residuals")
            before = gpu_snapshot()
            if any(row["memory_mib"] > int(self.cfg.get("idle_gpu_memory_mib", 512)) for row in before):
                raise RuntimeError(f"GPUs4-7 not released by outer RLT controller: {before}")
            for port in self.cfg["ports"]:
                with socket.socket() as check:
                    check.bind(("127.0.0.1", int(port)))
            atomic_json(self.run / f"gpu-before-{self.attempt}.json", before)
            monitor = threading.Thread(target=self.telemetry, daemon=True)
            monitor.start()
            # All summaries must exist before different lanes reach different seeds.
            for seed in SEEDS:
                self.summaries[seed] = {"run_id": self.cfg["run_id"], "seed": seed,
                    "expected_tasks": 54, "expected_episodes": 2100, "results": {
                        task: {"task": task, "seed": seed, "run_id": self.task_id(seed, task),
                               "status": "PENDING", "expected_episodes": self.plan["budgets"][task]}
                        for task in self.plan["tasks"]}}
                atomic_json(self.run / f"seed{seed}" / "summary.json", self.summaries[seed])
            with ThreadPoolExecutor(max_workers=len(set(GPUS))) as pool:
                futures = [pool.submit(self.gpu_lane, gpu) for gpu in sorted(set(GPUS))]
                errors = []
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as error:
                        self.stop.set()
                        self.event("gpu_lane_error", error=repr(error), error_context=error_context(error))
                        errors.append(repr(error))
                if errors:
                    raise RuntimeError("; ".join(errors))
            checks = [{"seed": seed, "task": task, **result_check(self.result_path(seed, task), self.plan["budgets"][task])}
                      for seed in SEEDS for task in self.plan["tasks"]]
            atomic_json(self.run / "result-audit.json", checks)
            self.official_summary(checks)
            if all(check["complete"] for check in checks):
                exit_code, state = 0, "COMPLETE"
            elif self.stop.is_set():
                exit_code, state = 3, "INTERRUPTED"
            else:
                exit_code, state = 2, "INCOMPLETE"
        except Exception as error:
            error_text = repr(error)
            self.event("sweep_error", error=error_text, error_context=error_context(error))
        finally:
            self.telemetry_stop.set()
            if monitor:
                monitor.join(timeout=30)
            try:
                cleanup_receipt = self.guard.cleanup(None, "controller_final_cleanup")
                processes_clear = not self.guard.scan()
            except Exception as error:
                self.event("cleanup_error", error=repr(error), error_context=error_context(error))
                cleanup_receipt = None
                exit_code, state = 5, "CLEANUP_FAILED"
                error_text = f"{error_text or ''}; {error!r}"
            try:
                processes_clear, released, after, release_error = self.release_status()
                if release_error:
                    error_text = f"{error_text or ''}; GPU release: {release_error}"
            except Exception as error:
                self.event("release_error", error=repr(error), error_context=error_context(error))
                processes_clear, released, after = False, False, []
                error_text = f"{error_text or ''}; release check: {error!r}"
            if not processes_clear or not released:
                exit_code, state = 5, "CLEANUP_FAILED"
            final = {"state": state, "exit_code": exit_code, "run_id": self.cfg["run_id"],
                     "finished_at": time.time(), "error": error_text,
                     "processes_clear": processes_clear, "gpus_released": released,
                     "gpu_after": after, "cleanup_receipt": cleanup_receipt,
                     "rlt_action": "none; outer controller decides and performs restoration"}
            atomic_json(self.run / "final.json", final)
            self.event("sweep_finished", **final)
            fcntl.flock(lockfile.fileno(), fcntl.LOCK_UN)
            lockfile.close()
        return exit_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--plan-only", action="store_true")
    modes.add_argument("--cleanup-only", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    try:
        if not SAFE_NAME.fullmatch(cfg["run_id"]):
            raise ValueError("Invalid run_id")
        if args.cleanup_only:
            return Sweep(cfg, None).cleanup_only()
        plan = build_plan(cfg)
        if args.plan_only:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return 0
        return Sweep(cfg, plan).run_all()
    except Exception as error:
        print(json.dumps({"state": "PREFLIGHT_OR_LOCK_ERROR", "exit_code": 4,
                          "error": repr(error), "error_context": error_context(error)}, ensure_ascii=False), file=sys.stderr, flush=True)
        return 4


if __name__ == "__main__":
    sys.exit(main())
