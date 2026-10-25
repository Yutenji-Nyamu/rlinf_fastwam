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

"""Four task lanes on two GPUs, reusing the accepted native policy evaluator."""

import copy
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, "/data/chenyiteng/deployment-20261008/bc-signal-tau-v1/ops")
from common import exact_signal, proc, read, same, save

ROOT = Path(__file__).resolve().parents[2]
PYTHON = "/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python"


def completed_checkpoint(output, method, stops):
    """Continue only from a saved evaluation boundary, preserving prior points."""
    output = Path(output)
    receipt = read(output / method / "checkpoint.json")
    step = receipt["step"]
    assert step in stops and Path(receipt["path"]).is_file()
    for stop in stops:
        if stop <= step:
            result = read(output / f"summary-{method}-s{stop:06d}.json")
            assert result["attempted_tasks"] == 50 and len(result["tasks"]) == 50
    return receipt


def token_processes(token):
    """Only this batch's process family; another task may share its GPU."""
    found = []
    needle = ("POLICY_EVAL_TOKEN=" + token).encode()
    for path in Path("/proc").glob("[0-9]*/environ"):
        try:
            if (
                path.stat().st_uid != os.getuid()
                or needle not in path.read_bytes().split(b"\0")
            ):
                continue
            identity = proc(int(path.parent.name))
            if identity and identity.get("state") != "Z":
                found.append(identity)
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
    return found


def cleanup(token):
    identities = token_processes(token)
    for identity in identities:
        if same(identity):
            exact_signal(identity, signal.SIGTERM)
    deadline = time.monotonic() + 20
    while token_processes(token) and time.monotonic() < deadline:
        time.sleep(0.5)
    for identity in token_processes(token):
        if same(identity):
            exact_signal(identity, signal.SIGKILL)
    return identities


def make_config(
    template,
    task,
    gpu,
    output,
    seeds_path,
    label,
    policy=None,
    steps=1,
    method="cp",
    *,
    options=None,
):
    options = options or {}
    episodes = int(options.get("episodes_per_task", 16))
    assert episodes in (8, 16)
    c = copy.deepcopy(template)
    limit = int(task["step_limit"])
    c.update(
        suite="pi05_distillation_multitask",
        task=task["task"],
        num_envs=episodes,
        gpu=gpu,
        step_limit=limit,
        official_step_limit=limit,
        condition=label,
        output=str(output),
        num_steps=steps,
        batch=0,
        warmup_prompt=task["task"].replace("_", " "),
    )
    for key in (
        "recording_mode",
        "record_decisions",
        "environment_step_timeout_seconds",
    ):
        if key in options:
            c[key] = options[key]
    if "execute_chunk" in options and not options.get("chunk_sweep"):
        c["execute_chunk"] = int(options["execute_chunk"])
    c.pop("distill_weights", None)
    c["model"]["openpi"].update(
        num_steps=steps, distill_target_time=bool(policy) and method == "cp"
    )
    if policy:
        c["distill_weights"] = policy
    if options.get("chunk_sweep"):
        assert policy is None and episodes == 8
        batch = int(options["seed_batch"])
        c.update(
            suite="pi05_chunk_sweep",
            execute_chunk=int(options["execute_chunk"]),
            batch=batch,
            noise_seed=int(template["noise_seed"]) + 1000003 * batch,
        )
    ec = c["env"]
    ec.update(
        total_num_envs=episodes,
        seeds_path=str(seeds_path),
        max_steps_per_rollout_epoch=limit,
        max_episode_steps=limit,
    )
    ec["task_config"].update(
        task_name=task["task"], step_lim=limit, save_path=str(output / "env")
    )
    ec["video_cfg"]["video_base_dir"] = str(output / "unused")
    return c


def sweep(
    deployment, label, *, policy=None, steps=1, tasks=None, method="cp", overrides=None
):
    """One batch per task; missing simulator batches are not policy failures."""
    deployment = Path(deployment)
    prepared = read(deployment / "prepared.json")
    sweep_started = time.time()
    settings_path = deployment / "eval-settings.json"
    options = read(settings_path) if settings_path.exists() else {}
    options.update(overrides or {})
    episodes = int(options.get("episodes_per_task", 16))
    lanes = int(options.get("tasks_per_gpu", 2))
    resident = bool(options.get("resident_model", False))
    assert episodes in (8, 16) and 1 <= lanes <= 4
    output = Path(prepared["output"])
    tx = deployment / ("eval-" + label)
    tx.mkdir(exist_ok=False)
    task_rows = tasks or read(deployment / "task-list.json")["tasks"]
    pending = list(task_rows)
    template = read(deployment / "eval-template.json")
    seeds_dir = deployment / (
        f"eval-seeds-b{int(options['seed_batch'])}"
        if options.get("chunk_sweep")
        else "eval-seeds"
    )
    save(
        tx / "settings.json",
        dict(options, episodes_per_task=episodes, tasks_per_gpu=lanes),
    )
    slots = [(g, lane) for g in prepared["physical_gpus"] for lane in range(lanes)]
    running, finished = {}, []
    workers, generations = {}, {}

    def publish():
        save(
            tx / "progress.json",
            {
                "time": time.time(),
                "total_tasks": len(task_rows),
                "completed_tasks": len(finished),
                "pending_tasks": len(pending),
                "running": [
                    {
                        "task": j["task"]["task"],
                        "gpu": s[0],
                        "lane": s[1],
                        "identity": j["identity"],
                    }
                    for s, j in running.items()
                ],
                "results": finished,
                "episodes_per_task": episodes,
                "tasks_per_gpu": lanes,
                "resident_model": resident,
            },
        )

    try:
        while pending or running:
            for slot in slots:
                if slot in running or not pending:
                    continue
                task = pending.pop(0)
                dest = output / "batches" / label / task["task"]
                token = f"{deployment.name}-{label}-{task['task']}"
                config = make_config(
                    template,
                    task,
                    slot[0],
                    dest,
                    seeds_dir / (task["task"] + ".json"),
                    label,
                    policy,
                    steps,
                    method,
                    options=options,
                )
                cfg_path = tx / (task["task"] + ".json")
                save(cfg_path, config)
                env = dict(
                    os.environ,
                    **read(output / f"environment-g{slot[0]}.json"),
                    POLICY_EVAL_TOKEN=token,
                )
                if resident:
                    if slot not in workers:
                        generation = generations.get(slot, 0) + 1
                        generations[slot] = generation
                        name = f"g{slot[0]}-lane{slot[1]}-worker{generation}"
                        token = f"{deployment.name}-{label}-{name}"
                        env["POLICY_EVAL_TOKEN"] = token
                        status = tx / (name + "-status.json")
                        log = (tx / (name + ".log")).open("x")
                        child = subprocess.Popen(
                            [
                                PYTHON,
                                "-u",
                                "-B",
                                str(ROOT / "tools/pi05_policy_eval/resident.py"),
                                str(status),
                            ],
                            cwd=ROOT,
                            env=env,
                            stdin=subprocess.PIPE,
                            text=True,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                        workers[slot] = {
                            "process": child,
                            "log": log,
                            "token": token,
                            "status": status,
                        }
                    worker = workers[slot]
                    child, log, token = (
                        worker["process"],
                        worker["log"],
                        worker["token"],
                    )
                    try:
                        child.stdin.write(json.dumps({"config": str(cfg_path)}) + "\n")
                        child.stdin.flush()
                    except (BrokenPipeError, OSError) as exc:
                        save(dest / "error.json", {"error": repr(exc)})
                else:
                    log = (tx / (task["task"] + ".log")).open("x")
                    child = subprocess.Popen(
                        [
                            PYTHON,
                            "-u",
                            "-B",
                            str(ROOT / "tools/pi05_policy_eval/run_batch.py"),
                            str(cfg_path),
                        ],
                        cwd=ROOT,
                        env=env,
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                running[slot] = {
                    "process": child,
                    "log": log,
                    "token": token,
                    "task": task,
                    "output": dest,
                    "identity": proc(child.pid),
                    "started": time.time(),
                }
            publish()
            time.sleep(5)
            for slot, job in list(running.items()):
                child, dest = job["process"], job["output"]
                heartbeat = dest / "progress.json"
                last = (
                    max(job["started"], heartbeat.stat().st_mtime)
                    if heartbeat.exists()
                    else job["started"]
                )
                error_path = dest / "error.json"
                error = read(error_path).get("error") if error_path.exists() else None
                if not error and time.time() - last > 1800:
                    error = "No inference/environment progress for 1800 seconds"
                # Longer native tasks get proportional time; no physical-action budget changes.
                if not error and time.time() - job["started"] > max(
                    10800, job["task"]["step_limit"] * 18
                ):
                    error = "Finite task wall-clock budget exceeded"
                ready = False
                if resident and workers[slot]["status"].exists():
                    state = read(workers[slot]["status"])
                    ready = state.get("state") == "idle" and state.get("output") == str(
                        dest
                    )
                if child.poll() is None and not error and not ready:
                    continue
                done = (
                    read(dest / "done.json") if (dest / "done.json").exists() else None
                )
                valid = (
                    not error
                    and (ready if resident else child.poll() == 0)
                    and done
                    and done.get("episodes") == episodes
                )
                if not valid:
                    error = (
                        error
                        or f"Evaluator exit {child.poll()} without complete {episodes}-episode receipt"
                    )
                    cleanup(job["token"])
                    child.wait(timeout=30)
                if not resident or not valid:
                    job["log"].close()
                if resident and not valid:
                    workers.pop(slot)
                row = {
                    "task": job["task"]["task"],
                    "gpu": slot[0],
                    "valid": bool(valid),
                    "episodes": episodes if valid else 0,
                    "successes": done["successes"] if valid else None,
                    "elapsed_seconds": time.time() - job["started"],
                    "error": error,
                    "output": str(dest),
                }
                finished.append(row)
                del running[slot]
            publish()
    finally:
        for job in running.values():
            cleanup(job["token"])
            job["log"].close()
        for worker in workers.values():
            child = worker["process"]
            if child.poll() is None:
                try:
                    child.stdin.write('{"stop":true}\n')
                    child.stdin.flush()
                    child.wait(timeout=30)
                except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                    cleanup(worker["token"])
                    child.wait(timeout=30)
            worker["log"].close()
    good = [row for row in finished if row["valid"]]
    summary = {
        "time": time.time(),
        "condition": label,
        "attempted_tasks": len(task_rows),
        "valid_tasks": len(good),
        "missing_tasks": len(task_rows) - len(good),
        "episodes": episodes * len(good),
        "episodes_per_task": episodes,
        "evaluation_settings": options,
        "elapsed_seconds": time.time() - sweep_started,
        "successes": sum(row["successes"] for row in good),
        "mean_success_rate": (
            sum(row["successes"] / row["episodes"] for row in good) / len(good)
            if good
            else None
        ),
        "aggregation": "equal task mean over valid tasks; missing tasks listed separately",
        "tasks": finished,
    }
    save(output / ("summary-" + label + ".json"), summary)
    # Preserve both all-valid means and a fixed common-task comparison for plotting.
    summaries = [
        read(p)
        for p in sorted(output.glob("summary-*.json"))
        if p.name.startswith(
            ("summary-cp-s", "summary-endpoint-s", "summary-onedp-s", "summary-raw-n")
        )
    ]
    if summaries:
        common = set.intersection(
            *[{r["task"] for r in s["tasks"] if r["valid"]} for s in summaries]
        )
        comparison = []
        for s in summaries:
            values = [
                r["successes"] / r["episodes"]
                for r in s["tasks"]
                if r["task"] in common
            ]
            comparison.append(
                {
                    "condition": s["condition"],
                    "valid_tasks": s["valid_tasks"],
                    "missing_tasks": s["missing_tasks"],
                    "all_valid_mean": s["mean_success_rate"],
                    "common_mean": sum(values) / len(values) if values else None,
                    "episodes_per_task": s.get("episodes_per_task", 16),
                }
            )
        save(
            output / "comparison.json",
            {"common_tasks": sorted(common), "points": comparison},
        )
    return summary
