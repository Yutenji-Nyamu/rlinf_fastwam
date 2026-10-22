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
    template, task, gpu, output, seeds_path, label, policy=None, steps=1, method="cp"
):
    c = copy.deepcopy(template)
    limit = int(task["step_limit"])
    c.update(
        suite="pi05_distillation_multitask",
        task=task["task"],
        num_envs=16,
        gpu=gpu,
        step_limit=limit,
        official_step_limit=limit,
        condition=label,
        output=str(output),
        num_steps=steps,
        batch=0,
        warmup_prompt=task["task"].replace("_", " "),
    )
    c.pop("distill_weights", None)
    c["model"]["openpi"].update(
        num_steps=steps, distill_target_time=bool(policy) and method == "cp"
    )
    if policy:
        c["distill_weights"] = policy
    ec = c["env"]
    ec.update(
        total_num_envs=16,
        seeds_path=str(seeds_path),
        max_steps_per_rollout_epoch=limit,
        max_episode_steps=limit,
    )
    ec["task_config"].update(
        task_name=task["task"], step_lim=limit, save_path=str(output / "env")
    )
    ec["video_cfg"]["video_base_dir"] = str(output / "unused")
    return c


def sweep(deployment, label, *, policy=None, steps=1, tasks=None, method="cp"):
    """One batch per task; missing simulator batches are not policy failures."""
    deployment = Path(deployment)
    prepared = read(deployment / "prepared.json")
    output = Path(prepared["output"])
    tx = deployment / ("eval-" + label)
    tx.mkdir(exist_ok=False)
    task_rows = tasks or read(deployment / "task-list.json")["tasks"]
    pending = list(task_rows)
    template = read(deployment / "eval-template.json")
    slots = [(g, lane) for g in prepared["physical_gpus"] for lane in range(2)]
    running, finished = {}, []

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
                    deployment / "eval-seeds" / (task["task"] + ".json"),
                    label,
                    policy,
                    steps,
                    method,
                )
                cfg_path = tx / (task["task"] + ".json")
                save(cfg_path, config)
                env = dict(
                    os.environ,
                    **read(output / f"environment-g{slot[0]}.json"),
                    POLICY_EVAL_TOKEN=token,
                )
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
                if child.poll() is None and not error:
                    continue
                done = (
                    read(dest / "done.json") if (dest / "done.json").exists() else None
                )
                valid = (
                    not error
                    and child.poll() == 0
                    and done
                    and done.get("episodes") == 16
                )
                if not valid:
                    error = (
                        error
                        or f"Evaluator exit {child.poll()} without complete 16-episode receipt"
                    )
                    cleanup(job["token"])
                    child.wait(timeout=30)
                job["log"].close()
                row = {
                    "task": job["task"]["task"],
                    "gpu": slot[0],
                    "valid": bool(valid),
                    "episodes": 16 if valid else 0,
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
    good = [row for row in finished if row["valid"]]
    summary = {
        "time": time.time(),
        "condition": label,
        "attempted_tasks": len(task_rows),
        "valid_tasks": len(good),
        "missing_tasks": len(task_rows) - len(good),
        "episodes": 16 * len(good),
        "successes": sum(row["successes"] for row in good),
        "mean_success_rate": (
            sum(row["successes"] / 16 for row in good) / len(good) if good else None
        ),
        "aggregation": "equal task mean over valid tasks; missing tasks listed separately",
        "tasks": finished,
    }
    save(output / ("summary-" + label + ".json"), summary)
    # Preserve both all-valid means and a fixed common-task comparison for plotting.
    summaries = [
        read(p)
        for p in sorted(output.glob("summary-*.json"))
        if p.name.startswith(("summary-cp-s", "summary-endpoint-s", "summary-raw-n"))
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
                }
            )
        save(
            output / "comparison.json",
            {"common_tasks": sorted(common), "points": comparison},
        )
    return summary
