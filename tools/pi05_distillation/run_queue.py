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

"""Own the authorized GPU pair: raw eval, two-method smoke, then formal training.

Reuses the accepted per-card evaluation lane, exact process identities and
finite failure handling. A failed scientific stage is recorded, never relabeled
as a valid evaluation or silently resumed with different hyperparameters.
"""

import copy
import json
import math
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, "/data/chenyiteng/deployment-20261008/bc-signal-tau-v1/ops")
from common import exact_signal, gpus, proc, read, same, save

R = Path(__file__).resolve().parents[2]
D = Path(sys.argv[1])
O = Path(read(D / "prepared.json")["output"])
CARDS = tuple(read(D / "prepared.json").get("physical_gpus", [2, 3]))
assert len(CARDS) == 2 and len(set(CARDS)) == 2
GPU_LIST = ",".join(map(str, CARDS))
PYTHON = "/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python"
child = None
owned_children = []


def stop_owned(*_):
    """Stop only processes launched by this queue, checking exact identities."""
    for identity in owned_children:
        if same(identity):
            exact_signal(identity, signal.SIGTERM)
    event("STOPPED")
    raise SystemExit(143)


signal.signal(signal.SIGTERM, stop_owned)
signal.signal(signal.SIGINT, stop_owned)


def event(phase, **fields):
    """Publish only useful progress and the exact current process identity."""
    row = dict(time=time.time(), phase=phase, **fields)
    save(D / "training-queue-status.json", row)
    print(json.dumps(row), flush=True)


def train(method, output, stop_step, resume=None, label=None, *, extra_args=()):
    """Run one complete two-rank segment and require its checkpoint receipt."""
    global child
    wait_for_cards()
    env = dict(os.environ, **read(O / f"environment-g{CARDS[0]}.json"))
    env.update(
        CUDA_VISIBLE_DEVICES=GPU_LIST,
        DISTILL_GPU_PAIR=GPU_LIST,
        FQL_GPU_PAIR=GPU_LIST,
        DISTILL_RUN_TOKEN=D.name,
        PYTHONPATH=str(R) + ":" + env["ROBOTWIN_PATH"],
        TORCH_NCCL_ASYNC_ERROR_HANDLING="1",
    )
    env.pop("DOJO_GPU_SCOPE", None)
    allocator = read(D / f"{method}.json").get("cuda_allocator_conf")
    if allocator:
        env["PYTORCH_CUDA_ALLOC_CONF"] = allocator
    command = [
        PYTHON,
        "-u",
        "-B",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        str(
            R
            / "tools/pi05_distillation"
            / ("train_fql.py" if method == "fql_fixed_teacher" else "train.py")
        ),
        "--config",
        str(D / f"{method}.json"),
        "--output",
        str(output),
        "--stop-step",
        str(stop_step),
    ]
    if resume:
        command += ["--resume", resume]
    command += list(extra_args)
    label = label or f"{method}-{stop_step}"
    with (D / f"{label}.log").open("x") as log:
        child = subprocess.Popen(
            command,
            cwd=R,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        identity = proc(child.pid)
        owned_children.append(identity)
        event(
            "TRAINING",
            method=method,
            stop_step=stop_step,
            output=str(output),
            child=identity,
            command=command,
        )
        launched = time.time()
        while child.poll() is None:
            time.sleep(float(read(D / "prepared.json").get("poll_seconds", 15)))
            progress = output / "progress.json"
            # Allow model load/checkpoint I/O, but stop a stalled owned segment.
            last = (
                max(launched, progress.stat().st_mtime)
                if progress.exists()
                else launched
            )
            if time.time() - last > 3600:
                exact_signal(identity, signal.SIGTERM)
                raise TimeoutError("No training update for an hour")
    if child.returncode:
        raise RuntimeError(f"{label}: training exited {child.returncode}; see its log")
    receipt = read(
        output
        / ("progress.json" if "--capacity-only" in extra_args else "checkpoint.json")
    )
    assert receipt["step"] == stop_step
    return receipt


def wait_for_cards(timeout_seconds=1800):
    """Wait for transient foreign contexts; never terminate another job."""
    deadline = time.monotonic() + timeout_seconds
    announced = False
    while True:
        snapshot = gpus()
        busy = {g: snapshot[g]["processes"] for g in CARDS
                if snapshot[g]["processes"]}
        if not busy:
            return
        if not announced:
            event("WAITING_GPU_CLEAR", processes=busy)
            announced = True
        if time.monotonic() >= deadline:
            raise TimeoutError("GPU pair occupied for 30 minutes; no foreign processes killed")
        time.sleep(2)


def evaluate(method, receipt, label):
    """128 paired episodes, 32 per card, using the established evaluator."""
    transaction = D / f"eval-{label}"
    transaction.mkdir()
    configs = []
    for index, g in enumerate(CARDS):
        group = []
        for b in [index, index + 2]:
            c = copy.deepcopy(read(O / f"configs/raw_n1_b{b}.json"))
            c.update(gpu=g, condition=label, distill_weights=receipt["policy"])
            c["model"]["openpi"].update(distill_target_time=method == "cp")
            dest = O / "batches" / f"{label}_b{b}"
            c["output"] = str(dest)
            c["env"]["task_config"]["save_path"] = str(dest / "env")
            c["env"]["video_cfg"]["video_base_dir"] = str(dest / "unused")
            path = O / f"configs/{label}_b{b}.json"
            save(path, c)
            configs.append(path)
            group.append(str(path))
        save(
            transaction / f"lane-g{g}-plan.json",
            {
                "gpu": g,
                "stages": [{"condition": label, "configs": group, "summarize": False}],
                "recovery": {
                    "attempts": 2,
                    "continue_errors": True,
                    "batch_seconds": 10800,
                    "stall_seconds": 1800,
                    "barrier_seconds": 86400,
                },
            },
        )
    children = []
    for g in CARDS:
        env = dict(
            os.environ,
            **read(O / f"environment-g{g}.json"),
            POLICY_EVAL_ROOT=str(O),
            POLICY_EVAL_TRANSACTION=str(transaction),
            POLICY_EVAL_GPUS=GPU_LIST,
            POLICY_EVAL_UID=str(os.getuid()),
        )
        with (transaction / f"lane-g{g}.log").open("x") as log:
            p = subprocess.Popen(
                [PYTHON, "-u", "-B", str(R / "tools/pi05_policy_eval/lane.py"), str(g)],
                cwd=R,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        children.append(p)
        owned_children.append(proc(p.pid))
    event(
        "EVALUATING",
        method=method,
        condition=label,
        checkpoint=receipt,
        children=[proc(p.pid) for p in children],
    )
    for p in children:
        p.wait()
    results = []
    for path in configs:
        done = Path(read(path)["output"]) / "done.json"
        if done.exists():
            results.append(read(done))
    total = sum(r["episodes"] for r in results)
    successes = sum(r["successes"] for r in results)
    summary = {
        "condition": label,
        "method": method,
        "checkpoint": receipt,
        "valid": total,
        "requested": 128,
        "successes": successes,
        "rate": successes / total if total else None,
        "complete": total == 128,
    }
    save(O / f"summary-{label}.json", summary)
    return summary


def accept_smoke(method):
    """Require measured finite updates, successful resume and native N1 exports."""
    out = O / f"smoke-{method}"
    checks = []
    for rank in [0, 1]:
        check = read(out / f"inference-check-rank{rank}.json")
        assert check["action_expert_calls"] == 1 and check["shape"][1:] == [50, 14]
        assert (
            check["loaded_export"] and check["finite"] and check["resumed_step"] == 14
        )
        assert check["teacher_frozen_during_training"]
        assert (
            check["cuda_visible_devices"] == GPU_LIST
            and check["local_cuda_device"] == rank
        )
        rows = [
            json.loads(s)
            for s in (out / f"metrics-rank{rank}.jsonl").read_text().splitlines()
        ]
        assert rows[-1]["step"] == 14 and max(r["grad_norm"] for r in rows) > 0
        assert all(
            math.isfinite(v)
            for row in rows
            for v in row.values()
            if isinstance(v, (float, int))
        )
        checks.append(
            {
                "rank": rank,
                "inference": check,
                "last_update": rows[-1],
                "peak_gib": max(r["peak_gib"] for r in rows),
            }
        )
    save(D / f"smoke-{method}-accepted.json", {"passed": True, "checks": checks})


def main():
    """Proceed automatically once both method smokes meet explicit criteria."""
    assert os.getuid() == 20001
    smoke_only = "--smoke-only" in sys.argv[2:]
    save(D / "training-queue-identity.json", proc(os.getpid()))
    if not smoke_only:
        event("WAIT_RAW_EVALUATION")
        for g in CARDS:
            while not (D / f"lane-g{g}-released.json").exists():
                time.sleep(15)
    # An incomplete simulator baseline remains explicitly incomplete; it does
    # not invalidate the independent offline training data or block its smoke.
    for n in [] if smoke_only else [1, 10]:
        results = [read(p) for p in (O / "batches").glob(f"raw_n{n}_b*/done.json")]
        valid = sum(r["episodes"] for r in results)
        successes = sum(r["successes"] for r in results)
        save(
            O / f"summary-raw_n{n}.json",
            {
                "condition": f"raw_n{n}",
                "requested": 128,
                "valid": valid,
                "successes": successes,
                "rate": successes / valid if valid else None,
                "complete": valid == 128,
            },
        )
    if not smoke_only and (D / "external-smoke-request.json").exists():
        event("WAIT_EXTERNAL_SMOKE")
        while not (D / "external-smoke.json").exists():
            time.sleep(15)
    # Both algorithms are smoke-tested before either long formal run.
    for method in ["cp", "onedp"]:
        output = O / f"smoke-{method}"
        if not smoke_only and (D / "external-smoke.json").exists():
            evidence = read(D / "external-smoke.json")
            assert evidence["host"] == "sz2" and evidence["passed"]
            measured = read(output / "resolved.json")["config"].copy()
            formal = read(D / f"{method}.json").copy()
            for field in ["log_every", "checkpoint_steps"]:
                measured.pop(field)
                formal.pop(field)
            assert measured == formal, "Measured training configuration changed"
            accept_smoke(method)
            event("REUSED_EXTERNAL_SMOKE", method=method, source=evidence)
            continue
        try:
            receipt = train(method, output, 12, label=f"smoke-{method}")
        except RuntimeError:
            log = (D / f"smoke-{method}.log").read_text(errors="replace")
            c = read(D / f"{method}.json")
            if "CUDA out of memory" not in log or c["microbatch_per_rank"] != 32:
                raise
            assert all(not gpus()[g]["processes"] for g in CARDS)
            output.rename(O / f"smoke-{method}-micro32-failed")
            c["microbatch_per_rank"] = (
                16  # global B64 and all learning settings stay fixed
            )
            save(D / f"{method}.json", c)
            event(
                "SMOKE_MICROBATCH_RETRY",
                method=method,
                microbatch_per_rank=16,
                global_batch_size=64,
            )
            receipt = train(method, output, 12, label=f"smoke-{method}-micro16")
        receipt = train(
            method, output, 14, receipt["path"], label=f"smoke-{method}-resume"
        )
        save(D / f"smoke-{method}-done.json", receipt)
        accept_smoke(method)
    event("SMOKE_ACCEPTED")
    save(
        D / "formal-ready.json",
        {
            "time": time.time(),
            "smokes": ["cp", "onedp"],
            "automatic_checks_passed": True,
        },
    )
    if smoke_only:
        event("SMOKE_ONLY_COMPLETE")
        return
    for method in ["cp", "onedp"]:
        c = read(D / f"{method}.json")
        shape = read(O / f"smoke-{method}/resolved.json")
        total = shape["total_steps"]
        interval = c["eval_epochs"] * shape["steps_per_epoch"]
        saved_path = O / method / "checkpoint.json"
        saved = read(saved_path) if saved_path.exists() else None
        resume = saved["path"] if saved else None
        if saved:
            assert 0 < saved["step"] <= total
            assert Path(saved["path"]).is_file() and Path(saved["policy"]).is_file()
            label = f"{method}-s{saved['step']:06d}"
            if not (O / f"summary-{label}.json").exists():
                evaluate(method, saved, label)
        for stop in sorted({*range(interval, total, interval), total}):
            if saved and stop <= saved["step"]:
                continue
            receipt = train(method, O / method, stop, resume)
            evaluate(method, receipt, f"{method}-s{stop:06d}")
            resume = receipt["path"]
    event("COMPLETE")
    save(
        D / "released.json",
        {"time": time.time(), "completed": True, "identity": proc(os.getpid())},
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        event("FAILED", error=traceback.format_exc())
        raise
