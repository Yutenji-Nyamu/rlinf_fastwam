"""Bounded offline replay between an existing training owner and its successor.

Uses the already accepted identity, GPU admission, and token cleanup helpers.
Every failure releases the pair so the next existing experiment can proceed.
"""

import concurrent.futures
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

OPS = "/data/chenyiteng/deployment-20261008/bc-signal-tau-v1/ops"
sys.path.insert(0, OPS)
from common import gpus, proc, read, same, save


def main(directory):
    d = Path(directory)
    p = read(d / "plan.json")
    tool = Path(__file__).parent
    sys.path.insert(0, str(Path(p["model_source"]) / "tools/pi05_distillation"))
    from multitask_eval import cleanup, PYTHON

    me = proc(os.getpid())
    save(d / "identity.json", me)
    save(d / "status.json", dict(phase="WAIT_CURRENT_TRAINING", time=time.time()))
    while not (d / "registered.json").exists():
        time.sleep(0.1)
    assert read(d / "registered.json")["identity"]["pid"] == me["pid"]
    while same(p["predecessor_identity"]) or any(
        gpus()[g]["processes"] for g in p["gpus"]
    ):
        time.sleep(0.25)
    save(d / "status.json", dict(phase="SMOKE_THEN_OFFLINE_REPLAY", time=time.time()))

    def lane(gpu):
        cfg = d / f"replay-g{gpu}.json"
        token = f"{d.name}-offline-g{gpu}"
        env = dict(os.environ, **read(d / "environment-template.json"))
        env.update(
            CUDA_VISIBLE_DEVICES=str(gpu),
            DOJO_GPU_SCOPE=str(gpu),
            POLICY_EVAL_TOKEN=token,
        )
        started = time.time()
        try:
            with (d / f"replay-g{gpu}.log").open("x") as log:
                child = subprocess.Popen(
                    [PYTHON, "-u", "-B", str(tool / "teacher_replay.py"), str(cfg)],
                    cwd=p["model_source"],
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                save(d / f"child-g{gpu}.json", proc(child.pid))
                rc = child.wait(timeout=900)
                return dict(gpu=gpu, rc=rc, seconds=time.time() - started)
        except Exception:
            return dict(
                gpu=gpu,
                rc=-1,
                error=traceback.format_exc(),
                seconds=time.time() - started,
            )
        finally:
            cleanup(token)

    result = {}
    try:
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            result["lanes"] = list(pool.map(lane, p["gpus"]))
        if not all(r["rc"] == 0 for r in result["lanes"]):
            raise RuntimeError(
                "Offline replay failed; see per-card logs; advance to existing queue"
            )
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES="",
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
        )
        check = subprocess.run(
            [
                PYTHON,
                "-B",
                str(tool / "entropy_analysis.py"),
                p["input_batch"],
                str(Path(p["output"]) / "analysis"),
                "--teacher",
                str(Path(p["output"]) / "teacher4"),
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        (d / "analysis.log").write_text(check.stdout + check.stderr)
        if check.returncode:
            raise RuntimeError("Analysis failed; raw replay retained")
        result.update(
            phase="COMPLETE", summary=read(Path(p["output"]) / "analysis/summary.json")
        )
    except Exception:
        result.update(phase="FAILED_ADVANCE", error=traceback.format_exc())
    finally:
        result["time"] = time.time()
        save(d / "status.json", result)
        save(d / "released.json", result)


if __name__ == "__main__":
    main(sys.argv[1])
