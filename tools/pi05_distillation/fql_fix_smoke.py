"""Accepted FQL pair launcher: twelve updates, resume to fourteen, evaluate, stop."""

import json
import os
import subprocess
import time

import run_queue as q
from multitask_eval import cleanup
from single_task_prepare import evaluation_config


def main():
    assert q.read(q.D / "cpu-checks.json")["passed"]
    q.save(q.D / "fql-fix-identity.json", q.proc(os.getpid()))
    config = q.read(q.D / "fql_fixed_teacher.json")
    assert config["alpha"] == 30 and config["q_diagnostics"]
    output = q.O / "fql-fix-smoke"
    first = q.train(
        "fql_fixed_teacher",
        output,
        12,
        label="fql-fix-12",
        extra_args=["--check-export"],
    )
    final = q.train(
        "fql_fixed_teacher",
        output,
        14,
        first["path"],
        label="fql-fix-resume14",
        extra_args=["--check-export"],
    )
    jobs = []
    results = []
    try:
        for index, gpu in enumerate(q.CARDS):
            assert not q.gpus()[gpu]["processes"]
            destination = q.O / f"fql-fix-eval/g{gpu}"
            c = evaluation_config(
                q.read(q.D / "eval-template.json"),
                destination,
                q.D / f"eval-seeds-g{gpu}.json",
                gpu,
                32,
                weights=final["policy"],
                method="fql_fixed_teacher",
            )
            c.update(
                fql_q_path=final["q"],
                batch=index,
                noise_seed=c["noise_seed"] + 1000003 * index,
            )
            path = q.D / f"fql-fix-eval-g{gpu}.json"
            q.save(path, c)
            token = q.D.name + f"-fql-eval-g{gpu}"
            env = dict(
                os.environ,
                **q.read(q.O / f"environment-g{gpu}.json"),
                POLICY_EVAL_TOKEN=token,
            )
            log = (q.D / f"fql-fix-eval-g{gpu}.log").open("x")
            p = subprocess.Popen(
                [
                    q.PYTHON,
                    "-u",
                    "-B",
                    str(q.R / "tools/pi05_policy_eval/run_batch.py"),
                    str(path),
                ],
                cwd=q.R,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            identity = q.proc(p.pid)
            q.owned_children.append(identity)
            jobs.append(
                dict(
                    process=p,
                    identity=identity,
                    log=log,
                    token=token,
                    output=destination,
                    gpu=gpu,
                    started=time.time(),
                )
            )
        q.event("FQL_FIX_EVALUATING", children=[j["identity"] for j in jobs])
        while jobs:
            for job in list(jobs):
                p = job["process"]
                expired = time.time() - job["started"] > 2400
                if p.poll() is None and not expired:
                    continue
                if expired:
                    cleanup(job["token"])
                p.wait(timeout=30)
                job["log"].close()
                cleanup(job["token"])
                result = {"gpu": job["gpu"], "rc": p.returncode, "timeout": expired}
                if p.returncode == 0:
                    result.update(
                        done=q.read(job["output"] / "done.json"),
                        calibration=q.read(job["output"] / "q-calibration.json"),
                        parity=q.read(job["output"] / "q-parity.json"),
                    )
                results.append(result)
                jobs.remove(job)
            if jobs:
                time.sleep(0.5)
        passed = all(
            r["rc"] == 0
            and r["done"]["episodes"] == 32
            and r["calibration"]["episodes"] == 32
            for r in results
        )
        metrics = [
            json.loads(s)
            for rank in (0, 1)
            for s in (output / f"metrics-rank{rank}.jsonl").read_text().splitlines()
        ]
        q.save(
            q.D / "fql-fix-smoke-summary.json",
            {
                "passed": passed,
                "checkpoint": final,
                "evaluations": results,
                "updates": 14,
                "alpha": 30,
                "microbatch_per_rank": 224,
                "global_batch": 448,
                "peak_gib": max(m["peak_gib"] for m in metrics),
                "metrics": metrics,
                "formal_started": False,
            },
        )
        q.event("FQL_FIX_SMOKE_COMPLETE", passed=passed, formal_started=False)
        if not passed:
            raise RuntimeError("FQL-fix evaluation failed; see exact receipts")
    finally:
        for job in jobs:
            cleanup(job["token"])
            job["log"].close()


if __name__ == "__main__":
    main()
