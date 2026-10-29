"""One accepted B32 shadow batch; bounded execution and exact owned cleanup."""

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/pi05_distillation"))
from multitask_eval import cleanup

sys.path.insert(0, "/data/chenyiteng/deployment-20261008/bc-signal-tau-v1/ops")
from common import gpus, proc, read, save


def main():
    D = Path(sys.argv[1])
    c = read(D / "shadow.json")
    O = Path(read(D / "prepared.json")["output"])
    gpu = c["gpu"]
    assert read(D / "cpu-checks.json")["passed"]
    python = "/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python"
    token = D.name + "-teacher-shadow"
    env = dict(
        os.environ, **read(O / f"environment-g{gpu}.json"), POLICY_EVAL_TOKEN=token
    )
    save(D / "shadow-owner.json", proc(os.getpid()))
    deadline = time.time() + 4 * 3600
    while gpus()[gpu]["processes"]:
        save(
            D / "shadow-status.json",
            {"phase": "waiting_for_other_user", "gpu": gpu, "time": time.time()},
        )
        if time.time() > deadline:
            raise TimeoutError("Assigned GPU remained busy; no other process touched")
        time.sleep(10)
    with (D / "shadow.log").open("x") as log:
        child = subprocess.Popen(
            [
                python,
                "-u",
                "-B",
                str(ROOT / "tools/pi05_policy_eval/run_batch.py"),
                str(D / "shadow.json"),
            ],
            cwd=ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        started = time.time()
        save(
            D / "shadow-status.json",
            {
                "phase": "running",
                "child": proc(child.pid),
                "gpu": gpu,
                "started": started,
            },
        )
        try:
            while child.poll() is None:
                time.sleep(1)
                progress = Path(c["output"]) / "progress.json"
                last = progress.stat().st_mtime if progress.exists() else started
                if time.time() - started > 3600 or time.time() - last > 900:
                    raise TimeoutError("Shadow batch timed out")
            assert child.returncode == 0, "Shadow batch failed; original log preserved"
        finally:
            cleanup(token)
    env.update(CUDA_VISIBLE_DEVICES="")
    env.pop("DOJO_GPU_SCOPE", None)
    result = subprocess.run(
        [
            python,
            "-B",
            str(ROOT / "tools/pi05_policy_eval/validate_shadow.py"),
            str(D / "shadow.json"),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    (D / "shadow-validation.log").write_text(result.stdout + result.stderr)
    save(
        D / "shadow-status.json",
        {
            "phase": "complete" if result.returncode == 0 else "validation_failed",
            "rc": result.returncode,
            "output": c["output"],
            "time": time.time(),
        },
    )
    assert result.returncode == 0


if __name__ == "__main__":
    main()
