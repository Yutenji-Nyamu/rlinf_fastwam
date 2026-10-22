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

"""Smoke, full-data CP training, paired all-task evaluation, then RLT handback."""

import importlib.util
import os
import signal
import subprocess
import time
import traceback
from pathlib import Path

from multitask_eval import sweep

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "accepted_queue", ROOT / "tools/pi05_distillation/run_queue.py"
)
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)
D, O = q.D, q.O
me = q.proc(os.getpid())
q.save(D / "identity.json", me)


def main():
    while not (D / "registered.json").exists():
        time.sleep(1)
    probe = """import os,sys,json,numpy as np,torch,sapien.core as sapien
g=int(os.environ['CUDA_VISIBLE_DEVICES']);assert os.environ['DOJO_GPU_SCOPE']==str(g)
x=torch.zeros(1,device='cuda');engine=sapien.Engine();renderer=sapien.SapienRenderer();engine.set_renderer(renderer)
scene=engine.create_scene();camera=scene.add_camera('binding',32,32,1.,.1,10.);scene.update_render();camera.take_picture();assert np.isfinite(camera.get_picture('Color')).all()
sys.path.insert(0,'/data/chenyiteng/deployment-20261008/bc-signal-tau-v1/ops');from common import gpus
bound=[i for i,row in gpus().items()if any(p['pid']==os.getpid()for p in row['processes'])];assert bound==[g],bound
print(json.dumps(dict(gpu=g,actual_bound=bound,rendered=True)))"""
    for gpu in q.CARDS:
        env = dict(os.environ, **q.read(O / f"environment-g{gpu}.json"))
        result = subprocess.run(
            [q.PYTHON, "-u", "-B", "-c", probe],
            env=env,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=90,
        )
        q.save(
            D / f"binding-g{gpu}.json",
            {
                "exit": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
        )
        assert result.returncode == 0, result.stdout + result.stderr
    smoke = q.train(
        "cp-smoke", O / "smoke-cp-multitask", 12, label="smoke-cp-multitask"
    )
    smoke = q.train(
        "cp-smoke",
        O / "smoke-cp-multitask",
        14,
        smoke["path"],
        label="smoke-cp-multitask-resume",
    )
    for rank in range(2):
        check = q.read(O / "smoke-cp-multitask" / f"inference-check-rank{rank}.json")
        assert (
            check["finite"]
            and check["loaded_export"]
            and check["action_expert_calls"] == 1
        )
    manifest = q.read(D / "task-list.json")
    names = ["adjust_bottle", "click_bell", "grab_roller", "press_stapler"]
    tasks = [next(t for t in manifest["tasks"] if t["task"] == name) for name in names]
    q.event(
        "EVALUATION_SMOKE", tasks=names, layout="two tasks x 16 environments per GPU"
    )
    result = sweep(D, "smoke-cp-multitask", policy=smoke["policy"], tasks=tasks)
    # Require all four ordinary tasks to exercise the layout before full training.
    assert result["valid_tasks"] == 4, (
        "Four-task smoke needs repair; see missing task errors"
    )
    q.save(
        D / "smoke-accepted.json",
        {"time": time.time(), "result": result, "train_batch_per_gpu": 64},
    )
    # Original-policy evaluations do not require offline training data; overlap
    # these required baselines with the remaining CPU-only data conversions.
    for steps in [1, 10]:
        q.event("EVALUATING", condition=f"raw-n{steps}", tasks=50)
        sweep(D, f"raw-n{steps}", steps=steps)
    assert (D / "data-ready.json").is_file(), (
        "Data preparation still incomplete; preserve baselines and repair before training"
    )
    shape = q.read(D / "data-ready.json")
    per_epoch, total = shape["steps_per_epoch"], shape["total_steps"]
    stops = sorted(
        {
            min(total, max(1, round(x * per_epoch)))
            for x in [0.5, 1, 2, 5, 10, 20, 35, 50]
        }
    )
    q.save(
        D / "formal-ready.json",
        {
            "time": time.time(),
            "total_steps": total,
            "eval_steps": stops,
            "epochs": 50,
            "baseline_steps": [1, 10],
            "tasks": 50,
            "eval_episodes_per_task": 16,
        },
    )
    resume = None
    for stop in stops:
        receipt = q.train("cp", O / "cp", stop, resume)
        label = f"cp-s{stop:06d}"
        q.event("EVALUATING", condition=label, tasks=50)
        sweep(D, label, policy=receipt["policy"])
        resume = receipt["path"]
    q.event("COMPLETE", method="multitask_cp")


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except BaseException:
        exit_code = 1
        q.event("FAILED", error=traceback.format_exc())
        for identity in q.owned_children:
            if q.same(identity):
                q.exact_signal(identity, signal.SIGTERM)
        deadline = time.time() + 90
        while any(q.gpus()[g]["processes"] for g in q.CARDS) and time.time() < deadline:
            time.sleep(2)
        raise
    finally:
        if all(not q.gpus()[g]["processes"] for g in q.CARDS):
            q.save(
                D / "released.json",
                {
                    "time": time.time(),
                    "identity": me,
                    "exit_code": exit_code,
                    "cards_clear": True,
                },
            )
