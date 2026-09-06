#!/usr/bin/env bash
set -euo pipefail

RUN=/data/chenyiteng/results/rlinf-shenzhen/ppo/ppo-formal100-4gpu128train64eval-official-v1
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin

TF_CPP_MIN_LOG_LEVEL=3 PYTHONDONTWRITEBYTECODE=1 "$VENV/bin/python" -B - "$RUN" <<'PY'
import pathlib
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

event = next(pathlib.Path(sys.argv[1]).rglob("events.out.tfevents.*"))
acc = EventAccumulator(str(event), size_guidance={"scalars": 0})
acc.Reload()
tags = [
    "env/success_once", "eval/success_once", "train/actor/approx_kl",
    "train/actor/clip_fraction", "train/actor/grad_norm", "train/actor/ratio",
    "train/critic/value_loss", "train/critic/explained_variance", "rollout/returns_mean",
    "time/generate_rollouts", "time/actor_training", "time/step",
]
values = {tag: {int(x.step) + 1: float(x.value) for x in acc.Scalars(tag)} for tag in tags}
print("global_step,train_success,eval_success,approx_kl,clip_fraction,grad_norm,ratio,value_loss,explained_variance,returns_mean,generate_rollouts_s,actor_training_s,step_s")
for step in sorted(values["env/success_once"]):
    row = [step] + [values[tag].get(step) for tag in tags]
    print(",".join("" if x is None else f"{x:.9g}" for x in row))
PY
