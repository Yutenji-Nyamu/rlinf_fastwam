#!/usr/bin/env bash
set -u

METRICS=/data/chenyiteng/results/rlinf-shenzhen/ppo/ppo-formal100-4gpu128train64eval-official-v1/metrics.log

/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python - "$METRICS" <<'PY'
import json
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8", errors="replace")
text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
starts = list(re.finditer(r"Global Step:\s*(\d+)/100", text))
rows = []

def number(pattern, block):
    match = re.search(pattern, block)
    return None if match is None else float(match.group(1))

def section(block, start_name, end_names):
    start = re.search(start_name, block)
    if start is None:
        return ""
    chunk = block[start.end():]
    stops = []
    for end_name in end_names:
        match = re.search(end_name, chunk)
        if match is not None:
            stops.append(match.start())
    return chunk[:min(stops)] if stops else chunk

for index, match in enumerate(starts):
    step = int(match.group(1))
    if step > 22:
        continue
    end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
    block = text[match.start():end]
    time_sec = section(block, r"Time", [r"Environment"])
    env_sec = section(block, r"Environment", [r"Rollout", r"Evaluation", r"Training/Actor"])
    eval_sec = section(block, r"Evaluation", [r"Training/Actor"])
    actor_sec = section(block, r"Training/Actor", [r"Training/Critic"])
    critic_sec = section(block, r"Training/Critic", [r"╰", r"Global Step"])
    row = {
        "step": step,
        "step_time_s": number(r"Step Time:\s*([0-9.eE+-]+)s", block),
        "train_num_trajectories": number(r"num_trajectories=([0-9.eE+-]+)", env_sec),
        "train_success_once": number(r"success_once=([0-9.eE+-]+)", env_sec),
        "train_return": number(r"return=([0-9.eE+-]+)", env_sec),
        "train_reward": number(r"reward=([0-9.eE+-]+)", env_sec),
        "eval_num_trajectories": number(r"num_trajectories=([0-9.eE+-]+)", eval_sec),
        "eval_success_once": number(r"success_once=([0-9.eE+-]+)", eval_sec),
        "eval_success_at_end": number(r"success_at_end=([0-9.eE+-]+)", eval_sec),
        "actor_approx_kl": number(r"actor/approx_kl=([0-9.eE+-]+)", actor_sec),
        "actor_clip_fraction": number(r"actor/clip_fraction=([0-9.eE+-]+)", actor_sec),
        "actor_grad_norm": number(r"actor/grad_norm=([0-9.eE+-]+)", actor_sec),
        "actor_policy_loss": number(r"actor/policy_loss=([0-9.eE+-]+)", actor_sec),
        "actor_total_loss": number(r"actor/total_loss=([0-9.eE+-]+)", actor_sec),
        "critic_explained_variance": number(r"critic/explained_variance=([0-9.eE+-]+)", critic_sec),
        "critic_value_loss": number(r"critic/value_loss=([0-9.eE+-]+)", critic_sec),
        "time_generate_rollouts_s": number(r"generate_rollouts=([0-9.eE+-]+)", time_sec),
        "time_actor_training_s": number(r"actor_training=([0-9.eE+-]+)", time_sec),
        "time_eval_s": number(r"(?:^|[│\s])eval=([0-9.eE+-]+)", time_sec),
        "time_step_s": number(r"(?:^|[│\s])step=([0-9.eE+-]+)", time_sec),
    }
    rows.append(row)

print(json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True))
print(f"rows={len(rows)} unique_steps={len({row['step'] for row in rows})}")
PY
