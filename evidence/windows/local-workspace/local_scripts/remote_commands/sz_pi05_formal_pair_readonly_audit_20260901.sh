#!/usr/bin/env bash
set -euo pipefail

WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-robotwin-rl
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
MODEL=/data/chenyiteng/models/rlinf/RLinf-Pi05-RoboTwin-SFT-adjust_bottle@fa8df6ed
ROOT=/data/chenyiteng/results/rlinf-shenzhen/pi05
CLEAN="$ROOT/packets/pi05-grpo-smoke1-2gpu64x4-g8-b512-u5-m5-phys23-localshard-v2/resolved.yaml"
DVAC="$ROOT/packets/pi05-grpo-dvac-action-adv-w0p5to1p5-smoke2-2gpu64x4-g8-b512-u5-m5-phys23-localshard-v2/resolved.yaml"
PI0=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/grpo-control-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v2/runtime/resolved.yaml

date --iso-8601=seconds
id
printf '%s\n' '=== source ==='
git -C "$WT" status --short --branch
printf 'head=%s\nbranch=%s\n' "$(git -C "$WT" rev-parse HEAD)" "$(git -C "$WT" branch --show-current)"
git -C "$WT" rev-parse personal/codex/sz-pi05-robotwin-rl 2>/dev/null || true
sha256sum "$WT/examples/embodiment/config/robotwin_adjust_bottle_grpo_openpi_pi05.yaml"

printf '%s\n' '=== model ==='
du -sh "$MODEL"
find "$MODEL" -maxdepth 4 -type f \( -name 'model.safetensors.index.json' -o -name 'model-*.safetensors' -o -name 'norm_stats.json' \) -printf '%P %s\n' | sort

printf '%s\n' '=== packets ==='
for path in "$CLEAN" "$DVAC" "$PI0"; do
  test -s "$path"
  sha256sum "$path"
done
find "$ROOT/packets" -maxdepth 2 -type f \( -name 'resolved.yaml' -o -name 'config_name.txt' -o -name 'pi05-smoke-contract.json' \) -printf '%p %s\n' | sort

"$VENV/bin/python" - "$CLEAN" "$DVAC" "$PI0" <<'PY'
import json, sys, yaml

clean, dvac, pi0 = [yaml.safe_load(open(p, encoding="utf-8")) for p in sys.argv[1:4]]

def flat(value, prefix=""):
    out = {}
    if isinstance(value, dict):
        for key, item in value.items():
            out.update(flat(item, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, list):
        out[prefix] = value
    else:
        out[prefix] = value
    return out

fc, fd = flat(clean), flat(dvac)
diff = []
for key in sorted(set(fc) | set(fd)):
    if fc.get(key) != fd.get(key):
        diff.append({"key": key, "clean": fc.get(key), "dvac": fd.get(key)})
print("=== clean_vs_dvac_smoke_all_leaf_diff ===")
print(json.dumps(diff, ensure_ascii=False, indent=2))

keys = [
    "runner.max_steps", "runner.val_check_interval", "runner.save_interval",
    "algorithm.group_size", "algorithm.adv_type", "algorithm.loss_type",
    "algorithm.filter_rewards", "algorithm.logprob_type", "algorithm.update_epoch",
    "algorithm.clip_ratio_high", "algorithm.clip_ratio_low",
    "env.train.total_num_envs", "env.train.rollout_epoch", "env.eval.total_num_envs",
    "actor.global_batch_size", "actor.micro_batch_size", "actor.optim.lr",
    "actor.model.model_path", "actor.model.num_steps", "actor.model.openpi.config_name",
    "actor.model.add_value_head", "actor.fsdp_config.checkpoint_format",
    "env.enable_offload", "rollout.enable_offload", "actor.enable_offload",
]
fpi = flat(pi0)
print("=== pi05_clean_vs_pi0_clean_key_fields ===")
print(json.dumps([
    {"key": key, "pi05": fc.get(key), "pi0": fpi.get(key)} for key in keys
], ensure_ascii=False, indent=2))
PY

printf '%s\n' '=== shared_ray_and_live_namespaces ==='
RAY_ADDRESS=172.17.0.1:6389 "$VENV/bin/ray" status | sed -n '1,35p'
RAY_ADDRESS=172.17.0.1:6389 "$VENV/bin/python" - <<'PY'
import collections, ray
ray.init(address="172.17.0.1:6389", namespace="codex_pi05_readonly_audit", logging_level="ERROR")
rows = ray.util.list_named_actors(all_namespaces=True)
print(dict(sorted(collections.Counter(row.get("namespace", "") for row in rows).items())))
ray.shutdown()
PY

printf '%s\n' '=== storage ==='
df -h / /home /data
awk '/MemTotal|MemAvailable|SwapTotal|SwapFree/ {print}' /proc/meminfo
