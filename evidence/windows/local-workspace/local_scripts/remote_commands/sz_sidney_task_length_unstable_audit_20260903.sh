set -euo pipefail

native=/data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711/third_party/RoboTwin
rlrt=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
lerobot=/data/chenyiteng/projects/lerobot-sidney/lerobot-v060-py310-compat
rlinf=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-rl-current

echo '=== source identities ==='
for p in "$native" "$rlrt" "$lerobot" "$rlinf"; do
  if test -d "$p"; then
    echo "PATH $p"
    git -C "$p" rev-parse HEAD 2>/dev/null || true
    git -C "$p" branch --show-current 2>/dev/null || true
  fi
done

echo '=== native task limits ==='
for p in "$native/task_config/_eval_step_limit.yml" "$native/env_cfg/task_config/_eval_step_limit.yml"; do
  test -f "$p" && { echo "FILE $p"; cat "$p"; }
done

echo '=== rlinf task limits ==='
for p in "$rlrt/task_config/_eval_step_limit.yml" "$rlrt/env_cfg/task_config/_eval_step_limit.yml"; do
  test -f "$p" && { echo "FILE $p"; cat "$p"; }
done

echo '=== lerobot reset and horizon ==='
grep -nE 'DEFAULT_EPISODE_LENGTH|episode_length|def reset|actual_seed|setup_demo|episode_index|def step|truncated|self.reset' \
  "$lerobot/src/lerobot/envs/robotwin.py" || true

echo '=== robotwin unstable definitions and setup ==='
for root in "$native" "$rlrt"; do
  echo "ROOT $root"
  rg -n 'class UnStableError|UnStableError|def setup_demo|setup_demo\(' "$root/envs" "$root/robotwin" 2>/dev/null | head -240 || true
done

echo '=== vector reset ==='
for p in "$rlrt/robotwin/envs/vector_env.py" "$rlrt/robotwin/envs/task_env.py" "$native/robotwin/envs/vector_env.py" "$native/robotwin/envs/task_env.py"; do
  test -f "$p" && { echo "FILE $p"; grep -nE 'def reset|setup_demo|UnStable|seed|retry|while|except' "$p"; sed -n '1,280p' "$p"; }
done

echo '=== rlinf wrapper and resolved seed config ==='
for p in \
 "$rlinf/rlinf/envs/robotwin/robotwin_env.py" \
 /data/chenyiteng/projects/rlinf-shenzhen/worktrees/grpo-pi0-robotwin-7d07a421/rlinf/envs/robotwin/robotwin_env.py; do
  test -f "$p" && { echo "FILE $p"; grep -nE '_init_reset_state_ids|success_seeds|reset_state_ids|def reset|venv.reset|update_reset_state_ids|seeds_path' "$p"; }
done

echo '=== unstable logs ==='
find /data/chenyiteng/results/lerobot-sidney/pi05_robotwin-e49e2ab -maxdepth 3 -type f -name 'eval.log' -print0 2>/dev/null | \
  xargs -0 grep -nH -A12 -B20 'UnStableError' 2>/dev/null | head -500 || true

echo '=== candidate seed bank presence ==='
for p in \
 /data/chenyiteng/projects/rlinf-shenzhen/worktrees/grpo-pi0-robotwin-7d07a421/rlinf/envs/robotwin/seeds/train_seeds.json \
 /data/chenyiteng/projects/rlinf-shenzhen/worktrees/grpo-pi0-robotwin-7d07a421/rlinf/envs/robotwin/seeds/eval_seeds.json; do
  if test -f "$p"; then
    echo "FILE $p"
    python - "$p" <<'PY'
import json, sys
d=json.load(open(sys.argv[1]))
for task in ['adjust_bottle','move_can_pot','move_stapler_pad','lift_pot','pick_diverse_bottles','place_a2b_left','place_a2b_right','place_fan','place_mouse_pad','stamp_seal','turn_switch']:
    x=d.get(task,{})
    s=x.get('success_seeds')
    print(task, 'n=', None if s is None else len(s), 'head=', None if s is None else s[:12])
PY
  fi
done
