set -eu
cd /root/autodl-tmp/RLinf_fastwam_rlinf
for task in move_stapler_pad turn_switch pick_diverse_bottles place_can_basket hanging_mug open_microwave; do
  test -f "/root/autodl-tmp/RoboTwin_RLinf/envs/$task.py" && printf '%s\n' "$task ENV_OK" || printf '%s\n' "$task ENV_MISSING"
done
/root/autodl-tmp/conda/envs/FastWAM-RLinf/bin/python - <<'PY'
import json
for p in [
    'rlinf/envs/robotwin/seeds/train_seeds.json',
    'rlinf/envs/robotwin/seeds/eval_seeds.json',
]:
    with open(p) as f:
        data=json.load(f)
    print(p)
    for task in ['move_stapler_pad','turn_switch','pick_diverse_bottles','place_can_basket','hanging_mug','open_microwave']:
        item=data.get(task)
        print(task, 'present' if item else 'missing', len(item.get('success_seeds',[])) if item else 0)
PY
cat /root/autodl-tmp/FastWAM/third_party/RoboTwin/task_config/demo_randomized.yml
cat /root/autodl-tmp/FastWAM/third_party/RoboTwin/task_config/demo_clean.yml
grep -nE '^(adjust_bottle|move_stapler_pad|turn_switch|pick_diverse_bottles|place_can_basket|hanging_mug|open_microwave):' /root/autodl-tmp/FastWAM/third_party/RoboTwin/task_config/_eval_step_limit.yml
