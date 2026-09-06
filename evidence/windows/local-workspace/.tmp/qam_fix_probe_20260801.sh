set -euo pipefail
repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
run=/root/autodl-tmp/experiments/qam_formal_20260731_v1/robotwin_adjust_bottle_qam_formal_20260731_v1
printf 'TIME=%s\n' "$(date --iso-8601=seconds)"
printf 'BRANCH=%s\n' "$(git -C "$repo" branch --show-current)"
printf 'HEAD=%s\n' "$(git -C "$repo" rev-parse HEAD)"
printf 'STATUS_BEGIN\n'
git -C "$repo" status --short
printf 'STATUS_END\n'
sha256sum \
  "$repo/rlinf/workers/actor/fsdp_qam_policy_worker.py" \
  "$repo/rlinf/models/embodiment/openpi/openpi_action_model.py"
printf 'REMOTE_BEGIN\n'
git -C "$repo" remote -v
printf 'REMOTE_END\n'
printf 'CHECKPOINT_FILES_BEGIN\n'
find "$run/checkpoints/global_step_50" -maxdepth 3 -type f -printf '%P %s\n' | sort
printf 'CHECKPOINT_FILES_END\n'
printf 'REPLAY_PROMPTS_BEGIN\n'
/root/autodl-tmp/RLinf/.venv/bin/python - <<'PY'
import glob
import torch

for path in sorted(glob.glob('/root/autodl-tmp/experiments/qam_formal_20260731_v1/robotwin_adjust_bottle_qam_formal_20260731_v1/checkpoints/global_step_50/actor/qam_components/replay_rank_*.pt')):
    state = torch.load(path, map_location='cpu', weights_only=False)
    observations = state['observations']
    prompts = sorted({value['prompt'] for value in observations.values()})
    first = next(iter(observations.values()))
    print(path)
    print('size', state['size'], 'total_inserted', state['total_inserted'], 'observations', len(observations))
    print('prompts', prompts)
    print('camera', tuple(first['cameras_uint8'].shape), first['cameras_uint8'].dtype, int(first['cameras_uint8'].min()), int(first['cameras_uint8'].max()))
PY
printf 'REPLAY_PROMPTS_END\n'
printf 'PROMPT_CODE_BEGIN\n'
rg -n --glob '*.py' 'task_descriptions|language_num|instruction|prompt' /root/autodl-tmp/RoboTwin_RLinf/rlinf /root/autodl-tmp/RoboTwin_RLinf | head -n 120 || true
printf 'PROMPT_CODE_END\n'
