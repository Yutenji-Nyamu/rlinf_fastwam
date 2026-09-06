#!/usr/bin/env bash
set -euo pipefail

OLD_RUN=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/dvac-global-z-formal100-4gpu32x8-g8-phys2367-v2
PACKET=/data/chenyiteng/results/rlinf-shenzhen/grpo/packets/dvac-global-z-formal100-4gpu32x8-g8-phys4567-v3
RUN=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/dvac-global-z-formal100-4gpu32x8-g8-phys4567-v3

printf 'MARKER=SZ_FIX_PACKET_LAUNCH_DVAC_PHYS4567_V3_V1\n'
test -d "$PACKET"
test ! -e "$RUN"
test ! -e "$OLD_RUN/runtime/stopped_by_user.txt"
printf 'stopped_at=%s\nreason=user_requested_gpu_relocation_2367_to_4567\nlast_complete_step=4\n' \
  "$(date --iso-8601=seconds)" > "$OLD_RUN/runtime/stopped_by_user.txt"

python3 - "$PACKET" <<'PY'
from pathlib import Path
import json, sys

root = Path(sys.argv[1])
contract_path = root / 'contract.json'
contract = json.loads(contract_path.read_text(encoding='utf-8'))
contract['physical_gpus'] = [4, 5, 6, 7]
contract_path.write_text(json.dumps(contract, indent=2) + '\n', encoding='utf-8')

command_path = root / 'command.txt'
command = command_path.read_text(encoding='utf-8')
command = command.replace(r'2\,3\,6\,7', r'4\,5\,6\,7')
command_path.write_text(command, encoding='utf-8')
PY

bash -n "$PACKET/launch.sh"
python3 - "$PACKET" <<'PY'
from pathlib import Path
import json, sys

root = Path(sys.argv[1])
c = json.loads((root / 'contract.json').read_text())
assert c['physical_gpus'] == [4, 5, 6, 7]
assert c['outer_steps'] == 100
assert c['train_envs'] == 32 and c['rollout_epochs'] == 8
assert c['trajectories_per_step'] == 256
assert c['global_batch'] == 512 and c['micro_batch'] == 32
assert c['dvac']['selected_l'] == 3 and c['dvac']['weight_range'] == [0, 2]
resolved = (root / 'resolved.yaml').read_text()
command = (root / 'command.txt').read_text()
launch = (root / 'launch.sh').read_text()
assert 'actor, env, rollout: 4,5,6,7' in resolved
assert r'4\,5\,6\,7' in command and r'2\,3\,6\,7' not in command
assert 'rollout:"4,5,6,7"' in launch
PY

if nvidia-smi -i 4,5,6,7 --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '^[[:space:]]*[0-9]+'; then
  echo 'physical GPUs 4,5,6,7 are not idle; refusing launch' >&2
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
  exit 30
fi

bash "$PACKET/launch.sh"
printf 'MARKER=SZ_FIX_PACKET_LAUNCH_DVAC_PHYS4567_V3_OK\n'
