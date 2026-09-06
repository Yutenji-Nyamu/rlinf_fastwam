set -eu
root=/data/chenyiteng/projects/rlinf-shenzhen/worktrees
date -Is
for rel in rlinf/data/online_bc.py rlinf/models/embodiment/openpi/openpi_action_model.py rlinf/workers/actor/fsdp_online_bc_policy_worker.py rlinf/workers/env/env_worker.py; do
  sha256sum "$root/pi0-online-bc/$rel"
done
sha256sum "$root/grpo-dvac-action-adv-fix/rlinf/algorithms/dvac_train_weighting.py" "$root/grpo-dvac-action-adv-fix/rlinf/models/embodiment/openpi/openpi_action_model.py" "$root/rlt-dvac-pure-single-gpu-7d07a421/rlinf/algorithms/rlt/dvac_weighting.py"
git -C "$root/rlt-checkpoint-diagnosis-7d07a421" diff --stat
git -C "$root/rlt-checkpoint-diagnosis-7d07a421" diff -- rlinf/hybrid_engines/fsdp/fsdp_model_manager.py rlinf/hybrid_engines/fsdp/strategy/checkpoint.py
git -C "$root/rlt-checkpoint-diagnosis-7d07a421" log -3 --oneline
wc -c "$root/robotwin-oidn-toggle-20260904/oidn_toggle_trial.py"
git -C "$root/sidney-pi05-current-rlinf" ls-files evidence docs/evidence
/usr/bin/python3 - <<'PY'
import json, os
from pathlib import Path
base=Path('/data/chenyiteng/results/rlinf-shenzhen')
starts=[base/'ppo'/n for n in ['ppo-oneopt-4gpu128train64eval-v1','ppo-reload-fixed64-4gpu4567-v1','sft-fixed64-4gpu4567-official-half-v1']]
starts += [base/'fastwam-grpo/diagnostics'/n for n in ['scene-fence-env-local-smoke-20260904','scene-fence-smoke-20260904','oidn-toggle-20260904']]
for root in starts:
    records=[]
    for d, dirs, files in os.walk(root):
        dirs[:]=[x for x in dirs if x not in {'checkpoints','video','videos','images','ray','ray_logs','robotwin_data'} and not x.startswith('ray_logs')]
        for n in files:
            p=Path(d)/n
            if p.suffix in {'.log','.json','.csv','.yaml','.sh','.txt'}:
                records.append([str(p.relative_to(root)),p.stat().st_size])
    print(json.dumps({'run':str(root),'small_files':records},ensure_ascii=False))
PY
