"""Read-only BC branch / historical pi0 GRPO config and resource audit."""
import csv
import datetime
import hashlib
import json
import os
import subprocess
from pathlib import Path

import yaml

assert os.getuid() == 1003
root = Path('/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-online-bc')
pin = 'dc9b87cc49334c7516487ead68ebeb060fd7c090'
def git(*args):
    p = subprocess.run(['git', '--no-optional-locks', '-C', str(root), *args], capture_output=True, text=True, timeout=20)
    return {'rc': p.returncode, 'out': p.stdout, 'err': p.stderr}
files = ['rlinf/data/online_bc.py', 'rlinf/workers/actor/fsdp_online_bc_policy_worker.py',
         'rlinf/models/embodiment/openpi/openpi_action_model.py', 'examples/embodiment/train_embodied_agent.py',
         'rlinf/workers/rollout/hf/huggingface_worker.py', 'rlinf/workers/env/env_worker.py',
         'examples/embodiment/config/robotwin_adjust_bottle_online_bc_openpi.yaml',
         'tests/unit_tests/test_online_bc.py', 'docs/online_bc.md']
data = {'time': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        'git': {k: git(*args) for k,args in {'head': ['rev-parse','HEAD'], 'branch': ['branch','--show-current'],
            'status': ['status','--porcelain'], 'numstat': ['diff','--numstat'], 'diff': ['diff','--',*files]}.items()},
        'files': {f: {'sha256': hashlib.sha256((root/f).read_bytes()).hexdigest(), 'lines': len((root/f).read_text().splitlines())} for f in files}}
runroot = Path('/data/chenyiteng/results/rlinf-shenzhen/online-bc')
data['bc_output_entries'] = [str(p.relative_to(runroot)) for p in runroot.iterdir()] if runroot.exists() else []
data['bc_smoke_dir_exists'] = (runroot/'pi0-adjust-bottle-smoke-v1').exists()
data['bc_yaml'] = (root/files[6]).read_text()
old = Path('/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/grpo-control-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v2')
data['pi0_grpo'] = {'path': str(old), 'config': yaml.safe_load((old/'runtime/resolved.yaml').read_text())}
res = old/'runtime/resource.csv'
if res.exists():
    with res.open() as handle:
        rows = list(csv.DictReader(handle))
    data['pi0_grpo']['resource'] = {'count': len(rows), 'first': rows[0], 'last': rows[-1],
        'gpu_peaks_mib': {k: max(float(r[k]) for r in rows if r.get(k)) for k in rows[0] if k.endswith('_used_mib')}}
ls = git('ls-tree','-r','--name-only',pin,'examples/embodiment/config')['out'].splitlines()
paths = [p for p in ls if ('sft' in p or 'dagger' in p) and ('openpi' in p or 'robotwin' in p)]
data['official_recipe_paths'] = paths
data['official_recipes'] = {p: git('show', f'{pin}:{p}')['out'] for p in paths if p.endswith('.yaml') and ('robotwin' in p or 'sft' in p)}
print('BC_REVIEW_JSON '+json.dumps(data, ensure_ascii=False))
