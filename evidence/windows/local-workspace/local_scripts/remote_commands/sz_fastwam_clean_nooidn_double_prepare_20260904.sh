#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo
PRIOR=$ROOT/runs/fastwam-grpo-control-fresh100-2gpu32x4-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-clean-nooidn-v1
RUN=$ROOT/runs/fastwam-grpo-control-fresh100-2gpu32x8-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-clean-nooidn-v1
PACKET=$ROOT/packets/fastwam-grpo-control-fresh100-2gpu32x8-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-clean-nooidn-v1
test ! -e "$PRIOR/runtime/wrapper.pid"
test ! -e "$PRIOR/runtime/driver.log"
test ! -e "$RUN"
test ! -e "$PACKET"
source "$PRIOR/runtime/environment.sh"
export PRIOR RUN PACKET
mkdir -p "$RUN/runtime" "$PACKET"
cd "$REPO_PATH"
"$VIRTUAL_ENV/bin/python" - <<'PY'
import os,shlex,subprocess,json,hashlib,shutil
from pathlib import Path
import yaml
prior,run,packet=(Path(os.environ[k]) for k in ['PRIOR','RUN','PACKET'])
prior_cfg=yaml.safe_load((prior/'runtime/resolved.yaml').read_text())
contract=json.loads((prior/'runtime/contract.json').read_text())
donor=Path(contract['donor'])
old=yaml.safe_load((donor/'runtime/resolved.yaml').read_text())
command=shlex.split((prior/'runtime/command.txt').read_text())
assert command.count('env.train.rollout_epoch=4')==1
command=[s.replace(str(prior),str(run)).replace(prior_cfg['runner']['logger']['experiment_name'],run.name.replace('-','_')) if s!='env.train.rollout_epoch=4' else 'env.train.rollout_epoch=8' for s in command]
(packet/'command.txt').write_text(shlex.join(command)+'\n')
out=subprocess.run(command+['--cfg','job','--resolve'],capture_output=True,text=True)
if out.returncode: print(out.stdout,out.stderr);raise SystemExit(out.returncode)
(packet/'resolved.yaml').write_text(out.stdout)
new=yaml.safe_load(out.stdout)
def flat(v,p=''):
 if isinstance(v,dict):return {k2:v2 for k,c in v.items() for k2,v2 in flat(c,p+'.'+str(k) if p else str(k)).items()}
 if isinstance(v,list):return {k2:v2 for i,c in enumerate(v) for k2,v2 in flat(c,p+'.'+str(i)).items()}
 return {p:v}
a,b=flat(old),flat(new)
diff={k:[a.get(k),b.get(k)] for k in sorted(a.keys()|b.keys()) if a.get(k)!=b.get(k)}
assert set(diff)==set(contract['allowed_resolved_differences'])|{'env.train.rollout_epoch'},diff
assert diff['env.train.rollout_epoch']==[4,8]
assert new['actor']['global_batch_size']==1024 and new['actor']['micro_batch_size']==2
assert new['algorithm']['update_epoch']==2 and new['env']['train']['total_num_envs']==32
assert new['runner']['resume_dir'] is None and new['runner']['max_steps']==100
contract['resolved_difference_count']=len(diff)
contract['allowed_resolved_differences']=diff
contract['latest_user_amendment']='Double trajectories via rollout4->8; keep parallel32 and all optimizer hyperparameters unchanged. GB1024/update_epoch2 implies4 optimizer calls per outer step.'
contract['prior_128_packet']='prepared but never launched'
contract['budget']={'outer_steps':100,'train_episodes':25600,'train_submitted_action_slots_max':4915200,'new_query_records_max':204800,'query_record_presentations_max':409600,'actor_optimizer_calls':400,'critic_updates':0,'eval_episodes':640,'checkpoint_generations':10,'wall_timeout_seconds':432000,'reserved_gpu_hours_cap':240,'expected_wall_hours_rough':50,'expected_gpu_hours_rough':100}
contract['risk']='Prior256-trajectory OIDN-on run OOMed; noOIDN may reduce renderer memory but long-run fit is unproven. On OOM stop, do not silently change parallelism or batches.'
(packet/'contract.json').write_text(json.dumps(contract,ensure_ascii=False,indent=2)+'\n')
for name in ['source_head.txt','robotwin.patch','environment.sh']:shutil.copyfile(prior/'runtime'/name,packet/name)
(packet/'packet_complete.txt').write_text('source/import preflight passed in128 preparation; amended256 config compose+leafdiff passed; not launched\n')
files=['command.txt','resolved.yaml','contract.json','source_head.txt','robotwin.patch','environment.sh','packet_complete.txt']
(packet/'sha256.txt').write_text(''.join(f'{hashlib.sha256((packet/n).read_bytes()).hexdigest()}  {n}\n' for n in files))
for n in files+['sha256.txt']:shutil.copyfile(packet/n,run/'runtime'/n)
print(json.dumps(contract,ensure_ascii=False,indent=2))
print('EXACT_COMMAND',shlex.join(command))
print('OUTPUT',run)
PY
date -Is
