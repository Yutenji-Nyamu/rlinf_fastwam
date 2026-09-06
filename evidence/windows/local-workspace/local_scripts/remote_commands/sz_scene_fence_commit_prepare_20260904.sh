#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo
export OLD=$ROOT/runs/fastwam-grpo-control-fresh100-2gpu32x8-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-clean-nooidn-v1
export RUN=$ROOT/runs/fastwam-grpo-control-fresh100-2gpu32x8-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-clean-nooidn-scene-fence-v3
export PACKET=$ROOT/packets/$(basename "$RUN")
export DIAG=$ROOT/diagnostics/scene-fence-env-local-smoke-20260904
source "$OLD/runtime/environment.sh"
cd "$REPO_PATH"
test "$(cat "$DIAG/exit_code")" = 0
test -s "$DIAG/native_binding.txt"
"$VIRTUAL_ENV/bin/python" - <<'PY'
import json, os
from pathlib import Path
r=json.loads((Path(os.environ['DIAG'])/'result.json').read_text())
assert r['status']=='passed' and r['camera_images']==384, r
assert r['torch_archive_before_after']=='passed',r
PY
test "$(git branch --show-current)" = codex/sz-fastwam-current-rlinf-grpo
test "$(git rev-parse HEAD)" = b60144fd60270f303fb5ea229ca92125f6b0a710
test -z "$(git diff --name-only -- . ':!tools/fastwam_scene_fence' ':!rlinf/envs/robotwin/robotwin_env.py')"
test -z "$(git diff --cached --name-only -- . ':!tools/fastwam_scene_fence' ':!rlinf/envs/robotwin/robotwin_env.py' ':!rlinf/envs/robotwin/scene_fence.py')"
git add tools/fastwam_scene_fence/scene_fence.cpp tools/fastwam_scene_fence/scene_fence.patch tools/fastwam_scene_fence/build.sh tools/fastwam_scene_fence/exports.map tools/fastwam_scene_fence/README.md tools/fastwam_scene_fence/smoke.py
git add rlinf/envs/robotwin/robotwin_env.py rlinf/envs/robotwin/scene_fence.py
# The .patch artifact contains required single-space blank context lines.
git diff --cached --check -- . ':!tools/fastwam_scene_fence/scene_fence.patch'
git diff --cached --stat
git commit -m "fix(fastwam): scope native fence loading to RoboTwin without global ZIP symbols"
timeout 60 git push personal HEAD:refs/heads/codex/sz-fastwam-current-rlinf-grpo
test "$(timeout 30 git ls-remote personal refs/heads/codex/sz-fastwam-current-rlinf-grpo | cut -f1)" = "$(git rev-parse HEAD)"
test -z "$(git status --porcelain)"
test ! -e "$RUN"
test ! -e "$PACKET"
mkdir -p "$RUN/runtime" "$PACKET"
"$VIRTUAL_ENV/bin/python" - <<'PY'
import os, shlex, subprocess, json, hashlib, shutil
from pathlib import Path
import yaml
old,run,packet=(Path(os.environ[k]) for k in ['OLD','RUN','PACKET'])
prior=yaml.safe_load((old/'runtime/resolved.yaml').read_text())
command=[v.replace(str(old),str(run)).replace(prior['runner']['logger']['experiment_name'],run.name.replace('-','_')) for v in shlex.split((old/'runtime/command.txt').read_text())]
p=subprocess.run(command+['--cfg','job','--resolve'],capture_output=True,text=True)
assert p.returncode==0,(p.stdout,p.stderr)
cfg=yaml.safe_load(p.stdout)
def flat(v,p=''):
 if isinstance(v,dict):return {a:b for k,x in v.items() for a,b in flat(x,p+'.'+str(k) if p else str(k)).items()}
 if isinstance(v,list):return {a:b for k,x in enumerate(v) for a,b in flat(x,p+'.'+str(k)).items()}
 return {p:v}
a,b=flat(prior),flat(cfg)
diff={k:[a.get(k),b.get(k)] for k in a.keys()|b.keys() if a.get(k)!=b.get(k)}
allowed={'runner.logger.log_path','runner.logger.experiment_name','env.train.task_config.save_path','env.eval.task_config.save_path','env.train.video_cfg.video_base_dir','env.eval.video_cfg.video_base_dir'}
assert set(diff)==allowed,diff
assert cfg['env']['train']['total_num_envs']*cfg['env']['train']['rollout_epoch']==256
assert cfg['runner']['resume_dir'] is None and cfg['runner']['max_steps']==100
head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
contract=json.loads((old/'runtime/contract.json').read_text())
contract.update(donor=str(old),resolved_difference_count=len(diff),allowed_resolved_differences=diff)
contract['source_heads']['rlinf']=head
contract['native_fix']={'source':'tools/fastwam_scene_fence','library':'/home/chenyiteng/builds/fastwam-scene-fence-20260904/release-final/librlinf_scene_fence.so','sha256':'45e6cac35ea2edea0724fa4cd92ec82cab401b6abd6f8001805d22306f375df0','loading':'RoboTwin init only: RTLD_LOCAL after PyTorch and before SAPIEN. No LD_PRELOAD or LD_LIBRARY_PATH changes; Actor/Rollout/driver unaffected.','scope':'timeline render scene-access fence only; no original libraries modified; no dependency upgrade','smoke':str(Path(os.environ['DIAG']))}
contract['restart']='fresh original SFT; old stalled run had no checkpoint; old artifacts preserved'
contract['stop']='100 steps, driver fatal/exception or inherited120h wall cap. If new hang is confirmed during this supervised launch, capture evidence and stop only this run; no parameter change or automatic restart.'
(packet/'command.txt').write_text(shlex.join(command)+'\n')
(packet/'resolved.yaml').write_text(p.stdout)
(packet/'contract.json').write_text(json.dumps(contract,ensure_ascii=False,indent=2)+'\n')
(packet/'environment.sh').write_text((old/'runtime/environment.sh').read_text()+'\nsource /home/chenyiteng/builds/fastwam-scene-fence-20260904/release-final/enable.sh\n')
(packet/'source_head.txt').write_text(head+'\n')
(packet/'packet_complete.txt').write_text('Native smoke passed; six path/name-only resolved differences; source pushed; not launched\n')
shutil.copyfile(old/'runtime/robotwin.patch',packet/'robotwin.patch')
files=['command.txt','resolved.yaml','contract.json','environment.sh','source_head.txt','packet_complete.txt','robotwin.patch']
(packet/'sha256.txt').write_text(''.join(f'{hashlib.sha256((packet/n).read_bytes()).hexdigest()}  {n}\n' for n in files))
for n in files+['sha256.txt']:shutil.copyfile(packet/n,run/'runtime'/n)
for n in ['wrapper.sh','observer.sh']:shutil.copyfile(old/'runtime'/n,run/'runtime'/n)
print('SOURCE_HEAD',head)
print('RESOLVED_DIFF',json.dumps(diff,ensure_ascii=False))
print('CONTRACT',json.dumps(contract,ensure_ascii=False))
print('EXACT_COMMAND',shlex.join(command))
print('OUTPUT',run)
PY
date -Is
