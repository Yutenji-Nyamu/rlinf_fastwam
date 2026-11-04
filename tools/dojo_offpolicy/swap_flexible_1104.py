"""One-time Nov4 flexible swap after GPU4 failed Isaac; reuse accepted queues."""
from pathlib import Path
_ROOT=Path(__file__).resolve().parents[2]
PAYLOAD={name:(_ROOT/name).read_text() for name in ['tools/rlt_disk_fallback/common.py', 'tools/rlt_disk_fallback/owner.py', 'tools/rlt_disk_fallback/priority_pair_handoff.py', 'tools/pi05_policy_eval/grpo_entropy_queue.py', 'tools/dojo_offpolicy/fql_queue_admission.py']}
import ast,copy,json,os,shutil,signal,subprocess,sys,time,fcntl
from pathlib import Path
from omegaconf import OmegaConf
B=Path('/data/chenyiteng/deployment-20261104')
R=Path('/data/chenyiteng/projects/rlinf-shenzhen/worktrees/dojo-dsrl-fullnoise-20261104')
S=Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1')
PY='/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python'
for name,content in PAYLOAD.items():
    ast.parse(content);(R/name).write_text(content)
for name in ['common.py','owner.py']:
    (S/'ops'/name).write_text(PAYLOAD['tools/rlt_disk_fallback/'+name])
sys.path.insert(0,str(S/'ops'));from common import read,save,proc,same,exact_signal
sys.path.insert(0,str(R));from tools.dojo_offpolicy.swap_4567 import launch

old=B/'dojo-dsrl-repeat-g4';D=B/'dojo-dsrl-repeat-g6'
assert not D.exists();D.mkdir();(D/'rpc').mkdir(mode=0o700);(D/'config').mkdir()
for name in ['plan.json','training-environment.json','sim-environment.json','native-g4.json','config/formal.yaml','config/smoke.yaml','config/smoke_resume.yaml']:
    dst=D/name.replace('native-g4','native-g6')
    dst.write_text((old/name).read_text().replace(str(old),str(D)).replace('dojo-dsrl-repeat-g4','dojo-dsrl-repeat-g6'))
(D/'rpc/authkey').write_bytes(os.urandom(32));(D/'rpc/authkey').chmod(0o600)
p=read(D/'plan.json');p.update(gpu=6,gpus=[6],render_gpus={'6':6},reviewed_graphics_residual_mib={'4':8},namespace='RLinf_dojo_dsrl_repeat_g6_20261104');save(D/'plan.json',p)
n=read(D/'native-g6.json');n.update(gpu=6,output=str(D/'native-g6'),socket=str(D/'rpc/gpu6.sock'));save(D/'native-g6.json',n)
for phase in ['smoke','smoke_resume','formal']:
    f=D/f'config/{phase}.yaml';c=OmegaConf.load(f);c.cluster.component_placement={'actor, env, rollout':'6'}
    for mode in ['train','eval']:c.env[mode].physical_gpus=[6]
    OmegaConf.save(c,f)
G0=B/'grpo-clean-entropy-g67';G=B/'grpo-clean-entropy-g47'
G.mkdir();(G/'config').mkdir();shutil.copytree(G0/'tools',G/'tools')
(G/'tools/grpo_entropy_queue.py').write_text(PAYLOAD['tools/pi05_policy_eval/grpo_entropy_queue.py'])
admission={'4':{'admission_graphics_sources':[dict(deployment=str(B/'dojo-dsrl-full-g5'),gpu=5,max_mib=8),dict(deployment=str(D),gpu=6,max_mib=8)]}}
gp=read(G0/'plan.json');gp.update(gpus=[4,7],predecessor_identity=read(G0/'identity.json'),driver_output=str(G/'driver'),namespace='RLinf_grpo_clean_entropy_g47_20261104',gpu_admission=admission)
save(G/'plan.json',gp);save(G/'training-environment.json',read(G0/'training-environment.json'))
c=OmegaConf.load(G0/'config/grpo_clean.yaml');c.cluster.component_placement={'actor, env, rollout':'4,7'};OmegaConf.save(c,G/'config/grpo_clean.yaml')
for oldg,newg in [(6,4),(7,7)]:
    e=read(G0/f'inference-environment-g{oldg}.json');e.update(CUDA_VISIBLE_DEVICES=str(newg),DOJO_GPU_SCOPE=str(newg));save(G/f'inference-environment-g{newg}.json',e)
    cfg=read(G0/f'pool-g{oldg}.json');cfg.update(gpu=newg,output=cfg['output'].replace(f'g{oldg}',f'g{newg}'));save(G/f'pool-g{newg}.json',cfg)
F0=B/'fql-v2-after-grpo-g67';F=B/'fql-v2-after-grpo-g47'
F.mkdir();shutil.copytree(F0/'seeds',F/'seeds')
for name in ['formal-plan.json','execution-source.json','fql_fixed_teacher.json','eval-template.json','prepared.json']:save(F/name,read(F0/name))
prepared=read(F/'prepared.json');prepared['physical_gpus']=[4,7];save(F/'prepared.json',prepared)

do=launch(D,[PY,'-u','-B',str(R/'tools/dojo_offpolicy/owner.py'),str(D)])
go=launch(G,['/usr/bin/python3','-u','-B',str(G/'tools/grpo_entropy_queue.py'),str(G)])
fp=read(F/'formal-plan.json');fp.update(predecessor_identity=go,predecessor_deployment=str(G),gpu_admission=admission);save(F/'formal-plan.json',fp)
fs=read(F/'execution-source.json')['source']
fo=launch(F,[PY,'-u','-B',str(R/'tools/dojo_offpolicy/fql_queue_admission.py'),str(F),fs],cwd=fs,name='queue.log')

serial=(S/'priority-handoff.lock').open('a');fcntl.flock(serial,fcntl.LOCK_EX)
oldowner=read(S/'owner-identity.json');exact_signal(oldowner,signal.SIGTERM)
deadline=time.monotonic()+30
while same(oldowner):
    assert time.monotonic()<deadline;time.sleep(.1)
lock=(S/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
p,state=read(S/'plan.json'),read(S/'status.json')
for g in ['4','7']:
    p['slots'][g].update(external_identity=fo,external_release=str(F/'released.json'),external_blockers=[go],**admission.get(g,{}));state['slots'][g]={'phase':'WAIT_EXTERNAL'}
p['slots']['6'].update(external_identity=do,external_release=str(D/'released.json'),external_blockers=[]);state['slots']['6']={'phase':'WAIT_EXTERNAL'}
save(S/'plan.json',p);save(S/'status.json',state)
for dd,who in [(D,do),(G,go),(F,fo)]:save(dd/'registered.json',dict(identity=who,time=time.time(),reason='user flexible allocation: Dojo5/6, RoboTwin4/7'))
targets=[read(F0/'identity.json'),read(G0/'identity.json'),read(G0/'status.json')['child']]
lock.close();launch(B,['/usr/bin/python3','-u','-B',str(S/'ops/owner.py'),'--plan',str(S/'plan.json')],name='flex-rlt-owner.log',identity=None,cwd=S);serial.close()
for who in targets:
    if same(who):exact_signal(who,signal.SIGTERM)
launch(D,[PY,'-u','-B',str(R/'tools/dojo_offpolicy/recover_queue.py'),str(D)],name='recovery.log',identity='recovery-identity.json')
save(B/'flex-swap-receipt.json',dict(time=time.time(),owners={'dojo6':do,'grpo47':go,'fql47':fo},stopped=targets,admission=admission))
print(json.dumps(read(B/'flex-swap-receipt.json')))
