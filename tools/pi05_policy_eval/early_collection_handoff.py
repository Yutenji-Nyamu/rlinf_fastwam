"""Nov4 authorized CP0/40/80/120 analysis; prepare before stopping GRPO.

PAYLOAD is supplied by the Windows source uploader, never contains credentials.
"""
import ast,fcntl,json,os,shutil,signal,subprocess,sys,time
from pathlib import Path
B=Path('/data/chenyiteng/deployment-20261104')
R=Path('/data/chenyiteng/projects/rlinf-shenzhen/worktrees/dojo-dsrl-fullnoise-20261104')
S=Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1')
PY='/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python'
sys.path.insert(0,str(S/'ops'));from common import read,save,proc,same,exact_signal
sys.path.insert(0,str(R));from tools.dojo_offpolicy.swap_4567 import launch

def main():
    assert os.getuid()==1003
    old=B/'grpo-clean-entropy-g47';oldf=B/'fql-v2-after-grpo-g47'
    d=B/'grpo-fourpoint-g47';f=B/'fql-v2-after-fourpoint-g47'
    assert not d.exists() and not f.exists()
    d.mkdir();shutil.copytree(old/'tools',d/'tools')
    for name,content in PAYLOAD.items():
        ast.parse(content);(d/'tools'/name).write_text(content)
    p=read(old/'plan.json')
    predecessors=[read(old/'identity.json'),read(old/'status.json')['child']]
    p.update(output='/data/chenyiteng/results/pi05-entropy/20261104-clean-fourpoint',milestones=[0,40,80,120],entropy_bins=20,artifact_prefix='four-checkpoint',predecessors=predecessors)
    for step in p['milestones'][1:]:
        assert (Path(p['checkpoint_root'])/f'global_step_{step}/actor/model_state_dict/full_weights.pt').is_file()
    save(d/'plan.json',p)
    for g in p['gpus']:
        save(d/f'inference-environment-g{g}.json',read(old/f'inference-environment-g{g}.json'))
        c=read(old/f'pool-g{g}.json')
        text=json.dumps(c).replace(read(old/'plan.json')['output'],p['output']).replace('global_step_200','global_step_120')
        save(d/f'pool-g{g}.json',json.loads(text))
    f.mkdir();shutil.copytree(oldf/'seeds',f/'seeds')
    for name in ['formal-plan.json','execution-source.json','fql_fixed_teacher.json','eval-template.json','prepared.json']:
        save(f/name,read(oldf/name))
    # New owners are CPU-only until the registration and predecessor handoff.
    co=launch(d,['/usr/bin/python3','-u','-B',str(d/'tools/checkpoint_collection_owner.py'),str(d)])
    fp=read(f/'formal-plan.json');fp.update(predecessor_identity=co,predecessor_deployment=str(d));save(f/'formal-plan.json',fp)
    fs=read(f/'execution-source.json')['source']
    fo=launch(f,[PY,'-u','-B',str(R/'tools/dojo_offpolicy/fql_queue_admission.py'),str(f),fs],cwd=fs,name='queue.log')
    serial=(S/'priority-handoff.lock').open('a');fcntl.flock(serial,fcntl.LOCK_EX)
    owner=read(S/'owner-identity.json');exact_signal(owner,signal.SIGTERM)
    deadline=time.monotonic()+30
    while same(owner):
        assert time.monotonic()<deadline;time.sleep(.1)
    lock=(S/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan,state=read(S/'plan.json'),read(S/'status.json')
    save(d/'previous-rlt-plan.json',plan)
    for g in ['4','7']:
        plan['slots'][g].update(external_identity=fo,external_release=str(f/'released.json'),external_blockers=[co],**p.get('gpu_admission',{}).get(g,{}));state['slots'][g]={'phase':'WAIT_EXTERNAL'}
    save(S/'plan.json',plan);save(S/'status.json',state)
    # Stop the old waiting FQL first, so GRPO's exit cannot launch it.
    oldfw=read(oldf/'identity.json');assert same(oldfw)
    exact_signal(oldfw,signal.SIGTERM)
    for directory,who in [(d,co),(f,fo)]:save(directory/'registered.json',dict(identity=who,time=time.time()))
    lock.close()
    launch(d,['/usr/bin/python3','-u','-B',str(S/'ops/owner.py'),'--plan',str(S/'plan.json')],name='rlt-owner.log',identity=None,cwd=S)
    serial.close()
    for who in predecessors:
        if same(who):exact_signal(who,signal.SIGTERM)
    receipt=dict(time=time.time(),stopped=[oldfw]+predecessors,collector=co,fql=fo,milestones=p['milestones'],gpus=p['gpus'],reason='user early stop; retain saved CP40/80/120; qualitative Table1 first')
    save(d/'handoff.json',receipt);save(old/'intentional-stop.json',receipt)
    print(json.dumps(receipt))

if __name__=='__main__':main()
