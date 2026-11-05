"""Prepare four same-card continuations without stopping the active trainers."""
import copy
import fcntl
import json
import os
import signal
import sys
import time
from pathlib import Path
from omegaconf import OmegaConf

R = Path('/data/chenyiteng/projects/rlinf-shenzhen/worktrees/parallel-dsrl-u-20261104')
B = Path('/data/chenyiteng/deployment-20261105/dsrl400')
OLD = Path('/data/chenyiteng/deployment-20261104')
S = Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1')
PY = '/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python'
sys.path.insert(0, str(S/'ops'))
from common import read, save, same, exact_signal
sys.path.insert(0, str(R))
from tools.dojo_offpolicy import swap_4567 as swap
swap.R = R


def main():
    assert os.getuid() == 1003
    B.mkdir(parents=True, exist_ok=False)
    source = {4:'rt2-dsrl-clean-g4', 5:'dojo-dsrl-full-g5', 6:'dojo-dsrl-u-g6', 7:'rt2-dsrl-u-g7'}
    original_rlt = read(S/'plan.json')
    entries = {}
    for gpu, name in source.items():
        old, d = OLD/name, B/name
        dojo = gpu in (5,6)
        oldowner, owner = (old,d) if dojo else (old/'owner',d/'owner')
        predecessor = read(oldowner/'identity.json')
        assert same(predecessor) and read(oldowner/'status.json')['phase'] == 'formal'
        assert original_rlt['slots'][str(gpu)]['external_identity']['pid'] == predecessor['pid']
        owner.mkdir(parents=True)
        (d/'config').mkdir()
        def relocate(obj):
            return json.loads(json.dumps(obj).replace(str(old),str(d))) if dojo else copy.deepcopy(obj)
        plan = relocate(read(old/'plan.json'))
        plan.update(namespace=plan['namespace']+'_to400', pre_registered=True, predecessors=[predecessor])
        c0 = OmegaConf.to_container(OmegaConf.load(old/'config/formal.yaml'), resolve=True)
        cfg = relocate(c0)
        end = c0['runner']['max_steps']
        assert end == (100 if dojo else 200)
        cp = Path(c0['runner']['logger']['log_path'])/c0['runner']['logger']['experiment_name']/'checkpoints'/f'global_step_{end}'
        cfg['runner'].update(max_steps=400, resume_dir=str(cp))
        # The only algorithm/config change is the total budget and full resume.
        check = copy.deepcopy(cfg)
        check['runner']['max_steps'] = end
        check['runner']['resume_dir'] = c0['runner'].get('resume_dir')
        assert check == relocate(c0)
        OmegaConf.save(OmegaConf.create(cfg), d/'config/formal.yaml')
        env = relocate(read(old/'training-environment.json'))
        env['DOJO_GRPO_DEPLOYMENT'] = str(d)
        save(d/'training-environment.json', env)
        if dojo:
            plan['phases'] = [x for x in plan['phases'] if x['name']=='formal']
            plan['phases'][0]['timeout'] = None
            save(d/'sim-environment.json', relocate(read(old/'sim-environment.json')))
            save(d/f'native-g{gpu}.json', relocate(read(old/f'native-g{gpu}.json')))
            (d/'rpc').mkdir(mode=0o700)
            (d/'rpc/authkey').write_bytes((old/'rpc/authkey').read_bytes())
            (d/'rpc/authkey').chmod(0o600)
        else:
            op = read(oldowner/'plan.json')
            op.update(from_rlt=False, predecessors=[predecessor], phases=[dict(role='clean' if gpu==4 else 'u', phase='formal', deployment=str(d), timeout=None)])
            save(owner/'plan.json', op)
        save(d/'plan.json', plan)
        meta=dict(gpu=gpu, kind='dojo' if dojo else 'rt2', predecessor=predecessor,
                  previous_release=str(oldowner/'released.json'), previous_deployment=str(old),
                  checkpoint=str(cp), deployment=str(d), target=400)
        save(owner/'continuation.json', meta)
        entries[str(gpu)]=dict(owner=str(owner), deployment=str(d), **{k:v for k,v in meta.items() if k!='deployment'})
    # Include only known tiny simulator discovery contexts, on the same GPU4.
    for gpu in (5,6):
        original_rlt['slots']['4'].setdefault('admission_graphics_sources',[]).append(dict(deployment=entries[str(gpu)]['deployment'],gpu=gpu,max_mib=8))
    o4=Path(entries['4']['owner']);p4=read(o4/'plan.json')
    p4['gpu_admission']=original_rlt['slots']['4'];save(o4/'plan.json',p4)
    save(B/'prepared.json',dict(time=time.time(),entries=entries,config_diff='max_steps=400; full resume; private Dojo IPC paths only'))
    owners={}
    for gpu,e in entries.items():
        owners[gpu]=swap.launch(Path(e['owner']),[PY,'-u','-B',str(R/'tools/robotwin_dsrl_full/continue_owner.py'),e['owner']],cwd=R)
    pair=OLD/'parallel-u47'
    oldjoin=read(pair/'join-identity.json')
    assert same(oldjoin) and read(pair/'join-status.json')['phase']=='WAIT_BOTH_U'
    exact_signal(oldjoin,signal.SIGTERM)
    while same(oldjoin):time.sleep(.1)
    # Only pause the CPU scheduler under its existing handoff lock. Trainers stay.
    serial=(S/'priority-handoff.lock').open('a');fcntl.flock(serial,fcntl.LOCK_EX)
    ro=read(S/'owner-identity.json');exact_signal(ro,signal.SIGTERM)
    deadline=time.monotonic()+30
    while same(ro):
        assert time.monotonic()<deadline
        time.sleep(.1)
    lock=(S/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        p,st=read(S/'plan.json'),read(S/'status.json')
        save(B/'previous-rlt-plan.json',p)
        for gpu,e in entries.items():
            p['slots'][gpu].update(external_identity=owners[gpu],external_release=str(Path(e['owner'])/'released.json'),external_blockers=[e['predecessor']])
            st['slots'][gpu]={'phase':'WAIT_EXTERNAL'}
        p['slots']['4']['admission_graphics_sources']=original_rlt['slots']['4']['admission_graphics_sources']
        save(S/'plan.json',p);save(S/'status.json',st)
        for gpu,e in entries.items():save(Path(e['owner'])/'registered.json',dict(identity=owners[gpu],time=time.time()))
    finally:
        lock.close()
        swap.launch(B,['/usr/bin/python3','-u','-B',str(S/'ops/owner.py'),'--plan',str(S/'plan.json')],cwd=S,name='rlt-owner.log',identity=None)
        serial.close()
    jp=read(pair/'join-plan.json');save(B/'previous-join-plan.json',jp)
    jp['jobs']=[entries['4']['owner'],entries['7']['owner']];save(pair/'join-plan.json',jp)
    f=Path(jp['fql']);fp=read(f/'formal-plan.json')
    fp.setdefault('gpu_admission',{})['4']=p['slots']['4'];save(f/'formal-plan.json',fp)
    join=swap.launch(pair,[PY,'-u','-B',str(R/'tools/robotwin_dsrl_full/join_fql.py'),str(pair)],cwd=R,name='join-to400.log',identity=None)
    fp['predecessor_identity']=join;save(f/'formal-plan.json',fp)
    for gpu in ('5','6'):
        d=entries[gpu]['owner']
        swap.launch(B,[PY,'-u','-B','-c',f'from tools.dojo_offpolicy.recover_queue import watch_owner; watch_owner({d!r})'],cwd=R,name=f'watch-g{gpu}.log',identity=None)
    assert all(same(e['predecessor']) and same(owners[g]) for g,e in entries.items())
    receipt=dict(time=time.time(),owners=owners,join=join,entries=entries,active_trainers_untouched=True)
    save(B/'armed.json',receipt)
    print(json.dumps(receipt))


if __name__=='__main__':
    main()
