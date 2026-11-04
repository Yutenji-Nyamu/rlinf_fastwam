"""Nov4 authorized immediate swap; prepare all queues before exact scoped stops.

No global Ray or GPU reset. Original experiment outputs/checkpoints stay intact.
Accepted RLT requests are cloned with their latest complete restore points.
"""
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from omegaconf import OmegaConf

B = Path('/data/chenyiteng/deployment-20261104')
S = Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1')
R = Path('/data/chenyiteng/projects/rlinf-shenzhen/worktrees/dojo-dsrl-fullnoise-20261104')
G0 = Path('/home/chenyiteng/deployment-20261102/grpo-clean-entropy-g45')
F0 = Path('/home/chenyiteng/deployment-20261101/fql-v2/formal-g45-after-grpo-1102')
D0 = Path('/data/chenyiteng/deployment-20261103/dojo-offpolicy-g6-r3-n8-recovery-1')
G, F = B/'grpo-clean-entropy-g67', B/'fql-v2-after-grpo-g67'
PY = '/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python'
sys.path.insert(0, str(S/'ops'))
from common import read, save, proc, same, exact_signal, gpus


def launch(d, command, env=None, name='outer.log', identity='identity.json', cwd=R):
    with (d/name).open('x') as log:
        child = subprocess.Popen(command, cwd=cwd,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(R), PYTHONDONTWRITEBYTECODE='1', **(env or {})),
            stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    if identity:
        deadline = time.monotonic()+60
        while not (d/identity).is_file():
            assert child.poll() is None and time.monotonic()<deadline, (str(d), child.poll())
            time.sleep(.1)
        who = read(d/identity)
        assert who['pid'] == child.pid and same(who)
        return who
    return proc(child.pid)


def prepare_rlt(d, plan, state):
    result = {}
    for g in map(str, [4, 5, 6, 7]):
        st = state['slots'][g]
        q = read(st.get('request', plan['slots'][g]['fallback']))
        if q.get('normal_request'):
            q = read(q['normal_request'])
        assert q['kind']=='rlt' and q['gpu']==int(g)
        rt = Path(q['runtime'])
        c = OmegaConf.to_container(OmegaConf.load(rt/'resolved.yaml'), resolve=True)
        run = Path(q['run']); logger=c['runner']['logger']
        existing=[Path(x) for x in [c['runner'].get('resume_dir'),q.get('checkpoint')] if x]
        roots={run/run.name/'checkpoints',Path(logger['log_path'])/logger['experiment_name']/'checkpoints',*[p.parent for p in existing]}
        candidates=existing+[p for root in roots for p in root.glob('global_step_*')]
        complete=[]
        for cp in candidates:
            shard=cp/'actor/local_shard_checkpoint/checkpoint_rank_0.pt'
            dcp=cp/'actor/dcp_checkpoint'
            if (shard.is_file() and zipfile.is_zipfile(shard)) or ((dcp/'.metadata').is_file() and any(dcp.glob('*.distcp'))):complete.append(cp)
        assert complete, f'No RLT restore for GPU{g}'
        cp=max(complete,key=lambda p:int(p.name.rsplit('_',1)[1]))
        new=run/('runtime-after-'+d.name+f'-g{g}');new.mkdir()
        c['runner']['resume_dir']=str(cp)
        if 'dvac_gradient_weighting' in c.get('algorithm',{}):c['algorithm']['dvac_gradient_weighting']['output_dir']=str(run/('dvac-after-'+d.name+f'-g{g}'))
        save(new/'resolved.yaml',c);save(new/'environment.json',read(rt/'environment.json'))
        q.update(runtime=str(new),checkpoint=str(cp),step=int(cp.name.rsplit('_',1)[1]))
        path=d/f'fallback-g{g}.json';save(path,q);result[g]=str(path)
    return result


def main():
    assert os.getuid()==1003
    d=B/'swap-dojo45-grpo67';d.mkdir(exist_ok=False)
    save(d/'identity.json',proc(os.getpid()))
    oldplan,oldstate=read(S/'plan.json'),read(S/'status.json')
    save(d/'previous-plan.json',oldplan);save(d/'previous-status.json',oldstate)
    fallbacks=prepare_rlt(d,oldplan,oldstate)
    assert oldstate['slots']['7']['phase']=='RLT_RUNNING'
    old_rlt=oldstate['slots']['7']['identity']
    old_grpo=read(G0/'status.json')['child']
    old_watch=read(D0.with_name('dojo-offpolicy-g6-r3-n8')/'recovery-identity.json')
    # All newly started owners remain CPU-only until registered below.
    owners={}
    for g,name in [(4,'repeat'),(5,'full')]:
        dd=B/f'dojo-dsrl-{name}-g{g}'
        owners[g]=launch(dd,[PY,'-u','-B',str(R/'tools/dojo_offpolicy/owner.py'),str(dd)])
    owners['grpo']=launch(G,['/usr/bin/python3','-u','-B',str(G/'tools/grpo_entropy_queue.py'),str(G)])
    fp=read(F/'formal-plan.json');fp.update(predecessor_identity=owners['grpo'],predecessor_deployment=str(G));save(F/'formal-plan.json',fp)
    fs=Path(read(F/'execution-source.json')['source'])
    owners['fql']=launch(F,[PY,'-u','-B',str(fs/'tools/pi05_distillation/single_formal_queue.py'),str(F)],
        env={'PI05_DISTILL_SOURCE_ROOT':str(fs)},cwd=fs,name='queue.log')

    serial=(S/'priority-handoff.lock').open('a');fcntl.flock(serial,fcntl.LOCK_EX)
    old_owner=read(S/'owner-identity.json')
    if same(old_owner):exact_signal(old_owner,signal.SIGTERM)
    deadline=time.monotonic()+30
    while same(old_owner):
        assert time.monotonic()<deadline
        time.sleep(.1)
    lock=(S/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p,state=read(S/'plan.json'),read(S/'status.json')
    for g in map(str,[4,5,6,7]):
        target=B/f'dojo-dsrl-{("repeat" if g=="4" else "full")}-g{g}' if g in ('4','5') else F
        who=owners[int(g)] if g in ('4','5') else owners['fql']
        p['slots'][g].update(fallback=fallbacks[g],external_identity=who,external_release=str(target/'released.json'),
                             external_blockers=[] if g in ('4','5') else [owners['grpo']])
        p['slots'][g].pop('priority_after_external',None)
        state['slots'][g]={'phase':'WAIT_EXTERNAL'}
    save(S/'plan.json',p);save(S/'status.json',state)
    targets=[old_watch,read(F0/'identity.json'),read(G0/'identity.json'),old_grpo,read(D0/'identity.json'),old_rlt]
    assert all(w['uid']==1003 for w in targets)
    save(d/'prepared.json',dict(time=time.time(),owners=owners,stopping=targets,fallbacks=fallbacks,grpo_resume=120))
    # Register before stopping: new owners observe card release and start at once.
    for key,dd in [(4,B/'dojo-dsrl-repeat-g4'),(5,B/'dojo-dsrl-full-g5'),('grpo',G),('fql',F)]:
        save(dd/'registered.json',dict(identity=owners[key],time=time.time(),swap=str(d)))
    lock.close()
    launch(d,['/usr/bin/python3','-u','-B',str(S/'ops/owner.py'),'--plan',str(S/'plan.json')],name='rlt-owner.log',identity=None,cwd=S)
    serial.close()
    for who in targets:
        if same(who):exact_signal(who,signal.SIGTERM)
    save(d/'stops-issued.json',dict(time=time.time(),targets=targets))
    for gpu,name in [(4,'repeat'),(5,'full')]:
        dd=B/f'dojo-dsrl-{name}-g{gpu}'
        launch(dd,[PY,'-u','-B',str(R/'tools/dojo_offpolicy/recover_queue.py'),str(dd)],name='recovery.log',identity='recovery-identity.json')
    deadline=time.monotonic()+180
    while any(same(w) for w in targets):
        assert time.monotonic()<deadline,'Exact old owner cleanup did not finish'
        time.sleep(.5)
    save(d/'complete.json',dict(time=time.time(),owners=owners,old_stopped=True,slots=read(S/'status.json')['slots'],gpu=gpus()))
    print(json.dumps(dict(swapped=True,owners=owners)),flush=True)


if __name__=='__main__':main()
