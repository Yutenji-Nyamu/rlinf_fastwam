"""Authorized correction: stable Dojo GPUs5/6, RT2 GPUs4/7, FQL after4/7."""
import copy,fcntl,hashlib,json,os,shutil,signal,sys,time
from pathlib import Path
from omegaconf import OmegaConf
B=Path('/data/chenyiteng/deployment-20261104');S=Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1')
R=Path('/data/chenyiteng/projects/rlinf-shenzhen/worktrees/parallel-dsrl-u-20261104')
PY='/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python'
sys.path.insert(0,str(S/'ops'));from common import read,save,same,exact_signal
sys.path.insert(0,str(R));from tools.dojo_offpolicy import swap_4567 as swap
swap.R=R

def scope4(d):
    old=Path('/data/chenyiteng/deployment-20261006/dsrl-u-lease-v1/scope-g6');new=d/'scope-g4';new.mkdir()
    shutil.copytree(old/'bootstrap',new/'bootstrap',ignore=shutil.ignore_patterns('__pycache__'))
    marker=new/'libdsrl_u_g4_20261104.so';shutil.copy2(old/'libdsrl_u_g6_20261006.so',marker)
    profile=Path('/home/chenyiteng/.nv/nvidia-application-profiles-rc.d/00-dsrl-u-g4-20261104.json')
    assert not profile.exists();save(profile,dict(rules=[dict(pattern=dict(feature='dso',matches=marker.name),profile=['EGLVisibleDGPUDevices',16])]))
    m=read(old/'scope.json');m.update(physical_gpu=4,minor=4,mask=16,gpu_uuid=read(S/'plan.json')['slots']['4']['gpu_uuid'],token='dsrl-u-g4-20261104',receipts_dir=str(new/'receipts'))
    (new/'receipts').mkdir()
    for key,p in [('marker',marker),('profile',profile),('bootstrap',new/'bootstrap/sitecustomize.py'),('runtime',new/'bootstrap/gpu_scope_runtime.py')]:
        p.chmod(0o600);m[key+'_path']=str(p);m[key+'_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
    save(new/'scope.json',m);(new/'scope.json').chmod(0o600)
    return old,new,marker

def main():
    pair=B/'parallel-u47';old6=B/'rt2-dsrl-full-g6-r3';old=old6/'clean';d4=B/'rt2-dsrl-clean-g4';d6=B/'dojo-dsrl-u-g6'
    for d in (d4,d6):d.mkdir(exist_ok=False)
    # Prepare full checkpoint migration, not an inference-only restart.
    p=read(old/'plan.json');cp_root=Path(p['run'])/'formal/formal/checkpoints'
    cps=[x for x in cp_root.glob('global_step_*') if (x/'dojo-complete.json').exists() and (x/'actor/local_shard_checkpoint/checkpoint_rank_0.pt').exists() and (x/'actor/sac_components/dsrl_trainer_state_rank_0.pt').exists()]
    cp=max(cps,key=lambda x:int(x.name.rsplit('_',1)[1]));oldscope,newscope,marker=scope4(d4)
    def rtmap(x):
        t=json.dumps(x).replace(p['training_source'],str(R)).replace(str(old),str(d4)).replace(str(oldscope),str(newscope)).replace('libdsrl_u_g6_20261006.so',marker.name).replace('dsrl_gpu6','dsrl_gpu4')
        return json.loads(t)
    env=rtmap(read(old/'training-environment.json'));c=rtmap(OmegaConf.to_container(OmegaConf.load(old/'config/formal.yaml'),resolve=True))
    c['runner']['resume_dir']=str(cp);c['cluster']['component_placement']={'actor,env,rollout':{'node_group':'dsrl_gpu4','placement':'4'}}
    c['cluster']['node_groups'][0]['env_configs'][0]['env_vars']=[{k:v} for k,v in sorted(env.items()) if k not in ('CUDA_VISIBLE_DEVICES','DOJO_GRPO_DEPLOYMENT')]
    rp=rtmap(p);rp.update(gpu=4,gpus=[4],namespace='RLinf_rt2_clean_g4_from_cp52',model_contract_required=False)
    save(d4/'plan.json',rp);save(d4/'training-environment.json',env);(d4/'config').mkdir();OmegaConf.save(OmegaConf.create(c),d4/'config/formal.yaml')
    o4=d4/'owner';o4.mkdir();predecessor=read(old6/'identity.json');assert same(predecessor)
    rules=read(S/'plan.json')['slots']['4']
    rules.setdefault('admission_graphics_sources',[]).append(dict(deployment=str(d6),gpu=6,max_mib=8))
    save(o4/'plan.json',dict(gpu=4,gpus=[4],training_source=str(R),from_rlt=True,predecessors=[predecessor],gpu_admission=rules,phases=[dict(role='clean',phase='formal',deployment=str(d4),timeout=604800)]))
    # Copy the newly prepared Dojo U configuration only, not failed simulator state.
    oldd=B/'dojo-dsrl-u-g4';dp=read(oldd/'plan.json');newrun=Path('/data/chenyiteng/results/dojo-offpolicy/dojo-dsrl-u-g6-20261104')
    def djmap(x):
        t=json.dumps(x).replace(str(oldd),str(d6)).replace(dp['run'],str(newrun)).replace('/native-g4','/native-g6').replace('/gpu4.sock','/gpu6.sock')
        return json.loads(t)
    np=djmap(dp);np.update(gpu=6,gpus=[6],render_gpus={'6':6},predecessors=[predecessor],namespace='RLinf_dojo_dsrl_u_g6',gpu_admission={},reviewed_graphics_residual_mib={'4':8},protected_output='/data/chenyiteng/models/protected-eval/dojo-dsrl-u-g6-20261104')
    save(d6/'plan.json',np)
    for n in ('sim-environment.json','training-environment.json'):save(d6/n,djmap(read(oldd/n)))
    nc=djmap(read(oldd/'native-g4.json'));nc['gpu']=6;save(d6/'native-g6.json',nc)
    (d6/'rpc').mkdir();(d6/'rpc/authkey').write_bytes(os.urandom(32));(d6/'rpc/authkey').chmod(0o600);(d6/'config').mkdir()
    for phase in ('smoke','smoke_resume','formal'):
        x=djmap(OmegaConf.to_container(OmegaConf.load(oldd/f'config/{phase}.yaml'),resolve=True));x['cluster']['component_placement']={'actor, env, rollout':'6'}
        for mode in ('train','eval'):x['env'][mode]['physical_gpus']=[6]
        OmegaConf.save(OmegaConf.create(x),d6/f'config/{phase}.yaml')
    # Stop only the waiting CPU join before changing its predecessor pair.
    join=read(pair/'join-identity.json');exact_signal(join,signal.SIGTERM)
    while same(join):time.sleep(.1)
    owners={6:swap.launch(d6,[PY,'-u','-B',str(R/'tools/dojo_offpolicy/owner.py'),str(d6)],cwd=R),4:swap.launch(o4,[PY,'-u','-B',str(R/'tools/robotwin_dsrl_full/queue.py'),str(o4)],cwd=R)}
    deadline=time.monotonic()+240
    while not (o4/'registered.json').exists():
        assert same(owners[4]) and time.monotonic()<deadline;time.sleep(.2)
    serial=(S/'priority-handoff.lock').open('a');fcntl.flock(serial,fcntl.LOCK_EX)
    ro=read(S/'owner-identity.json');exact_signal(ro,signal.SIGTERM)
    while same(ro):time.sleep(.1)
    lock=(S/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    sp,st=read(S/'plan.json'),read(S/'status.json');sp['slots']['6'].update(external_identity=owners[6],external_release=str(d6/'released.json'),external_blockers=[]);st['slots']['6']={'phase':'WAIT_EXTERNAL'}
    # Only known tiny G-only neighbor contexts; no healthy process is signalled.
    sp['slots']['4'].setdefault('admission_graphics_sources',[]).append(dict(deployment=str(d6),gpu=6,max_mib=8))
    save(S/'plan.json',sp);save(S/'status.json',st);save(d6/'registered.json',dict(identity=owners[6],time=time.time()))
    lock.close();swap.launch(pair,['/usr/bin/python3','-u','-B',str(S/'ops/owner.py'),'--plan',str(S/'plan.json')],cwd=S,name='rlt-owner-flex.log',identity=None);serial.close()
    exact_signal(predecessor,signal.SIGTERM)
    jp=read(pair/'join-plan.json');jp['jobs']=[str(o4),str(B/'rt2-dsrl-u-g7/owner')];save(pair/'join-plan.json',jp)
    f=Path(jp['fql']);fp=read(f/'formal-plan.json');fp.setdefault('gpu_admission',{})['4']=sp['slots']['4'];save(f/'formal-plan.json',fp)
    j=swap.launch(pair,[PY,'-u','-B',str(R/'tools/robotwin_dsrl_full/join_fql.py'),str(pair)],cwd=R,name='join-flex.log',identity=None)
    deadline=time.monotonic()+30
    while read(pair/'join-identity.json')['pid']!=j['pid']:assert time.monotonic()<deadline;time.sleep(.1)
    fp['predecessor_identity']=j;save(f/'formal-plan.json',fp)
    swap.launch(pair,[PY,'-u','-B','-c',f'from tools.dojo_offpolicy.recover_queue import watch_owner; watch_owner({str(d6)!r})'],cwd=R,name='dojo6-watch.log',identity=None)
    receipt=dict(time=time.time(),checkpoint=str(cp),owners=owners,join=j,dojo_gpus=[5,6],rt2_gpus=[4,7]);save(pair/'flexible-handoff.json',receipt);save(old6/'migrated-to-g4.json',receipt);print(receipt)

if __name__=='__main__':main()
