"""Exclusive Norm6/7 lease: failures hold the cards; both formal jobs then return RLT."""
import argparse,fcntl,hashlib,importlib.util,json,math,os,signal,socket,subprocess,sys,threading,time,traceback
from pathlib import Path

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(x,indent=2));tmp.replace(p)
def load(p):
    spec=importlib.util.spec_from_file_location('norm_cycle',p);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m
def scalars(run):
    sys.modules.setdefault('tensorflow',None)
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    out={};root=Path(run)
    for p in (root/'tensorboard',root/root.name/'tensorboard'):
        if p.is_dir():
            a=EventAccumulator(str(p),size_guidance={'scalars':0});a.Reload()
            for k in a.Tags()['scalars']:out[k]=[{'step':r.step,'value':r.value,'time':r.wall_time} for r in a.Scalars(k)]
    return out
def smoke_gate(req):
    m=scalars(req['run']);assert m,'No smoke scalar evidence'
    assert all(math.isfinite(v['value']) for k,vs in m.items() if k.startswith('train/') for v in vs),'Nonfinite train metric'
    prefix='train/norm/' if req['role']=='bc' else 'train/dsrl_u/'
    effect=m.get(prefix+'weight_std',[])
    assert any(v['value']>0 for v in effect),'No nonuniform Norm weights'
    grad=m.get('train/actor/grad_norm',[]);assert any(v['value']>0 for v in grad),'No actor gradient'
    root=Path(req['run']);cp=root/root.name/'checkpoints'/('global_step_'+str(req['steps']))/'actor'
    assert cp.is_dir(),str(cp)
    if req['role']=='bc':
        assert len(m.get('train/bc/actor_loss',[]))>=2
        files=[cp/'online_bc/rank_0/norm.pt',cp/'online_bc/rank_0/success_replay.pt',cp/'online_bc/rank_0/learner.pt',cp/'local_shard_checkpoint/checkpoint_rank_0.pt']
        assert all(p.is_file() and p.stat().st_size>0 for p in files)
    else:
        assert any(v['value']>0 for v in m.get('train/critic/grad_norm',[]))
        audit_path=Path(req['repo'])/'tools/norm_20261007/audit_checkpoint.py'
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',PYTHONPATH=req['repo'])
        result=subprocess.run([sys.executable,'-B',str(audit_path),str(cp.parent),'--role','u','--expected-runner-step',str(req['steps'])],env=env,text=True,capture_output=True,timeout=240)
        audit=json.loads(result.stdout.strip().splitlines()[-1]);assert result.returncode==0 and audit['ok'],audit
        assert audit['update_step']>0 and audit['replay']['u']['signal_spec']['name']=='norm_residual_t5_l3'
        save(Path(req['runtime'])/'checkpoint-audit.json',audit)
        files=[cp/'sac_components/dsrl_trainer_state_rank_0.pt',cp/'sac_components/replay_buffer/rank_0/dsrl_transition_replay.pt']
    return {'time':time.time(),'weights':effect,'gradients':grad,'checkpoint':str(cp),'files':[{'path':str(p),'bytes':p.stat().st_size} for p in files],'metrics':{k:vs[-1] for k,vs in m.items() if k.startswith(prefix) or k in ('train/sac/actor_loss','train/sac/critic_loss','train/bc/actor_loss','env/success_once')}}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--control',required=True);args=parser.parse_args();C=Path(args.control);p=read(C/'plan.json')
    assert os.getuid()==p['uid']==20001 and socket.gethostname()==p['hostname']
    assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==p['boot_id']
    for name,digest in p['pins'].items():assert sha(name)==digest,name
    cycle=Path(p['cycle']);m=load(cycle/'rlt_returned_cycle.py');m.install_helper(cycle);H=m.H;m.load_plan(cycle)
    lock=(C/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lease=(cycle/'operation.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (C/'owner-identity.json').exists(),'Never duplicate owner'
    identity=H.proc(os.getpid());H.save(C/'owner-identity.json',dict(identity,boot_id=p['boot_id']))
    state={'time':time.time(),'owner':identity,'phase':'BORROWING','roles':{r:{'state':'DEV_HOLD'} for r in p['roles']},'active':{},'completed':{},'managed':{}}
    for name in ('requests','receipts'):(C/name).mkdir(exist_ok=True)
    done=threading.Event()
    def heartbeat():
        while not done.is_set():
            save(C/'heartbeat.json',{'time':time.time(),'owner':identity,'phase':state['phase']});done.wait(5)
    threading.Thread(target=heartbeat,daemon=True).start()
    stop_requested=False
    def signal_stop(*_):
        nonlocal stop_requested;stop_requested=True
    signal.signal(signal.SIGTERM,signal_stop)
    children={};seen=set();attempts={};last_resource=0
    try:
        if not (cycle/'rlt-stopped.json').exists():m.stop(cycle)
        assert not H.gpu_processes([6,7]);state['phase']='DEV_HOLD';save(C/'status.json',state)

        def launch(req,path):
            role=req['role'];approved=p['roles'][role]
            assert req['gpu']==approved['gpu'] and req['repo']==approved['repo']
            assert role not in children and req['kind'] in ('probe','smoke','formal')
            assert subprocess.check_output(['git','-C',req['repo'],'rev-parse','HEAD'],text=True).strip()==req['head']
            assert not subprocess.check_output(['git','-C',req['repo'],'status','--porcelain'],text=True).strip()
            for name,digest in req['pins'].items():assert sha(name)==digest,name
            assert not H.gpu_processes([req['gpu']]) and not H.active(H.actors(p),req['namespace'])
            rt=Path(req['runtime']);rt.mkdir(parents=True,exist_ok=True);assert not (rt/'launch.json').exists()
            env=read(rt/'environment.json');env.pop('CUDA_VISIBLE_DEVICES',None)
            if req['kind']=='probe':
                env['CUDA_VISIBLE_DEVICES']=str(req['gpu'])
                argv=[p['python'],'-u','-B',str(Path(p['root'])/'ops/gpu_probe.py'),'--config',str(rt/'resolved.yaml'),'--role',role,'--gpu',str(req['gpu']),'--output',str(rt/'probe.json')]
            else:argv=[p['python'],'-u','-B',str(Path(p['root'])/'ops/driver.py'),'--control',str(C),'--request',str(path)]
            H.save(rt/'launch.json',{'time':time.time(),'request':str(path),'argv':argv})
            log=(rt/'driver.log').open('x');child=subprocess.Popen(argv,cwd=req['repo'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close()
            ident=H.proc(child.pid);assert ident and ident['uid']==20001
            state['managed'][str(ident['pid'])]=ident
            children[role]=(child,req,path,ident,time.time());state['roles'][role]={'state':req['kind'].upper(),'request':req['id'],'driver':ident,'runtime':req['runtime'],'namespace':req['namespace']}
            H.save(C/'receipts'/(req['id']+'-launched.json'),state['roles'][role])

        while True:
            if children and time.time()-last_resource>30:
                last_resource=time.time()
                try:
                    gpu=H.gpu_processes(list(range(8)));actors=H.actors(p);owned={}
                    for role,(child,req,path,ident,started) in children.items():
                        selected=H.active(actors,req['namespace']);jobs={a['job_id'] for a in selected}
                        assert len(jobs)<=1;H.validate_actor_rows(selected,req['namespace'],jobs)
                        roots={ident['pid']}|{a['pid'] for a in selected if a.get('pid')}
                        tree=H.process_tree(roots);state['managed'].update({str(k):v for k,v in tree.items()})
                        owned[role]=[g for g in gpu if g['pid'] in tree]
                        outside=[g for g in owned[role] if g['gpu']!=req['gpu']]
                        if outside and H.same(ident):
                            save(C/'receipts'/(req['id']+'-binding-error.json'),outside);os.kill(ident['pid'],signal.SIGTERM)
                    save(C/'resources.json',{'time':time.time(),'owned':owned,'all_gpu_processes':gpu})
                except Exception as exc:save(C/'resource-query-error.json',{'time':time.time(),'error':repr(exc)})
            if stop_requested:
                state['phase']='STOP_REQUESTED_HOLD'
                for role,(child,req,path,ident,started) in children.items():
                    if H.same(ident):os.kill(ident['pid'],signal.SIGTERM)
                stop_requested=False
            for path in sorted((C/'requests').glob('*.json')):
                if path.name in seen:continue
                req=read(path);role=req['role']
                if role in children:continue
                seen.add(path.name)
                try:launch(req,path)
                except Exception as exc:
                    state['roles'][role]={'state':'FAILED_HOLD','request':req.get('id'),'error':repr(exc)}
                    save(C/'receipts'/(req['id']+'-error.json'),{'time':time.time(),'error':repr(exc),'traceback':traceback.format_exc()})
            for role,(child,req,path,ident,started) in list(children.items()):
                if child.poll() is None:
                    if time.time()-started>req['timeout_seconds'] and H.same(ident):os.kill(ident['pid'],signal.SIGTERM)
                    continue
                try:
                    live=H.active(H.actors(p),req['namespace']);contexts=H.gpu_processes([req['gpu']])
                    if live or contexts:
                        attempts[req['id']]=attempts.get(req['id'],0)+1
                        assert attempts[req['id']]<60,'Driver exited but scoped resources did not release'
                        continue
                    if child.returncode:raise RuntimeError('Driver exit '+str(child.returncode))
                    rt=Path(req['runtime'])
                    if req['kind']=='probe':assert read(rt/'probe.json')['status']=='PASS';proof=read(rt/'probe.json')
                    else:
                        finish=read(rt/'finished.json');assert finish['exit_code']==0 and finish['cleanup_error'] is None
                        proof=smoke_gate(req)
                        if req['kind']=='formal':proof['formal_exit0']=True
                    save(C/'receipts'/(req['id']+'-accepted.json'),proof);state['completed'][req['id']]=proof
                    state['roles'][role]={'state':req['kind'].upper()+'_PASSED','request':req['id']}
                    if req['kind']=='smoke' and req.get('auto_formal'):
                        nextreq=read(req['auto_formal']);assert nextreq['role']==role and nextreq['kind']=='formal'
                        H.save(C/'requests'/(nextreq['id']+'.json'),nextreq)
                    if req['kind']=='formal':state['roles'][role]['state']='FORMAL_COMPLETE'
                except Exception as exc:
                    state['roles'][role]={'state':'FAILED_HOLD','request':req['id'],'error':repr(exc)}
                    save(C/'receipts'/(req['id']+'-error.json'),{'time':time.time(),'error':repr(exc),'traceback':traceback.format_exc()})
                del children[role]
            if all(row['state']=='FORMAL_COMPLETE' for row in state['roles'].values()):
                assert not H.gpu_processes([6,7]);assert all(not H.same(i) for i in state['managed'].values())
                release={'cycle_id':cycle.name,'gpus':[6,7],'terminal_status':'completed','all_workers_stopped':True,'managed_processes':list(state['managed'].values()),'time':H.now()}
                H.save(C/'release.json',release);state['phase']='RETURNING_RLT';save(C/'status.json',state)
                m.H.resume(cycle,C/'release.json');state['phase']='RLT_RETURNED';save(C/'status.json',state);break
            state['phase']='RUNNING' if children else 'DEV_HOLD'
            state['time']=time.time();state['active']={r:{'request':v[1]['id'],'pid':v[3]['pid']} for r,v in children.items()};save(C/'status.json',state)
            time.sleep(5)
    except BaseException as exc:
        state['phase']='OWNER_FAILED_HOLD';state['error']=repr(exc);save(C/'status.json',state);save(C/'owner-error.json',{'error':repr(exc),'traceback':traceback.format_exc(),'time':time.time()});raise
    finally:done.set()
if __name__=='__main__':main()
