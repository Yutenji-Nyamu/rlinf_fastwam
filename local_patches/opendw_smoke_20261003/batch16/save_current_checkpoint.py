"""Save both FSDP actor ranks while the exact driver is paused in sampling."""
import argparse,datetime,hashlib,importlib.util,json,os,re,signal,socket,sys,time
from pathlib import Path

S=Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003')
O=S/'runs/formal-v2'
EXPECTED_OWNER={'pid':3242048,'uid':20001,'start':706364464}
EXPECTED_DRIVER={'pid':2340489,'uid':20001,'start':706555360}
ACTORS=[{'pid':2443415,'actor_id':'b076de40d13fa66110d35ad26f000000'},
        {'pid':2443418}]

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def record(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def now():return datetime.datetime.now().astimezone().isoformat()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    assert os.getuid()==20001 and socket.gethostname()=='h100-gpu01'
    assert not args.output.exists() and args.output.resolve().is_relative_to(S.resolve())
    plan=read(O/'owner-plan.json');spec=importlib.util.spec_from_file_location('cp_owner',plan['base_owner_module'])
    M=importlib.util.module_from_spec(spec);sys.modules[spec.name]=M;spec.loader.exec_module(M);M.load_lifecycle(plan)
    owner=read(O/'owner-identity.json');driver=read(O/'formal/driver-identity.json')
    for expected,actual in [(EXPECTED_OWNER,owner),(EXPECTED_DRIVER,driver)]:
        assert all(actual[k]==v for k,v in expected.items()) and M.H.same(actual)
    assert not (O/'error.json').exists() and not (O/'final.json').exists()
    actors=M.H.active(M.H.actors(plan),driver['namespace'])
    rows=[]
    for rank,expected in enumerate(ACTORS):
        matches=[r for r in actors if r['pid']==expected['pid'] and r['name']==f'OpenDWActor_opendw-adjust-bottle-formal-v2:{rank}']
        assert len(matches)==1
        row=matches[0];assert row['state']=='ALIVE'
        worker_env=json.loads(row['serialized_runtime_env'])['env_vars']
        assert worker_env['OPENDW_SMOKE_OWNER_TOKEN']==plan['token']
        assert worker_env['CUDA_VISIBLE_DEVICES']==str(rank+4)
        if expected.get('actor_id'):assert row['actor_id']==expected['actor_id']
        ident=M.H.proc(row['pid']);assert ident and ident['uid']==20001
        cmd=(Path('/proc')/str(row['pid'])/'cmdline').read_bytes().decode().replace('\0',' ')
        assert 'EmbodiedFSDPActor.recv_rollout_trajectories' in cmd,cmd
        rows.append({'rank':rank,'name':row['name'],'actor_id':row['actor_id'],'identity':ident})
    log=(O/'formal/driver.log').read_text()
    steps=[int(x) for x in re.findall(r'Global Step:\s+(\d+)/200',log)]
    assert steps and 0<max(steps)<200
    completed=max(steps);suffix=log[log.rfind('Global Step:'):]
    assert 'Generating Rollout Epochs:' in suffix and '| 8/8 ' not in suffix
    sys.path.insert(0,plan['repo']);import ray
    ray.init(address=plan['ray_address'],namespace=driver['namespace'],log_to_driver=False,logging_level='ERROR')
    handles=[ray.get_actor(r['name'],namespace=driver['namespace']) for r in rows]
    assert all(hasattr(h,'save_checkpoint') for h in handles)
    args.output.mkdir(parents=True,mode=0o700)
    cp=args.output/f'global_step_{completed}';actor_path=cp/'actor';actor_path.mkdir(parents=True)
    record(args.output/'intent.json',{'time':now(),'owner':owner,'driver':driver,'actors':rows,'completed_step':completed,'path':str(cp)})
    fd=M.pidfd_open(driver['pid']);paused=False;save_dispatched=False;save_finished=False
    try:
        assert M.H.same(driver) and M.H.same(owner)
        M.pidfd_send(fd,signal.SIGSTOP);paused=True
        deadline=time.monotonic()+10
        while M.H.proc(driver['pid'])['state'] not in ('T','t'):
            assert time.monotonic()<deadline;time.sleep(.1)
        assert max(int(x) for x in re.findall(r'Global Step:\s+(\d+)/200',(O/'formal/driver.log').read_text()))==completed
        for r in rows:
            assert M.H.same(r['identity'])
            cmd=(Path('/proc')/str(r['identity']['pid'])/'cmdline').read_bytes().decode()
            assert 'EmbodiedFSDPActor.recv_rollout_trajectories' in cmd
        print(json.dumps({'time':now(),'phase':'saving_both_ranks','completed_step':completed}),flush=True)
        save_dispatched=True
        refs=[h.save_checkpoint.remote(str(actor_path),completed) for h in handles]
        ray.get(refs,timeout=600)
        save_finished=True
        rank_rows=[]
        for rank in range(2):
            p=actor_path/'local_shard_checkpoint'/f'checkpoint_rank_{rank}.pt'
            st=p.stat();assert st.st_size>1000000000
            rank_rows.append({'rank':rank,'complete':True,'files':[{'path':str(p),'bytes':st.st_size,'mtime_ns':st.st_mtime_ns}]})
        full=actor_path/'model_state_dict/full_weights.pt';assert full.stat().st_size>1000000000
        receipt={'schema':1,'kind':'opendw-formal-resume-checkpoint','time':now(),'source_owner':str(O),
            'source_owner_plan_sha256':sha(O/'owner-plan.json'),'checkpoint_path':str(cp),'completed_step':completed,
            'complete':True,'actor_world_size':2,'runner_state':{'global_step':completed,'optimizer_state_saved':True,'scheduler_state_saved':True},
            'ranks':rank_rows,'full_weights':{'path':str(full),'bytes':full.stat().st_size,'mtime_ns':full.stat().st_mtime_ns},
            'save_method':'both original actors save_checkpoint while exact driver SIGSTOP during sampling',
            'owner_identity':owner,'driver_identity':driver,'actors':rows}
        assert M.H.same(owner) and M.H.same(driver)
        record(args.output/'resume-receipt.json',receipt)
        print(json.dumps(receipt),flush=True)
    except BaseException as exc:
        record(args.output/'failure.json',{'time':now(),'type':type(exc).__name__,'error':str(exc)})
        raise
    finally:
        may_resume=not save_dispatched or save_finished
        if paused and may_resume and M.H.same(driver):M.pidfd_send(fd,signal.SIGCONT)
        os.close(fd);ray.shutdown()
        record(args.output/('driver-resumed.json' if may_resume else 'needs-attention.json'),
               {'time':now(),'same_original_driver':M.H.same(driver),'driver_resumed':may_resume,
                'save_dispatched':save_dispatched,'save_finished':save_finished})
if __name__=='__main__':main()
