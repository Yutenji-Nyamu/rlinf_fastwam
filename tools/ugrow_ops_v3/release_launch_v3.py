import importlib.util,json,os,subprocess,time
from pathlib import Path
S=Path('/data/chenyiteng/deployment-20261006/ugrow-rlt-g5-formal-v3');ls=S/'rlt'
p=json.loads((ls/'plan.json').read_text());assert p['gpu']==5 and p['lane']=='rlt'
sp=importlib.util.spec_from_file_location('runtime',p['ops']);rt=importlib.util.module_from_spec(sp);sp.loader.exec_module(rt)
op,b=rt.configure(ls);op.checked()
sp=importlib.util.spec_from_file_location('lease',S/'tools/returned_rlt_lease.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
proof=m.verify_release_returned(Path(p['lease_dir']))
out={'ready':proof['ready'],'release_evidence':proof['release_evidence'],'gpu':proof['gpu']}
if proof['ready']:
    launch=ls/'owner-launch.json'
    if launch.exists():out['launch']=json.loads(launch.read_text())
    else:
        assert not (ls/'launch-authorized.json').exists(),'Ambiguous launch attempt; inspect'
        rt.save(ls/'launch-authorized.json',{'time':time.time(),'operation_id':p['operation_id'],'gpu':5,'lease_dir':p['lease_dir']},True)
        env={k:v for k,v in os.environ.items() if k not in ('CUDA_VISIBLE_DEVICES','LD_PRELOAD','PYTHONPATH','RLINF_OPENDW_GPU_SCOPE_MANIFEST')}
        env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1')
        with (ls/'owner.log').open('x') as f:
            child=subprocess.Popen([p['python'],'-u','-B',p['ops'],'--stage',str(ls),'owner'],cwd=p['repo'],env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        record={'time':time.time(),'identity':b.proc(child.pid),'plan':str(ls/'plan.json')}
        rt.save(launch,record,True);out['launch']=record
print(json.dumps(out))
