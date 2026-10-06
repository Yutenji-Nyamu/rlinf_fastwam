import importlib.util,json,os,subprocess,time
from pathlib import Path
S=Path('/data/chenyiteng/deployment-20261006/ugrow-budget-400-2000-v1')
spec=importlib.util.spec_from_file_location('extension',S/'tools/extend_budget.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
out={}
for lane in ('bc','rlt'):
    stage=S/lane;p=m.read(stage/'plan.json');rt=m.module(p['ops'],'rt_'+lane);op,b=rt.configure(stage);op.checked()
    assert not (stage/'extension-launch.json').exists()
    e=m.read(stage/'extension.json');assert b.same(e['original_owner']) and b.same(e['original_driver'])
    with (stage/'extension.log').open('x') as f:
        child=subprocess.Popen([p['python'],'-u','-B',str(S/'tools/extend_budget.py'),'--stage',str(stage)],cwd=p['repo'],env=m.cpu_env(),
            stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    result={'time':time.time(),'identity':b.proc(child.pid),'stage':str(stage)}
    m.save(stage/'extension-launch.json',result,True);out[lane]=result
print(json.dumps(out))
