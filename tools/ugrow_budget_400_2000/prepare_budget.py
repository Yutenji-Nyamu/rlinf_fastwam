import copy,hashlib,importlib.util,json,os,time
from pathlib import Path
import yaml
S=Path('/data/chenyiteng/deployment-20261006/ugrow-budget-400-2000-v1')
def load(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
rt=load(S/'tools/runtime_v3.py','rt');base=load(S/'tools/rlt_lease.py','lease');ext=load(S/'tools/extend_budget.py','extension')
out={}
for lane,prior,start,target in (('bc','ugrow-bc-rlt-g45-v2',100,400),('rlt','ugrow-rlt-g5-formal-v3',800,2000)):
    oldstage=S.with_name(prior)/lane;old=rt.read(oldstage/'plan.json');oldrt=load(old['ops'],'old_'+lane);oldop,b=oldrt.configure(oldstage);oldop.checked()
    owner=rt.read(oldstage/'owner-identity.json');oldrun=Path(old['runs']['formal']['run']);driver=rt.read(oldrun/'runtime/driver-identity.json')
    assert b.same(owner) and b.same(driver) and not (oldstage/'terminal.json').exists()
    assert not (Path(old['lease_dir'])/'return-attempt.json').exists()
    stage=S/lane
    if stage.exists():assert not any(stage.iterdir()),'Prepared stage already contains records'
    else:stage.mkdir()
    run=oldrun.with_name(f'ugrow-{lane}-g{old["gpu"]}-{target}-1006-v1');runtime=run/'runtime'
    if runtime.exists():assert not any(runtime.iterdir()),'Runtime already contains records'
    else:runtime.mkdir(parents=True)
    before=yaml.safe_load((oldrun/'runtime/resolved.yaml').read_text());assert min(before['runner']['max_epochs'],before['runner']['max_steps'] if before['runner']['max_steps']>0 else 999999)==start
    cfg=ext.changed_config(before,oldrun,run,start,target)
    diff={k:[v,base._flatten(cfg)[k]] for k,v in base._flatten(before).items() if v!=base._flatten(cfg)[k]}
    allowed=base.OUTPUT_KEYS|{'runner.max_steps','runner.max_epochs','runner.resume_dir'}
    if lane=='bc':allowed=allowed|{'algorithm.online_bc.data_path'}
    assert set(diff)<=allowed,diff
    (runtime/'resolved.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    env=rt.read(oldrun/'runtime/environment.json')
    env={k:v.replace(str(oldrun),str(run)).replace(oldrun.name,run.name) if isinstance(v,str) else v for k,v in env.items()}
    rt.save(runtime/'environment.json',env,True);(runtime/'environment.json').chmod(0o600)
    p=copy.deepcopy(old);p['ops']=str(S/'tools/runtime_v3.py');p['runs']['formal']={**old['runs']['formal'],'run':str(run),'namespace':run.name}
    e={'time':time.time(),'lane':lane,'original_stage':str(oldstage),'original_run':str(oldrun),'original_owner':owner,'original_driver':driver,
       'from_step':start,'total_steps':target,'retention_authorized':'latest2+resume+final; only this BC; preserve logs' if lane=='bc' else None,
       'user_instruction':'RLT total2000, BC total400; preserve existing progress. BC latest2 + resume + final checkpoints explicitly approved.'}
    rt.save(stage/'extension.json',e,True);rt.save(stage/'config-diff.json',diff,True)
    p['pins'].update({str(f):base._sha(f) for f in [*sorted((S/'tools').glob('*.py')),stage/'extension.json',runtime/'resolved.yaml',runtime/'environment.json',oldstage/'plan.json']})
    rt.save(stage/'plan.json',p,True);rt.configure(stage)[0].checked()
    out[lane]={'stage':str(stage),'from':start,'total':target,'run':str(run),'resume':cfg['runner']['resume_dir'],'diff':diff,'original_driver_alive':b.same(driver)}
print(json.dumps(out))
