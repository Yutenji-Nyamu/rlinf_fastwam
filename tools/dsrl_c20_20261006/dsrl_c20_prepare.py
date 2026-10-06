"""Prepare a fresh C20 pair from the exact running C10 configs; no GPU/stop/launch."""
import copy, json, os, subprocess, sys, time
from pathlib import Path
import yaml

os.umask(0o077)
OLD = Path('/data/chenyiteng/deployment-20261006/dsrl-u-lease-v1')
NEW = OLD.with_name('dsrl-u-c20-lease-v2')
REPO = Path('/data/chenyiteng/projects/rlinf-shenzhen/worktrees/dsrl-pi05-u-sz1-20261006')
CODE = REPO/'tools/dsrl_u_20261006/ops'
sys.path.insert(0, str(CODE))
from lease_common import checked, read, save, sha
from prepare_lease import replace_outputs

p, b, op = checked(OLD)
state = read(OLD/'status.json')
assert b.same(state['owner']) and time.time()-state['time'] < 60
assert state['owner']['pid'] == 2475292 and state['owner']['start'] == 414936945
assert all(r['state'] == 'FORMAL' for r in state['slots'].values())
assert subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip() == '8bcd99a6df38a1700a988d8982bde9368bf18025'
assert subprocess.check_output(['git','-C',str(REPO),'status','--porcelain'],text=True) == ''
assert not NEW.exists()
for gpu in ('6','7'):
    assert not (OLD/'receipts'/('gpu'+gpu+'-restore-attempt.json')).exists()
    assert not (Path(p['slots'][gpu]['fallback']['run'])/'runtime/driver-identity.json').exists()

protected = {}
for pid,start in ((3603824,418090820),(2665696,415431106),(3574244,418006423)):
    ident = b.proc(pid)
    assert ident and ident['uid'] == 1003 and ident['start'] == start and b.same(ident)
    protected[str(pid)] = ident
snapshot = b.gpu_snapshot()
for gpu in (4,5):
    for item in snapshot[gpu]['processes']:
        ident = b.proc(item['pid'])
        assert ident and ident['uid'] == 1003 and b.same(ident)
        protected[str(ident['pid'])] = ident

targets = {}
for gpu,role in (('6','clean'),('7','u')):
    row = state['slots'][gpu]
    assert row['request_id'] == role+'-formal-200-mb256-v1' and b.same(row['driver'])
    req = read(OLD/'requests'/(row['request_id']+'.json'))
    for path,digest in req['pins'].items():
        assert sha(path) == digest, path
    actors = op.active_actors(p,row['namespace'])
    op.validate_scoped_actors(p,actors)
    assert actors
    roots = {row['driver']['pid']} | {a['pid'] for a in actors if a.get('pid')}
    tree = op.process_tree(roots,p['uid'])
    assert not any(g['gpu'] != int(gpu) and any(v['pid'] in tree for v in g['processes']) for g in snapshot)
    targets[gpu] = {'role':role,'driver':row['driver'],'namespace':row['namespace'],
                    'runtime':row['runtime'],'actors':actors,'process_tree':tree}

NEW.mkdir()
for name in ('requests','receipts'):
    (NEW/name).mkdir()
capture = {'time':time.time(),'owner':state['owner'],'old_control':str(OLD),
           'targets':targets,'protected':protected,'gpu':snapshot,'old_plan_sha256':sha(OLD/'plan.json')}
save(NEW/'c10-capture.json',capture,True)

plan = copy.deepcopy(p)
plan['prepared_at'] = time.time()
plan['predecessor'] = {'control':str(OLD),'capture':str(NEW/'c10-capture.json'),
                       'reason':'User requested both DSRL C20 fresh from SFT and empty replay; formal 200 unchanged'}
plan['pins'][str(NEW/'c10-capture.json')] = sha(NEW/'c10-capture.json')
save(NEW/'plan.json',plan,True)
save(NEW/'prepared.json',{'time':time.time(),'plan_sha256':sha(NEW/'plan.json'),
                         'predecessor_plan_sha256':sha(OLD/'plan.json'),'reuse':'existing proven scopes and unused RLT fallback'},True)

def differences(x,y,prefix=''):
    if isinstance(x,dict) and isinstance(y,dict):
        assert set(x)==set(y)
        return [v for k in x for v in differences(x[k],y[k],prefix+'.'+str(k) if prefix else str(k))]
    if isinstance(x,list) and isinstance(y,list):
        assert len(x)==len(y)
        return [v for i,(a,c) in enumerate(zip(x,y)) for v in differences(a,c,prefix+'.'+str(i))]
    return [] if x==y else [{'key':prefix,'old':x,'new':y}]

prepared = {}
for gpu,role in (('6','clean'),('7','u')):
    oldrt = Path(targets[gpu]['runtime']); oldrun = oldrt.parent
    run = oldrun.with_name(role+'-c20-formal-200-mb256-v2'); rt=run/'runtime'
    assert not run.exists()
    before = yaml.safe_load((oldrt/'resolved.yaml').read_text())
    cfg = replace_outputs(before,oldrun,run)
    assert cfg['runner']['max_steps']==200 and cfg['runner']['resume_dir'] is None
    assert cfg['actor']['model']['num_action_chunks']==cfg['actor']['model']['openpi']['action_chunk']==10
    cfg['actor']['model']['num_action_chunks']=20
    cfg['actor']['model']['openpi']['action_chunk']=20
    allowed={'actor.model.num_action_chunks','actor.model.openpi.action_chunk'}
    if role=='u':
        for spec,key in ((cfg['actor']['model']['openpi']['dsrl_u_spec'],'actor.model.openpi.dsrl_u_spec.chunk_length'),
                         (cfg['algorithm']['dsrl_u']['spec'],'algorithm.dsrl_u.spec.chunk_length')):
            assert spec['chunk_length']==10
            spec['chunk_length']=20
            allowed.add(key)
    diff=differences(before,cfg)
    for row in diff:
        assert row['key'] in allowed or (isinstance(row['old'],str) and row['new']==replace_outputs(row['old'],oldrun,run)),row
    assert {v['key'] for v in diff if v['key'] in allowed}==allowed
    rt.mkdir(parents=True)
    (rt/'resolved.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    req=copy.deepcopy(read(OLD/'requests'/(role+'-formal-200-mb256-v1.json')))
    req.update(request_id=run.name,runtime=str(rt),namespace='dsrl-u-20261006-'+run.name,created_at=time.time())
    req['pins'].pop(str(oldrt/'resolved.yaml'))
    req['pins'][str(rt/'resolved.yaml')]=sha(rt/'resolved.yaml')
    reqpath=NEW/(run.name+'-request.json')
    save(reqpath,req,True)
    prepared[role]={'gpu':int(gpu),'request':str(reqpath),'run':str(run),'namespace':req['namespace'],
                    'config_sha256':sha(rt/'resolved.yaml'),'diff':diff}
save(NEW/'c20-configs.json',{'time':time.time(),'runs':prepared},True)
print(json.dumps({'control':str(NEW),'prepared':prepared,'no_stop_or_launch':True}))
