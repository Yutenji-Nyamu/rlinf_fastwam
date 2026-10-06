"""Honor the user's explicit minimal-smoke gate and stop only these two drivers."""
import importlib.util,json,os,signal,time
from pathlib import Path
S=Path('/data/chenyiteng/projects/norm-bc-dsrl-sz3-20261007');C=S/'control';cycle=S/'rlt-after-norm-g67-v1'
def read(p):return json.loads(Path(p).read_text())
spec=importlib.util.spec_from_file_location('norm_promote',cycle/'rlt_returned_cycle.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.install_helper(cycle);H=m.H
s=read(C/'status.json');assert all(r['state']=='SMOKE' for r in s['roles'].values()),s['roles']
assert read(C/'receipts/bc-probe-v2-accepted.json')['status']=='PASS'
assert read(C/'receipts/dsrl-probe-v1-accepted.json')['status']=='PASS'
proof={'time':time.time(),'user_instruction':'Why so many smoke rounds? It only needs to run. Work concisely.','gate':'real-model action/RNG passivity probes, full-size startup, environment sampling, Norm production/replay path and CPU mapping tests; no requirement to finish warmup or a smoke checkpoint','formal':'fresh original SFT and empty replay; BC300, DSRL200; original formal parameters unchanged','dsrl_gpu_weighted_update':'not yet exercised; it will follow the original formal warmup500','stopped_drivers':{}}
targets=[]
for role,row in s['roles'].items():
    assert row['request']==role+'-smoke-v1'
    rt=Path(row['runtime']);identity=read(rt/'driver-identity.json');identity['match_cmdline']=True
    assert identity['namespace']==row['namespace'] and identity['pid']==row['driver']['pid'] and H.same(identity)
    log=(rt/'driver.log').read_text(errors='replace')
    assert 'Generating Rollout Epochs:' in log and 'Loaded norm stats' in log
    assert 'Traceback (most recent call last)' not in log
    proof['stopped_drivers'][role]=identity;targets.append(identity)
with (C/'minimal-smoke-approval.json').open('x') as f:json.dump(proof,f,indent=2)
for identity in targets:
    assert H.same(identity);os.kill(identity['pid'],signal.SIGTERM)
print(json.dumps({'intentional_smoke_stop':proof['stopped_drivers'],'next':'wait for exact namespace and GPU release, then fresh formal immediately'}),flush=True)
