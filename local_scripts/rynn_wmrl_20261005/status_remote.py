"""Read-only current Rynn transition and trial status, bounded log tails."""
import datetime,json,os,socket,subprocess
from pathlib import Path
assert os.getuid()==20001 and socket.gethostname()=='h100-gpu01'
S=Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003')
version=2 if (S/'runs/rynn-success-v2').exists() else 1
D=S/f'rynn-control-v{version}';O=S/f'runs/rynn-success-v{version}'
out={'time':datetime.datetime.now().astimezone().isoformat(),'version':version,'json':{},'logs':{}}
for p in [D/'launch-receipt.json',D/'handoff/status.json',D/'handoff/failure.json',D/'handoff/final.json',D/'handoff/handoff-completed.json',
 O/'owner-identity.json',O/'state.json',O/'error.json',O/'final.json',O/'rm_gate/result.json',O/'rm_gate/frozen-result.json',
 O/'startup_smoke/result.json',O/'startup_smoke/driver-finished.json',O/'rynn-learning-smoke-gate.json',O/'formal/driver-identity.json']:
    if p.is_file():out['json'][str(p)]=json.loads(p.read_text())
for p in [D/'handoff-launch.log',D/'handoff/formal-owner.log',D/'owner-launch.log',O/'rm_gate/client.log',O/'startup_smoke/driver.log',O/'formal/driver.log']:
    if p.is_file():out['logs'][str(p)]=p.read_text(errors='replace')[-5000:]
for key in ['wm6','wm7','rm4','rm5']:
    for name in ['service.log','records/infer.jsonl']:
        p=O/'services'/key/name
        if p.is_file():
            with p.open('rb') as f:
                f.seek(max(0,p.stat().st_size-4000));out['logs'][str(p)]=f.read().decode(errors='replace')
out['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True)
out['rm_steps']={}
for key in ['reference-b1','sanity-b4','throughput-b4','throughput-b8','throughput-b16','second-service','tail-order']:
    p=O/'rm_gate'/(key+'.json')
    if p.is_file():
        value=json.loads(p.read_text());items=value.get('items',[])
        out['rm_steps'][key]={'timings':value.get('timings'),'items':items,'mtime':p.stat().st_mtime}
print(json.dumps(out,ensure_ascii=False))
