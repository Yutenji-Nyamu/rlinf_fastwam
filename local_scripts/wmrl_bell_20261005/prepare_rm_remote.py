"""Fetch the exact public bell RM and check the existing loader on CPU."""
import datetime,hashlib,importlib.util,json,os,socket,sys,time
from pathlib import Path
import requests
assert os.getuid()==20001 and socket.gethostname()=='h100-gpu01'
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
S=Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003');D=S/'click-bell-v1';D.mkdir(exist_ok=True)
dest=S/'models/worldarena/reward_model/click_bell/resnet_rm.pth';dest.parent.mkdir(exist_ok=True)
expected='5c4141b599b35f6f46064e11410e376668659e669fdda6e938f497bf7d34d464'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
    return h.hexdigest()
env=json.loads((S/'formal-b16-control-v1/prepared/environment.json').read_text())
for k,v in env.items():
    if k.lower() in ('http_proxy','https_proxy','no_proxy'):os.environ[k]=v
started=time.monotonic()
if not dest.exists():
    url='https://huggingface.co/WorldArena/WorldArena2.0/resolve/main/reward_model/click_bell/resnet_rm.pth'
    temp=dest.with_suffix('.download')
    assert not temp.exists()
    with requests.get(url,stream=True,timeout=(20,120)) as r:
        r.raise_for_status()
        with temp.open('xb') as f:
            for chunk in r.iter_content(1024*1024):f.write(chunk)
    assert sha(temp)==expected,'Downloaded checkpoint hash mismatch'
    temp.rename(dest)
assert sha(dest)==expected
print(json.dumps({'download':'verified','bytes':dest.stat().st_size,'seconds':time.monotonic()-started}),flush=True)
p=S/'multigpu-payload-v1/multigpu/tools/opendw_reward.py'
spec=importlib.util.spec_from_file_location('bell_rm',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
model=m.RoboTwinT5Reward(dest,S/'models/worldarena/reward_model/t5-base')
out={'time':datetime.datetime.now().astimezone().isoformat(),'checkpoint':str(dest),'sha256':expected,'bytes':dest.stat().st_size,'strict_load':True,'device':str(next(model.parameters()).device),'parameters':sum(p.numel() for p in model.parameters()),'loader':str(p),'loader_sha256':sha(p),'seconds':time.monotonic()-started}
(D/'rm-asset.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
