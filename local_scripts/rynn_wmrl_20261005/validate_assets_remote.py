"""CPU-only small-file/shard metadata verification; no model loading."""
import json,os,socket,sys
from pathlib import Path
assert os.getuid()==20001 and socket.gethostname()=='h100-gpu01'
D=Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003/rynn-control-v1')
sys.path.insert(0,str(D/'code'))
from rynn_success_service import validate_assets
model=Path('/data/chenyiteng/models/RynnValue-8B-8738c5e4')
receipt=validate_assets(model,model/'manifest.json')
(D/'prepared/assets-verified.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'model_id':receipt['model_id'],'revision':receipt['revision'],'metadata_sha256':receipt['metadata_sha256'],'shards':len(receipt['shards'])}))
