set -eu
PY=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python
RUN=/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/smokes/parity-core224-m10-phys4-v7
NATIVE=/data/chenyiteng/models/lerobot/sidney-pi05-robotwin-e49e2ab
CONVERTED=/data/chenyiteng/models/rlinf-native/sidney-pi05-robotwin-e49e2ab
echo '=== norm-related files ==='
find "$NATIVE" -maxdepth 3 -type f \( -iname '*norm*' -o -iname '*processor*' -o -iname '*config*' \) -printf '%p\t%s\n' | sort
find "$CONVERTED" -maxdepth 4 -type f \( -iname '*norm*' -o -iname '*manifest*' \) -printf '%p\t%s\n' | sort
"$PY" - "$RUN/input.npz" "$RUN/native.pt" "$RUN/rlinf.pt" "$NATIVE" "$CONVERTED" <<'PY'
import json, pathlib, sys
import numpy as np, torch
inp=np.load(sys.argv[1])
n=torch.load(sys.argv[2],map_location='cpu',weights_only=True)
r=torch.load(sys.argv[3],map_location='cpu',weights_only=True)
print('input keys', inp.files)
for k in inp.files:
 x=inp[k]
 print('input', k, x.shape, x.dtype, x.reshape(-1)[:20].tolist() if x.size < 100 else '')
print('native_state', n['normalized_state14'].tolist())
print('rlinf_state', r['normalized_state14'].tolist())
print('delta', (n['normalized_state14'].float()-r['normalized_state14'].float()).tolist())
for base in map(pathlib.Path, sys.argv[4:6]):
 print('BASE',base)
 for p in base.rglob('*'):
  if p.is_file() and ('norm' in p.name.lower() or 'processor' in p.name.lower() or 'manifest' in p.name.lower()):
   print(p, p.stat().st_size)
   if p.suffix=='.json' and p.stat().st_size < 200000:
    try:
     data=json.loads(p.read_text())
     print(json.dumps(data,ensure_ascii=False)[:12000])
    except Exception as e: print('jsonerr',e)
PY
