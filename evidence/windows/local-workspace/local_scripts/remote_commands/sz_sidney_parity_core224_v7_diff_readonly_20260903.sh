set -eu
PY=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python
RUN=/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/smokes/parity-core224-m10-phys4-v7
"$PY" - "$RUN/native.pt" "$RUN/rlinf.pt" <<'PY'
import sys, torch
n=torch.load(sys.argv[1],map_location='cpu',weights_only=True)
r=torch.load(sys.argv[2],map_location='cpu',weights_only=True)
for k in ['images','image_masks','normalized_state14','padded_state32','tokens','token_mask','model_actions32','final_actions14']:
 a=n[k]; b=r[k]
 d=(a.float()-b.float()).abs()
 print(k, 'shape', tuple(a.shape), 'max', float(d.max()) if d.numel() else 0,
       'mean', float(d.mean()) if d.numel() else 0, 'exact', torch.equal(a,b),
       'sample_close', torch.allclose(a.float(),b.float(),rtol=1e-2,atol=5e-3))
PY
