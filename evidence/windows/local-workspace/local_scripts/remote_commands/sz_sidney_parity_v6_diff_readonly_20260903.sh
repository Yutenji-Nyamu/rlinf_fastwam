set -eu
PY=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python
RUN=/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/smokes/parity-m10-phys4-v6-chw
"$PY" - "$RUN/native.pt" "$RUN/rlinf.pt" <<'PY'
import sys, torch
n=torch.load(sys.argv[1],map_location='cpu',weights_only=True)
r=torch.load(sys.argv[2],map_location='cpu',weights_only=True)
for k in ['images','image_masks','normalized_state14','padded_state32','tokens','token_mask','model_actions32','final_actions14']:
 a=n[k]; b=r[k]
 print('\n',k,'shape',tuple(a.shape),tuple(b.shape),'dtype',a.dtype,b.dtype)
 if a.shape==b.shape:
  d=(a.float()-b.float()).abs()
  print('max',float(d.max()) if d.numel() else 0,'mean',float(d.mean()) if d.numel() else 0,'close_exact',torch.equal(a,b),'close_sample',torch.allclose(a.float(),b.float(),rtol=1e-2,atol=5e-3))
print('\nimage_pair_mae')
for i in range(n['images'].shape[0]):
 print(i,[round(float((n['images'][i].float()-r['images'][j].float()).abs().mean()),6) for j in range(r['images'].shape[0])])
print('native image range/means',[(float(x.min()),float(x.max()),float(x.mean())) for x in n['images']])
print('rlinf image range/means',[(float(x.min()),float(x.max()),float(x.mean())) for x in r['images']])
PY
