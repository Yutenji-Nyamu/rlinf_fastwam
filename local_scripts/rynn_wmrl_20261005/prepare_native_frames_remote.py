"""CPU data conversion in the existing RLinf environment; Rynn env stays unchanged."""
import hashlib,json,os,socket
from pathlib import Path
import cv2,h5py,numpy as np
assert os.getuid()==20001 and socket.gethostname()=='h100-gpu01'
S=Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003');P=S/'rynn-diagnosis-v2/prepared'
assert not (P/'plan.json').exists() and not (P/'native_frames.npz').exists()
controls=json.loads((S/'rynn-control-v1/prepared/rm-sanity-clips.json').read_text())
arrays={};meta={}
for ep,i in [(0,1),(4,9)]:
    item=controls['items'][i];path=Path(item['source'])
    with h5py.File(path,'r') as f:
        frames=f['observation/head_camera/rgb'];indices=np.linspace(0,len(frames)-1,8,dtype=int)
        decoded=[cv2.imdecode(np.frombuffer(bytes(frames[j]),dtype=np.uint8),cv2.IMREAD_COLOR) for j in indices]
        assert all(x is not None for x in decoded)
        key=f'expert{ep}';arrays[key]=np.stack(decoded)
        meta[key]={'source':str(path),'episode_uid':item['episode_uid'],'source_total_frames':len(frames),
            'frame_indices':indices.tolist(),'source_frame_size_wh':[decoded[0].shape[1],decoded[0].shape[0]],
            'pixel_contract':'Original cv2 decode without channel permutation; no resize',
            'array_sha256':hashlib.sha256(arrays[key].tobytes()).hexdigest()}
np.savez(P/'native_frames.npz',**arrays)
(P/'native_frames.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps({'native_frames':meta,'npz_sha256':hashlib.sha256((P/'native_frames.npz').read_bytes()).hexdigest()}))
