"""CPU-only initial diagnostic clips; labels identify simple controls, not policy accuracy."""
import datetime,hashlib,json,os,socket
from pathlib import Path
import cv2,h5py,numpy as np
assert os.getuid()==20001 and socket.gethostname()=='h100-gpu01'
S=Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003');D=S/'rynn-control-v1'
target=D/'prepared/rm-sanity-clips.npz';assert not target.exists()
reset=json.loads((S/'data/adjust-bottle-clean50-reset.json').read_text())
frames=[];instructions=[];uids=[];ends=[];times=[];labels=[];provenance=[]
for ep in reset['episodes'][:16]:
    p=Path(ep['path']);instruction=json.loads((p.parent.parent/'instructions'/(p.stem+'.json')).read_text())['seen'][0]
    with h5py.File(p,'r') as f:
        images=f['observation/head_camera/rgb'];N=len(images);assert N>=8
        if not frames: print(json.dumps({'hdf5_keys':list(f.keys()),'attributes':{k:str(v) for k,v in f.attrs.items()},'frames':N}),flush=True)
        indices=np.linspace(0,N-1,8,dtype=np.int64)
        def decode(i):
            raw=images[int(i)].tobytes();assert b'XPL-RGB1' not in raw
            image=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
            assert image is not None
            return cv2.resize(image,(320,256))
        for mode,label,idx in [('initial',0,np.zeros(8,dtype=np.int64)),('expert_episode',1,indices)]:
            frames.append(np.stack([decode(i) for i in idx]));instructions.append(instruction)
            uid=f'sanity/{p.stem}/{mode}';uids.append(uid);ends.append(int(idx[-1]));times.append(idx);labels.append(label)
            provenance.append({'episode_uid':uid,'source':str(p),'indices':idx.tolist(),'expected_success':bool(label),
                'label_basis':'initial unmoved reset control' if label==0 else 'expert clean50 successful demonstration; not a newly read physics check_success label',
                'decode':'same untagged cv2 decode/no channel permutation as current reset pipeline'})
np.savez_compressed(target,frames=np.asarray(frames,np.uint8),instructions=np.asarray(instructions),episode_uids=np.asarray(uids),
    end_action_indices=np.asarray(ends,np.int64),frame_action_indices=np.asarray(times,np.int64),expected_success=np.asarray(labels,np.int8))
receipt={'time':datetime.datetime.now().astimezone().isoformat(),'path':str(target),'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
 'samples':len(frames),'K':8,'head_shape':[256,320,3],'purpose':'sanity and batching; no estimate of native policy or WM success accuracy',
 'selection':'first 16 clean50 episodes by existing manifest order; initial and complete demo paired', 'items':provenance}
target.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k!='items'}),flush=True)
