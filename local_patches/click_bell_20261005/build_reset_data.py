"""Extract simultaneous initial views/state from unfiltered RoboTwin clean50 episodes."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import h5py
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--task-name',choices=('adjust_bottle','click_bell'),required=True)
    a=p.parse_args()
    assert not a.output.exists()
    files=sorted((a.source/'data').glob('episode*.hdf5'),key=lambda f:int(f.stem[7:]))
    assert len(files)==50,'Use the complete existing clean50 set, without seed filtering'
    views=[];states=[];instructions=[];provenance=[]
    for f in files:
        instruction=json.loads((a.source/'instructions'/(f.stem+'.json')).read_text())['seen'][0]
        with h5py.File(f,'r') as d:
            # Match the established RoboTwin raw->Sidney conversion exactly:
            # cv2 decode returns the saved channel array; do not add a color swap.
            encoded=[d[f'observation/{cam}/rgb'][0].tobytes() for cam in ('head_camera','left_camera','right_camera')]
            assert all(b'XPL-RGB1' not in raw for raw in encoded), 'New tagged RGB JPEG: use official tagged decoder'
            row=[cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR) for raw in encoded]
            assert all(v is not None and v.ndim==3 and v.shape[-1]==3 for v in row)
            state=d['joint_action/vector'][0].astype(np.float32)
            assert state.shape==(14,) and np.isfinite(state).all()
            provenance.append({'episode':f.stem,'path':str(f),'frame':0,'source_shapes':[list(v.shape) for v in row],
                               'raw_initial_image_sha256':[hashlib.sha256(v.tobytes()).hexdigest() for v in row]})
            views.append([cv2.resize(v,(256,256)) for v in row]);states.append(state);instructions.append(instruction)
    data=np.asarray(views,dtype=np.uint8)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output,main_images=data[:,0],wrist_images=data[:,1:],states=np.asarray(states),
                        instructions=np.asarray(instructions),reset_ids=np.asarray([f.stem for f in files]))
    receipt={'task':a.task_name,'count':len(files),'frame':0,'selection':'all 50 episodes; no success or policy filtering',
             'state':'joint_action/vector at same frame','cameras':['head_camera','left_camera','right_camera'],
             'instruction':'seen[0] of matching episode','decode':'cv2.imdecode, no channel permutation',
             'output':str(a.output),'sha256':hashlib.sha256(a.output.read_bytes()).hexdigest(),'episodes':provenance}
    a.output.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='episodes'}))


if __name__=='__main__':main()
