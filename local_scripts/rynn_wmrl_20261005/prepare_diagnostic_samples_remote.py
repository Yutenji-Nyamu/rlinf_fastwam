"""CPU-only finite diagnostic clips; reference labels are not physics labels."""
import datetime,hashlib,json,os,socket
from pathlib import Path
import cv2,h5py,numpy as np
assert os.getuid()==20001 and socket.gethostname()=='h100-gpu01'
S=Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003')
D=S/'rynn-diagnosis-v1';P=D/'prepared';P.mkdir(parents=True,exist_ok=True)
assert not (P/'cases.json').exists() and not (P/'samples.npz').exists()
with np.load(S/'rynn-control-v1/prepared/rm-sanity-clips.npz',allow_pickle=False) as old:
    old_frames=old['frames'].copy();instructions=old['instructions'].copy()
meta=json.loads((S/'rynn-control-v1/prepared/rm-sanity-clips.json').read_text())
cases=[];arrays={}
def add(key,frames,instruction,description,reference=None,meta_mode='current'):
    arrays[key]=np.asarray(frames,dtype=np.uint8)
    cases.append(dict(id=key,frames_key=key,instruction=instruction,description=description,
        expected_reference=reference,meta_mode=meta_mode))
def decode(value):
    raw=np.frombuffer(bytes(value),dtype=np.uint8)
    frame=cv2.imdecode(raw,cv2.IMREAD_COLOR)
    assert frame is not None
    return cv2.resize(frame,(320,256),interpolation=cv2.INTER_AREA)
for ep,idx in [(0,1),(4,9)]:
    source=Path(meta['items'][idx]['source'])
    with h5py.File(source,'r') as file:
        values=file['observation/head_camera/rgb']
        n=len(values)
        def sampled(end,k):return np.stack([decode(values[int(i)]) for i in np.linspace(0,end,k,dtype=np.int64)])
        add(f'initial{ep}_k8',old_frames[idx-1],str(instructions[idx]),'Repeated initial reset; no motion',False)
        add(f'expert{ep}_k8',old_frames[idx],str(instructions[idx]),'Full clean50 expert, same input as failed production gate',True)
        visible=int(meta['items'][idx]['indices'][-2])
        add(f'visible{ep}_prefix_k8',sampled(visible,8),str(instructions[idx]),
            f'Prefix through source frame {visible}; upright bottle visible, no physics label')
        add(f'visible{ep}_repeat_k8',np.repeat(sampled(visible,1)[-1:,:,:,:],8,axis=0),str(instructions[idx]),
            'Repeated visible upright frame; static-state diagnostic, no motion claim')
        # Correct the repeat to the intended source frame, rather than linspace(0,end,1)==0.
        arrays[f'visible{ep}_repeat_k8'][:]=decode(values[visible])
        add(f'final{ep}_repeat_k8',np.repeat(decode(values[n-1])[None],8,axis=0),str(instructions[idx]),
            'Repeated final frame; object partly outside head view')
        if ep==0:
            add('expert0_k64',sampled(n-1,64),str(instructions[idx]),'Full same expert with 64 frames',True)
            add('expert0_default_meta_k8',old_frames[idx],str(instructions[idx]),
                'Official default metadata omitted; same eight pixels',True,'official_default')
            add('expert0_canonical_k8',old_frames[idx],
                'Pick up the bottle and hold it upright in the air.',
                'Simplified task wording, omits arm and appearance qualifiers; diagnostic only',True)
example_root=Path('/data/chenyiteng/projects/RynnValue-10e0d333/example')
examples=sorted(example_root.glob('*.mp4'))
official=None
if examples:
    video=examples[0];cap=cv2.VideoCapture(str(video));frames=[]
    while True:
        ok,frame=cap.read()
        if not ok:break
        frames.append(cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),(320,256),interpolation=cv2.INTER_AREA))
    cap.release();assert frames
    instruction=video.stem.replace('_',' ')
    official=dict(path=str(video),frames=len(frames),instruction=instruction)
    for k in (8,64):
        sampled=np.stack([frames[int(i)] for i in np.linspace(0,len(frames)-1,k,dtype=np.int64)])
        add(f'official_k{k}',sampled,instruction,'Bundled official example; label not independently verified',None,'official_default')
np.savez(P/'samples.npz',**arrays)
result=dict(time=datetime.datetime.now().astimezone().isoformat(),cases=cases,official_example=official,
    pixel_contract='native HDF5 keeps current cv2 channel contract; standard mp4 explicitly decoded to RGB',
    sample_sha256=hashlib.sha256((P/'samples.npz').read_bytes()).hexdigest(),
    caveat='Reference expert labels reflect clean50 provenance, not newly simulated check_success; prefix labels remain unknown')
(P/'cases.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({key:value for key,value in result.items() if key!='cases'}));print(json.dumps(cases))
