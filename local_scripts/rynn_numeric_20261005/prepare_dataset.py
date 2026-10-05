"""CPU-only fixed expert-prefix controls; preserves the existing RGB contract."""
import hashlib,json,os,socket
from pathlib import Path
import cv2,h5py,numpy as np

S=Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003')
D=S/'rynn-numeric-v1/prepared'

def main():
    assert os.getuid()==20001 and socket.gethostname()=='h100-gpu01'
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert D.is_dir() and not (D/'cases.json').exists() and not (D/'samples.npz').exists()
    manifest=json.loads((S/'data/adjust-bottle-clean50-reset.json').read_text())
    arrays={};cases=[]
    for index,entry in enumerate(manifest['episodes'][:16]):
        source=Path(entry['path']);source_sha=hashlib.sha256(source.read_bytes()).hexdigest()
        instruction_file=source.parent.parent/'instructions'/source.with_suffix('.json').name
        instruction=json.loads(instruction_file.read_text())['seen'][0]
        with h5py.File(source,'r') as h:
            images=h['observation/head_camera/rgb'];count=len(images)
            configs=[]
            for fraction in [0,.25,.5,.75,1]:
                end=round((count-1)*fraction)
                ids=np.linspace(0,end,8).astype(int).tolist()
                configs.append(('expert_prefix',fraction,ids))
            configs += [('repeat_final',None,[count-1]*8),
                ('reversed',None,np.linspace(count-1,0,8).astype(int).tolist()),
                ('return_to_initial',None,np.linspace(0,count-1,7).astype(int).tolist()+[0])]
            needed={i for _,_,ids in configs for i in ids};cache={}
            for i in needed:
                frame=cv2.imdecode(np.frombuffer(images[i].tobytes(),np.uint8),cv2.IMREAD_COLOR)
                assert frame is not None and frame.shape==(240,320,3)
                cache[i]=cv2.resize(frame,(320,256))
            for kind,fraction,ids in configs:
                uid=f'expert-{index:02d}';key=f'{uid}-{kind}'+('-%03d'%round(fraction*100) if fraction is not None else '')
                pixels=np.stack([cache[i] for i in ids]);arrays[key]=pixels
                reference=True if (kind=='repeat_final' or kind=='expert_prefix' and fraction==1) else None
                cases.append(dict(id=key,frames_key=key,episode_uid=uid,kind=kind,
                    prefix_fraction=fraction,frame_indices=ids,instruction=instruction,
                    source=str(source),source_sha256=source_sha,reference_success=reference,
                    label_origin='clean50 successful demonstration endpoint' if reference else 'unlabeled prefix or synthetic temporal control; not a physical failure',
                    array_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),source_frame_count=count))
    assert len(cases)==128
    np.savez_compressed(D/'samples.npz',**arrays)
    info=dict(schema_version=1,cases=cases,
        input_contract='K8 320x256; JPEG cv2 decode, no channel permutation, same as current reset and prior probes',
        native_eval_excluded='Existing CP70 videos are tiled with reward/termination overlays; all displayed rewards are zero under wrapper path and no per-env success mapping was recovered. No native success/failure AUC is claimed.',
        synthetic_caveat='Reversed and return_to_initial are temporal consistency controls, not real physical failures.')
    (D/'cases.json').write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps(dict(cases=len(cases),episodes=16,samples_bytes=(D/'samples.npz').stat().st_size)))

if __name__=='__main__':main()
