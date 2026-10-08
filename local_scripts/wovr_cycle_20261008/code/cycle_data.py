"""One conversion per offline stage; training reads prepacked arrays directly."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image


def episodes(roots):
    return [(p,json.loads(p.read_text())) for root in roots for p in sorted(Path(root).glob('*/episode.json'))]


def prepare(roots,output,heldout,policy_eval_seeds,allow_partial=False):
    from dexbotic.policy.dw05_policy import compose_robotwin_image
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    records=episodes(roots)
    excluded=set(policy_eval_seeds); hold=set(heldout)
    rows=[]; rms={key:[] for key in ['train','val','test']}
    stats=dict(episodes=0,complete=0,successes=0,train_chunks=0,heldout_chunks=0,excluded=0)
    for ep_path,ep in records:
        seed=ep['actual_seed']
        if seed in excluded:
            stats['excluded']+=1; continue
        if not ep['complete'] and not allow_partial:
            continue
        split='val' if seed in hold else 'train'
        # Holdout is by actual simulator seed, including reset fallback.
        stats['episodes']+=1; stats['complete']+=int(ep['complete']); stats['successes']+=int(ep['success'])
        rm_split=('val' if sorted(hold).index(seed)<len(hold)//2 else 'test') if seed in hold else 'train'
        for chunk in ep['chunks']:
            src=ep_path.parent/chunk['file']
            with np.load(src,allow_pickle=False) as z:
                images=np.stack([compose_robotwin_image([z['head'][i],z['left'][i],z['right'][i]],layout='robotwin_resize',image_size_hw=(384,320)) for i in range(9)])
                index=len(rows); file=output/f'wm_{index:06d}.npz'
                np.savez(file,video=images,action=z['action'],state=z['state'][0:1],image_is_pad=z['image_is_pad'])
                # Failed/incomplete episodes teach dynamics, not action imitation.
                rows.append(dict(file=file.name,split=split,prompt=ep['instruction'],
                    success=bool(ep['complete'] and ep['success']),actual_seed=seed,
                    episode=str(ep_path.parent),chunk=chunk['file']))
                stats['train_chunks' if split=='train' else 'heldout_chunks']+=1
                if ep['complete']:
                    for i in range(9):
                        if z['image_is_pad'][i] or (i==0 and chunk['requested_action_start']>0): continue
                        im=np.asarray(Image.fromarray(z['head'][i]).resize((224,224),Image.Resampling.BILINEAR),dtype=np.uint8)
                        rms[rm_split].append((im,int(z['success'][i]),str(ep_path.parent),chunk['requested_action_start']+i*4))
    (output/'manifest.json').write_text(json.dumps(dict(schema=1,rows=rows,stats=stats,
        timebase='C32 request to normalized native-controller progress; not individual action timestamps'),indent=2)+'\n')
    # Reuse the established RM trainer's NPZ schema.
    rm=output/'rm';rm.mkdir(exist_ok=True)
    manifest=dict(schema_version=1,task_name='lift_pot',task_config='demo_clean',splits={})
    import hashlib
    for split,values in rms.items():
        if not values: continue
        file=rm/(split+'.npz')
        np.savez(file,images=np.stack([x[0] for x in values]),labels=np.asarray([x[1] for x in values],dtype=np.float32),
            episode_uids=np.asarray([x[2] for x in values]),action_steps=np.asarray([x[3] for x in values]))
        manifest['splits'][split]=dict(path=file.name,sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
            count=len(values),positives=sum(x[1] for x in values),episodes=len(set(x[2] for x in values)),samples=[dict(group_id=x) for x in sorted(set(v[2] for v in values))])
    (rm/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(stats),flush=True)
    return stats


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--roots',nargs='+',required=True);p.add_argument('--output',required=True)
    p.add_argument('--seed-plan',required=True);p.add_argument('--allow-partial',action='store_true');a=p.parse_args()
    plan=json.loads(Path(a.seed_plan).read_text());prepare(a.roots,a.output,plan['heldout'],plan['policy_eval'],a.allow_partial)
