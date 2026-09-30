"""Download only pinned official assets; no model execution."""
import json,os,time
from pathlib import Path
from huggingface_hub import snapshot_download

root=Path('/data/chenyiteng/projects/wan-goal-sz3')
items=[
 ('RLinf/RLinf-Wan-LIBERO-Goal','bd395971c3467de3dd19e7e6c7562af48a2894a6','wan-goal'),
 ('Haozhan72/Openvla-oft-SFT-libero-goal-traj1','d20e1d447dfd87c0daa121b0739e2a379f7fe334','oft-goal'),
 ('RLinf/RLinf-Pi05-LIBERO-SFT','45ccfcc4e28634f1576ebf78cab0fbe2fd82432d','pi05-libero'),
]
for repo,rev,name in items:
    d=root/'models'/name
    print(json.dumps({'time':time.time(),'phase':'download','repo':repo,'revision':rev,'destination':str(d)}),flush=True)
    for attempt in range(1,9):
        try:
            snapshot_download(repo_id=repo,revision=rev,local_dir=d,max_workers=8,token=False)
            break
        except Exception as exc:
            print(json.dumps({'phase':'retry','repo':repo,'attempt':attempt,'error_type':type(exc).__name__}),flush=True)
            if attempt==8:raise
            time.sleep(15)
    files=[{'path':str(p.relative_to(d)),'bytes':p.stat().st_size} for p in sorted(d.rglob('*')) if p.is_file() and '.cache' not in p.parts]
    receipt={'repo':repo,'revision':rev,'finished':time.time(),'files':files}
    (root/'logs'/f'{name}-download.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps({'phase':'complete','repo':repo,'files':len(files),'bytes':sum(x['bytes'] for x in files)}),flush=True)
print('ALL_PINNED_ASSETS_DOWNLOADED',flush=True)
