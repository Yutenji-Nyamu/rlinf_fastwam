"""Finish only missing Clean50 tasks after the first preparation worker exits."""
import concurrent.futures
import importlib.util
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

D = Path('/data/chenyiteng/deployment-20261023/endpoint-multitask-sz1-g45/data')
S = Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1')
sys.path.insert(0, str(S/'ops'))
from common import proc, read, same, save

save(D/'data-finish-identity.json', proc(os.getpid()))
deadline=time.time()+12*3600
spec = importlib.util.spec_from_file_location('accepted_data_worker', D/'data_worker.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
tasks = read(D/'task-list.json')['tasks']
can = Path('/data/chenyiteng/datasets/robotwin2/canonical')

def scan():
    rows = []
    for t in tasks:
        candidates = [can/'pi0-aloha-clean50-v1'] if t['task']=='adjust_bottle' else sorted(can.glob(t['task'].replace('_','-')+'-aloha-clean50-*'), reverse=True)
        for p in candidates:
            if not (p/'meta/info.json').is_file() or read(p/'meta/info.json')['total_episodes'] != 50 or len(list(p.glob('data/*/episode_*.parquet'))) != 50:
                continue
            episodes=[json.loads(s)for s in(p/'meta/episodes.jsonl').read_text().splitlines()if s]
            windows=sum(max(0,e['length']-49)for e in episodes)
            if len(episodes)==50 and windows>0:
                rows.append(dict(**t,path=str(p),windows=windows,episodes=50))
                break
    save(D/'data-live-progress.json',dict(time=time.time(),completed=len(rows),tasks=rows,initial_worker_alive=False,finishing_worker=True))
    return rows

def retry(t):
    archive=m.download(t['task'])
    with (D/f"data-{t['task']}-finish.log").open('a') as log:
        result=subprocess.run([sys.executable,'-u','-B',str(D/'data_worker.py'),'--convert',t['task'],str(archive)],stdout=log,stderr=subprocess.STDOUT,timeout=10800)
    assert result.returncode==0, f"{t['task']}: inspect preserved partial conversion"
    # This run creates these temporary inputs; retain the completed canonical dataset.
    import shutil
    root=Path('/data/chenyiteng/datasets/robotwin2')
    temps=[root/kind/m.REV/t['task']/('clean50-'+m.DATE) for kind in ['raw','intermediate']]
    temps.append(D/'downloads'/m.REV/'dataset'/t['task'])
    for p in temps:
        assert p.resolve().is_relative_to(root.resolve()) or p.resolve().is_relative_to((D/'downloads').resolve())
        if p.is_dir() and not p.is_symlink(): shutil.rmtree(p)


try:
    for attempt in range(1,5):
        rows=scan()
        if len(rows)==50:
            break
        present={r['task']for r in rows}
        missing=[t for t in tasks if t['task']not in present]
        errors={}
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures={pool.submit(retry,t):t['task']for t in missing}
            for f in concurrent.futures.as_completed(futures):
                try:
                    f.result()
                except Exception:
                    errors[futures[f]]=traceback.format_exc()
                scan()
                save(D/'data-finish-attempt.json',dict(attempt=attempt,errors=errors))
        assert time.time()<deadline, 'Data completion exceeded12h'
    rows=scan()
    assert len(rows)==50, dict(missing=[t['task']for t in tasks if t['task']not in {r['task']for r in rows}],errors=errors)
    rows.sort(key=lambda x:x['task'])
    n=sum(r['windows']for r in rows)
    save(D/'data-manifest.json',dict(revision=m.REV,tasks=rows,total_windows=n,horizon=50,sampler='task_balanced_shuffled_window_cycles'))
    save(D/'data-ready.json',dict(time=time.time(),tasks=50,episodes=2500,windows=n,steps_per_epoch=(n+127)//128,total_steps=50*((n+127)//128),complete=True))
    print(json.dumps(read(D/'data-ready.json')),flush=True)
except BaseException:
    save(D/'data-finish-error.json',dict(time=time.time(),error=traceback.format_exc()))
    raise
