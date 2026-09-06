set -eu
nice -n 19 ionice -c 3 /usr/bin/python3 - <<'PY'
import datetime,json,os,re,subprocess
from pathlib import Path
base=Path('/data/chenyiteng')
def du(p,depth):
    r=subprocess.run(['du','-x','-B1',f'--max-depth={depth}','--',str(p)],capture_output=True,text=True,timeout=600)
    rows=[]
    for line in r.stdout.splitlines():
        size,name=line.split('\t',1);rows.append({'path':name,'bytes':int(size)})
    return {'rc':r.returncode,'rows':sorted(rows,key=lambda x:x['bytes'],reverse=True),'errors':r.stderr[:1500]}
result={'time_started':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'scope':'metadata only, own /data and /home directories; no deletes','data_structure':du(base,3),'home_structure':du(Path('/home/chenyiteng'),2),'checkpoint_runs':[]}
results=base/'results'
groups={}
for current,dirs,files in os.walk(results,followlinks=False):
    path=Path(current)
    dirs[:]=[d for d in dirs if d not in ('video','videos','.git')]
    if 'checkpoints' in dirs:
        checkpoint_root=path/'checkpoints'
        generations=[]
        for gen in sorted(checkpoint_root.iterdir()):
            if not gen.is_dir():continue
            big=[];small=[];allocated=0
            for f in gen.rglob('*'):
                if not f.is_file() or f.is_symlink():continue
                st=f.stat();allocated+=st.st_blocks*512
                row={'relative':str(f.relative_to(gen)),'bytes':st.st_size,'allocated_bytes':st.st_blocks*512}
                (big if st.st_size>=1024**2 else small).append(row)
            generations.append({'generation':gen.name,'path':str(gen),'mtime':gen.stat().st_mtime,'allocated_bytes':allocated,'big_files':big,'small_files':small})
        result['checkpoint_runs'].append({'root':str(checkpoint_root),'generations':generations})
        dirs.remove('checkpoints')
result['time_finished']=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
print(json.dumps(result,ensure_ascii=False))
PY
