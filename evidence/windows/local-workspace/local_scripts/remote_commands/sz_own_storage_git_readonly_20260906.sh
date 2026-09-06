set -eu
export PYTHONDONTWRITEBYTECODE=1 GIT_OPTIONAL_LOCKS=0 GIT_TERMINAL_PROMPT=0
nice -n 19 ionice -c 3 /usr/bin/python3 -B - <<'PY'
import concurrent.futures,datetime,json,os,re,subprocess
from pathlib import Path
base=Path('/data/chenyiteng');project=base/'projects/rlinf-shenzhen'
def cmd(a,timeout=60,env=None):
    try:
        p=subprocess.run(a,capture_output=True,text=True,timeout=timeout,env=env)
        return {'rc':p.returncode,'out':p.stdout.strip(),'err':p.stderr.strip()[:1500]}
    except subprocess.TimeoutExpired as e:return {'timeout':True,'out':(e.stdout or b'').decode(errors='replace') if isinstance(e.stdout,bytes) else e.stdout or ''}
def disk():
    du=cmd(['du','-x','-B1','--max-depth=5','--',str(base)],600)
    rows=[]
    for line in du['out'].splitlines():
        sz,p=line.split('\t',1)
        if int(sz)>=100*1024**2 or Path(p).parent==base:rows.append({'path':p,'allocated_bytes':int(sz)})
    ckpts=[];large=[];runs=[]
    for current,dirs,files in os.walk(base/'results',followlinks=False):
        dirs[:]=[x for x in dirs if x not in ['.git','video','videos','success_data','tensorboard']]
        here=Path(current)
        if 'driver.log' in files or 'started_at.txt' in files:
            state={n:(here/n).read_text(errors='replace').strip() if (here/n).is_file() else None for n in ['wrapper.pid','started_at.txt','finished_at.txt','exit_code.txt']}
            state['wrapper_alive']=bool(state['wrapper.pid']) and Path('/proc',state['wrapper.pid']).exists()
            runs.append({'path':str(here),'state':state})
        for name in files:
            p=here/name
            try:
                s=p.stat()
                if s.st_size>=1024**3 and not p.is_symlink():large.append({'path':str(p),'bytes':s.st_size,'allocated_bytes':s.st_blocks*512,'nlink':s.st_nlink})
            except FileNotFoundError:pass
        if 'checkpoints' not in dirs:continue
        cr=here/'checkpoints';gens=[]
        for gen in cr.iterdir():
            if not gen.is_dir() or not re.fullmatch('global_step_\d+',gen.name):continue
            fs=[]
            for p in gen.rglob('*'):
                if p.is_file() and not p.is_symlink():
                    st=p.stat();fs.append({'file':str(p.relative_to(gen)),'bytes':st.st_size,'allocated_bytes':st.st_blocks*512,'nlink':st.st_nlink})
            gens.append({'step':int(gen.name.rsplit('_',1)[-1]),'path':str(gen),'files':fs})
        ckpts.append({'root':str(cr),'generations':sorted(gens,key=lambda g:g['step'])});dirs.remove('checkpoints')
    return {'du':rows,'du_rc':du.get('rc'),'du_errors':du.get('err'),'checkpoint_runs':ckpts,'large_noncheckpoint_files':large,'run_states':runs,'finished_at':datetime.datetime.now().astimezone().isoformat()}
def git():
    candidates=[project/'RLinf',project/'RoboTwin-RLinf-support']+[p for p in (project/'worktrees').iterdir() if p.is_dir() and (p/'.git').exists()]
    groups={};result=[]
    for p in candidates:
        common=cmd(['git','-C',str(p),'rev-parse','--path-format=absolute','--git-common-dir'],10)
        if common.get('rc')==0:groups.setdefault(common['out'],p)
    for common,root in groups.items():
        worktree=cmd(['git','-C',str(root),'worktree','list','--porcelain'])
        blocks=worktree['out'].split('\n\n');trees=[]
        for b in blocks:
            fields={l.split(' ',1)[0]:l.split(' ',1)[1] for l in b.splitlines() if ' ' in l}
            if 'worktree' not in fields:continue
            p=Path(fields['worktree'])
            if not p.exists():continue
            tracked=cmd(['git','-C',str(p),'ls-files','-s'])['out'].splitlines()
            evidence=[];mapping=[]
            for row in tracked:
                if '\t' not in row:continue
                meta,f=row.split('\t',1)
                if any(x in f.lower() for x in ['evidence/','/evidence','manifest','server_run_path']):
                    evidence.append(f)
                    fp=p/f
                    if fp.is_file() and fp.stat().st_size<200000 and (f.endswith(('.md','.json','.txt','.csv')) or 'SERVER_RUN_PATH' in f):
                        text=fp.read_text(errors='replace')
                        refs=re.findall(r'/data/chenyiteng/results/[^\s\"\'`<>]+',text)
                        if refs:mapping.append({'file':f,'run_refs':sorted(set(refs))[:40]})
            trees.append({'path':str(p),'head':fields.get('HEAD'),'branch':fields.get('branch'),
              'status':cmd(['git','-C',str(p),'status','--porcelain']),
              'tracked_files':len(tracked),'evidence_count':len(evidence),'evidence_dirs':sorted(set('/'.join(f.split('/')[:4]) for f in evidence)),
              'run_mapping':mapping})
        branches=cmd(['git','-C',str(root),'for-each-ref','--format=%(refname) %(objectname)','refs/heads/codex/'])
        remotes=cmd(['git','-C',str(root),'remote','-v'])
        remote_names=cmd(['git','-C',str(root),'remote'])['out'].splitlines();remote='personal' if 'personal' in remote_names else 'origin'
        env={k:v for k,v in os.environ.items() if k.lower() not in ['http_proxy','https_proxy','all_proxy']}
        live=cmd(['git','-C',str(root),'-c','http.proxy=','-c','https.proxy=','ls-remote','--heads',remote],45,env)
        result.append({'common_dir':common,'root':str(root),'trees':trees,'branches':branches,'remotes':remotes,'checked_remote':remote,'live_remote_heads':live})
    return result
d={'time':datetime.datetime.now().astimezone().isoformat()}
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    a=pool.submit(disk);b=pool.submit(git);d['storage']=a.result();d['git']=b.result()
d['df']=cmd(['df','-B1','/data','/home']);d['finished_at']=datetime.datetime.now().astimezone().isoformat()
print(json.dumps(d,ensure_ascii=False))
PY
