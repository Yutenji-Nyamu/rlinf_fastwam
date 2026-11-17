"""Publish generic collectors and completed figure bundles; private real data stay private."""
import sys,json,time,base64,fcntl,zipfile,os,subprocess
from pathlib import Path
def main(config,kind):
    j=json.loads(Path(config).read_text());root=Path(j['root']);code=root/'code';sys.path.insert(0,j['publisher_library'])
    import publish_results as lib
    files={};prefix='research/appendix_followup_20261117/'
    if kind=='code':
        files={f'tools/appendix_followup/{p.name}':p for p in code.glob('*.py')};files[prefix+'README.md']=root/'README.md'
    else:
        stage=j['stages'][kind];dest=Path(stage['output'])
        if not dest.exists():return
        if kind=='batched-round-cost' and (dest/'complete.json').exists():
            from render_results import timing
            rendered=dest/'figures';rendered.mkdir(exist_ok=True);timing(dest,rendered,Path(stage['memory']))
        if kind=='geowam-dense-offline':
            lib.repo=j['private_repo'];lib.branch=j['private_branch'];prefix='materials/appendix_geowam_dense_20261117/'
            # Read current repo visibility before uploading any private RGB-D/recordings.
            r=subprocess.run(['gh','api','repos/'+lib.repo],capture_output=True,text=True,env=lib.env,timeout=30)
            info=json.loads(r.stdout);assert info['private'] is True
        contents=[p for p in dest.rglob('*')if p.is_file()and p.suffix not in ['.zip','.log']]
        pack=root/'packages'/kind;pack.mkdir(parents=True,exist_ok=True)
        groups=[];group=[];size=0
        for p in contents:
            if size+p.stat().st_size>35*1024**2 and group:groups.append(group);group=[];size=0
            group.append(p);size+=p.stat().st_size
        if group:groups.append(group)
        for i,group in enumerate(groups):
            archive=pack/f'{kind}-{i:03d}.zip'
            with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED)as z:
                for p in group:z.write(p,str(p.relative_to(dest)))
            files[prefix+kind+'/'+archive.name]=archive
        for p in contents:
            if p.suffix in ['.png','.pdf','.svg','.json','.csv','.md'] and p.stat().st_size<2*1024**2:files[prefix+kind+'/'+str(p.relative_to(dest))]=p
    with(Path(j['publisher_library'])/'publish.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);base=lib.api('git/ref/heads/'+lib.branch)['object']['sha'];parent=lib.api('git/commits/'+base);tree=[]
        for name,p in files.items():
            blob=lib.api('git/blobs',dict(content=base64.b64encode(p.read_bytes()).decode(),encoding='base64'));tree.append(dict(path=name,mode='100644',type='blob',sha=blob['sha']));print('uploaded',name,flush=True)
        t=lib.api('git/trees',dict(base_tree=parent['tree']['sha'],tree=tree));c=lib.api('git/commits',dict(message='Add appendix collection '+kind,tree=t['sha'],parents=[base]));lib.api('git/refs/heads/'+lib.branch,dict(sha=c['sha'],force=False),method='PATCH')
        (root/f'published-{kind}.json').write_text(json.dumps(dict(commit=c['sha'],repo=lib.repo,branch=lib.branch,files=list(files),time=time.time()),indent=2));print('PUBLISHED',c['sha'],flush=True)
if __name__=='__main__':main(*sys.argv[1:])
