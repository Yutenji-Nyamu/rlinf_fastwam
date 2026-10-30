"""Move completed own outputs; route unstarted MFD without touching live training."""
import os,json,shutil,stat,time,subprocess
from pathlib import Path
assert os.getuid()==1003
os.nice(10)
subprocess.run(['ionice','-c','3','-p',str(os.getpid())],capture_output=True)
H=Path('/home/chenyiteng/results/pi05-distill/single-extend400-fqlfix-sz1-g67-20261029')
B=Path('/data/chenyiteng/results/pi05-distill/single-extend400-fqlfix-sz1-g67-20261029')
D=Path('/home/chenyiteng/deployment-20261029/single-extend400-fqlfix-sz1-g67')
T=Path('/home/chenyiteng/deployment-20261030/storage-audit-1030f')
def read(p):return json.loads(p.read_text())
def save(p,d):
 t=p.with_suffix(p.suffix+'.route-tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)
def free():return {p:round(os.statvfs(p).f_bavail*os.statvfs(p).f_frsize/2**30,2) for p in ['/home','/data']}
assert H.resolve()==H and B.parent.resolve()==B.parent
assert read(D/'training-queue-status.json')['method']=='onedp'
assert not (T/'storage-route1030n.json').exists()
B.mkdir(exist_ok=True)
record={'time':time.time(),'before_free':free(),'moves':[],'route':{},'processes_stopped':0}
for method in ['endpoint','cp_v2']:
 source=H/method; dest=B/method; backup=H/(method+'.route-original-1030n')
 assert source.resolve()==source and source.is_dir() and not dest.exists() and not backup.exists()
 assert read(source/'segment-done.json')['complete']
 files=[]
 for f in source.iterdir():
  s=f.lstat();assert stat.S_ISREG(s.st_mode) and s.st_uid==1003 and s.st_nlink==1
  files.append({'name':f.name,'size':s.st_size,'inode':s.st_ino,'mtime_ns':s.st_mtime_ns,'allocated':s.st_blocks*512})
 for p in Path('/proc').glob('[0-9]*'):
  try:
   if p.stat().st_uid!=1003:continue
   for fd in (p/'fd').iterdir():
    try:target=os.readlink(fd)
    except OSError:continue
    assert not target.startswith(str(source)+'/'),('open file',p.name,target)
  except (FileNotFoundError,PermissionError):pass
 print(json.dumps({'copying':method,'GiB':sum(x['size'] for x in files)/2**30}),flush=True)
 shutil.copytree(source,dest)
 for x in files:
  s=(source/x['name']).stat()
  assert (s.st_ino,s.st_size,s.st_mtime_ns)==(x['inode'],x['size'],x['mtime_ns'])
  assert (dest/x['name']).stat().st_size==x['size']
 assert read(D/'training-queue-status.json')['method']=='onedp'
 source.rename(backup)
 try:source.symlink_to(dest,target_is_directory=True)
 except BaseException:
  backup.rename(source);raise
 assert source.resolve()==dest
 receipt=read(source/'checkpoint.json')
 assert all(Path(receipt[k]).is_file() for k in ['path','policy'])
 assert backup.resolve().parent==H
 for x in files:
  p=backup/x['name'];s=p.lstat();assert s.st_ino==x['inode'];p.unlink()
 backup.rmdir()
 record['moves'].append({'method':method,'from':str(source),'to':str(dest),'GiB':sum(x['allocated'] for x in files)/2**30,'files':files,'old_path_preserved_by_symlink':True})
 save(T/'storage-route1030n-progress.json',record)
 print(json.dumps({'moved':method,'free':free()}),flush=True)
assert not (H/'mfd').exists() and not (H/'mfd').is_symlink() and not (B/'mfd').exists()
(B/'mfd').mkdir();(H/'mfd').symlink_to(B/'mfd',target_is_directory=True)
# Exercise future write location using a tiny own file, then remove it.
test=H/'mfd/.route-write-test';test.write_text('route test');assert test.stat().st_dev==Path('/data').stat().st_dev;test.unlink()
record['route']={'g45':'existing Data output unchanged','g67_onedp':'active Home output unchanged; free space replenished','g67_mfd':str(B/'mfd'),'g67_mfd_compatibility_path':str(H/'mfd')}
plan=read(D/'formal-plan.json');plan['storage']='Active DMD-new stays Home; completed Direct/CPv2 and upcoming MFD live Data via original-path directory symlinks. Latest2+best unchanged.'
plan['storage_routing_receipt']=str(T/'storage-route1030n.json');save(D/'formal-plan.json',plan)
record['after_free']=free();record['queue_after']=read(D/'training-queue-status.json');record['completed']=True
save(T/'storage-route1030n.json',record)
print(json.dumps({'completed':True,'moved_GiB':sum(x['GiB'] for x in record['moves']),'free':record['after_free'],'route':record['route'],'active_method':record['queue_after']['method']}),flush=True)
