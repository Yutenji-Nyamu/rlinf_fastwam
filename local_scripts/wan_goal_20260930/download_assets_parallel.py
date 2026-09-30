"""Fetch fixed public assets with bounded aria2 HTTP ranges and official hashes."""
import hashlib,json,os,subprocess,time
from pathlib import Path
from urllib.parse import quote
from huggingface_hub import snapshot_download

root=Path('/data/chenyiteng/projects/wan-goal-sz3')
manifest=json.loads((root/'scripts/asset_manifest.json').read_text())
aria=root/'tools/aria2/root/usr/bin/aria2c'
stage=root/'downloads/aria2';stage.mkdir(parents=True,exist_ok=True)
env=os.environ.copy();env['LD_LIBRARY_PATH']=str(root/'tools/aria2/root/usr/lib/x86_64-linux-gnu')
for key in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):
 env.pop(key,None)  # aria2 uses its explicit HTTP proxy, not an inherited SOCKS URL.
pending=[]; lines=[]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
for repo in manifest['repositories']:
 destination=root/repo['local_subdir'];destination.mkdir(parents=True,exist_ok=True)
 for file in repo['files']:
  if file['size']<100_000_000:continue
  target=destination/file['path'];expected=file['lfs_sha256']
  if target.exists():
   assert target.stat().st_size==file['size'] and sha(target)==expected,str(target)
   print(json.dumps({'phase':'existing_verified','file':str(target)}),flush=True);continue
  staged=stage/destination.name/file['path'];staged.parent.mkdir(parents=True,exist_ok=True)
  if staged.exists() and staged.stat().st_size==file['size'] and not Path(str(staged)+'.aria2').exists():
   assert sha(staged)==expected,str(staged)
   os.replace(staged,target)
   print(json.dumps({'phase':'completed_stage_verified','file':str(target)}),flush=True);continue
  url=f"https://huggingface.co/{repo['repo_id']}/resolve/{repo['revision']}/{quote(file['path'])}"
  lines.extend([url,' dir='+str(staged.parent),' out='+staged.name,' checksum=sha-256='+expected])
  pending.append((repo,file,staged,target))
input_file=stage/('urls-'+str(int(time.time()))+'.txt');input_file.write_text('\n'.join(lines)+'\n')
if pending:
 command=[str(aria),'--input-file='+str(input_file),'--max-concurrent-downloads=2',
 '--max-connection-per-server=8','--split=8','--min-split-size=16M','--continue=true',
 '--auto-file-renaming=false','--allow-overwrite=false','--file-allocation=none',
 '--check-integrity=true','--all-proxy=http://127.0.0.1:7897',
 '--connect-timeout=15','--timeout=60','--max-tries=12','--retry-wait=10',
 '--summary-interval=30','--download-result=full','--console-log-level=warn']
 print(json.dumps({'phase':'aria2_ranges','files':len(pending),'connections_per_file':8,'parallel_files':2}),flush=True)
 for attempt in range(1,9):
  result=subprocess.run(command,env=env)
  if result.returncode==0:break
  print(json.dumps({'phase':'aria2_retry','attempt':attempt,'exit_code':result.returncode,'partials_preserved':True}),flush=True)
  if attempt==8 or result.returncode not in (1,2,6,19):
   raise subprocess.CalledProcessError(result.returncode,command)
  time.sleep(min(15*attempt,60))
 for repo,file,staged,target in pending:
  assert staged.stat().st_size==file['size'] and not Path(str(staged)+'.aria2').exists()
  assert not target.exists();os.replace(staged,target)
  print(json.dumps({'phase':'large_downloaded_hash_checked','file':str(target),'sha256':file['lfs_sha256']}),flush=True)
for repo in manifest['repositories']:
 destination=root/repo['local_subdir']
 small=[f['path'] for f in repo['files'] if f['size']<100_000_000]
 for attempt in range(1,9):
  try:
   snapshot_download(repo_id=repo['repo_id'],revision=repo['revision'],local_dir=destination,
                     allow_patterns=small,max_workers=8,token=False);break
  except Exception as exc:
   print(json.dumps({'phase':'small_retry','repo':repo['repo_id'],'attempt':attempt,'error':type(exc).__name__}),flush=True)
   if attempt==8:raise
   time.sleep(15)
 files=[{'path':f['path'],'bytes':(destination/f['path']).stat().st_size} for f in repo['files']]
 receipt={'repo':repo['repo_id'],'revision':repo['revision'],'finished':time.time(),'files':files,'transport':'aria2 official URLs+LFS SHA256; HF snapshot small files'}
 (root/'logs'/f'{destination.name}-download.json').write_text(json.dumps(receipt,indent=2))
 print(json.dumps({'phase':'complete','repo':repo['repo_id'],'files':len(files),'bytes':sum(f['bytes'] for f in files)}),flush=True)
print('ALL_PINNED_ASSETS_DOWNLOADED',flush=True)
