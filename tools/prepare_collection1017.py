import json,datetime,base64,zlib
from pathlib import Path
R=Path('E:/Codex/home/visualizations/2026/10/13/01a1354d-42a1-74a3-adc8-6e9c3421e45a');cut=datetime.datetime(2026,9,26,tzinfo=datetime.timezone(datetime.timedelta(hours=8))).timestamp()
body=r'''
import os,io,json,time,tarfile,hashlib,re,base64,zlib
from pathlib import Path
os.nice(15);uid=os.getuid();D=Path('/data/chenyiteng/archives/post-iclr-20261017-v1');D.mkdir(parents=True,exist_ok=False)
files=json.loads(zlib.decompress(base64.b64decode(PAYLOAD)));manifest=[];skipped=[];archives=[];tar=None;count=0;size=0
secret=re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{24,}|-----BEGIN (?:OPENSSH |RSA )?PRIVATE KEY-----|(?:password|api_key|access_token)\s*[:=]\s*["\x27]?[A-Za-z0-9/+_-]{8,})',re.I)
def close():
 global tar
 if tar:
  tar.close();p=D/('part-%03d.tar.gz'%count);archives.append({'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()});tar=None
for row in files:
 p=Path(row['path'])
 try:
  s=p.stat();assert s.st_uid==uid and str(p).startswith(('/data/chenyiteng/results/','/home/chenyiteng/results/')) and not p.is_symlink()
  if s.st_size>20*1024**2:skipped.append({'path':str(p),'reason':'grew past 20 MiB'});continue
  with p.open('rb') as f:data=f.read(s.st_size)
  if secret.search(data):skipped.append({'path':str(p),'reason':'possible credential; withheld for review'});continue
  if not tar or size+len(data)>40*1024**2:
   close();count+=1;size=0;tar=tarfile.open(D/('part-%03d.tar.gz'%count),'w:gz',compresslevel=3)
  name=str(p).lstrip('/');info=tarfile.TarInfo(name);info.size=len(data);info.mtime=s.st_mtime;info.mode=0o644;tar.addfile(info,io.BytesIO(data));size+=len(data)
  manifest.append({'path':str(p),'bytes':len(data),'mtime':s.st_mtime,'sha256':hashlib.sha256(data).hexdigest(),'part':count,'changed_during_read':p.stat().st_mtime!=s.st_mtime})
 except (OSError,AssertionError) as e:skipped.append({'path':str(p),'reason':type(e).__name__})
close();out={'time':time.time(),'files':manifest,'skipped':skipped,'archives':archives,'cutoff':'2026-09-26 CST','snapshot':'active files copied as observed prefixes; no pause of training'};(D/'manifest.json').write_text(json.dumps(out));print(json.dumps(out))
'''
def runkey(p):
 a=p.split('/results/',1)[1].split('/')
 if a[0]=='rlinf-shenzhen':
  if 'runs' in a:return '/'.join(a[:a.index('runs')+2])
  return '/'.join(a[:3])
 return '/'.join(a[:2]) if a[0] in ['rlinf-rlt','rlinf-dsrl-pi0-single','rlinf-dsrl-pi05-u-20261006'] else a[0]
for h in ['sz1','sz2','sz3']:
 d=json.loads((R/h/'artifact-inventory1017.out').read_text());groups={runkey(f['path']) for f in d['files'] if f['mtime']>=cut and 'server-maintenance-' not in f['path']}
 files=[f for f in d['files'] if runkey(f['path']) in groups and 'server-maintenance-' not in f['path']]
 payload=base64.b64encode(zlib.compress(json.dumps(files).encode())).decode();Path(f'.tmp/collect1017_{h}.py').write_text('PAYLOAD='+repr(payload)+'\n'+body,encoding='utf-8')
 (R/'archive1017'/f'{h}-selection.json').write_text(json.dumps({'groups':sorted(groups),'files':files}),encoding='utf-8')
 print(h,len(groups),'groups',len(files),'files',round(sum(f['bytes'] for f in files)/2**20),'MiB')
