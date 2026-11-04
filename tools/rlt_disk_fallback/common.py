"""Small shared lifecycle primitives, reused from the proven RLT wrappers."""
import errno,hashlib,json,os,signal,subprocess,urllib.request,xml.etree.ElementTree as ET
from pathlib import Path

def emergency_path(p):
 root=Path(os.environ.get('RLT_EMERGENCY_ROOT',f'/tmp/chenyiteng-rlt-control-{os.getuid()}'))
 return root/(hashlib.sha256(str(Path(p).absolute()).encode()).hexdigest()+'-'+Path(p).name)

def read(p):
 p=Path(p);fallback=emergency_path(p)
 if fallback.exists() and (not p.exists() or fallback.stat().st_mtime_ns>p.stat().st_mtime_ns):p=fallback
 return json.loads(p.read_text())
def save(p,v):
 p=Path(p)
 try:
  p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2));os.replace(tmp,p)
 except OSError as e:
  if e.errno not in (errno.ENOSPC,errno.EDQUOT):raise
  p=emergency_path(p);p.parent.mkdir(mode=0o700,parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2));os.replace(tmp,p)
def proc(pid):
 try:
  p=Path('/proc')/str(pid);s=(p/'stat').read_text().rsplit(')',1)[1].split();return dict(pid=int(pid),uid=p.stat().st_uid,start=int(s[19]),state=s[0])
 except OSError:return None
def same(q):
 a=proc(q['pid']);return bool(a and all(a[k]==q[k] for k in ['pid','uid','start']) and a['state'] not in ['Z','X'])
def exact_signal(q,s):
 if not same(q):return
 if not hasattr(os,'pidfd_open') or not hasattr(signal,'pidfd_send_signal'):
  subprocess.run(['/usr/bin/python3','-B',__file__,'signal',json.dumps(q),str(int(s))],check=True);return
 fd=os.pidfd_open(q['pid'])
 try:
  assert same(q) and q['uid']==os.getuid();signal.pidfd_send_signal(fd,s)
 finally:os.close(fd)
def actors(p,ns):
 url=p['ray_dashboard_url'].rstrip('/')+'/api/v0/actors?limit=10000&detail=1'
 with urllib.request.urlopen(url,timeout=20) as f:r=json.load(f)['data']['result']
 assert r.get('num_after_truncation',r.get('total',0))<10000
 return [a for a in r['result'] if a.get('ray_namespace')==ns and a.get('state')!='DEAD']
def gpus():
 x=ET.fromstring(subprocess.check_output(['nvidia-smi','-q','-x'],timeout=25))
 return {i:dict(uuid=g.findtext('uuid'),processes=[dict(pid=int(p.findtext('pid')),type=p.findtext('type'),mib=int(p.findtext('used_memory').split()[0]) if p.findtext('used_memory','').endswith('MiB') else -1) for p in g.findall('processes/process_info')]) for i,g in enumerate(x.findall('gpu'))}

def admission_card(card,slot):
 # Isaac can leave a known 7 MiB Vulkan discovery context on a sibling card.
 # Admit work alongside only that exact live, owned G-only process; never C/C+G.
 allowed=[]
 for source in slot.get('admission_graphics_sources',[]):
  path=Path(source['deployment'])/f"native-g{source['gpu']}-child.json"
  if not path.is_file():continue
  who=read(path)
  if who['uid']==os.getuid() and same(who):allowed.append((who,source.get('max_mib',8)))
 ignored=[q for q in card['processes'] if q['type']=='G' and any(q['pid']==who['pid'] and 0<=q.get('mib',-1)<=limit for who,limit in allowed)]
 return dict(card,processes=[q for q in card['processes'] if q not in ignored],ignored_graphics=ignored)
def releasable(identity,rows,card):return not same(identity) and not rows and not card['processes']

if __name__=="__main__":
 import sys
 assert sys.argv[1]=="signal";exact_signal(json.loads(sys.argv[2]),int(sys.argv[3]))
