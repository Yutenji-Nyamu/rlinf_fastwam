"""One-off GPU4 BC replacement; old results and all GPU5-7 processes are preserved."""
import importlib.util,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
S=Path('/data/chenyiteng/deployment-20261006/ugrow-bc8u5-300-v1');D=S/'bc'
def read(p):return json.loads(Path(p).read_text())
def mod(p,n):
 spec=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def save(p,v):
 p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2));tmp.replace(p)
def run():
 info=read(S/'prepared.json');p=read(D/'plan.json');rt=mod(p['ops'],'rt');op,b=rt.configure(D);op.checked()
 ex=mod(S/'tools/extend_budget.py','ex');base=mod(S/'tools/rlt_lease.py','lease');oldstage=Path(info['old_stage']);oldp=read(oldstage/'plan.json');oldrt=mod(oldp['ops'],'oldrt');oldop,_=oldrt.configure(oldstage);oldop.checked()
 ext=Path(info['old_extension']);extid=read(ext/'extension-identity.json');oid=read(oldstage/'owner-identity.json');driver=read(Path(oldp['runs']['formal']['run'])/'runtime/driver-identity.json')
 protected=[]
 for r in ('/data/chenyiteng/results/rlinf-rlt/ugrow-rlt-g5-formal-1006-v3', '/data/chenyiteng/results/rlinf-dsrl-pi05-u-20261006/clean-formal-200-mb256-v1','/data/chenyiteng/results/rlinf-dsrl-pi05-u-20261006/u-formal-200-mb256-v1'):
  protected.append(read(Path(r)/'runtime/driver-identity.json'))
 for g in b.gpu_snapshot():
  if g['gpu'] in (5,6,7):protected.extend(b.proc(q['pid']) for q in g['processes'])
 assert all(b.same(q) for q in protected)
 me=b.proc(os.getpid());save(S/'cutover-identity.json',me);state={'time':time.time(),'identity':me,'phase':'PREPARED','protected':protected}
 child=None;stopped=False
 def sig(q):
  assert q['uid']==1003 and b.same(q)
  code='import os,signal,sys;from pathlib import Path;p=Path("/proc")/sys.argv[1];fd=os.pidfd_open(int(sys.argv[1]));s=(p/"stat").read_text().rsplit(")",1)[1].split();assert p.stat().st_uid==1003 and int(s[19])==int(sys.argv[2]) and s[0] not in ("Z","X");signal.pidfd_send_signal(fd,signal.SIGTERM);os.close(fd)'
  subprocess.run(['/usr/bin/python3','-B','-c',code,str(q['pid']),str(q['start'])],check=True)
 try:
  assert read(S/'cpu-tests.json')['rc']==0 and read(S/'cpu-tests.json')['combined']['rc']==0
  assert b.same(oid) and b.same(driver) and b.same(extid)
  assert not (Path(p['lease_dir'])/'return-attempt.json').exists()
  save(S/'stop-old-extension-intent.json',{'time':time.time(),'identity':extid,'original_owner':oid,'reason':'User replaces old BC400 continuation with fresh N8/U5/300'})
  sig(extid)
  for _ in range(60):
   if not b.same(extid):break
   time.sleep(1)
  assert not b.same(extid) and read(ext/'extension-status.json')['phase']=='CANCELLED'
  with base._lock(Path(p['lease_dir'])):
   assert b.same(oid) and b.same(driver)
   state.update(phase='STOPPING_OLD_BC',time=time.time());save(S/'cutover-status.json',state)
   save(S/'stop-old-owner-intent.json',{'time':time.time(),'identity':oid,'driver':driver,'preserve_old_results':True})
   sig(oid);stopped=True
   for _ in range(180):
    if not b.same(oid):break
    state.update(time=time.time());save(S/'cutover-status.json',state);time.sleep(2)
   assert not b.same(oid) and not b.same(driver),'Old BC did not terminate cleanly'
   terminal=read(oldstage/'terminal.json');scope=oldrt.scope_state(oldop,b,oldp,oldp['runs']['formal'],driver)
   assert terminal['released'] and scope['released'] and not scope['outside']
   assert all(not b.same(q) for row in terminal['runs'] for q in [row['identity'],*row.get('processes',[])])
   assert all(b.same(q) for q in protected)
   save(S/'old-bc-released.json',{'time':time.time(),'terminal':terminal,'scope':scope,'protected_alive':True,'intentional_stop':True})
   op.checked();save(D/'launch-authorized.json',{'time':time.time(),'gpu':4,'operation_id':p['operation_id'],'lease_dir':p['lease_dir']})
  assert not (D/'owner-identity.json').exists()
  with (D/'owner.log').open('x') as f:
   child=subprocess.Popen([p['python'],'-u','-B',p['ops'],'--stage',str(D),'owner'],cwd=p['repo'],env=ex.cpu_env(),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  ci=b.proc(child.pid);save(S/'new-owner-launch.json',{'time':time.time(),'identity':ci,'stage':str(D)})
  state.update(phase='NEW_BC_RUNNING_RETENTION',new_owner=ci,time=time.time());save(S/'cutover-status.json',state)
  e={'original_run':str(Path(p['runs']['formal']['run']).with_name('unused-bc8-original-placeholder')),'from_step':0,'total_steps':300,'retention_authorized':'latest2+resume+final; only this BC; preserve logs'}
  while child.poll() is None:
   ex.prune_bc(p,e,rt,b,S)
   state.update(time=time.time(),training_state=read(D/'status.json') if (D/'status.json').exists() else None);save(S/'cutover-status.json',state);time.sleep(20)
  ex.prune_bc(p,e,rt,b,S);state.update(phase='NEW_OWNER_FINISHED',exit_code=child.returncode)
 except BaseException:
  state.update(phase='NEEDS_ATTENTION',error=traceback.format_exc())
  if stopped and child is None and (oldstage/'terminal.json').exists() and read(oldstage/'terminal.json').get('released'):
   try:state['fallback']=base.restore(Path(p['lease_dir']),oldstage/'terminal.json')
   except BaseException:state['fallback_error']=traceback.format_exc()
 finally:state['time']=time.time();save(S/'cutover-status.json',state)
if __name__=='__main__':run()
