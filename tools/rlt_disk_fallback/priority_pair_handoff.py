"""One authorized pair: preserve RLT resume points, register priority, stop exact drivers."""
import os,sys,time,signal,subprocess,fcntl,zipfile
from pathlib import Path
from omegaconf import OmegaConf
S=Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1');sys.path.insert(0,str(S/'ops'));from common import read,save,proc,same,exact_signal,gpus

def handoff(D,cards,identity,reservations=None):
 assert same(identity)and identity['uid']==os.getuid()
 if reservations:
  assert set(map(str,cards))==set(reservations)
  assert all(same(r['identity'])and r['identity']['uid']==os.getuid() for r in reservations.values())
 serial=(S/'priority-handoff.lock').open('a');fcntl.flock(serial,fcntl.LOCK_EX)
 lock=None
 try:
  owner=read(S/'owner-identity.json')
  if same(owner):exact_signal(owner,signal.SIGTERM)
  for _ in range(60):
   if not same(owner):break
   time.sleep(1)
  assert not same(owner)
  lock=(S/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  p=read(S/'plan.json');state=read(S/'status.json');save(D/'previous-rlt-plan.json',p);save(D/'previous-rlt-status.json',state)
  targets=[]
  for g in map(str,cards):
   st=state['slots'][g];assert st['phase']in ['RLT_RUNNING','RLT_EXITED','WAITING_RLT'],st
   old=read(st.get('request',p['slots'][g]['fallback']));assert old['kind']=='rlt'and old['gpu']==int(g),old
   if st['phase']=='RLT_RUNNING'and same(st['identity']):
    command=Path(f"/proc/{st['identity']['pid']}/cmdline").read_bytes();assert str(S/'ops/driver.py').encode()in command and str(st['request']).encode()in command
    targets.append(st['identity'])
  for who in targets:exact_signal(who,signal.SIGTERM)
  deadline=time.time()+150
  while any(same(who)for who in targets)or any(gpus()[g]['processes']for g in cards):
   assert time.time()<deadline,'RLT cleanup incomplete';time.sleep(2)
  restores=[]
  for g in map(str,cards):
   st=state['slots'][g];old=read(st.get('request',p['slots'][g]['fallback']))
   if old.get('normal_request'):old=read(old['normal_request'])
   rt=Path(old['runtime']);c=OmegaConf.to_container(OmegaConf.load(rt/'resolved.yaml'),resolve=True);cp_root=Path(old['run']);found=[]
   existing=[Path(x)for x in [c['runner'].get('resume_dir'),old.get('checkpoint')]if x]
   logger=c['runner']['logger'];roots={cp_root/cp_root.name/'checkpoints',Path(logger['log_path'])/logger['experiment_name']/'checkpoints',*[x.parent for x in existing]}
   candidates=existing+[cp for root in roots for cp in root.glob('global_step_*')]
   for cp in candidates:
    shard=cp/'actor/local_shard_checkpoint/checkpoint_rank_0.pt'
    dcp=cp/'actor/dcp_checkpoint'
    if cp.is_dir()and ((shard.is_file()and zipfile.is_zipfile(shard))or((dcp/'.metadata').is_file()and any(dcp.glob('*.distcp')))):found.append(cp)
   assert found,'RLT full resume point missing'
   latest=max(found,key=lambda x:int(x.name.split('_')[-1]));newrt=cp_root/('runtime-after-'+D.name);newrt.mkdir(exist_ok=False)
   c['runner']['resume_dir']=str(latest)
   if 'dvac_gradient_weighting'in c.get('algorithm',{}):c['algorithm']['dvac_gradient_weighting']['output_dir']=str(cp_root/('dvac-after-'+D.name))
   save(newrt/'resolved.yaml',c);save(newrt/'environment.json',read(rt/'environment.json'));old.update(runtime=str(newrt),checkpoint=str(latest),step=int(latest.name.split('_')[-1]));req=D/f'fallback-g{g}.json';save(req,old)
   reservation=reservations[g] if reservations else dict(identity=identity,release=str(D/'released.json'))
   p['slots'][g].update(fallback=str(req),external_identity=reservation['identity'],external_release=reservation['release']);p['slots'][g].pop('priority_after_external',None);state['slots'][g]={'phase':'WAIT_EXTERNAL'};restores.append(dict(gpu=int(g),checkpoint=str(latest),request=str(req)))
  save(S/'plan.json',p);save(S/'status.json',state)
 finally:
  if lock is not None:lock.close()
  with(D/'owner.log').open('x')as log:child=subprocess.Popen(['/usr/bin/python3','-u','-B',str(S/'ops/owner.py'),'--plan',str(S/'plan.json')],cwd=S,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
  # The next pair must observe the new owner, not the just-stopped identity.
  ready=time.time()+15
  while read(S/'owner-identity.json').get('pid')!=child.pid:
   assert child.poll()is None and time.time()<ready,'RLT owner restart not registered'
   time.sleep(.1)
  serial.close()
 receipt=dict(time=time.time(),identity=identity,owner=proc(child.pid),gpus=cards,rlt_restores=restores,stopped=targets);save(D/'registered.json',receipt);return receipt
