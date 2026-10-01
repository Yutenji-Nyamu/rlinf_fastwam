"""Reuse the already restored four RLT runs and their fully verified snapshots.

Only this imported helper reuses a frozen checkpoint; shared installations and
the original helper bytes remain unchanged. Runtime private environment files
are copied server-side and never logged or published.
"""
import argparse,copy,hashlib,importlib.util,json,os,signal,socket,sys,types
from pathlib import Path
from expo_process import pidfd_open,pidfd_send,pidfd_probe
ROOT=Path('/data/chenyiteng/projects/expo-ft-sz2-20261001/formal-20261001')
PREVIOUS=ROOT.parent/'scale-smoke-20261001/rlt-cycle-scale-20261001-v1'
CYCLE=ROOT/'rlt-cycle-expo-formal-20261001-v1'
UID=20001

def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def boot():return Path('/proc/sys/kernel/random/boot_id').read_text().strip()

def helper(stage):
 path=stage/'rlt_cycle.py'
 spec=importlib.util.spec_from_file_location('_frozen_rlt_cycle',path)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 return module

def guard_checkpoint(h,checked,cfg,repo):
 path=Path(checked['path']).resolve(strict=True)
 assert path.is_relative_to(Path('/data/chenyiteng/recovered-rlt').resolve())
 assert path.stat().st_uid==UID and not path.is_symlink()
 _,contract=h.runtime_contract(repo,cfg)
 assert contract==checked['contract_sha256'],'Frozen RLT config contract drift'
 for relative,expected in checked['small_sha256'].items():
  file=path/'actor'/relative
  assert file.stat().st_uid==UID and sha(file)==expected,relative
 marker=read(path/'actor/sac_components/rlt_trainer_state/complete.json')
 assert marker['complete'] and marker['saved_runner_step']==checked['step']
 for relative,size in checked['dcp_shards'].items():
  assert (path/'actor/dcp_checkpoint'/relative).stat().st_size==size
 assert (path/'actor/sac_components/target_model/checkpoint_rank_0.pt').stat().st_size==checked['target_bytes']
 return copy.deepcopy(checked)

def prepare():
 from omegaconf import OmegaConf
 assert not CYCLE.exists(),'A formal cycle already exists; inspect receipts'
 assert read(PREVIOUS.parent/'rlt-guardian.json')['state']=='RESTORED'
 prior=read(PREVIOUS/'plan.json');stopped=read(PREVIOUS/'rlt-stopped.json')
 helper_bytes=(PREVIOUS/'rlt_cycle.py').read_bytes()
 assert hashlib.sha256(helper_bytes).hexdigest()==prior['script_sha256']
 CYCLE.mkdir(mode=0o700);(CYCLE/'rlt_cycle.py').write_bytes(helper_bytes)
 (CYCLE/'rlt_cycle.py').chmod(0o500);h=helper(CYCLE)
 plan=copy.deepcopy(prior);plan.update(cycle_id=CYCLE.name,time=h.now(),management_namespace='expo-rlt-ops-formal-20261001')
 h.source_check(plan);live=h.actors(plan)
 for key,row in plan['runs'].items():
  old=prior['runs'][key];old_run=Path(old['new_run']);rt=old_run/'runtime'
  ident=read(rt/'driver-identity.json')
  assert h.same(ident) and ident['uid']==UID
  words=(Path('/proc')/str(ident['pid'])/'cmdline').read_bytes().split(b'\0')
  assert str(PREVIOUS/'rlt_cycle.py').encode() in words and b'driver' in words and key.encode() in words
  ident.update(match_cmdline=True,cmdline_sha256=h.proc(ident['pid'])['cmdline_sha256'])
  owned=h.active(live,old['namespace']);jobs={a['job_id'] for a in owned}
  assert owned and len(jobs)==1;h.validate_actor_rows(owned,old['namespace'],jobs)
  cfg=h.config(rt/'resolved.yaml');checked=stopped['runs'][key]['checkpoint']
  guard_checkpoint(h,checked,cfg,plan['repo'])
  assert h.dependency_snapshot(cfg)==old['dependencies']
  new_run=Path('/data/chenyiteng/results/rlinf-rlt')/h.resumed_name(old_run,CYCLE)
  namespace='er-rlt-expo-formal-20261001-v1-'+key.replace('gpu','g')
  assert not new_run.exists() and not h.active(live,namespace)
  newcfg,changes=h.resumed_config(cfg,old_run,new_run,checked['path'])
  pre=CYCLE/'prepared'/key;pre.mkdir(parents=True,mode=0o700)
  OmegaConf.save(OmegaConf.create(cfg),pre/'original.yaml',resolve=True)
  OmegaConf.save(OmegaConf.create(newcfg),pre/'resolved.yaml',resolve=True)
  environment=read(rt/'environment.json')
  assert not any(name in environment for name in h.MASKS)
  environment={name:value.replace(str(old_run),str(new_run)).replace(old['namespace'],namespace)
    for name,value in environment.items()}
  h.save(pre/'environment.json',environment)
  row.update(original_run=str(old_run),original_namespace=old['namespace'],original_identity=ident,
    original_jobs=sorted(jobs),original_config_sha256=sha(rt/'resolved.yaml'),new_run=str(new_run),namespace=namespace,
    checkpoint=checked,config_changes=changes,latest_metrics_before=h.latest_metrics(old_run),
    prepared_sha256={name:sha(pre/name) for name in ('original.yaml','resolved.yaml','environment.json')})
 h.save(CYCLE/'plan.json',plan)
 h.save(CYCLE/'reuse-frozen-proof.json',{'boot_id':boot(),'previous_cycle':str(PREVIOUS),
   'previous_stop_sha256':sha(PREVIOUS/'rlt-stopped.json'),'policy':'Prior full payload validation and successful real RLT restore reused; small states/contracts/dependencies rechecked',
   'steps':{key:row['checkpoint']['step'] for key,row in plan['runs'].items()}})
 print(json.dumps({'prepared':True,'cycle':str(CYCLE),'steps':{key:row['checkpoint']['step'] for key,row in plan['runs'].items()}}))

def patched():
 h=helper(CYCLE);plan=h.load_plan(CYCLE);proof=read(CYCLE/'reuse-frozen-proof.json')
 assert proof['boot_id']==boot() and proof['previous_stop_sha256']==sha(PREVIOUS/'rlt-stopped.json')
 originals={row['original_run']:row['checkpoint'] for row in plan['runs'].values()}
 checked_by_path={row['checkpoint']['path']:row['checkpoint'] for row in plan['runs'].values()}
 h.latest_complete=lambda run:Path(originals[str(run)]['path'])
 h.inspect_checkpoint=lambda cp,cfg,repo:guard_checkpoint(h,checked_by_path[str(cp)],cfg,repo)
 # Register only the helper's proven descendant snapshots before pidfd signals.
 original_tree=h.process_tree;anchors={}
 def tree(roots):
  result=original_tree(roots);anchors.update(result);return result
 def safe_signal(pid,sig):
  assert sig in (signal.SIGTERM,signal.SIGKILL) and pid in anchors
  expected=anchors[pid];assert expected['uid']==UID
  if not h.same(expected):return
  fd=pidfd_open(pid)
  try:
   if h.same(expected):pidfd_send(fd,sig)
  finally:os.close(fd)
 h.process_tree=tree;h.os=types.SimpleNamespace(**{**os.__dict__,'kill':safe_signal})
 return h

def main():
 parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','stop','resume','status'])
 parser.add_argument('--release');args=parser.parse_args()
 assert os.getuid()==UID and socket.gethostname()=='h100-gpu02';pidfd_probe()
 if args.action=='prepare':prepare();return
 h=patched()
 if args.action=='stop':result=h.stop(CYCLE)
 elif args.action=='resume':
  assert args.release;result=h.resume(CYCLE,args.release)
 else:result=h.status(CYCLE)
 print(json.dumps(result))

if __name__=='__main__':main()
