"""Read-only CPU preflight while waiting for returned RLT checkpoints."""
import copy,json,os,time,sys
from pathlib import Path
ROOT=Path('/data/chenyiteng/projects/expo-ft-sz2-20261001/eval10-continuation-20261003')
sys.path.insert(0,str(ROOT))
import migrate_eval10 as m
def main():
 assert os.getuid()==20001 and os.environ.get('CUDA_VISIBLE_DEVICES')==''
 import torch
 torch.set_num_threads(4);assert not torch.cuda.is_initialized()
 old=m.read(m.ROOT/'inputs.json');resolved=m.read(m.ROOT/'run/resolved.json')
 assert resolved['inputs']==old and resolved['contract']['inputs_sha256']==m.sha(m.ROOT/'inputs.json')
 assert m.jhash(resolved['contract'])==resolved['contract_sha256']
 for name,h in old['port_source_manifest'].items():assert m.sha(m.ROOT/'source'/name)==h,name
 original=(m.ROOT/'source'/m.DRIVER).read_bytes();patch=(ROOT/'train_expo_formal.py').read_bytes()
 expected=original.replace(b"evaluation.get('every_episodes') != 25",b"evaluation.get('every_episodes') != 10").replace(b'Fixed initial/25-episode/final evaluation contract differs',b'Fixed initial/10-episode/final evaluation contract differs')
 assert patch==expected and patch!=original
 new=copy.deepcopy(old);new['evaluation']['every_episodes']=10;new['port_source_manifest'][m.DRIVER]=m.sha(ROOT/'train_expo_formal.py');m.check_input_diff(old,new)
 digest=m.load_digest(torch);rows={};index=m.read(m.ROOT/'replay/index.json');stopped=m.read(m.ROOT/'run/stopped.json');receipt=m.read(m.ROOT/'run/checkpoint.json')
 for name in ['checkpoint-latest.pt','checkpoint-last1.pt']:
  p=m.ROOT/'run'/name;before=p.stat();saved=torch.load(p,map_location='cpu',weights_only=False)
  assert saved['version']==4 and saved['contract']==resolved['contract'] and saved['contract_sha256']==resolved['contract_sha256']
  hashes={k:digest(saved[k]) for k in m.PAYLOAD};assert hashes==saved['hashes']
  if name=='checkpoint-latest.pt':
   assert saved['cadence']==stopped['cadence']==receipt['cadence'] and hashes==receipt['hashes']
   assert index['online_entries']==saved['replay']['online_entries']
   assert len(index['online_entries'])==saved['cadence']['counters']['episodes_completed']
   assert saved['progress']['stopped_episodes']==0
  rows[name]={'bytes':before.st_size,'hashes':hashes,'counters':saved['cadence']['counters'],'progress':saved['progress']}
  del saved
  after=p.stat();assert (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
 assert not torch.cuda.is_initialized()
 result={'ok':True,'time':time.time(),'read_only_training_state':True,'cpu_only':True,'source_manifest_count':len(old['port_source_manifest']),'driver_patch_sha256':m.sha(ROOT/'train_expo_formal.py'),'checkpoints':rows}
 with (ROOT/'migration-preflight.json').open('x') as f:json.dump(result,f,indent=2)
 print(json.dumps(result))
if __name__=='__main__':main()
