"""Let the already-running early-only lanes finish, then return SZ1 to RLT."""
import sys,time,traceback
from pathlib import Path
S=Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1');sys.path.insert(0,str(S/'ops'))
from common import read,save,proc,same,gpus
D=Path(sys.argv[1]);p=read(D/'prepared.json');old=Path(p['old_transaction']);me=proc(__import__('os').getpid())
save(D/'sequence-identity.json',me)
try:
 while not(D/'enqueued.json').exists():time.sleep(1)
 save(D/'status.json',dict(time=time.time(),phase='EARLY_ONLY_EXISTING_LANES'))
 while any(same(v)for v in p['lanes'].values()):time.sleep(10)
 rows=[read(old/'early'/f'lane-g{g}-released.json')for g in(4,5)]
 assert all(r['completed']for r in rows)
 assert all(not gpus()[g]['processes']for g in(4,5))
 save(old/'early-complete.json',dict(time=time.time(),receipts=rows))
 save(D/'released.json',dict(time=time.time(),completed=True,cards_clear=True,identity=me,c20_moved_to='sz2 4/5'))
 save(D/'status.json',dict(time=time.time(),phase='COMPLETE_RLT_HANDOFF'))
except BaseException:
 save(D/'error.json',dict(time=time.time(),error=traceback.format_exc()))
 raise
