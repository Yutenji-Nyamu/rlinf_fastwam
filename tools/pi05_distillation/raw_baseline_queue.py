"""Run paired raw N1/N10 baselines through the accepted multitask evaluator."""
import json,os,sys,time,traceback,shutil
from pathlib import Path
D=Path(sys.argv[1]);prepared=json.loads((D/'prepared.json').read_text());R=Path(prepared['source']);O=Path(prepared['output'])
sys.path.insert(0,str(R/'tools/pi05_distillation'))
from multitask_eval import sweep
from common import save,proc,gpus
def event(phase,**kw):save(D/'status.json',dict(time=time.time(),phase=phase,**kw))
save(D/'identity.json',proc(os.getpid()))
try:
 assert all(not gpus()[g]['processes']for g in prepared['physical_gpus'])
 for n in [1,10]:
  event('EVALUATING',condition=f'raw-n{n}',gpus=prepared['physical_gpus'])
  summary=sweep(D,f'raw-n{n}',steps=n,method='endpoint')
  target=Path(prepared['baseline_source']);target.mkdir(exist_ok=True)
  src=O/f'summary-raw-n{n}.json';dst=target/src.name
  assert not dst.exists(),str(dst)
  shutil.copyfile(src,dst)
  event('POINT_COMPLETE',condition=f'raw-n{n}',successes=summary['successes'],episodes=summary['episodes'],elapsed_seconds=summary['elapsed_seconds'])
 event('COMPLETE',conditions=['raw-n1','raw-n10'])
except BaseException:
 event('FAILED',error=traceback.format_exc());raise
finally:
 save(D/'released.json',dict(time=time.time(),cards_clear=all(not gpus()[g]['processes']for g in prepared['physical_gpus'])))
