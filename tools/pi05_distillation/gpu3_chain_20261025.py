"""One GPU: existing Rand queue, existing Dojo queue, then owner-managed RLT."""
import os,sys,time,subprocess,signal,json
from pathlib import Path
sys.path.insert(0,'/data/chenyiteng/deployment-20261008/bc-signal-tau-v1/ops')
from common import read,save,proc,same,exact_signal,gpus
D=Path(sys.argv[1]);me=proc(os.getpid());save(D/'identity.json',me)
signal.signal(signal.SIGTERM,lambda *_:sys.exit(143))
while not(D/'registered.json').exists():time.sleep(.2)
assert read(D/'registered.json')['identity']==me
for name,entry in [('rand-chunk-sz1-g3-r4','tools/pi05_distillation/rand_chunk_queue.py'),('dojo-chunk-sz1-g3-after-rand','tools/dojo_chunk_eval/scan_queue.py')]:
 d=D.parent/name;assert not gpus()[3]['processes']
 save(D/'status.json',{'time':time.time(),'phase':'RUNNING','deployment':str(d)})
 with(d/'queue.log').open('x')as log:
  child=subprocess.Popen([sys.executable,'-u','-B',str(d/'source'/entry),str(d)],cwd=d/'source',env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 while not(d/'identity.json').exists():assert child.poll()is None;time.sleep(.1)
 identity=read(d/'identity.json');save(d/'registered.json',{'time':time.time(),'identity':identity,'gpu':3,'chain':str(D)})
 save(D/'child.json',{'identity':identity,'deployment':str(d)})
 rc=child.wait();assert not gpus()[3]['processes'],'Owned queue exited without a clear card'
 save(D/(name+'-finished.json'),{'time':time.time(),'returncode':rc,'status':read(d/'status.json')})
save(D/'status.json',{'time':time.time(),'phase':'COMPLETE','next':'RLT'})
save(D/'released.json',{'time':time.time(),'identity':me,'cards_clear':True})
