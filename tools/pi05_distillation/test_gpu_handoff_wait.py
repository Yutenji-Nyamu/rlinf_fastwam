"""CPU-only regression: a foreign graphical context may outlive our evaluation."""
import ast
from pathlib import Path
import pytest

def load_wait(states, timeout=10):
 source=Path(__file__).with_name('run_queue.py').read_text()
 fn=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='wait_for_cards')
 class Clock:
  value=0
  def monotonic(self):return self.value
  def sleep(self,n):self.value+=n
 clock=Clock();events=[];snapshots=iter(states);last=states[-1]
 def snapshot():return next(snapshots,last)
 ns={'time':clock,'gpus':snapshot,'CARDS':(4,5),'event':lambda *a,**k:events.append((a,k))}
 exec(compile(ast.Module(body=[fn],type_ignores=[]),'<queue-wait>','exec'),ns)
 return ns['wait_for_cards'],clock,events

def test_foreign_context_exits_then_training_can_continue():
 busy={4:{'processes':[{'pid':12,'type':'G'}]},5:{'processes':[]}}
 clear={4:{'processes':[]},5:{'processes':[]}}
 wait,clock,events=load_wait([busy,busy,clear]);wait(10)
 assert clock.value==4 and len(events)==1

def test_unidentified_context_is_never_killed():
 busy={4:{'processes':[{'pid':12,'type':'C'}]},5:{'processes':[]}}
 wait,clock,events=load_wait([busy])
 with pytest.raises(TimeoutError):wait(4)
 assert clock.value==4

def test_empty_pair_has_no_wait():
 wait,clock,events=load_wait([{4:{'processes':[]},5:{'processes':[]}}]);wait(10)
 assert clock.value==0 and events==[]
