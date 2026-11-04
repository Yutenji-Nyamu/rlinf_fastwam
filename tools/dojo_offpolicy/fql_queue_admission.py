"""Run the unchanged accepted FQL queue beside recorded tiny Isaac G contexts."""
import runpy
import sys
from pathlib import Path

deployment, source = Path(sys.argv[1]), Path(sys.argv[2])
sys.argv = [str(source/'tools/pi05_distillation/single_formal_queue.py'),str(deployment)]
sys.path.insert(0,str(source/'tools/pi05_distillation'))
import run_queue as queue
from common import admission_card

raw_gpus = queue.gpus

def admitted_gpus():
    rules = queue.read(deployment/'formal-plan.json').get('gpu_admission',{})
    return {g:admission_card(card,rules.get(str(g),{})) for g,card in raw_gpus().items()}

queue.gpus = admitted_gpus
runpy.run_path(sys.argv[0],run_name='__main__')
