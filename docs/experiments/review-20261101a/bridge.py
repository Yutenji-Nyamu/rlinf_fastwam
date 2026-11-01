"""Use the accepted one-card RLT lifecycle while its sibling finishes collection."""
import json, os, sys, time
from pathlib import Path
S=Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1')
D=Path('/home/chenyiteng/deployment-20261101/gad-k4-v1')
B=D/'g7-collection-wait-rlt'
sys.path.insert(0,str(S/'ops'))
from common import read,save,proc,same
from priority_pair_handoff import handoff

while not (B/'armed.json').exists():time.sleep(.2)
lane=read(D/'signal-recovery/lane-g6-identity.json')
while same(lane) and not (D/'signal-recovery/lane-g6-released.json').exists():time.sleep(.5)
# Existing handoff stops only the exact 7-card RLT driver, preserves its latest
# complete checkpoint, and returns ownership to the already-running GAD queue.
queue=read(D/'pair67/identity.json')
receipt=handoff(B,[7],queue)
plan=read(S/'plan.json')
assert plan['slots']['7']['external_identity']['pid']==queue['pid']
plan['slots']['7']['external_release']=str(D/'pair67/released.json')
save(S/'plan.json',plan)
save(B/'bridge-complete.json',dict(time=time.time(),receipt=receipt))
