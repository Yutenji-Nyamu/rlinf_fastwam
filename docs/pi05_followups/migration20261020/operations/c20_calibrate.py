"""Reuse accepted candidate selection; merge frozen compatible SZ1 history."""
import importlib.util,sys,json
from pathlib import Path
D=Path(__file__).resolve().parent;p=json.loads((D/'prepared.json').read_text());R=Path(p['source'])
spec=importlib.util.spec_from_file_location('accepted_prepare',R/'tools/pi05_followups/prepare.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.D=D;m.OLD=D/'templates';m.ROOTS={k:Path(v)for k,v in p['roots'].items()}
prior=m.read(D/'prior-c20-pool.json');original=m.deduplicate
m.deduplicate=lambda pools:original(pools+[prior])
m.prepare_c20()
