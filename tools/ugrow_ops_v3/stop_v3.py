import importlib.util,json
from pathlib import Path
S=Path('/data/chenyiteng/deployment-20261006/ugrow-rlt-g5-formal-v3')
sp=importlib.util.spec_from_file_location('returned_lease',S/'tools/returned_rlt_lease.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
p=json.loads((S/'rlt/plan.json').read_text());assert p['gpu']==5 and p['lane']=='rlt'
proof=m.stop(Path(p['lease_dir']))
print(json.dumps(proof))
