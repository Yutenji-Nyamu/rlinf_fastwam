"""Reuse the successful Clean driver and its namespace-scoped Ray cleanup."""
import importlib.util,os,sys,resource
from pathlib import Path
import json
d=Path(os.environ['GRPO_ENTROPY_DEPLOYMENT']);p=json.loads((d/'plan.json').read_text())
soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE,(min(4096,hard),hard))
spec=importlib.util.spec_from_file_location('accepted_clean_ops',p['legacy_ops'])
ops=importlib.util.module_from_spec(spec);spec.loader.exec_module(ops)
ops.RUN=Path(p['run']);ops.ROOT=Path(p['training_source']);ops.NAMESPACE=p['namespace']
# The accepted entry expects argv[1] == formal_driver; preserve its argument contract.
sys.argv=[sys.argv[0],'formal_driver',*sys.argv[1:]]
ops.formal_driver()
