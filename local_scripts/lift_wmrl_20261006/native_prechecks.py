"""Reuse the proven finite native phase for one independent lift_pot baseline.

The existing phase runner retains token/namespace cleanup and the owner's
finally. Only the allowed phase list changes; no Rynn/bell phase is scheduled.
"""
import hashlib
import importlib.util
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def install(owner, plan):
    assert [(r['key'], r['kind']) for r in plan['prechecks']] == [('native_lift32', 'native')]
    source = Path(plan['native_prechecks_donor'])
    assert sha(source) == plan['source_sha256'][str(source)]
    spec = importlib.util.spec_from_file_location('lift_reused_native_precheck', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.ORDER == [('native_rynn32', 'native'), ('rynn32', 'rynn_binary'),
                            ('native_bell32', 'native'), ('bell_rm', 'bell_reward')]
    module.ORDER = [('native_lift32', 'native')]
    return module.install(owner, plan)
