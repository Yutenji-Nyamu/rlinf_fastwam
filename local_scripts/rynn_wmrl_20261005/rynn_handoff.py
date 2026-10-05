"""Exact B16-owner -> Rynn-owner handoff; reuse the reviewed RLT adoption.

No training source or old receipt is modified. The root supplies a fully
prepared plan; all validation runs before the exact previous owner is signaled.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

S = Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003')
D = S / 'rynn-control-v1'
CODE = D / 'code'
OLD_OWNER = S / 'runs/formal-b16-v1'
NEW_OWNER = S / 'runs/rynn-success-v1'
OLD_IDENTITY = dict(pid=394537, uid=20001, start=707417273)
OLD_WRAPPER = S / 'formal-b16-control-v1/code/batch16_formal_owner.py'
DONOR = S / 'formal-b16-control-v1/code/batch16_handoff.py'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def adapted_main_source(source):
    replacements = [
        ("assert Path(__file__).resolve() == (CODE / 'batch16_handoff.py').resolve()",
         "assert Path(__file__).resolve() == (CODE / 'rynn_handoff.py').resolve()"),
        ("state = D / 'prepared'", "state = D / 'handoff'"),
        ("support = OLD_WRAPPER.parent", "support = Path(old['base_formal_wrapper']).parent"),
    ]
    for before, after in replacements:
        assert source.count(before) == 1, 'Frozen handoff source differs: ' + before
        source = source.replace(before, after, 1)
    assert source.count("CODE / 'batch16_formal_owner.py'") == 2
    return source.replace("CODE / 'batch16_formal_owner.py'", "CODE / 'rynn_formal_owner.py'")


def install():
    previous = read(OLD_OWNER / 'owner-plan.json')
    assert previous['source_sha256'][str(DONOR)] == sha(DONOR)
    assert previous['owner_script_sha256'] == sha(OLD_WRAPPER)
    H = load('rynn_exact_batch16_handoff', DONOR)
    H.S, H.D, H.CODE = S, D, CODE
    H.OLD_OWNER, H.NEW_OWNER, H.OLD_IDENTITY, H.OLD_WRAPPER = OLD_OWNER, NEW_OWNER, OLD_IDENTITY, OLD_WRAPPER
    H.__file__ = str(CODE / 'rynn_handoff.py')

    def ready_manifest(path):
        value = read(path)
        assert value['schema'] == 1 and value['all_cpu_tests_passed'] is True
        assert value['wm_batch_size'] == 16
        for name in ('rynn_handoff.py', 'rynn_formal_owner.py', 'rynn_gate_client.py'):
            target = CODE / name
            assert value['source_sha256'].get(str(target)) == sha(target), 'Unready source: ' + name
        for name, digest in value['source_sha256'].items():
            assert sha(name) == digest, 'Ready source changed: ' + name
        return value

    def make_inputs(old, ready, resume_path, state, template_path=None):
        assert template_path is not None, 'Reviewed plan template is required'
        plan = read(template_path)
        assert plan['owner_dir'] == str(NEW_OWNER)
        assert plan['repo'] == str(S / 'rlinf-rynn-v1')
        assert plan['lifecycle_path'] == old['lifecycle_path']
        assert plan['lifecycle_module'] == old['lifecycle_module']
        assert plan['resume_checkpoint'] == dict(receipt=str(resume_path), sha256=sha(resume_path))
        assert plan['base_formal_wrapper'] == old['base_formal_wrapper']
        assert plan['base_owner_module'] == old['base_owner_module']
        assert plan['physical_gpus'] == [4, 5, 6, 7]
        for path, digest in ready['source_sha256'].items():
            assert plan['source_sha256'].get(path) == digest, 'Ready source omitted from launch plan'
        F = load('rynn_checkpoint_preflight', CODE / 'rynn_formal_owner.py')
        resumed = F.verify_resume(plan)
        assert resumed['completed_step'] >= 70, 'Do not resume an older B16 checkpoint'
        module = F.install(plan)
        module.validate(plan)
        H.record(state / 'preflight-plan.json', plan)
        H.record(state / 'preflight-passed.json', dict(time=module.H.now(), old_owner_untouched=True,
            plan_sha256=sha(state / 'preflight-plan.json'), resume_receipt_sha256=sha(resume_path),
            completed_step=resumed['completed_step'], graphics_scope_reused=True,
            selected_reward='rynn_success', max_steps=200))
        return plan

    H.ready_manifest, H.make_inputs = ready_manifest, make_inputs
    raw = DONOR.read_text()
    node = next(n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    source = ast.get_source_segment(raw, node) + '\n'
    changed = adapted_main_source(source)
    exec(compile(changed, str(CODE / 'rynn_handoff.py') + ':adapted-exact-handoff', 'exec'), H.__dict__)
    return H


def main():
    install().main()


if __name__ == '__main__':
    main()
