"""CPU regression: optional memory timeout must retain strict GPU checks."""
import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace


def check(path):
    tree = ast.parse(Path(path).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'resource_snapshot')
    rows = []
    managed = []
    query = ['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_gpu_memory', '--format=csv,noheader']

    def timed_out(args, **kwargs):
        assert args == query and kwargs == {'text': True, 'timeout': 20}
        raise subprocess.TimeoutExpired(args, 20)

    ns = {'Path': Path, 'json': json, 'GPUS': [4, 5],
          'H': SimpleNamespace(now=lambda: 'fixture', gpu_processes=lambda _: rows),
          'subprocess': SimpleNamespace(check_output=timed_out, TimeoutExpired=subprocess.TimeoutExpired)}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), ns)
    catalog = SimpleNamespace(scan=lambda: None, live=lambda *args: managed)
    with tempfile.TemporaryDirectory() as tmp:
        plan = {'owner_dir': tmp, 'services': [{'key': 'wm5', 'physical_gpu': 5}]}
        result = ns['resource_snapshot'](plan, catalog, 'formal')
        assert result['compute_memory_csv'] is None
        assert result['compute_memory_error']['type'] == 'TimeoutExpired'
        assert json.loads((Path(tmp) / 'resources.jsonl').read_text())['compute_memory_error']
        managed.append({'pid': 999999999})
        rows.append({'pid': 999999999, 'gpu': 0})
        try:
            ns['resource_snapshot'](plan, catalog, 'formal')
        except AssertionError as exc:
            assert 'outside' in str(exc)
        else:
            raise AssertionError('Timeout bypassed the wrong-GPU guard')
        rows[0]['gpu'] = 4
        try:
            ns['resource_snapshot'](plan, catalog, 'formal')
        except AssertionError as exc:
            assert 'different physical GPU' in str(exc)
        else:
            raise AssertionError('Timeout bypassed the service-GPU guard')
    print(json.dumps({'optional_timeout_logged': True, 'own_wrong_gpu_rejected': True,
                      'service_wrong_gpu_rejected': True, 'cpu_only': True}))


if __name__ == '__main__':
    check(sys.argv[1])
