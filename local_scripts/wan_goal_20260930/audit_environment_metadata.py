"""Record final versions and exact reviewed dependency warnings before GPU smoke.

Does not call pip-check success a requirement: upstream intentionally overrides
older packages' torch/protobuf metadata. Unknown warnings fail this audit.
Runtime compatibility is still tested by the separate GPU smoke.
"""
import argparse
from importlib import metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('--kind', choices=['oft', 'pi05'], required=True)
p.add_argument('--imports', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
root = Path('/data/chenyiteng/projects/wan-goal-sz3')
assert Path(sys.prefix).resolve() == (root / 'envs' / (a.kind + '-wan')).resolve()
imports = json.loads(a.imports.read_text())
assert imports['kind'] == a.kind and imports['ok'] is True
assert all(row['ok'] for row in imports['imports'])
uv = shutil.which('uv')
assert uv, 'Use the same installed uv executable as preparation'
check = subprocess.run([uv, '--no-config', 'pip', 'check', '--python', sys.executable],
                       capture_output=True, text=True, timeout=60)
raw = check.stdout + check.stderr
diagnostics = re.findall(r'The package `([^`]+)` requires `([^`]+)`, but `([^`]+)` is installed', raw)
assert check.returncode in (0, 1)
assert ('Found ' not in raw) or diagnostics
common = {'torch': '2.11.0+cu130', 'torchvision': '0.26.0+cu130',
          'mujoco': '3.3.7'}
if a.kind == 'oft':
    common['torchaudio'] = '2.11.0+cu130'
for name, version in common.items():
    assert metadata.version(name) == version, (name, metadata.version(name))
allowed = {
 'pi05': {
    ('rlinf-openpi', 'torch'): 'Pinned RLinf pyproject torch override.',
    ('lerobot', 'torch'): 'Pinned RLinf pyproject torch override.',
    ('lerobot', 'torchvision'): 'Pinned RLinf pyproject torchvision override.',
    ('lerobot', 'torchcodec'): 'Pinned RLinf pyproject torchcodec override.',
    ('transformers', 'tokenizers'): 'Official install.sh 2416-2421 explicitly selects OpenPI fork files and tokenizers<0.22.',
 },
 'oft': {
    ('openvla-oft', 'torch'): 'Pinned RLinf pyproject torch override.',
    ('openvla-oft', 'torchvision'): 'Pinned RLinf pyproject torchvision override.',
    ('openvla-oft', 'torchaudio'): 'Pinned RLinf pyproject torchaudio override.',
    ('tensorflow', 'protobuf'): 'Pinned RLinf protobuf>=6.33.5,<7 for Ray. TensorFlow emits nonfatal import diagnostics; GPU smoke remains required.',
    ('tyro', 'typeguard'): 'OFT TensorFlow Addons requires older typeguard; this run uses Hydra, not the tyro CLI.',
    ('swanlab', 'wrapt'): 'OFT TensorFlow requires older wrapt; this run selects TensorBoard, not SwanLab.',
 },
}[a.kind]
rows = []
for package, requirement, installed in diagnostics:
    dependency = re.match(r'[A-Za-z0-9_.-]+', requirement).group()
    assert (package, dependency) in allowed, (package, requirement, installed)
    rows.append(dict(package=package, requirement=requirement, installed=installed,
                     assessment=allowed[(package, dependency)]))
assert len(rows) == (5 if a.kind == 'pi05' else 6), 'Reinspect changed dependency diagnostics'
names = ['torch', 'torchvision', 'torchcodec', 'flash-attn', 'transformers',
         'tokenizers', 'mujoco', 'rlinf-libero', 'diffsynth', 'deepspeed', 'ray', 'protobuf']
names += ['rlinf-openpi', 'rlinf-transformer-openpi', 'jax', 'jaxlib', 'orbax-checkpoint', 'dm-control'] if a.kind == 'pi05' else ['torchaudio', 'openvla-oft', 'tensorflow', 'tensorflow-addons', 'tyro', 'typeguard', 'swanlab', 'wrapt']
versions = {name: metadata.version(name) for name in names}
if a.kind == 'oft':
    for package, dependency in [('tensorflow-addons', 'typeguard'), ('tensorflow', 'wrapt')]:
        from packaging.requirements import Requirement
        requirements = [Requirement(x) for x in metadata.requires(package)]
        selected = [r for r in requirements if r.name == dependency]
        assert selected and all(r.specifier.contains(versions[dependency]) for r in selected)
sources = {}
for name in names:
    info = metadata.distribution(name).read_text('direct_url.json')
    if info:
        data = json.loads(info)
        if 'vcs_info' in data:
            sources[name] = data['vcs_info']
report = dict(time=time.time(), kind=a.kind, status='REVIEWED_FOR_GPU_SMOKE',
              pip_check_exit=check.returncode, pip_check_clean=False,
              versions=versions, vcs_revisions=sources, warnings=rows,
              imports_receipt=str(a.imports), imports_passed=True,
              limitation='This is a scoped readiness decision, not a clean pip check or verified GRPO/GPU execution.')
with a.output.open('x') as out:
    json.dump(report, out, indent=2); out.write('\n')
print(json.dumps(report, indent=2))
