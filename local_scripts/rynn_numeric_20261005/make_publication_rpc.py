"""Build an isolated numeric-reward audit release; never execute SSH or Git.

Run only after the three topical documents, HANDOFF, and sanitized light receipt
are final. Reuses the already reviewed publication stage/push implementation.
Generated RPC files are transport artifacts and are deliberately not published.
"""
import ast
import base64
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
PRIOR = 'e9e9faf4826695d552f19fe50366b2dc1969b2bd'
SOURCE_BRANCH = 'codex/rynnvalue-wmrl-release-20261005'
BRANCH = 'codex/rynn-numeric-reward-audit-20261005'
OPS = 'local_scripts/rynn_numeric_20261005/'
DOC = 'docs/world-model/robotwin_pipeline_20261003/'
EVIDENCE = 'docs/world-model/publication_rynn_numeric_20261005/'
MANIFEST = EVIDENCE + 'source_manifest.json'
LIGHT_RECEIPT = EVIDENCE + 'light_receipt.json'
SOURCE_FILES = [OPS + name for name in (
    'rynn_numeric_owner.py', 'rynn_numeric_probe.py',
    'test_numeric_owner.py', 'test_numeric_probe.py',
    'prepare_dataset.py', 'plot_results.py', 'repair_return.py', 'repair_return_v2.py',
    'graphics_scope_runtime.py', 'test_named_scope.py', 'build_return_v2.py', 'complete_return_repair.py',
    'build_stage_rpc.py', 'build_owner.py', 'make_publication_rpc.py')]
DOC_FILES = [DOC + name for name in (
    'rynn_numeric_probe_20261005.md', 'robometer_audit_20261005.md',
    'click_bell_reward_audit_20261005.md')]
PINNED_SOURCE_NAMES = {'rynn_numeric_owner.py', 'rynn_numeric_probe.py',
    'test_numeric_owner.py', 'test_numeric_probe.py', 'prepare_dataset.py'}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def previous_generator():
    path = ROOT / 'local_scripts/rynn_wmrl_20261005/make_publication_rpc.py'
    spec = importlib.util.spec_from_file_location('previous_rynn_publication', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


IDENTITIES = r'''def identities():
    assert os.getuid()==20001 and socket.gethostname()=='h100-gpu01'
    assert P['prior_commit']=='e9e9faf4826695d552f19fe50366b2dc1969b2bd'
    assert P['branch']=='codex/rynn-numeric-reward-audit-20261005'
    for path in (S,BASE,R,N):
        assert path.is_dir() and path.stat().st_uid==20001 and path.resolve().is_relative_to(S.resolve())
    assert W.resolve().is_relative_to((S/'publication').resolve())
    assert E.resolve().is_relative_to((S/'publication').resolve())
    assert git(BASE,'remote','get-url','personal').decode().strip()=='git@github.com:Yutenji-Nyamu/rlinf_fastwam.git'
'''

SNAPSHOT = r'''def runtime_snapshot():
    plan_path=N/'run/owner-plan.json'
    tests_path=N/'prepared/cpu-tests.json'
    plan=read(plan_path);tests=read(tests_path)
    frozen=dict(plan['source_sha256']);frozen.update(tests['source_sha256'])
    selected={p:sha for p,sha in frozen.items() if Path(p).is_relative_to(N/'code')}
    for path,sha in selected.items():
        assert digest(Path(path).read_bytes())==sha,'Frozen numeric source changed: '+path
    head=git(R,'rev-parse','HEAD').decode().strip()
    assert head==EXPECTED_RUNTIME_HEAD,'Runtime checkout HEAD changed'
    return {'head':head,'status_sha256':digest(git(R,'status','--porcelain','--untracked-files=all')),
        'diff_sha256':digest(git(R,'diff','HEAD','--')),
        'owner_plan_sha256':digest(plan_path.read_bytes()),
        'cpu_tests_sha256':digest(tests_path.read_bytes()),'frozen_numeric_sources':selected}
'''

BINDINGS = r'''def check_source_bindings():
    plan=read(N/'run/owner-plan.json');tests=read(N/'prepared/cpu-tests.json')
    assert plan['mode']=='rynn-single-gpu-diagnostic' and plan['physical_gpus']==[4]
    assert plan['owner_dir']==str(N/'run')
    assert tests['all_cpu_tests_passed'] is True
    assert plan['source_sha256'][str(N/'prepared/cpu-tests.json')]==digest((N/'prepared/cpu-tests.json').read_bytes())
    frozen=dict(plan['source_sha256']);frozen.update(tests['source_sha256'])
    for name,path in P['numeric_bindings'].items():
        assert Path(path).is_relative_to(N/'code')
        assert frozen.get(path)==P['files'][name]['sha256'],'Numeric source differs from tested/frozen file: '+name
        assert digest(Path(path).read_bytes())==P['files'][name]['sha256']
    # A publication is evidence preservation, not permission to relaunch training.
    final_path=N/'run/final.json'
    assert final_path.is_file(),'Wait for the small numeric probe to finish before publishing its receipt'
    assert digest(final_path.read_bytes())==P['numeric_final_sha256'],'Numeric final receipt differs from reviewed evidence'
    for name,path in P['postrun_bindings'].items():
        assert Path(path).parent in (N/'postrun-code',N/'postrun-code-v2')
        assert digest(Path(path).read_bytes())==P['files'][name]['sha256']
    repaired=N/'rlt-return-repair-v2/repaired.json'
    assert digest(repaired.read_bytes())==P['return_repair_sha256']
'''

RESTAGE = r'''
def main():
    identities()
    assert W.is_dir() and E.is_dir() and not (E/'staged.json').exists()
    failure=read(E/'failed.json')
    assert failure['error_type']=='CalledProcessError' and "'diff', '--cached', '--check'" in failure['error']
    assert git(W,'rev-parse','HEAD').decode().strip()==P['prior_commit']
    assert git(W,'branch','--show-current').decode().strip()==P['branch'] and not remote_head()
    previous=read(W/P['manifest_path'])
    for name,row in previous['files'].items():
        assert digest((W/name).read_bytes())==row['sha256']
        assert digest(git(W,'show',':'+name))==row['sha256']
    before=runtime_snapshot();check_source_bindings();contents=decoded_files()
    for name,data in contents.items():
        path=W.joinpath(*relative(name).parts);assert path.resolve().is_relative_to(W.resolve()) and not path.is_symlink()
        path.write_bytes(data)
    git(W,'add','-f','--',*sorted(contents))
    names=[x.decode() for x in git(W,'diff','--cached','--name-only','-z').split(b'\0') if x]
    assert names and set(names).issubset(contents)
    for name,data in contents.items():assert git(W,'show',':'+name)==data
    assert not git(W,'diff','--name-only').strip() and not git(W,'ls-files','--others','--exclude-standard').strip()
    # Preserve byte-identical Matplotlib SVG and the tested runtime EOF. All
    # substantive source/document whitespace remains checked.
    git(W,'-c','core.whitespace=cr-at-eol,-blank-at-eof','diff','--cached','--check','--','.',
        ':(exclude)docs/world-model/publication_rynn_numeric_20261005/rynn-numeric-probe.svg')
    diff=git(W,'diff','--cached','--binary');stat=git(W,'diff','--cached','--stat')
    after=runtime_snapshot();assert after==before
    receipt={'time':datetime.now(timezone.utc).isoformat(),'status':'staged_for_review',
        'prior_commit':P['prior_commit'],'branch':P['branch'],'manifest_sha256':P['manifest_sha256'],
        'files':{name:{'bytes':len(raw),'sha256':digest(raw)} for name,raw in contents.items()},
        'staged_names':names,'staged_diff_sha256':digest(diff),'runtime_before':before,'runtime_after':after,
        'committed':False,'pushed':False,'runtime_unchanged':True,
        'format_only_exception':'Generated SVG trailing path spaces and frozen runtime extra EOF newline are preserved byte-for-byte.'}
    record(E/'staged.json',receipt);(E/'staged.diff').write_bytes(diff);(E/'staged.stat.txt').write_bytes(stat)
    print(json.dumps({'status':'staged_for_review','files':len(contents),'diff_sha256':digest(diff),'runtime_unchanged':True}))
    print(stat.decode())
if __name__=='__main__':main()
'''


def replace_function(source, name, next_name, replacement):
    source, count = re.subn(r'^def ' + name + r'\(.*?(?=^def ' + next_name + r'\()',
                            lambda _: replacement, source, count=1, flags=re.M | re.S)
    assert count == 1, 'Previous publication function changed: ' + name
    return source


def main():
    prior = previous_generator()
    published = json.loads((ROOT / 'docs/world-model/publication_rynn_20261005/published.json').read_text())
    assert published['commit'] == PRIOR and published['verified_remote_commit'] == PRIOR and published['pushed'] is True
    receipt_path = ROOT / LIGHT_RECEIPT
    receipt = json.loads(receipt_path.read_text(encoding='utf-8-sig'))
    assert isinstance(receipt, dict)
    prior.evidence_shape(receipt)
    # Root copies the exact remote final.json SHA into this small sanitized receipt.
    final_sha = receipt['numeric_final_sha256']
    assert re.fullmatch(r'[0-9a-f]{64}', final_sha)
    files, bindings, postrun = {}, {}, {}
    for name in SOURCE_FILES + DOC_FILES + ['HANDOFF.md', LIGHT_RECEIPT, EVIDENCE+'rynn-numeric-probe.svg']:
        prior.relative(name)
        path = ROOT / name
        assert path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(ROOT)
        data = path.read_bytes()
        assert 0 < len(data) <= 262144
        text = prior.scan(name, data)
        if name.endswith('.py'):
            ast.parse(text, filename=name)
        files[name] = dict(source_local=name, bytes=len(data), sha256=digest(data),
                           base64=base64.b64encode(data).decode())
        if Path(name).name in PINNED_SOURCE_NAMES:
            bindings[name] = '/data/chenyiteng/projects/opendw-robotwin-smoke-20261003/rynn-numeric-v1/code/' + Path(name).name
        if Path(name).name in ('plot_results.py','repair_return.py'):
            postrun[name] = '/data/chenyiteng/projects/opendw-robotwin-smoke-20261003/rynn-numeric-v1/postrun-code/' + Path(name).name
        if Path(name).name in ('repair_return_v2.py','graphics_scope_runtime.py','test_named_scope.py','complete_return_repair.py'):
            postrun[name] = '/data/chenyiteng/projects/opendw-robotwin-smoke-20261003/rynn-numeric-v1/postrun-code-v2/' + Path(name).name
    manifest = dict(schema_version=1, created_time_utc=datetime.now(timezone.utc).isoformat(),
        prior_remote_commit=PRIOR, source_branch=SOURCE_BRANCH, branch=BRANCH,
        purpose='Offline Rynn numeric-reward probe and Robometer/click_bell reward research',
        files={name: {key: value for key, value in row.items() if key != 'base64'} for name, row in files.items()},
        numeric_bindings=bindings, numeric_final_sha256=final_sha, postrun_bindings=postrun,
        return_repair_sha256=receipt['return_repair_sha256'],
        execution_evidence=LIGHT_RECEIPT,
        execution_claim='Only the listed light receipt establishes execution results; publication does not establish formal WMRL success.',
        excluded=['runtime checkout/index mutation', 'one-shot remote deployment and discovery scripts',
                  'generated RPC payloads', 'credentials/full environments', 'raw logs/media',
                  'weights/checkpoints/datasets', 'unrelated local dirty files'],
        handoff_notice='Explicitly included as the current local routing snapshot; other local dirty files are excluded.',
        self_hash_rule='Manifest excludes itself; generated RPC payload pins its exact bytes.')
    raw = (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode()
    prior.scan(MANIFEST, raw)
    (ROOT / MANIFEST).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / MANIFEST).write_bytes(raw)
    files[MANIFEST] = dict(source_local=MANIFEST, bytes=len(raw), sha256=digest(raw), base64=base64.b64encode(raw).decode())
    payload = dict(prior_commit=PRIOR, source_branch=SOURCE_BRANCH, branch=BRANCH, files=files,
        secret_patterns=prior.SECRET_PATTERNS, manifest_path=MANIFEST, manifest_sha256=digest(raw),
        numeric_bindings=bindings, numeric_final_sha256=final_sha, postrun_bindings=postrun,
        return_repair_sha256=receipt['return_repair_sha256'])
    common = prior.REMOTE_COMMON.replace("BASE=S/'publication/opendw-v2'", "BASE=S/'publication/rynn-success-release-v1'")
    common = common.replace("W=S/'publication/rynn-success-release-v1'", "W=S/'publication/rynn-numeric-release-v1'")
    common = common.replace("E=S/'publication/rynn-success-release-20261005-v1'", "E=S/'publication/rynn-numeric-release-20261005-v1'")
    common = common.replace("ND=S/'rynn-diagnosis-v2'", "ND=S/'rynn-diagnosis-v2'\nN=S/'rynn-numeric-v1'")
    common = replace_function(common, 'identities', 'runtime_snapshot', IDENTITIES)
    common = replace_function(common, 'runtime_snapshot', 'remote_head', SNAPSHOT)
    common = replace_function(common, 'check_source_bindings', 'staged_contract', BINDINGS)
    stage = prior.REMOTE_STAGE
    push = prior.REMOTE_PUSH.replace('Add RynnValue reward adapters and semantic diagnostics',
                                     'Audit Rynn numeric rewards and alternative success models')
    header = '# Generated isolated numeric-reward audit publication.\nPAYLOAD = ' + repr(
        base64.b64encode(json.dumps(payload, ensure_ascii=False).encode()).decode()) + '\n'
    outputs = {}
    for name, body in (('publication_stage_remote.py', stage), ('publication_restage_remote.py', RESTAGE), ('publication_push_remote.py', push)):
        source = header + common + body
        ast.parse(source, filename=name)
        path = Path(__file__).with_name(name)
        path.write_text(source, encoding='utf-8', newline='\n')
        outputs[name] = dict(path=str(path), bytes=path.stat().st_size, sha256=digest(path.read_bytes()))
    print(json.dumps(dict(generator_only=True, files=len(files), numeric_bindings=len(bindings),
        scripts=outputs, manifest=str(ROOT / MANIFEST), ssh=False, committed=False, pushed=False)))


if __name__ == '__main__':
    main()
