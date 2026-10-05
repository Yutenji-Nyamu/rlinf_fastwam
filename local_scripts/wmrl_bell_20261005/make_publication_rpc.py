"""Build an explicit bell/binary release allowlist and isolated publication RPCs.

--inventory-only creates a candidate list even while final documents are pending.
Default mode requires final documents plus a root-reviewed light receipt. It only
serializes local files; generated stage/push RPCs are never executed here.
"""
import argparse
import ast
import base64
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
S = '/data/chenyiteng/projects/opendw-robotwin-smoke-20261003'
PRIOR = '382b2698a1ca8853343e32ae7c0b688a38521933'
SOURCE_BRANCH = 'codex/rynn-numeric-reward-audit-20261005'
BRANCH = 'codex/wmrl-bell-reward-20261005'
OPS = 'local_scripts/wmrl_bell_20261005/'
BINARY = 'local_scripts/rynn_binary_20261005/'
BELL = 'local_patches/click_bell_20261005/'
DOC = 'docs/world-model/robotwin_pipeline_20261003/'
EVIDENCE = 'docs/world-model/publication_bell_20261005/'
MANIFEST = EVIDENCE + 'source_manifest.json'
ALLOWLIST = EVIDENCE + 'allowlist.json'
LIGHT = EVIDENCE + 'light_receipt.json'
V1_LIGHT = EVIDENCE + 'light_receipt_v1_launch.json'
RYNN_RESULT = EVIDENCE + 'rynn_binary_result.json'

BINARY_FILES = [BINARY + name for name in (
    'README.md', 'install_capture.py', 'native_binary_recorder.py', 'prepare_native_eval.py',
    'run_native_eval.py', 'merge_native_dataset.py', 'rynn_binary_probe.py', 'test_binary_probe.py',
    'prepare_bell_reward_samples.py', 'summarize_bell_reward_probe.py')]
BELL_FILES = [BELL + name for name in (
    'README.md', 'build_config.py', 'build_reset_data.py', 'extract_sz1_assets.py',
    'patch_click_bell.py', 'probe_reward.py', 'test_click_bell.py')]
OPS_FILES = [OPS + name for name in (
    'prepare_cycles.py', 'test_prepare_cycles.py', 'prechecks.py', 'test_prechecks.py',
    'reuse_scope_hook.py', 'prepare_formal_remote.py', 'build_stage.py', 'build_cycles_rpc.py',
    'build_formal_stage.py', 'prepare_rm_remote.py', 'validate_prepared_remote.py',
    'launch_remote.py', 'status_remote.py', 'prepare_formal_v2.py', 'prepare_cycles_v2.py',
    'build_stage_v2.py', 'result_summary_remote.py', 'make_followup_publication_rpc.py', 'make_publication_rpc.py')]
BACKFILL = [
    'docs/world-model/publication_batch16_20261004/published.json',
    'docs/world-model/publication_formal_20261004/published.json',
    'docs/world-model/publication_formal_repair_20261004/acceptance_published.json',
    'docs/world-model/publication_rynn_20261005/published.json',
    'docs/world-model/publication_rynn_numeric_20261005/published.json',
    'docs/world-model/publication_rynn_numeric_20261005/return_status_20261005_2030.json',
    'local_scripts/rynn_wmrl_20261005/owner_implementation_notes.md',
    DOC + 'publication_inventory_20261005.md']
FINAL_DOCS = [DOC + 'rynn_binary_probe_20261005.md', DOC + 'click_bell_execution_20261005.md']
GENERATED_FILES = [BELL + name for name in (
    'generated/rlinf/envs/world_model/opendw_adapter.py',
    'generated/rlinf/envs/world_model/opendw_robotwin_env.py',
    'generated/owner/opendw_multigpu_owner.py', 'generated/owner/opendw_formal_owner.py',
    'config/formal.yaml', 'config/startup_smoke.yaml')]
# The two real environment implementations also occupy the branch's canonical
# paths. Keeping the inherited Rynn env there would misrepresent this release.
SOURCE_MAP = {
    'rlinf/envs/world_model/opendw_adapter.py': BELL + 'generated/rlinf/envs/world_model/opendw_adapter.py',
    'rlinf/envs/world_model/opendw_robotwin_env.py': BELL + 'generated/rlinf/envs/world_model/opendw_robotwin_env.py',
}
FILES = BINARY_FILES + BELL_FILES + OPS_FILES + BACKFILL + FINAL_DOCS + GENERATED_FILES + list(SOURCE_MAP) + [EVIDENCE + 'README.md', LIGHT, V1_LIGHT, RYNN_RESULT]
EXCLUDED = ['references/environment files', 'generated RPC payloads', 'SSH/relay/session helpers',
    'failed relay helpers', 'credentials/full environment dictionaries/full owner plans',
    'weights/checkpoints/reset data/native videos/raw logs', 'unrelated HANDOFF and other workspace changes']


def digest(data):
    return hashlib.sha256(data).hexdigest()


def prior_generator():
    path = ROOT / 'local_scripts/rynn_wmrl_20261005/make_publication_rpc.py'
    spec = importlib.util.spec_from_file_location('reviewed_rynn_publisher', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')


def inventory(prior):
    rows, missing = [], []
    for name in FILES:
        prior.relative(name)
        local_source = SOURCE_MAP.get(name, name)
        path = ROOT / local_source
        if not path.is_file():
            missing.append(name)
            rows.append(dict(path=name, source_local=local_source, status='pending_final_source_result_or_document'))
            continue
        assert not path.is_symlink() and path.resolve().is_relative_to(ROOT)
        raw = path.read_bytes()
        assert 0 < len(raw) <= 262144
        text = prior.scan(name, raw)
        if path.suffix == '.py':
            ast.parse(text, filename=name)
        rows.append(dict(path=name, source_local=local_source, bytes=len(raw), sha256=digest(raw), status='local_candidate'))
    value = dict(schema_version=1, time=datetime.now(timezone.utc).isoformat(), prior_commit=PRIOR,
        branch=BRANCH, source_branch=SOURCE_BRANCH, files=rows, missing=missing, excluded=EXCLUDED,
        remote_checkout=S + '/publication/wmrl-bell-release-v1',
        remote_receipts=S + '/publication/wmrl-bell-release-20261005-v1',
        committed=False, pushed=False)
    write_json(ROOT / ALLOWLIST, value)
    return value


IDENTITIES = r'''def identities():
    assert os.getuid()==20001 and socket.gethostname()=='h100-gpu01'
    assert P['prior_commit']=='382b2698a1ca8853343e32ae7c0b688a38521933'
    assert P['branch']=='codex/wmrl-bell-reward-20261005'
    for path in (S,BASE,R,O,D):
        assert path.is_dir() and path.stat().st_uid==20001 and path.resolve().is_relative_to(S.resolve())
    assert W.resolve().is_relative_to((S/'publication').resolve())
    assert E.resolve().is_relative_to((S/'publication').resolve())
    assert git(BASE,'remote','get-url','personal').decode().strip()=='git@github.com:Yutenji-Nyamu/rlinf_fastwam.git'
'''

SNAPSHOT = r'''def runtime_snapshot():
    snapshots={}
    for label,row in P['runtime_repositories'].items():
        path=Path(row['path']);assert path.resolve().is_relative_to(S.resolve()) and path.stat().st_uid==20001
        head=git(path,'rev-parse','HEAD').decode().strip();assert head==row['head']
        snapshots[label]={'path':str(path),'head':head,
            'status_sha256':digest(git(path,'status','--porcelain','--untracked-files=all')),
            'diff_sha256':digest(git(path,'diff','HEAD','--'))}
    return {'repositories':snapshots,'owner_plan_sha256':digest((O/'owner-plan.json').read_bytes()),
        'source_bindings':{name:digest(Path(item['remote_path']).read_bytes()) for name,item in P['source_bindings'].items()}}
'''

BINDINGS = r'''def check_source_bindings():
    plan=read(O/'owner-plan.json')
    assert plan['owner_dir']==str(O) and plan['physical_gpus']==[4,5,6,7]
    assert plan['prechecks_module']==str(D/'code/ops/prechecks.py')
    assert digest((O/'owner-plan.json').read_bytes())==P['owner_plan_sha256']
    assert plan['repo']==str(R) and plan['repo_head']==EXPECTED_RUNTIME_HEAD
    for name,item in P['source_bindings'].items():
        path=Path(item['remote_path']);assert path.is_absolute() and path.resolve().is_relative_to(S.resolve())
        assert not path.is_symlink() and path.stat().st_uid==20001
        actual=digest(path.read_bytes());assert actual==P['files'][name]['sha256'],'Published bytes differ from used source: '+name
        if item['owner_frozen']:
            assert plan['source_sha256'][str(path)]==actual,'Source was not frozen by owner: '+name
    for path,expected in P['evidence_pins'].items():
        target=Path(path);assert target.is_absolute() and target.resolve().is_relative_to(S.resolve())
        assert target.stat().st_uid==20001 and not target.is_symlink()
        assert digest(target.read_bytes())==expected,'Reviewed experiment evidence changed: '+path
'''


def replace_function(source, name, next_name, replacement):
    source, count = re.subn(r'^def ' + name + r'\(.*?(?=^def ' + next_name + r'\()',
        lambda _: replacement, source, count=1, flags=re.M | re.S)
    assert count == 1
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory-only', action='store_true')
    args = parser.parse_args()
    prior = prior_generator()
    listing = inventory(prior)
    if args.inventory_only:
        print(json.dumps(dict(inventory_only=True, files=len(FILES), missing=listing['missing'], allowlist=ALLOWLIST)))
        return
    assert not listing['missing'], 'Final documents and light receipt required: ' + repr(listing['missing'])
    receipt = json.loads((ROOT / LIGHT).read_text(encoding='utf-8-sig'))
    prior.evidence_shape(receipt)
    assert receipt['ready_for_publication'] is True
    assert receipt['owner_dir'] == S + '/runs/click-bell-v2'
    assert re.fullmatch('[0-9a-f]{64}', receipt['owner_plan_sha256'])
    assert set(receipt['runtime_repositories']) == {'wm', 'native'}
    expected_repos = {'wm': S + '/rlinf-opendw-bell-v1', 'native': S + '/rlinf-rynn-binary-v1'}
    for label, row in receipt['runtime_repositories'].items():
        assert row['path'] == expected_repos[label]
        assert row['head'] == '2151a08ee1bd75df1bef0d8190e594bd5c7f7977'
    bindings = receipt['source_bindings']
    assert OPS + 'prechecks.py' in bindings and BINARY + 'rynn_binary_probe.py' in bindings
    assert BELL + 'probe_reward.py' in bindings and OPS + 'prepare_formal_remote.py' in bindings
    assert OPS + 'prepare_formal_v2.py' in bindings and OPS + 'prepare_cycles_v2.py' in bindings
    assert bindings and receipt['evidence_pins']
    for name, item in bindings.items():
        assert name in FILES and isinstance(item['owner_frozen'], bool)
        assert item['remote_path'].startswith(S + '/') and not any(x in item['remote_path'] for x in ('/../', '\\'))
    for path, value in receipt['evidence_pins'].items():
        assert path.startswith(S + '/') and re.fullmatch('[0-9a-f]{64}', value)
        assert Path(path).name not in ('state.json', 'resources.jsonl'), 'Pin stable completed evidence only'
    files = {}
    for name in FILES + [ALLOWLIST]:
        local_source = SOURCE_MAP.get(name, name)
        data = (ROOT / local_source).read_bytes()
        prior.scan(name, data)
        files[name] = dict(source_local=local_source, bytes=len(data), sha256=digest(data), base64=base64.b64encode(data).decode())
    manifest = dict(schema_version=1, created_time_utc=datetime.now(timezone.utc).isoformat(), prior_remote_commit=PRIOR,
        source_branch=SOURCE_BRANCH, branch=BRANCH, purpose='Rynn native binary diagnosis and click_bell sparse-success WMRL',
        files={name: {k: v for k, v in row.items() if k != 'base64'} for name, row in files.items()},
        source_bindings=bindings, execution_evidence=LIGHT, excluded=EXCLUDED,
        execution_claim='Only the dated light receipt and topical documents establish actual progress and limitations.',
        self_hash_rule='Manifest excludes itself; generated payload pins its exact bytes.')
    write_json(ROOT / MANIFEST, manifest)
    data = (ROOT / MANIFEST).read_bytes()
    files[MANIFEST] = dict(source_local=MANIFEST, bytes=len(data), sha256=digest(data), base64=base64.b64encode(data).decode())
    payload = dict(prior_commit=PRIOR, source_branch=SOURCE_BRANCH, branch=BRANCH, files=files,
        secret_patterns=prior.SECRET_PATTERNS, manifest_path=MANIFEST, manifest_sha256=digest(data),
        source_bindings=bindings, evidence_pins=receipt['evidence_pins'], owner_plan_sha256=receipt['owner_plan_sha256'],
        runtime_repositories=receipt['runtime_repositories'])
    common = prior.REMOTE_COMMON.replace("BASE=S/'publication/opendw-v2'", "BASE=S/'publication/rynn-numeric-release-v1'")
    common = common.replace("W=S/'publication/rynn-success-release-v1'", "W=S/'publication/wmrl-bell-release-v1'")
    common = common.replace("E=S/'publication/rynn-success-release-20261005-v1'", "E=S/'publication/wmrl-bell-release-20261005-v1'")
    common = common.replace("R=S/'rlinf-rynn-v1'", "R=S/'rlinf-opendw-bell-v1'")
    common = common.replace("O=S/'runs/rynn-success-v2'", "O=S/'runs/click-bell-v2'")
    common = common.replace("D=S/'rynn-control-v2'", "D=S/'click-bell-v2'")
    common = replace_function(common, 'identities', 'runtime_snapshot', IDENTITIES)
    common = replace_function(common, 'runtime_snapshot', 'remote_head', SNAPSHOT)
    common = replace_function(common, 'check_source_bindings', 'staged_contract', BINDINGS)
    header = '# Generated isolated bell reward publication; execute only after root review.\nPAYLOAD = ' + repr(
        base64.b64encode(json.dumps(payload, ensure_ascii=False).encode()).decode()) + '\n'
    outputs = {}
    push = prior.REMOTE_PUSH.replace('Add RynnValue reward adapters and semantic diagnostics',
        'Add native binary reward checks and click-bell WMRL experiment')
    # Preserve exact historic receipt bytes, including their final blank lines.
    stage = prior.REMOTE_STAGE.replace('core.whitespace=cr-at-eol', 'core.whitespace=cr-at-eol,-blank-at-eof')
    for filename, body in [('publication_stage_remote.py', stage), ('publication_push_remote.py', push)]:
        source = header + common + body
        ast.parse(source, filename=filename)
        path = Path(__file__).with_name(filename)
        path.write_text(source, encoding='utf-8', newline='\n')
        outputs[filename] = dict(path=str(path), bytes=path.stat().st_size, sha256=digest(path.read_bytes()))
    print(json.dumps(dict(generator_only=True, scripts=outputs, files=len(files), committed=False, pushed=False)))


if __name__ == '__main__':
    main()
