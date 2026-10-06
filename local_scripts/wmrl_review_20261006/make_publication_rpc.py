"""Serialize a reviewed WMRL result/research update; never execute Git or SSH.

Reuses the isolated bell publication checkout and refuses a changed parent,
dirty checkout, altered runtime source, or replayed receipt directory.
"""
import ast
import base64
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve().parent
PREFIX = 'docs/world-model/publication_reward_review_20261006/'
PRIOR = '66a240c2fe256f3dbccb09c0d3ea4f369979fe3a'
DOC = 'docs/world-model/robotwin_pipeline_20261003/'
MANIFEST = PREFIX + 'manifest.json'
FILES = (
    PREFIX + 'README.md', PREFIX + 'wmrl-curves-1108.svg',
    PREFIX + 'wmrl-scalars-1108.csv', PREFIX + 'wmrl-summary-1108.json',
    PREFIX + 'live_status.json',
    DOC + 'reward_sources_20261006.md', DOC + 'rynn_options_20261006.md',
    'docs/world-model/publication_bell_20261005/results_published.json',
    'local_scripts/wmrl_review_20261006/make_publication_rpc.py',
)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    initial = load_module('bell_initial', ROOT / 'local_scripts/wmrl_bell_20261005/make_publication_rpc.py')
    followup = load_module('bell_results', ROOT / 'local_scripts/wmrl_bell_20261005/make_followup_publication_rpc.py')
    prior = initial.prior_generator()
    published = json.loads((ROOT / 'docs/world-model/publication_bell_20261005/results_published.json').read_text(encoding='utf-8-sig'))
    assert published['commit'] == PRIOR == published['verified_remote_commit']
    assert published['pushed'] is True and published['branch'] == initial.BRANCH
    runtime = json.loads((ROOT / 'docs/world-model/publication_bell_20261005/light_receipt.json').read_text(encoding='utf-8-sig'))
    prior.evidence_shape(runtime)
    assert runtime['ready_for_publication'] is True
    assert runtime['owner_dir'] == initial.S + '/runs/click-bell-v2'
    files = {}
    for name in FILES:
        prior.relative(name)
        path = ROOT / name
        assert path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(ROOT)
        raw = path.read_bytes()
        assert 0 < len(raw) <= 262144, name
        text = prior.scan(name, raw)
        if path.suffix == '.json':
            prior.evidence_shape(json.loads(text))
        if path.suffix == '.py':
            ast.parse(text, filename=name)
        files[name] = dict(bytes=len(raw), sha256=sha(raw), base64=base64.b64encode(raw).decode())
    manifest = dict(schema_version=1, prepared_at_utc=datetime.now(timezone.utc).isoformat(),
        parent=PRIOR, branch=initial.BRANCH,
        files={name: {k: v for k, v in row.items() if k != 'base64'} for name, row in files.items()},
        scope='Dated WMRL scalar/curve snapshot, live light status, Rynn and reward-source research; no training changes.',
        runtime_source_check='Compare all source bindings and completed evidence pinned by the prior bell release.',
        excluded=['EXPO data', 'weights/checkpoints', 'raw logs/media', 'credentials/full environments/plans',
                  'generated RPC payloads', 'unrelated workspace changes'],
        committed=False, pushed=False,
        self_hash_rule='Manifest excludes itself; generated RPC pins its exact bytes.')
    raw = (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode()
    prior.scan(MANIFEST, raw)
    (ROOT / MANIFEST).write_bytes(raw)
    files[MANIFEST] = dict(bytes=len(raw), sha256=sha(raw), base64=base64.b64encode(raw).decode())
    payload = dict(prior_commit=PRIOR, branch=initial.BRANCH, files=files,
        secret_patterns=prior.SECRET_PATTERNS, manifest_path=MANIFEST, manifest_sha256=sha(raw),
        source_bindings=runtime['source_bindings'], evidence_pins=runtime['evidence_pins'],
        owner_plan_sha256=runtime['owner_plan_sha256'], runtime_repositories=runtime['runtime_repositories'])
    common = prior.REMOTE_COMMON
    for old, new in [
        ("BASE=S/'publication/opendw-v2'", "BASE=S/'publication/rynn-numeric-release-v1'"),
        ("W=S/'publication/rynn-success-release-v1'", "W=S/'publication/wmrl-bell-release-v1'"),
        ("E=S/'publication/rynn-success-release-20261005-v1'", "E=S/'publication/wmrl-reward-review-20261006-v1'"),
        ("R=S/'rlinf-rynn-v1'", "R=S/'rlinf-opendw-bell-v1'"),
        ("O=S/'runs/rynn-success-v2'", "O=S/'runs/click-bell-v2'"),
        ("D=S/'rynn-control-v2'", "D=S/'click-bell-v2'")]:
        assert old in common
        common = common.replace(old, new)
    identities = initial.IDENTITIES.replace(initial.PRIOR, PRIOR)
    bindings = initial.BINDINGS.replace("P['files'][name]['sha256']", "item['sha256']")
    common = initial.replace_function(common, 'identities', 'runtime_snapshot', identities)
    common = initial.replace_function(common, 'runtime_snapshot', 'remote_head', initial.SNAPSHOT)
    common = initial.replace_function(common, 'check_source_bindings', 'staged_contract', bindings)
    stage = followup.STAGE.replace("publication/wmrl-bell-release-20261005-v1/published.json",
        "publication/wmrl-bell-results-20261005-v1/published.json")
    stage = stage.replace('assert len(contents)==4', 'assert len(contents)==' + str(len(files)))
    push = prior.REMOTE_PUSH.replace('Add RynnValue reward adapters and semantic diagnostics',
        'Record WMRL training snapshot and reward model review')
    old = "staged_contract(staged);assert not remote_head(),'Release branch appeared before this commit'"
    new = "staged_contract(staged);assert remote_head()==[P['prior_commit'],'refs/heads/'+P['branch']],'Published parent changed'"
    assert old in push
    push = push.replace(old, new)
    old = "if not remote:git(W,'push','personal','HEAD:refs/heads/'+P['branch'])"
    new = "if remote==[P['prior_commit'],'refs/heads/'+P['branch']]:git(W,'push','personal','HEAD:refs/heads/'+P['branch'])"
    assert old in push
    push = push.replace(old, new)
    header = '# Generated exact WMRL review publication; root reviews then executes.\nPAYLOAD = ' + repr(
        base64.b64encode(json.dumps(payload, ensure_ascii=False).encode()).decode()) + '\n'
    outputs = {}
    for name, body in [('publication_stage_remote.py', stage), ('publication_push_remote.py', push)]:
        source = header + common + body
        ast.parse(source, filename=name)
        path = SCRIPT / name
        path.write_text(source, encoding='utf-8', newline='\n')
        outputs[name] = dict(bytes=path.stat().st_size, sha256=sha(path.read_bytes()))
    print(json.dumps(dict(generator_only=True, files=len(files), parent=PRIOR, branch=initial.BRANCH,
        scripts=outputs, receipt_directory=initial.S + '/publication/wmrl-reward-review-20261006-v1',
        ssh=False, committed=False, pushed=False)))


if __name__ == '__main__':
    main()
