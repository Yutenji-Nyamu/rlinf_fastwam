"""Serialize a later four-file result update on the already-published bell branch.

Requires the initial verified commit, reviewed bell result, and current light
receipt. Generates separate stage/push RPCs without executing Git or SSH.
"""
import argparse
import ast
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve().parent
PREFIX = 'docs/world-model/publication_bell_20261005/'
FILES = (
    'docs/world-model/robotwin_pipeline_20261003/click_bell_execution_20261005.md',
    PREFIX + 'bell_reward_result.json', PREFIX + 'light_receipt.json', PREFIX + 'published.json',
)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


STAGE = r'''
def main():
    identities()
    assert W.is_dir() and not W.is_symlink() and not E.exists(),'Do not replay result staging'
    assert remote_head()==[P['prior_commit'],'refs/heads/'+P['branch']]
    assert git(W,'branch','--show-current').decode().strip()==P['branch']
    assert git(W,'rev-parse','HEAD').decode().strip()==P['prior_commit']
    assert not git(W,'status','--porcelain','--untracked-files=all').strip()
    initial=read(S/'publication/wmrl-bell-release-20261005-v1/published.json')
    assert initial['commit']==P['prior_commit'] and initial['verified_remote_commit']==P['prior_commit']
    before=runtime_snapshot();check_source_bindings();contents=decoded_files()
    assert len(contents)==4
    E.mkdir(mode=0o700)
    try:
        for name,data in contents.items():
            path=W.joinpath(*relative(name).parts)
            assert path.resolve().is_relative_to(W.resolve()) and not path.is_symlink()
            for parent in path.parents:
                if parent==W:break
                assert not parent.is_symlink()
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
            assert path.read_bytes()==data
        git(W,'add','-f','--',*sorted(contents))
        names=[x.decode() for x in git(W,'diff','--cached','--name-only','-z').split(b'\0') if x]
        assert names and set(names).issubset(contents)
        for name,data in contents.items():assert git(W,'show',':'+name)==data
        assert not git(W,'diff','--name-only').strip() and not git(W,'ls-files','--others','--exclude-standard').strip()
        git(W,'-c','core.whitespace=cr-at-eol,-blank-at-eof','diff','--cached','--check')
        diff=git(W,'diff','--cached','--binary');stat=git(W,'diff','--cached','--stat')
        after=runtime_snapshot();assert after==before
        receipt=dict(status='staged_for_review',prior_commit=P['prior_commit'],branch=P['branch'],
            manifest_sha256=P['manifest_sha256'],staged_names=names,staged_diff_sha256=digest(diff),
            runtime_before=before,runtime_after=after,committed=False,pushed=False,runtime_unchanged=True)
        record(E/'staged.json',receipt);(E/'staged.diff').write_bytes(diff);(E/'staged.stat.txt').write_bytes(stat)
        print(json.dumps(dict(status='staged_for_review',changed_files=len(names),worktree=str(W),receipt=str(E/'staged.json'))))
        print(stat.decode())
    except BaseException as exc:
        record(E/'failed.json',dict(error_type=type(exc).__name__,error=str(exc),committed=False,pushed=False))
        raise
if __name__=='__main__':main()
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', required=True, help='Exact verified first-release commit')
    args = parser.parse_args()
    assert re.fullmatch('[0-9a-f]{40}', args.parent)
    spec = importlib.util.spec_from_file_location('bell_initial_publisher', SCRIPT / 'make_publication_rpc.py')
    initial = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(initial)
    prior = initial.prior_generator()
    published = json.loads((ROOT / (PREFIX + 'published.json')).read_text(encoding='utf-8-sig'))
    assert published['commit'] == args.parent == published['verified_remote_commit']
    assert published['pushed'] is True and published['branch'] == initial.BRANCH
    receipt = json.loads((ROOT / (PREFIX + 'light_receipt.json')).read_text(encoding='utf-8-sig'))
    prior.evidence_shape(receipt)
    assert receipt['ready_for_publication'] is True and receipt['owner_dir'] == initial.S + '/runs/click-bell-v2'
    files = {}
    for name in FILES:
        path = ROOT / name
        assert path.is_file() and not path.is_symlink()
        raw = path.read_bytes();assert 0 < len(raw) <= 262144
        text = prior.scan(name, raw)
        if path.suffix == '.json':prior.evidence_shape(json.loads(text))
        files[name] = dict(bytes=len(raw), sha256=sha(raw), base64=base64.b64encode(raw).decode())
    payload = dict(prior_commit=args.parent, branch=initial.BRANCH, files=files,
        secret_patterns=prior.SECRET_PATTERNS, manifest_path=PREFIX + 'light_receipt.json',
        manifest_sha256=files[PREFIX + 'light_receipt.json']['sha256'],
        source_bindings=receipt['source_bindings'], evidence_pins=receipt['evidence_pins'],
        owner_plan_sha256=receipt['owner_plan_sha256'], runtime_repositories=receipt['runtime_repositories'])
    common = prior.REMOTE_COMMON
    for old, new in [
        ("BASE=S/'publication/opendw-v2'", "BASE=S/'publication/rynn-numeric-release-v1'"),
        ("W=S/'publication/rynn-success-release-v1'", "W=S/'publication/wmrl-bell-release-v1'"),
        ("E=S/'publication/rynn-success-release-20261005-v1'", "E=S/'publication/wmrl-bell-results-20261005-v1'"),
        ("R=S/'rlinf-rynn-v1'", "R=S/'rlinf-opendw-bell-v1'"),
        ("O=S/'runs/rynn-success-v2'", "O=S/'runs/click-bell-v2'"),
        ("D=S/'rynn-control-v2'", "D=S/'click-bell-v2'")]:
        assert old in common;common = common.replace(old, new)
    identities = initial.IDENTITIES.replace("assert P['prior_commit']=='" + initial.PRIOR + "'",
        "assert re.fullmatch('[0-9a-f]{40}',P['prior_commit'])")
    bindings = initial.BINDINGS.replace("P['files'][name]['sha256']", "item['sha256']")
    common = initial.replace_function(common, 'identities', 'runtime_snapshot', identities)
    common = initial.replace_function(common, 'runtime_snapshot', 'remote_head', initial.SNAPSHOT)
    common = initial.replace_function(common, 'check_source_bindings', 'staged_contract', bindings)
    push = prior.REMOTE_PUSH.replace('Add RynnValue reward adapters and semantic diagnostics',
        'Record click-bell reward validation and experiment outcome')
    old = "staged_contract(staged);assert not remote_head(),'Release branch appeared before this commit'"
    new = "staged_contract(staged);assert remote_head()==[P['prior_commit'],'refs/heads/'+P['branch']],'Published parent changed'"
    assert old in push;push = push.replace(old, new)
    old = "if not remote:git(W,'push','personal','HEAD:refs/heads/'+P['branch'])"
    new = "if remote==[P['prior_commit'],'refs/heads/'+P['branch']]:git(W,'push','personal','HEAD:refs/heads/'+P['branch'])"
    assert old in push;push = push.replace(old, new)
    header = '# Generated four-file bell result follow-up; review before execution.\nPAYLOAD = ' + repr(
        base64.b64encode(json.dumps(payload, ensure_ascii=False).encode()).decode()) + '\n'
    outputs = {}
    for name, body in [('publication_results_stage_remote.py', STAGE), ('publication_results_push_remote.py', push)]:
        source = header + common + body
        ast.parse(source, filename=name)
        path = SCRIPT / name;path.write_text(source, encoding='utf-8', newline='\n')
        outputs[name] = dict(bytes=path.stat().st_size, sha256=sha(path.read_bytes()))
    print(json.dumps(dict(generator_only=True, parent=args.parent, branch=initial.BRANCH,
        files=list(FILES), scripts=outputs, committed=False, pushed=False)))


if __name__ == '__main__':
    main()
