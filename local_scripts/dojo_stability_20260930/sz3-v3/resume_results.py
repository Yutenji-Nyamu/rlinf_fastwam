"""Keep native Dojo result details and reconstruct only a missing resume file."""
import json
import math
from pathlib import Path
from dojo_sweep import result_check
from process_guard import atomic_json


def prepare_result(repo, task, seed, run_id, expected, archive, apply=False):
    base = Path(repo)/'eval_result/RoboDojo'/task/'Pi_05/arx_x5'/(
        f'{seed}_ckpt_name=sim,action_type=joint')
    result = base/run_id/'_result.json'
    resume = base/f'_resume_{run_id}.json'
    row = {'task':task,'seed':seed,'run_id':run_id,'expected':expected,
           'result':str(result),'manifest':str(resume),'action':'fresh'}
    if not result.exists():
        assert not resume.exists(), 'Resume without a result needs manual reconciliation'
        return row
    raw = result.read_bytes()
    data = json.loads(raw)
    count = data['eval_time']
    assert type(count) is int and 0 < count <= expected
    checked = result_check(result, count)
    assert checked['complete'], f'Invalid completed episode/video evidence: {checked}'
    details = data['details']
    assert set(details) == {str(i) for i in range(count)}, 'Non-contiguous episode keys'
    successes = sum(v['success'] for v in details.values())
    total_score = math.fsum(v['score'] for v in details.values())
    assert math.isclose(data['success_rate'], successes/count, abs_tol=1e-8)
    assert math.isclose(data['score'], total_score/count*100, abs_tol=1e-6)
    layout_ids = sorted(v['layout_id'] for v in details.values())
    layout_root = Path(repo)/f'Assets/Eval_Layout/RoboDojo/arx_x5/{seed}'
    assert all((layout_root/f'{task}_{i}.json').is_file() for i in layout_ids)
    row.update(episodes=count,successes=successes,action='skip_complete' if count==expected else 'reuse_manifest')
    if apply:
        target=Path(archive)/f'seed{seed}'/task
        target.mkdir(parents=True,exist_ok=True)
        with (target/'_result.json').open('xb') as f:f.write(raw)
        if resume.exists():
            with (target/'resume-original.json').open('xb') as f:f.write(resume.read_bytes())
    if count == expected:
        return row
    if resume.exists():
        manifest = json.loads(resume.read_text())
        assert manifest['run_id']==run_id and manifest['task_name']==task
        assert manifest['policy_name']=='Pi_05' and manifest['config_name']=='arx_x5'
        assert manifest['eval_seed']==seed and manifest['details']==details
        assert manifest['success_nums']==successes and manifest['fail_nums']==count-successes
        assert math.isclose(manifest['total_score'],total_score,abs_tol=1e-6)
        assert sorted(manifest['completed_layout_ids'])==layout_ids
        assert (Path(repo)/manifest['save_dir']).resolve()==result.parent.resolve()
        return row
    # A normal upstream early exit deletes the manifest even below the native
    # budget. Recover exactly the persisted episodes, never invent a success.
    manifest = {'run_id':run_id,'save_dir':str(result.parent.relative_to(repo)),
                'task_name':task,'policy_name':'Pi_05','config_name':'arx_x5',
                'eval_seed':seed,'additional_info':'ckpt_name=sim,action_type=joint',
                'success_nums':successes,'fail_nums':count-successes,'unstable_nums':0,
                'total_score':total_score,'completed_layout_ids':layout_ids,
                'abandoned_layout_ids':[],'details':details,'restart_count':0}
    row.update(action='rebuild_missing_manifest',caveat='Only completed layouts are known; no fabricated abandoned-layout history')
    if apply:
        assert not resume.exists()
        atomic_json(resume,manifest)
        atomic_json(target/'resume-created.json',manifest)
    return row
