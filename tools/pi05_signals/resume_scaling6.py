"""Resume only frozen incomplete GPU6 scaling cases after signal collection."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(plan):
    if plan['gpu'] != 6 or not plan['remaining']:
        raise RuntimeError('Only nonempty GPU6 continuation supported')
    full=[4,9,16,25,36]
    complete=[row['num_envs'] for row in plan['completed']]
    if complete != full[:len(complete)] or plan['remaining'] != full[len(complete):]:
        raise RuntimeError('Complete prefix and remaining suffix do not partition original order')
    for row in plan['completed']:
        if sha(row['result_path']) != row['result_sha256']:
            raise RuntimeError('Completed native result changed')
    for path, expected in plan['frozen_sha256'].items():
        if sha(path) != expected:
            raise RuntimeError('Frozen scaling input changed: '+path)
    root=Path(plan['project']).resolve()/'runs'
    run=Path(plan['run_dir']).resolve()
    if not run.is_relative_to(root) or run==root or run.exists():
        raise RuntimeError('Fresh owned run scope required')
    if sha(plan['source']) != plan['source_sha256']:
        raise RuntimeError('Original scaling source changed')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--plan-sha256',required=True)
    p.add_argument('--check',action='store_true')
    args=p.parse_args()
    if sha(args.plan)!=args.plan_sha256:
        raise RuntimeError('Continuation plan changed')
    plan=json.loads(args.plan.read_text());validate(plan)
    if args.check:
        print(json.dumps({'check':'PASS','remaining':plan['remaining'],'no_gpu_launch':True}));return
    spec=importlib.util.spec_from_file_location('original_gpu6_scaling',plan['source'])
    base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
    if tuple(base.ORDER)!=(4,9,16,25,36) or base.GPU!=6:
        raise RuntimeError('Unexpected original scaling defaults')
    base.RUN=Path(plan['run_dir']);base.ORDER=tuple(plan['remaining'])
    sys.argv=[plan['source'],*plan['base_arguments']]
    rc=base.main()
    if rc==0:
        summary=json.loads((base.RUN/'summary.json').read_text())
        released=json.loads((base.RUN/'released.json').read_text())
        rows=summary['cases']
        if summary.get('state')!='COMPLETE' or summary.get('released') is not True or released.get('owned_processes')!=[] or released.get('source_unchanged') is not True:
            raise RuntimeError('Native scaling did not prove complete release')
        if [row['num_envs'] for row in rows]!=plan['remaining'] or any(row['state']!='COMPLETE' or row['episodes']!=50 or not row['check']['complete'] or row['check']['eval_time']!=50 for row in rows):
            raise RuntimeError('Incomplete native remaining cases')
    sys.exit(rc)


if __name__=='__main__':main()
