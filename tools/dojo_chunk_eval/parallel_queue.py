# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Success-only scan: private simulator lanes, one resident policy, exact cleanup."""

import copy
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

from common import read, save
from layouts import make_plan
from run_task import check_policy_port_available, ops


def pending_jobs(tasks, output):
    return [(name, c, n, f'{name}-c{c}-n{n}') for name in tasks
            for c, n in [(10, 1), (10, 10), (50, 1), (50, 10)]
            if not (output / 'tasks' / f'{name}-c{c}-n{n}' / 'result.json').exists()]


def next_distinct(pending, active):
    busy = {v['task'] for v in active.values()}
    return next((i for i, job in enumerate(pending) if job[0] not in busy), None)


def main(deployment):
    d = Path(deployment).resolve()
    p = read(d / 'prepared.json')
    template = read(d / 'template.json')
    output = Path(p['output'])
    source = Path(__file__).resolve().parent
    o = ops()
    me = o.proc(os.getpid())
    save(d / 'identity.json', me)
    g = p['gpu']
    assert template['episodes'] == template['num_envs'] == 4
    active = {}
    server = None
    server_id = None
    clean = False
    start = time.time()

    def halt(*_):
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, halt)
    signal.signal(signal.SIGINT, halt)

    def summary():
        rows = [dict(read(x), condition=x.parent.name)
                for x in sorted(output.glob('tasks/*/result.json'))
                if read(x)['state'] != 'PROBE']
        save(output / 'summary.json', dict(
            time=time.time(), completed_jobs=len(rows), planned_jobs=216,
            episodes=sum(r['episodes'] for r in rows),
            successes=sum(r['successes'] for r in rows), rows=rows))
        return len(rows)

    try:
        while not (d / 'registered.json').exists():
            time.sleep(1)
        assert read(d / 'registered.json')['identity'] == me
        assert not o.gpus()[g]['processes'], 'GPU must be clear at private handoff'
        for i in range(p['lanes']):
            check_policy_port_available(template['policy']['port'] + i)
        e = dict(os.environ, **read(template['policy_environment']))
        e.update(CUDA_VISIBLE_DEVICES=str(g), DOJO_GPU_SCOPE=str(g),
                 DOJO_U_TOKEN=d.name + '-policy', DOJO_PARALLEL_ROOT=str(d),
                 XLA_PYTHON_CLIENT_PREALLOCATE='false',
                 XLA_PYTHON_CLIENT_MEM_FRACTION='0.45', OMP_NUM_THREADS='1',
                 PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
                 PYTHONPATH=':'.join([template['bootstrap'], str(source), template['repo'],
                                      template['repo'] + '/XPolicyLab',
                                      template['repo'] + '/XPolicyLab/policy/Pi_05/openpi/src']))
        with (d / 'policy.log').open('a') as log:
            server = subprocess.Popen([template['policy_python'], '-u', '-B',
                                        str(source / 'shared_policy.py'), str(d)],
                                       cwd=template['repo'], env=e, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        server_id = o.proc(server.pid)
        save(d / 'policy-identity.json', server_id)
        deadline = time.monotonic() + 900
        while not (d / 'policy-ready.json').exists():
            assert server.poll() is None, 'Shared policy initialization failed'
            assert time.monotonic() < deadline, 'Shared policy loading timeout'
            time.sleep(3)

        pending = pending_jobs(read(d / 'task-list.json'), output)
        last_launch = 0
        completed = summary()
        while pending or active:
            assert server.poll() is None, 'Shared policy exited'
            for lane, info in list(active.items()):
                code = info['process'].poll()
                if code is None:
                    continue
                dest = output / 'tasks' / info['label']
                assert (dest / 'clearance.json').exists() and read(dest / 'clearance.json')['clear'], (
                    f'Incomplete private cleanup: {info["label"]}')
                assert (dest / 'result.json').exists(), f'Missing result: {info["label"]}'
                del active[lane]
                completed = summary()
                # A simulator-local device error is already cleaned and recorded.
                # Keep healthy siblings and reset this lane for the next task.
                assert code in (0, 79), f'Task driver failed: {info["label"]}'

            used, total, util = map(float, subprocess.check_output(
                ['nvidia-smi', '-i', str(g), '--query-gpu=memory.used,memory.total,utilization.gpu',
                 '--format=csv,noheader,nounits'], text=True).strip().split(','))
            control = read(d / 'control.json')
            limit = min(p['lanes'], max(1, int(control['lanes'])))
            free = next((i for i in range(limit) if i not in active), None)
            index = next_distinct(pending, active)
            # Reserve one measured native simulator plus 3 GiB before admission.
            # Already running healthy tasks are never killed to chase utilization.
            fits = used + p.get('sim_estimate_mib', 11200) + 3072 < total
            if (free is not None and index is not None and fits
                    and time.monotonic() - last_launch >= p.get('launch_spacing_seconds', 20)):
                name, c, n, label = pending.pop(index)
                cfg = copy.deepcopy(template)
                cfg.update(task=name, execute_chunk=c, num_steps=n,
                           output=str(output / 'tasks' / label), parallel_root=str(d),
                           resident_policy_identity=server_id,
                           decision_observations_only=p['decision_observations_only'],
                           layout_plan=make_plan(p['layouts'], name, 4))
                cfg['policy'].update(task_name=name, port=template['policy']['port'] + free)
                path = d / 'jobs' / (label + '.json')
                save(path, cfg)
                save(d / f'lane-{free}' / 'current-job.json', {'config': str(path)})
                with (d / (label + '-driver.log')).open('x') as log:
                    proc = subprocess.Popen([sys.executable, '-u', '-B', str(source / 'run_task.py'), str(path)],
                                            env=dict(os.environ, CUDA_VISIBLE_DEVICES=''),
                                            stdin=subprocess.DEVNULL, stdout=log,
                                            stderr=subprocess.STDOUT, start_new_session=True)
                active[free] = dict(process=proc, identity=o.proc(proc.pid), task=name,
                                    label=label, started=time.time())
                last_launch = time.monotonic()
            state = dict(time=time.time(), phase='EVALUATING' if pending or active else 'COMPLETE',
                         gpu=g, active={str(k): {a:b for a,b in v.items() if a != 'process'}
                                        for k,v in active.items()}, pending=len(pending),
                         completed_jobs=completed, target_lanes=limit,
                         memory_mib=used, utilization=util, admission_has_space=fits)
            save(d / 'status.json', state)
            time.sleep(5)
        clean = True
        save(d / 'complete.json', dict(time=time.time(), elapsed_seconds=time.time()-start, completed_jobs=summary()))
    except BaseException:
        save(d / 'failure.json', dict(time=time.time(), error=traceback.format_exc()))
        raise
    finally:
        for info in active.values():
            if o.same(info['identity']):
                o.exact_signal(info['identity'], signal.SIGTERM)
        for info in active.values():
            info['process'].wait(timeout=90)
        if server_id and o.same(server_id):
            o.exact_signal(server_id, signal.SIGTERM)
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                o.exact_signal(server_id, signal.SIGKILL)
                server.wait(timeout=20)
        clear = not o.gpus()[g]['processes']
        save(d / 'released.json', dict(time=time.time(), identity=me, cards_clear=clear,
                                       complete=clean, next='release GPU; no RLT on GPU3'))


if __name__ == '__main__':
    main(sys.argv[1])
