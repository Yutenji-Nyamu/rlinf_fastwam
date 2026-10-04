"""Activate staged scope after the exact RLT borrowing/adoption proof.

Direct-start mode performs no native GPU probe. The formal driver's real native
initialization is the first test, with failure handled by the owner return path.
The optional strict mode retains a separately identified reset/close probe.
"""
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback


PHASE = 'native_scope_probe'


def install_and_verify(plan):
    M = sys.modules['frozen_multigpu_owner']
    H = M.H
    assert os.getuid() == 20001 and socket.gethostname() == 'h100-gpu01'
    scope = plan['graphics_scope']
    owner, cycle = Path(plan['owner_dir']), Path(plan['lifecycle_path'])
    direct = plan.get('start_mode') == 'direct_start_user_override_20261004'
    directory = owner / ('scope-install' if direct else 'native-probe')
    directory.mkdir(mode=0o700, exist_ok=False)
    stop = M.owned_path(cycle / 'rlt-stopped.json')
    assert not H.gpu_processes([4, 5, 6, 7])
    provenance = plan['handoff_evidence'] if direct else plan['learning_evidence']
    previous_final = M.owned_path(provenance['owner_final_path'])
    assert M.sha(previous_final) == provenance['owner_final_sha256']
    previous_identity = M.read(M.owned_path(previous_final.parent / 'owner-identity.json'))
    assert not H.same(previous_identity)
    retirement = dict(schema=1, uid=os.getuid(), hostname=socket.gethostname(),
                      boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                      previous_owner=previous_identity,
                      previous_owner_final=dict(path=str(previous_final), sha256=M.sha(previous_final)),
                      rlt_stop_receipt=dict(path=str(stop), sha256=M.sha(stop)))
    if scope.get('audited_unreadable_cpu_processes'):
        retirement['audited_unreadable_cpu_processes'] = scope['audited_unreadable_cpu_processes']
    retirement_path = directory / 'retirement-proof.json'
    M.record(retirement_path, retirement)
    prepare_path = M.owned_path(scope['prepare_module'])
    assert M.sha(prepare_path) == plan['source_sha256'][str(prepare_path)]
    runtime_path = prepare_path.parent / 'graphics_scope_runtime.py'
    assert M.sha(runtime_path) == plan['source_sha256'][str(runtime_path)]
    prior = sys.modules.get('graphics_scope_runtime')
    if prior is not None:
        assert M.sha(prior.__file__) == M.sha(runtime_path), 'Different graphics runtime already imported'
    sys.path.insert(0, str(prepare_path.parent))
    spec = importlib.util.spec_from_file_location('formal_graphics_scope_preparation', prepare_path)
    prepare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare)
    activation = M.owned_path(scope['activation_receipt'], exists=False)
    active = prepare.activate(scope['scope_dir'], retirement_path, activation)
    fragment_path = M.owned_path(scope['environment_fragment_file'])
    assert M.sha(fragment_path) == scope['environment_fragment_sha256']
    fragment = M.read(fragment_path)
    assert active['environment_fragment'] == fragment
    if direct:
        manifest = fragment['RLINF_OPENDW_FORMAL_GRAPHICS_MANIFEST']
        assert active['status'] == 'active' and active['native_probe_verified'] is False
        assert not H.gpu_processes([4, 5, 6, 7])
        return dict(time=H.now(), status='scope_activated', native_probe_verified=False,
                    native_env_reset_completed=False, native_success_evaluated=False,
                    base_environment_sha256=M.sha(plan['environment_file']),
                    environment_fragment_sha256=scope['environment_fragment_sha256'],
                    scope_manifest_path=manifest, scope_manifest_sha256=M.sha(manifest),
                    activation_receipt=str(activation), activation_receipt_sha256=M.sha(activation))
    # Once active, preserve its immutable receipt even if native probing fails.
    # The new RLT cycle uses this receipt to select the new return environment.
    probe_path = M.owned_path(scope['native_probe_module'])
    assert M.sha(probe_path) == plan['source_sha256'][str(probe_path)]
    config = M.owned_path(plan['trials'][0]['config'])
    assert M.sha(config) == plan['trials'][0]['config_sha256']
    timeout = scope.get('native_probe_timeout_seconds', 900)
    assert 0 < timeout <= 1800
    environment = M.read(M.owned_path(plan['environment_file']))
    assert not any(k in environment for k in M.MASKS)
    assert 'RLINF_OPENDW_GPU_SCOPE_MANIFEST' not in environment
    environment.update(fragment)
    # Keep the staged bootstrap first while making the exact training repo
    # importable in both direct probes and their spawned VectorEnv subprocesses.
    paths = [p for p in environment.get('PYTHONPATH', '').split(':') if p]
    if plan['repo'] not in paths:
        paths.append(plan['repo'])
    environment['PYTHONPATH'] = ':'.join(paths)
    environment.pop('DISPLAY', None)
    environment[M.TOKEN], environment[M.PHASE] = plan['token'], PHASE
    catalog = M.Catalog(directory / 'process-catalog.json', plan['token'])
    children, snapshots, failure, cleanup = [], [], None, None
    started = time.monotonic()
    try:
        for rank, gpu in enumerate((6, 7)):
            child_env = dict(environment, CUDA_VISIBLE_DEVICES=str(gpu))
            argv = [plan['python'], '-u', '-B', str(probe_path), '--config', str(config),
                    '--env-rank', str(rank), '--physical-gpu', str(gpu),
                    '--output', str(directory / ('rank' + str(rank))), '--hold-seconds', '4']
            with (directory / ('rank' + str(rank) + '.log')).open('x') as log:
                child = subprocess.Popen(argv, cwd=plan['repo'], env=child_env,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            identity = H.proc(child.pid)
            assert identity and identity['uid'] == 20001
            catalog.add(identity, PHASE, 'exact native-probe Popen child')
            children.append((rank, gpu, child, identity))
            M.record(directory / ('rank' + str(rank) + '-launched.json'), dict(identity=identity, argv=argv))
        deadline = time.monotonic() + timeout
        while True:
            catalog.scan()
            probe_pids = {r['pid'] for r in catalog.live(PHASE)}
            contexts = [r for r in H.gpu_processes(list(range(8))) if r['pid'] in probe_pids]
            for context in contexts:
                env = M.proc_env(context['pid'])
                expected = env.get(b'CUDA_VISIBLE_DEVICES', b'').decode()
                assert expected in ('6', '7') and context['gpu'] == int(expected), 'Native compute/graphics context escaped its rank GPU'
            snapshots.append(dict(time=H.now(), processes=list(probe_pids), contexts=contexts))
            with (directory / 'gpu-contexts.jsonl').open('a') as stream:
                stream.write(json.dumps(snapshots[-1]) + '\n')
            codes = [child.poll() for _, _, child, _ in children]
            assert not any(code is not None and code != 0 for code in codes), 'Native probe child failed'
            if all(code is not None for code in codes):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError('Native N32 reset/close deadline')
            time.sleep(1)
        results, seeds = [], []
        for rank, gpu, child, ident in children:
            child.wait(timeout=3)
            ready = M.read(directory / ('rank' + str(rank)) / 'reset-ready.json')
            result = M.read(directory / ('rank' + str(rank)) / 'result.json')
            assert result['status'] == 'passed' and result['close_completed'] and result['error'] is None and result['close_error'] is None
            assert result['config_sha256'] == M.sha(config) and ready['config_sha256'] == M.sha(config)
            assert result['env_rank'] == rank and result['physical_gpu'] == gpu
            assert all(result['identity'][k] == ident[k] for k in ('pid', 'uid', 'start'))
            assert ready['seed_offset'] == rank and ready['total_num_processes'] == 2 and ready['num_envs'] == 16
            assert ready['observations'] == result['observations']
            seeds.extend(ready['reset_state_ids'])
            results.append(dict(rank=rank, gpu=gpu, result_path=str(directory / ('rank' + str(rank)) / 'result.json'),
                                result_sha256=M.sha(directory / ('rank' + str(rank)) / 'result.json'),
                                observations=result['observations'], seconds=result['seconds']))
        assert len(seeds) == len(set(seeds)) == 32, 'Native ranks must use 32 distinct fixed reset states'
        assert any(r['contexts'] for r in snapshots), 'No live compute/graphics sample was captured'
    except BaseException as exc:
        failure = dict(type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        M.record(directory / 'failure.json', failure)
        raise
    finally:
        # Filter by this phase: never terminate the already CPU-ready WM services.
        cleanup = M.cleanup(plan, catalog, PHASE)
        for _, _, child, _ in children:
            child.wait(timeout=3)
        M.record(directory / 'cleanup.json', cleanup)
    assert cleanup['all_stopped'] and not catalog.live(PHASE)
    assert not H.gpu_processes([4, 5, 6, 7]), 'Native contexts remain after exact probe cleanup'
    manifest = fragment['RLINF_OPENDW_FORMAL_GRAPHICS_MANIFEST']
    return dict(time=H.now(), native_env_reset_completed=True, physical_gpus=[6, 7],
                total_num_envs=32, env_ranks=results, reset_state_ids=seeds,
                owned_outside_gpu4_7_contexts=[], post_native_probe_all_workers_stopped=True,
                task_actions_executed=0, native_success_evaluated=False,
                base_environment_sha256=M.sha(plan['environment_file']),
                environment_fragment_sha256=scope['environment_fragment_sha256'],
                native_eval_seeds_sha256=plan['native_eval_seeds_sha256'],
                scope_manifest_path=manifest, scope_manifest_sha256=M.sha(manifest),
                activation_receipt=str(activation), activation_receipt_sha256=M.sha(activation),
                probe_source_sha256=M.sha(probe_path), config_sha256=M.sha(config),
                seconds=time.monotonic() - started, snapshot_count=len(snapshots))
