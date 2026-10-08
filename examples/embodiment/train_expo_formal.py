"""Formal EXPO training from the validated four-device, B64 native smoke.

The only budgeted events are real online env.step commands. Immutable complete
episodes and every completed Q20/FM/editor/temperature call have a full boundary
checkpoint. Eval has a separate environment, output directory and RNG stream.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
from omegaconf import OmegaConf
import torch

from train_expo_ft import check_finite, digest, frozen_sample, log
from rlinf.algorithms.expo_ft.backend import (
    Pi05Backend, clone_env_observation, create_robotwin_env,
)
from rlinf.algorithms.expo_ft.core import ExpoConfig, ExpoLearner
from rlinf.algorithms.expo_ft.formal_cadence import FormalCadence
from rlinf.algorithms.expo_ft.formal_replay import FormalReplay
from rlinf.algorithms.expo_ft.lifecycle import close_robotwin_env


VERSION = 4


class StopRequested(Exception):
    pass


class StopControl:
    """SIGTERM finishes a real action or an atomic commit, then unwinds."""
    def __init__(self):
        self.requested = False
        self.signum = None
        self.collecting = False
        self.depth = 0

    def handler(self, signum, _frame):
        self.requested = True
        self.signum = signum
        if not self.collecting and not self.depth:
            raise StopRequested('Signal ' + str(signum))

    def check(self):
        if self.requested and not self.collecting and not self.depth:
            raise StopRequested('Signal ' + str(self.signum))

    @contextmanager
    def transaction(self):
        self.depth += 1
        try:
            yield
        finally:
            self.depth -= 1
        self.check()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.partial')
    with temp.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


def finite(value):
    if torch.is_tensor(value):
        check_finite(value)
    elif isinstance(value, dict):
        for item in value.values():
            finite(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            finite(item)
    elif isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.floating):
        if not np.isfinite(value).all():
            raise FloatingPointError('Nonfinite NumPy state')
    elif isinstance(value, (float, np.floating)) and not math.isfinite(value):
        raise FloatingPointError('Nonfinite scalar state/metric')


def rng_state(generator):
    return dict(candidate=generator.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all(), numpy=np.random.get_state(),
                python=random.getstate())


def restore_rng(state, generator):
    generator.set_state(state['candidate'].cpu())
    torch.set_rng_state(state['torch'].cpu())
    torch.cuda.set_rng_state_all([item.cpu() for item in state['cuda']])
    np.random.set_state(state['numpy']); random.setstate(state['python'])


def native_boundary(terminal, timeout):
    """A success on native action 200 remains a terminal eligible Q window."""
    terminal, timeout = bool(terminal), bool(timeout)
    return terminal, timeout and not terminal, terminal and timeout


def periodic_evaluation_due(cadence, evaluation, progress):
    return (cadence.remaining_actions > 0 and cadence.episodes_completed > 0 and
            cadence.episodes_completed % evaluation['every_episodes'] == 0 and
            cadence.episodes_completed not in progress['evaluation_periodic'])


def validate_inputs(inputs, budget):
    cfg = inputs['formal']
    devices = cfg.get('parallel_devices')
    if type(devices) is not int or devices not in (1, 2, 4):
        raise ValueError('Validated EXPO device count must be 1, 2 or 4')
    expected = dict(batch_size=64, parallel_devices=devices, candidate_microbatch=8,
                    observation_microbatch=64, fm_microbatch=64, base_lr=2.5e-5,
                    num_envs=1, warmup_episodes=10, physical_actions_per_update_call=40,
                    critic_updates_per_call=20, augmentation=False,
                    max_physical_actions=budget)
    for name, value in expected.items():
        if cfg.get(name) != value:
            raise ValueError('Approved formal contract differs: ' + name)
    core = ExpoConfig(**inputs['core'])
    for name, value in dict(chunk_length=10, action_dim=14, proprio_dim=14,
                            n_base=8, n_edit=8, critic_updates=20, parallel_devices=devices,
                            critic_microbatch_size=64, editor_microbatch_size=64,
                            selection_observation_microbatch_size=64,
                            selection_candidate_microbatch_size=8).items():
        if getattr(core, name) != value:
            raise ValueError('Validated learner contract differs: ' + name)
    if inputs['env']['max_episode_steps'] != 200 or inputs['model'].get('is_lora', False):
        raise ValueError('Inherited 200-action/no-LoRA contract differs')
    if not inputs.get('port_source_manifest'):
        raise ValueError('A pinned port source manifest is required')
    required_source = {'examples/embodiment/train_expo_ft.py',
                       'examples/embodiment/train_expo_formal.py',
                       *('rlinf/algorithms/expo_ft/' + name + '.py'
                         for name in ('backend', 'core', 'formal_replay', 'formal_cadence'))}
    if not required_source.issubset(inputs['port_source_manifest']):
        raise ValueError('Formal driver/helpers/learner/replay/cadence must all be source-pinned')
    source = Path(__file__).resolve().parents[2]
    for name, expected_hash in inputs['port_source_manifest'].items():
        path = (source / name).resolve()
        if not path.is_relative_to(source) or hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise ValueError('Port source fingerprint differs: ' + name)
    return cfg, core


def validate_boundary(backend, learner, replay, cadence):
    cadence.state_dict()
    entries = replay.online_entries
    if (len(entries) != cadence.episodes_completed or
            sum(item['frames'] for item in entries) != cadence.physical_actions or
            learner.update_calls != cadence.completed_calls or
            backend.base_updates != cadence.completed_calls or
            learner.critic_steps != cadence.completed_calls * 20 or
            learner.editor_steps != cadence.completed_calls or
            learner.temperature_steps != cadence.completed_calls):
        raise ValueError('Committed replay/cadence/base/core counters differ')
    for index, entry in enumerate(entries):
        if entry['episode_id'] != f'online-{index:06d}':
            raise ValueError('Replay episode sequence differs from fixed seed rotation')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--max-physical-actions', type=int, choices=(20000, 60000), required=True)
    parser.add_argument('--resume')
    parser.add_argument('--enable-evaluation', action='store_true')
    args = parser.parse_args()
    if args.max_physical_actions == 60000 and not args.resume:
        parser.error('The approved 60000-action extension requires the migrated complete checkpoint')
    inputs_path = Path(args.inputs).resolve(strict=True)
    inputs_bytes = inputs_path.read_bytes(); inputs = json.loads(inputs_bytes)
    inputs_hash = hashlib.sha256(inputs_bytes).hexdigest()
    cfg, core_config = validate_inputs(inputs, args.max_physical_actions)
    run = Path(args.run).resolve(); run.mkdir(parents=True, exist_ok=True)
    replay_root = run.parent / 'replay'
    contract = dict(version=VERSION, run=str(run), inputs_path=str(inputs_path),
                    inputs_sha256=inputs_hash, max_physical_actions=args.max_physical_actions,
                    evaluation_enabled=args.enable_evaluation)
    contract_hash = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    if not args.resume and any(path.exists() for path in (
            run / 'resolved.json', run / 'checkpoint-latest.pt', replay_root / 'index.json')):
        raise ValueError('Fresh training requires a fresh run/replay; use its complete checkpoint to resume')
    if args.resume:
        resume = Path(args.resume).resolve(strict=True)
        if resume.parent != run or resume.name not in ('checkpoint-latest.pt', 'checkpoint-last1.pt'):
            raise ValueError('Resume must use latest/last1 from this exact formal run')
        resolved = json.loads((run / 'resolved.json').read_text())
        if resolved['contract'] != contract or resolved['contract_sha256'] != contract_hash:
            raise ValueError('Resume run/inputs/budget/evaluation contract differs')
    else:
        atomic_json(run / 'resolved.json', dict(inputs=inputs, cli=vars(args), contract=contract,
                                               contract_sha256=contract_hash))
    evaluation = inputs.get('evaluation')
    if args.enable_evaluation:
        if (not isinstance(evaluation, dict) or evaluation.get('every_episodes') != 10 or
                evaluation.get('episodes') != 20 or evaluation.get('initial_base_only') is not True or
                not evaluation.get('config') or not evaluation.get('seed_path')):
            raise ValueError('Fixed initial/10-episode/final evaluation contract differs')
        from rlinf.algorithms.expo_ft.formal_eval import evaluate
        if 'rlinf/algorithms/expo_ft/formal_eval.py' not in inputs['port_source_manifest']:
            raise ValueError('Enabled evaluation helper must be source-pinned')

    stop = StopControl()
    previous_handlers = {sig: signal.signal(sig, stop.handler) for sig in (signal.SIGTERM, signal.SIGINT)}
    env = None; backend = learner = replay = generator = None
    started = time.time()
    cadence = FormalCadence(max_physical_actions=args.max_physical_actions,
                            warmup_episodes=cfg['warmup_episodes'],
                            actions_per_call=cfg['physical_actions_per_update_call'],
                            max_episode_actions=200, minimum_online_actions=64,
                            prior_budget_truncations=1 if args.max_physical_actions == 60000 else 0)
    progress = dict(online_success=0, stopped_episodes=0, evaluation_initial=False,
                    evaluation_periodic=[], evaluation_final=False, evaluation_summaries={},
                    training_seed_sha256=None, horizon_term_precedence=0)
    heartbeat_path = inputs_path.parent / 'driver-heartbeat'

    def heartbeat(*_args, **_kwargs):
        heartbeat_path.touch()

    def status(phase, **fields):
        heartbeat()
        atomic_json(run / 'status.json', dict(time=time.time(), pid=os.getpid(), phase=phase,
                    cadence=cadence.state_dict(), core_updates=getattr(learner, 'update_calls', 0),
                    base_updates=getattr(backend, 'base_updates', 0), **fields))

    def save_checkpoint(reason):
        """Replace latest atomically and retain exactly one previous complete file."""
        status('checkpoint_started', reason=reason)
        validate_boundary(backend, learner, replay, cadence)
        payload = dict(base=backend.state_dict(), core=learner.state_dict(), replay=replay.state_dict(),
                       cadence=cadence.state_dict(), rng=rng_state(generator), progress=copy.deepcopy(progress))
        finite(payload)
        hashes = {name: digest(value) for name, value in payload.items()}
        payload.update(version=VERSION, contract=contract, contract_sha256=contract_hash, hashes=hashes)
        latest = run / 'checkpoint-latest.pt'; last1 = run / 'checkpoint-last1.pt'
        temp = run / 'checkpoint-latest.pt.partial'
        with temp.open('wb') as stream:
            torch.save(payload, stream); stream.flush(); os.fsync(stream.fileno())
        if latest.exists():
            previous = run / 'checkpoint-last1.pt.partial'
            if previous.exists():
                previous.unlink()
            os.link(latest, previous)
            os.replace(previous, last1)
            # POSIX rename between two links to the same inode is a no-op.
            if previous.exists():
                previous.unlink()
        os.replace(temp, latest)
        directory_fd = os.open(run, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        receipt = dict(path=str(latest), bytes=latest.stat().st_size, hashes=hashes, reason=reason,
                       cadence=cadence.state_dict(), core_updates=learner.update_calls,
                       base_updates=backend.base_updates, finite=True)
        atomic_json(run / 'checkpoint.json', receipt)
        status('checkpoint_finished', reason=reason)
        log(run, 'checkpoint_committed', **receipt)

    def close_env():
        nonlocal env
        if env is not None:
            closing = env; env = None
            close_robotwin_env(closing)

    def run_evaluation(label, *, base_only=False):
        # The helper owns/offloads its four-env vector. Never overlap vectors.
        close_env(); status('evaluation_started', label=label, base_only=base_only)
        saved_rng = rng_state(generator)
        owned_rng = learner.generator.get_state()
        saved_selection = list(learner.last_selection_q_indices)
        saved_bootstrap = list(learner.last_bootstrap_q_indices)
        try:
            log(run, 'evaluation_native_enter', label=label)
            result = evaluate(backend, learner, evaluation['config'], run / 'evaluations' / label,
                              evaluation['seed_path'], base_only=base_only, heartbeat=heartbeat)
            log(run, 'evaluation_native_returned', label=label)
        finally:
            log(run, 'evaluation_rng_restore_started', label=label)
            restore_rng(saved_rng, generator); learner.generator.set_state(owned_rng.cpu())
            log(run, 'evaluation_rng_restore_finished', label=label)
            learner.last_selection_q_indices = saved_selection
            learner.last_bootstrap_q_indices = saved_bootstrap
        if isinstance(result, (str, Path)):
            summary_path = Path(result); summary = json.loads(summary_path.read_text())
            result = dict(path=str(summary_path), summary=summary)
        finite(result)
        progress['evaluation_summaries'][label] = result
        status('evaluation_finished', label=label)
        log(run, 'evaluation_finished', label=label, summary=result, budgeted_actions=0)

    def drain_calls():
        while cadence.pending_calls and cadence.can_learn:
            stop.check(); status('learner_started', call=cadence.completed_calls + 1)
            call_started = time.time()
            times = dict(replay=0.0, candidates=0.0)
            cache_before = replay.cache_metrics()
            def next_candidates(next_obs):
                heartbeat()
                started = time.perf_counter()
                try:
                    return backend.sample_normalized(next_obs['env_obs'], num_candidates=8,
                                                      generator=generator)[:, :, :10, :14]
                finally:
                    times['candidates'] += time.perf_counter() - started
            def sample_batch():
                heartbeat()
                started = time.perf_counter()
                try:
                    return replay.sample(cfg['batch_size'], backend, 'cuda:0')
                finally:
                    times['replay'] += time.perf_counter() - started
            def fm_callback():
                heartbeat()
                started = time.perf_counter()
                fm_obs, fm_actions, counts = replay.sample_fm(cfg['batch_size'])
                times['replay'] += time.perf_counter() - started
                log(run, 'base_fm_source', global_batch=cfg['batch_size'], sources=counts)
                return backend.fm_update(fm_obs, fm_actions)
            update_started = time.perf_counter()
            metrics = learner.update_call(sample_batch, next_candidates, fm_callback)
            elapsed = time.perf_counter() - update_started
            # Host wall times; no extra GPU synchronization in the training path.
            metrics.update(replay_seconds=times['replay'], candidate_seconds=times['candidates'],
                           update_other_seconds=elapsed - sum(times.values()))
            cache = replay.cache_metrics()
            metrics.update(cache)
            hits = cache['cache_hits'] - cache_before['cache_hits']
            misses = cache['cache_misses'] - cache_before['cache_misses']
            metrics.update(cache_call_hits=hits, cache_call_misses=misses,
                           cache_hit_rate=hits / max(1, hits + misses))
            finite(metrics)
            if metrics.get('base/fm_callback_called') != 1.0:
                raise ValueError('Formal learner omitted its native FM step')
            if frozen_sample(backend) != frozen_before:
                raise AssertionError('Frozen prefix samples changed')
            # A signal before this boundary discards partial learner state and
            # leaves its previous full checkpoint pending. No partial save.
            with stop.transaction():
                cadence.complete_call()
                log(run, 'learner_finished', metrics=metrics, call=cadence.completed_calls,
                    learner_seconds=time.time() - call_started)
                save_checkpoint('learner-call')

    try:
        status('initializing')
        seed = cfg['seed']; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        torch.set_num_threads(4)
        if torch.cuda.device_count() != cfg['parallel_devices']:
            raise ValueError('Visible device count differs from validated runtime contract')
        for device in range(torch.cuda.device_count()):
            with torch.cuda.device(device):
                torch.cuda.manual_seed(seed + 1009 * device)
        backend = Pi05Backend(OmegaConf.create(inputs['model']), device='cuda:0', lr=cfg['base_lr'],
            candidate_microbatch=cfg['candidate_microbatch'], observation_microbatch=cfg['observation_microbatch'],
            fm_microbatch=cfg['fm_microbatch'], parallel_devices=cfg['parallel_devices'],
            image_augmentation=False, source_head=inputs['source_head'])
        learner = ExpoLearner(core_config, device='cuda:0', seed=seed)
        generator = torch.Generator(device='cuda:0').manual_seed(seed)
        replay = FormalReplay(root=replay_root, demo_path=inputs['demo_path'], seed=seed)
        heartbeat()
        if args.resume:
            saved = torch.load(resume, map_location='cpu', weights_only=False)
            if (saved.get('version') != VERSION or saved.get('contract') != contract or
                    saved.get('contract_sha256') != contract_hash):
                raise ValueError('Formal checkpoint run/inputs/budget contract differs')
            finite(saved)
            if {name: digest(saved[name]) for name in saved['hashes']} != saved['hashes']:
                raise ValueError('Full checkpoint state digest differs')
            backend.load_state_dict(saved['base']); learner.load_state_dict(saved['core'])
            replay.load_state_dict(saved['replay']); cadence.load_state_dict(saved['cadence'])
            progress = copy.deepcopy(saved['progress']); restore_rng(saved['rng'], generator)
            restored = dict(base=backend.state_dict(), core=learner.state_dict(), replay=replay.state_dict(),
                            cadence=cadence.state_dict(), rng=rng_state(generator), progress=progress)
            if {name: digest(value) for name, value in restored.items()} != saved['hashes']:
                raise ValueError('Full formal state restore differs')
            validate_boundary(backend, learner, replay, cadence)
            log(run, 'resume_verified', checkpoint=str(resume), hashes=saved['hashes'],
                cadence=cadence.state_dict(), core_updates=learner.update_calls)
            del saved, restored
        else:
            with stop.transaction():
                save_checkpoint('fresh-base')
        frozen_before = frozen_sample(backend)
        log(run, 'initialized', base_contract=backend.contract, core_config=inputs['core'], formal=cfg,
            resumed=bool(args.resume), initialization='fresh-native-base/new-learner' if not args.resume else 'full-restore')
        if args.enable_evaluation and not progress['evaluation_initial']:
            if cadence.physical_actions or cadence.completed_calls:
                raise ValueError('Initial raw-base evaluation was not committed before training')
            run_evaluation('initial-base', base_only=True)
            with stop.transaction():
                progress['evaluation_initial'] = True; save_checkpoint('initial-evaluation')

        while True:
            # Resume drains the same published episode's pending calls first.
            drain_calls()
            if args.enable_evaluation and periodic_evaluation_due(cadence, evaluation, progress):
                episode_count = cadence.episodes_completed
                run_evaluation(f'episode-{episode_count:06d}')
                with stop.transaction():
                    progress['evaluation_periodic'].append(episode_count)
                    save_checkpoint('periodic-evaluation')
            if not cadence.remaining_actions:
                break
            stop.check()
            if env is None:
                status('environment_started')
                env = create_robotwin_env(OmegaConf.create(inputs['env']), num_envs=1, seed_offset=0)
                log(run, 'simulator_binding', **env.expo_renderer_binding)
            if env.success_seeds is None or env.success_seeds.numel() == 0:
                raise ValueError('Inherited fixed training seed table is unavailable')
            seed_hash = digest(env.success_seeds.detach().cpu())
            if progress['training_seed_sha256'] not in (None, seed_hash):
                raise ValueError('Fixed training seed table changed across environment recreation/resume')
            progress['training_seed_sha256'] = seed_hash
            episode = cadence.episodes_completed
            env_seed = int(env.success_seeds[episode % env.success_seeds.numel()])
            stop.collecting = True
            obs, _info = env.reset(env_seeds=[env_seed])
            frames = []; actions = []; rewards = []; terminated = []; truncated = []
            done = success = budget_truncated = stopped = False; chunks = 0
            status('episode_started', episode=episode, env_seed=env_seed)
            log(run, 'episode_started', episode=episode, env_seed=env_seed, warmup=cadence.in_warmup)
            while not done:
                base = backend.sample_normalized(obs, num_candidates=8, generator=generator)[:, :, :10, :14]
                selected = learner.select_actions(backend.critic_observation(obs), base)
                canonical = backend.decode(obs, selected['actions']); finite(canonical)
                executed = 0
                for index in range(10):
                    frames.append(clone_env_observation(obs)); command = canonical[:, index:index + 1, :].clone()
                    obs, reward, term, trunc, _info = env.step(command, auto_reset=False)
                    actions.append(command[0, 0].cpu())
                    rewards.append(float(torch.as_tensor(reward).reshape(-1)[0]))
                    terminal, timeout, horizon_term_precedence = native_boundary(
                        bool(torch.as_tensor(term).any()), bool(torch.as_tensor(trunc).any()))
                    # Budget/stop truncation preserves the real final observation;
                    # the terminal flag continues to mean native success only.
                    budget_truncated = len(actions) == cadence.remaining_actions and not (terminal or timeout)
                    stopped = stop.requested and not (terminal or timeout or budget_truncated)
                    timeout = timeout or budget_truncated or stopped
                    if len(actions) == 200 and not (terminal or timeout):
                        raise ValueError('Inherited native 200-action episode timeout is missing')
                    terminated.append(terminal); truncated.append(timeout); executed += 1
                    success = success or terminal; done = terminal or timeout
                    if done:
                        break
                chunks += 1; finite(rewards)
                status('real_chunk', episode=episode, chunk=chunks, episode_actions=len(actions))
                log(run, 'real_chunk', episode=episode, chunk=chunks, physical_steps=len(actions),
                    executed_K=executed, selected=int(selected['index'][0]), terminated=terminal,
                    truncated=timeout, budget_truncated=budget_truncated, stop_truncated=stopped,
                    horizon_term_precedence=horizon_term_precedence)
            with stop.transaction():
                entry = replay.append_episode(frames=frames, canonical_actions=torch.stack(actions), rewards=rewards,
                    terminated=terminated, truncated=truncated, success=success,
                    episode_id=f'online-{episode:06d}', final_obs=clone_env_observation(obs))
                cadence.finish_episode(len(actions), budget_truncated=budget_truncated)
                progress['online_success'] += int(success); progress['stopped_episodes'] += int(stopped)
                progress['horizon_term_precedence'] += int(horizon_term_precedence)
                log(run, 'episode_finished', episode=episode, env_seed=env_seed, physical_steps=len(actions),
                    chunks=chunks, success=success, budget_truncated=budget_truncated, stop_truncated=stopped,
                    horizon_term_precedence=horizon_term_precedence, replay_entry=entry,
                    cadence=cadence.state_dict())
                save_checkpoint('episode-boundary')
            stop.collecting = False
            del frames, actions, rewards, terminated, truncated, obs, base, selected, canonical
            stop.check()

        close_env()
        if args.enable_evaluation and not progress['evaluation_final']:
            run_evaluation('final-expo' if args.max_physical_actions == 20000 else 'final-expo-60000')
            with stop.transaction():
                progress['evaluation_final'] = True; save_checkpoint('final-evaluation')
        result = dict(ok=True, budget_completed=True, cadence=cadence.state_dict(),
                      online_success=progress['online_success'], core_updates=learner.update_calls,
                      horizon_term_precedence=progress['horizon_term_precedence'],
                      base_updates=backend.base_updates, elapsed_seconds=time.time() - started,
                      checkpoint=str(run / 'checkpoint-latest.pt'), env_closed=True,
                      evaluation=progress['evaluation_summaries'])
        atomic_json(run / 'complete.json', result); status('complete')
        log(run, 'complete', **result)
    except StopRequested as error:
        stop.depth += 1
        status('stopping', signal=stop.signum)
        atomic_json(run / 'stopped.json', dict(time=time.time(), reason=str(error), signal=stop.signum,
                    cadence=cadence.state_dict(), checkpoint=str(run / 'checkpoint-latest.pt')))
        log(run, 'stopped', signal=stop.signum, cadence=cadence.state_dict())
        raise SystemExit(128 + int(stop.signum or signal.SIGTERM)) from error
    except BaseException as error:
        stop.depth += 1
        atomic_json(run / 'failure.json', dict(time=time.time(), type=type(error).__name__, error=str(error),
                    cadence=cadence.state_dict(), checkpoint=str(run / 'checkpoint-latest.pt')))
        log(run, 'failure', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        stop.depth += 1
        try:
            close_env()
        finally:
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
            heartbeat()


if __name__ == '__main__':
    main()
