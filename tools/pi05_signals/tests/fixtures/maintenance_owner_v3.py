"""Adopt SZ1 GPU4/5, GPU6 or GPU7 without repeating RLT prepare/stop.

The owner waits READY for one-shot command.json requests. A probe normally
returns to the original formal lane; return_to_ready supports bounded diagnosis.
An untouched READY state expires after 1800 seconds and resumes the saved lane.
"""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import time


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def exact_signal(module, identity, sig):
    current = module.identity(identity['pid'])
    assert current and current == identity, 'Exact process identity changed'
    fd = os.pidfd_open(current['pid'])
    try:
        assert module.identity(current['pid']) == current
        signal.pidfd_send_signal(fd, sig)
    finally:
        os.close(fd)


class Adopted:
    def __init__(self, module, identity, final, started):
        self.module, self.identity, self.pid = module, identity, identity['pid']
        self.final, self.started = final, started

    def poll(self):
        if self.module.same(self.identity):
            return None
        if self.final.exists():
            data = self.module.read(self.final)
            if data.get('finished_at', 0) > self.started:
                return 0 if data.get('state') == 'COMPLETE' else 1
        return 1


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, choices=(4, 5, 6, 7), required=True)
    for name in ('owner-source', 'owner-dir', 'config-path', 'directory', 'expected-owner-json'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--check', action='store_true')
    return parser.parse_args()


class Maintenance:
    def __init__(self, args):
        self.args, self.gpu = args, args.gpu
        self.project = Path('/data/chenyiteng/projects/robodojo-openwam-sz1')
        self.directory, self.owner_dir = args.directory.resolve(), args.owner_dir.resolve()
        assert os.getuid() == 1003 and socket.gethostname() == 'admin'
        assert hasattr(os, 'pidfd_open'), 'Use the RLT Python with pidfd support'
        for path in (self.directory, self.owner_dir):
            assert path.is_relative_to(self.project / 'runs') and path != self.project / 'runs'
        assert self.directory != self.owner_dir and not self.directory.exists()
        assert args.owner_source.resolve().is_relative_to(self.project)
        self.M = load_module('maintenance_original_owner_' + str(self.gpu), args.owner_source)
        self.M.GPUS = (4, 5) if self.gpu in (4, 5) else (self.gpu,)
        source = self.project / 'scripts/lanes' / ('openwam' if self.gpu in (4, 5) else 'pi05')
        sys.path.insert(0, str(source))
        self.Guard = load_module('maintenance_guard_' + str(self.gpu), source / 'process_guard.py').ProcessGuard
        self.fatal_pattern = load_module('maintenance_fatal_' + str(self.gpu), source / 'eval_recovery.py').FATAL
        self.phase, self.probe, self.probe_guard = 'PREFLIGHT', None, None
        self.probe_cmd, self.cancel, self.ready_since = None, False, None
        self.resume_environment = {}

    def idle_gpu4_proof(self, state, lane, guard):
        """Accept a finished probe blocked only at the later empty-GPU check."""
        M = self.M
        saved = lane.get('maintenance_idle_receipt')
        directory = Path(saved['directory'] if saved else state['maintenance']).resolve()
        run = Path(lane['probe_run']).resolve()
        assert directory.is_relative_to(self.project / 'runs') and directory != self.owner_dir
        assert run.is_relative_to(self.project / 'runs') and run != self.project / 'runs'
        nonce = lane['command_nonce']
        assert re.fullmatch(r'[A-Za-z0-9_-]{1,80}', nonce)
        command = M.read(directory / 'commands' / (nonce + '.json'))['command']
        assert command['nonce'] == nonce and command['action'] == 'probe'
        assert Path(command['run_dir']).resolve() == run and command['sweep_id'] == run.name
        cleanup = Path(command['cleanup_dir']).resolve()
        assert cleanup.is_relative_to(run) and cleanup != run
        proof_path = directory / 'pause-proof.json'
        summary_path, release_path = run / 'summary.json', run / 'released.json'
        identity_path = directory / ('probe-' + nonce + '-identity.json')
        files = {str(p): M.sha(p) for p in (proof_path, summary_path, release_path, identity_path)}
        if saved:
            assert files == saved['files'], 'Idle probe evidence changed'
        else:
            assert M.read(directory / 'status.json')['identity'] == state['identity']
        proof, summary, release = M.read(proof_path), M.read(summary_path), M.read(release_path)
        assert proof['gpu']['gpu'] == 4 and proof['results']
        for row in proof['results']:
            assert M.sha(Path(row['path'])) == row['sha256'], 'Paused formal results changed'
        assert summary['state'] == 'COMPLETE' and summary['finished'] is True and summary['released'] is True
        assert not summary.get('fatal') and not summary.get('error')
        assert summary['cases'] and all(row['state'] == 'COMPLETE' for row in summary['cases'])
        assert release['source_unchanged'] is True and release['owned_processes'] == []
        assert M.identity(M.read(identity_path)['pid']) is None, 'Finished probe PID still exists'
        assert (cleanup / 'ownership-anchor.json').is_file()
        assert not self.Guard(command['sweep_id'], 1003, cleanup).scan()
        assert not guard.scan(), 'Paused formal processes remain'
        return dict(directory=str(directory), probe_run=str(run), nonce=nonce, files=files)

    def preflight(self):
        M = self.M
        state = M.read(self.owner_dir / 'status.json')
        expected = M.read(self.args.expected_owner_json)
        old = state['identity']
        assert old == expected and M.identity(old['pid']) == expected, 'Owner identity differs'
        assert state['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        cfgs, controllers, guards = {}, {}, {}
        for gpu in M.GPUS:
            lane = state['lanes'][str(gpu)]
            cfg_path = self.args.config_path if gpu == self.gpu else self.project / f'scripts/lanes/configs/gpu{gpu}.json'
            cfg = M.read(cfg_path)
            assert cfg['lane_gpu'] == gpu and cfg['uid'] == 1003 and cfg['workers_per_gpu'] == 1
            assert Path(cfg['project']).resolve() == self.project
            argv = lane['controller_argv']
            assert Path(argv[argv.index('--config') + 1]).resolve() == cfg_path.resolve()
            if any(Path(x).name == 'diagnostic_then_formal.py' for x in argv):
                assert gpu == 7
                assert Path(argv[argv.index('--sweep-script') + 1]).resolve() == self.project / 'scripts/lanes/pi05/dojo_sweep.py'
            plan = M.read(Path(cfg['control_dir']) / 'plan.json')
            assert plan['config'] == cfg
            for path, digest in plan['source_sha256'].items():
                assert M.sha(Path(path)) == digest, 'Existing formal source changed: ' + path
            guard = self.Guard(lane['sweep_id'], 1003, Path(cfg['control_dir']) / 'cleanup')
            if lane['state'] == 'DOJO_RUNNING':
                saved = lane['managed_processes'][0]
                current = guard.identity(saved['pid'])
                assert current and current['start_ticks'] == saved['start_ticks'] and current['role'] == 'controller'
                controllers[gpu] = M.identity(current['pid'])
            elif self.gpu == gpu == 4 and lane['state'] == 'SCALING_TEST':
                assert not guard.scan(), 'Old GPU4 formal processes remain'
            else:
                assert gpu == 4 and lane['state'] in (
                    'MAINTENANCE_PROBE_STOPPING', 'NEEDS_ATTENTION', 'MAINTENANCE_IDLE_AFTER_PROBE')
                lane['maintenance_idle_receipt'] = self.idle_gpu4_proof(state, lane, guard)
                lane.update(state='MAINTENANCE_IDLE_AFTER_PROBE', maintenance_phase='IDLE_AFTER_COMPLETE_PROBE')
            cycle = self.project / ('rlt-cycle-' + self.owner_dir.name + '-gpu' + str(gpu))
            assert all((cycle / name).is_file() for name in ('prepared.json', 'plan.json', 'rlt-stopped.json', 'rlt_cycle_sz1.py'))
            cfgs[gpu], guards[gpu] = cfg, guard
        old_probe = None
        if self.gpu == 4 and state['lanes']['4']['state'] == 'SCALING_TEST':
            pointer = M.read(self.owner_dir / 'active-gpu4-scaling.json')
            assert pointer['identity'] == old
            old_probe = M.read(Path(pointer['directory']) / 'probe-process.json')
            assert M.identity(old_probe['pid']) == old_probe
            assert pointer['probe'] == state['lanes']['4']['probe_run']
        return state, old, cfgs, controllers, guards, old_probe

    def take_over(self):
        M = self.M
        state, old, cfgs, controllers, guards, old_probe = self.preflight()
        self.directory.mkdir(mode=0o700)
        stopped = retired = False
        try:
            exact_signal(M, old, signal.SIGSTOP)
            stopped = True
            for _ in range(30):
                if Path(f"/proc/{old['pid']}/stat").read_text().rsplit(')', 1)[1].split()[0] == 'T':
                    break
                time.sleep(.1)
            else:
                raise RuntimeError('Previous owner did not stop')
            state, old, cfgs, controllers, guards, old_probe = self.preflight()
            allowed = {row['pid'] for row in controllers.values()}
            if old_probe:
                allowed.add(old_probe['pid'])
            children = {int(x) for x in Path(f"/proc/{old['pid']}/task/{old['pid']}/children").read_text().split()}
            assert children <= allowed, 'Active helper or unexpected owner child; restore old owner'
            M.atomic(self.directory / 'previous-owner-status.json', state)
            M.atomic(self.directory / 'handoff-ready.json', dict(old=old, new=M.identity(os.getpid()), controllers=controllers, old_probe=old_probe, time=time.time()))
            exact_signal(M, old, signal.SIGKILL)
            retired = True
            for _ in range(50):
                if not M.same(old):
                    break
                time.sleep(.1)
            assert not M.same(old)
            self.lock = (self.owner_dir / 'owner.lock').open('a')
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except Exception:
            if stopped and not retired and M.same(old):
                exact_signal(M, old, signal.SIGCONT)
            raise
        self.owner = M.Owner(self.project, self.owner_dir)
        self.owner.state, self.owner.configs, self.owner.guards = state, cfgs, guards
        state.update(identity=M.identity(os.getpid()), stage='RUNNING', maintenance=str(self.directory))
        self.owner.processes = {g: Adopted(M, identity, Path(cfgs[g]['control_dir']) / 'final.json', state['lanes'][str(g)]['started_at']) for g, identity in controllers.items()}
        self.owner.save()
        M.atomic(self.owner_dir / f'active-gpu{self.gpu}-maintenance.json', dict(identity=state['identity'], directory=str(self.directory), resume_original=True))
        self.old_probe = old_probe
        self.phase, self.pause_started = 'PAUSING', time.time()
        self.set_state('PAUSING_FOR_MAINTENANCE')
        target = old_probe or controllers.get(self.gpu)
        try:
            if target and M.same(target):
                exact_signal(M, target, signal.SIGTERM)
        except (AssertionError, ProcessLookupError):
            if target and M.same(target):
                raise

    def set_state(self, state, **extra):
        self.owner.state['lanes'][str(self.gpu)].update(state=state, maintenance_phase=self.phase, **extra)
        self.owner.save()
        self.M.atomic(self.directory / 'status.json', dict(identity=self.owner.state['identity'], gpu=self.gpu, phase=self.phase, time=time.time(), **extra))

    def snapshot_results(self):
        M, cfg = self.M, self.owner.configs[self.gpu]
        plan = M.read(Path(cfg['control_dir']) / 'plan.json')
        tasks = {t['task'] for group in plan['groups'] if group['gpu'] == self.gpu for t in group['tasks']}
        records = []
        for task in sorted(tasks):
            for seed in (0, 1, 2):
                rid = f"{cfg['run_id']}_s{seed}_{task}"
                root = self.project / 'RoboDojo/eval_result/RoboDojo' / task
                for path in root.glob('*/*/*/' + rid + '/*.json'):
                    target = self.directory / 'saved-results' / str(seed) / task / path.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, target)
                    records.append(dict(path=str(path), sha256=M.sha(path), snapshot=str(target)))
        assert records, 'No formal result records found; inspect the layout before proceeding'
        return records

    def pause_tick(self):
        M = self.M
        target = self.old_probe
        process = self.owner.processes.get(self.gpu)
        alive = M.same(target) if target else bool(process and process.poll() is None)
        if alive and time.time() - self.pause_started < 180:
            return
        if target:
            old_run = Path(self.owner.state['lanes'][str(self.gpu)]['probe_run'])
            guard = self.Guard(old_run.name, 1003, old_run / 'cleanup')
            guard.cleanup(None, 'operator_gpu_scope_maintenance', grace=20)
            assert not guard.scan()
            if M.same(target):
                exact_signal(M, target, signal.SIGKILL)
            assert not M.same(target)
        guard = self.owner.guards[self.gpu]
        guard.cleanup(None, 'operator_gpu_scope_maintenance', grace=20)
        assert not guard.scan()
        self.owner.processes.pop(self.gpu, None)
        health = M.gpu_health(self.gpu)
        records = self.snapshot_results()
        M.atomic(self.directory / 'pause-proof.json', dict(time=time.time(), gpu=health, results=records, old_probe=target))
        self.phase, self.ready_since = 'READY', time.time()
        self.set_state('MAINTENANCE_READY', ready_deadline=self.ready_since + 1800)

    def validate_environment(self, environment):
        assert isinstance(environment, dict)
        assert set(environment) <= {
            'DOJO_GPU_SCOPE', 'PYTHONPATH', 'ROBODOJO_RENDER_GPU', 'LD_PRELOAD',
            '__NV_PRIME_RENDER_OFFLOAD', '__VK_LAYER_NV_optimus',
            '__GL_APPLICATION_PROFILE', '__GL_APPLICATION_PROFILE_LOG',
            '__EGL_VENDOR_LIBRARY_FILENAMES', 'VK_LOADER_DRIVERS_SELECT',
            'VK_ICD_FILENAMES', 'VK_INSTANCE_LAYERS', 'DRI_PRIME', 'EGL_DEVICE_ID',
        }
        assert all(isinstance(v, str) and '\0' not in v for v in environment.values())
        if 'ROBODOJO_RENDER_GPU' in environment:
            render = environment['ROBODOJO_RENDER_GPU']
            assert render == str(self.gpu) or (render == '0' and environment.get('DOJO_GPU_SCOPE') == str(self.gpu))
        return environment

    def resume_formal(self, environment):
        M = self.M
        M.gpu_health(self.gpu)
        for row in M.read(self.directory / 'pause-proof.json')['results']:
            assert M.sha(Path(row['path'])) == row['sha256'], 'Saved formal result changed'
        environment = self.validate_environment(environment)
        old_env = os.environ.copy()
        log = self.owner_dir / f'gpu{self.gpu}-dojo.log'
        if log.exists():
            shutil.copy2(log, self.directory / ('formal-before-resume-' + str(time.time_ns()) + '.log'))
        self.phase = 'FORMAL_RESUMING'
        original_popen = M.subprocess.Popen
        direct_formal = None
        def checked_popen(argv, *args, **kwargs):
            nonlocal direct_formal
            assert isinstance(argv, (list, tuple)) and argv
            if '--config' in argv:
                actual = Path(argv[argv.index('--config') + 1]).resolve()
                assert actual == self.args.config_path.resolve(), 'Original owner would launch a different config'
                diagnostic = [i for i, x in enumerate(argv) if Path(str(x)).name == 'diagnostic_then_formal.py']
                if diagnostic:
                    assert self.gpu == 7 and len(diagnostic) == 1
                    sweep = self.project / 'scripts/lanes/pi05/dojo_sweep.py'
                    assert Path(argv[argv.index('--sweep-script') + 1]).resolve() == sweep
                    # The previous GPU7 adapter already completed its one-off
                    # diagnostic and exec'd this exact formal sweep. Preserve
                    # the outer owner's gate, guards and RLT cycle; never replay
                    # its old diagnostic ID or modify the formal configuration.
                    argv = list(argv[:diagnostic[0]]) + [str(sweep), '--config', str(actual)]
                    direct_formal = [str(Path(self.owner.configs[7]['policy_env']) / 'bin/python'), '-u', '-B', str(sweep), '--config', str(actual)]
            else:
                assert Path(argv[0]).name == 'nvidia-smi', 'Unexpected subprocess while resuming the saved lane'
            return original_popen(argv, *args, **kwargs)
        try:
            os.environ.update(environment)
            M.subprocess.Popen = checked_popen
            self.owner.start_lane(self.gpu)
            if direct_formal:
                self.owner.state['lanes'][str(self.gpu)].update(controller_argv=direct_formal, original_diagnostic_not_replayed=True)
                self.owner.save()
        finally:
            M.subprocess.Popen = original_popen
            os.environ.clear()
            os.environ.update(old_env)
        self.phase = 'FORMAL_RESUMED'
        M.atomic(self.directory / 'formal-resumed.json', dict(time=time.time(), lane=self.owner.state['lanes'][str(self.gpu)], config=self.owner.configs[self.gpu], environment=environment))
        self.set_state('DOJO_RUNNING')

    def command_tick(self):
        path = self.directory / 'command.json'
        if not path.exists():
            if self.cancel or time.time() - self.ready_since >= 1800:
                self.resume_formal(self.resume_environment)
            return
        command = self.M.read(path)
        nonce = command['nonce']
        assert isinstance(nonce, str) and re.fullmatch(r'[A-Za-z0-9_-]{1,80}', nonce)
        receipt = self.directory / 'commands' / (nonce + '.json')
        if receipt.exists():
            if time.time() - self.ready_since >= 1800:
                self.resume_formal(self.resume_environment)
            return
        assert command['action'] in ('resume', 'probe')
        environment = self.validate_environment(command.get('environment', {}))
        if command['action'] == 'resume':
            self.M.atomic(receipt, dict(command=command, accepted_at=time.time()))
            self.resume_formal(environment)
            return
        argv = command['argv']
        assert isinstance(argv, list) and argv and all(isinstance(x, str) and '\0' not in x for x in argv)
        assert Path(argv[0]).is_absolute()
        run = Path(command['run_dir']).resolve()
        cleanup = Path(command['cleanup_dir']).resolve()
        assert run.is_relative_to(self.project / 'runs') and run != self.project / 'runs'
        assert cleanup.is_relative_to(run) and cleanup != run
        assert command['sweep_id'] == run.name
        assert 0 < command.get('timeout_s', 28800) <= 28800
        self.resume_environment = self.validate_environment(command.get('resume_environment', {}))
        self.M.gpu_health(self.gpu)
        self.M.atomic(receipt, dict(command=command, accepted_at=time.time()))
        self.probe_cmd, self.probe_started = command, time.time()
        # The probe creates its own fresh run directory and ownership anchor.
        # Creating a guard now would pre-create RUN and break its no-replay gate.
        self.probe_guard = None
        gate_read, gate_write = os.pipe()
        wrapper = ('import os,sys; fd=int(sys.argv[1]); b=os.read(fd,1); os.close(fd); '
                   'b==b"G" or sys.exit(125); os.execv(sys.argv[2],sys.argv[2:])')
        # Keep the supervisor outside its own worker guard. Probe children must
        # use the supplied sweep_id; otherwise cleanup cannot prove release.
        env = {**os.environ, **environment}
        for name in ('DOJO_SWEEP_ID', 'DOJO_ROLE', 'ROBODOJO_RUN_ID'):
            env.pop(name, None)
        try:
            with (self.directory / ('probe-' + nonce + '.log')).open('xb') as stream:
                self.probe = subprocess.Popen([sys.executable, '-u', '-B', '-c', wrapper, str(gate_read), *argv],
                    env=env, cwd=self.project, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                    start_new_session=True, pass_fds=(gate_read,))
            os.close(gate_read)
            gate_read = None
            self.probe_identity = self.M.identity(self.probe.pid)
            assert self.probe_identity
            self.M.atomic(self.directory / ('probe-' + nonce + '-identity.json'), self.probe_identity)
            self.phase = 'PROBE'
            self.set_state('MAINTENANCE_PROBE', probe_run=str(run), command_nonce=nonce)
            os.write(gate_write, b'G')
        finally:
            if gate_read is not None:
                os.close(gate_read)
            os.close(gate_write)

    def probe_tick(self):
        command = self.probe_cmd
        if self.probe.poll() is None:
            if self.cancel or time.time() - self.probe_started > command.get('timeout_s', 28800):
                current = self.M.identity(self.probe.pid)
                if current:
                    exact_signal(self.M, current, signal.SIGTERM)
                if (Path(command['cleanup_dir']) / 'ownership-anchor.json').exists():
                    self.cleanup_probe('maintenance_probe_stop')
                self.probe.wait(timeout=30)
            else:
                return
        self.cleanup_probe('maintenance_probe_terminal')
        health = self.M.gpu_health(self.gpu)
        fatal = self.probe.returncode in (99, 134, 139, -6, -11)
        logs = list(Path(command['run_dir']).glob('**/*.log'))
        logs.append(self.directory / ('probe-' + command['nonce'] + '.log'))
        for path in logs:
            text = path.read_text(errors='replace')
            fatal |= bool(self.fatal_pattern.search(text) or 'Invalid PhysX transform' in text)
        self.M.atomic(self.directory / ('probe-' + command['nonce'] + '-finished.json'), dict(time=time.time(), returncode=self.probe.returncode, fatal=fatal, gpu=health))
        if fatal:
            self.phase = 'RLT_RETURNING'
            self.owner.return_lane(self.gpu, 'failed')
        elif command.get('return_to_ready') and not self.cancel:
            self.phase, self.ready_since = 'READY', time.time()
            self.set_state('MAINTENANCE_READY', ready_deadline=self.ready_since + 1800)
        else:
            self.resume_formal(self.resume_environment)

    def cleanup_probe(self, reason):
        command = self.probe_cmd
        cleanup = Path(command['cleanup_dir'])
        if not (cleanup / 'ownership-anchor.json').exists():
            # An early Python startup failure can leave no worker anchor. The
            # independent all-process GPU check still gates any next dispatch.
            self.M.gpu_health(self.gpu)
            return
        if self.probe_guard is None:
            self.probe_guard = self.Guard(command['sweep_id'], 1003, cleanup)
        self.probe_guard.cleanup(None, reason, grace=20)
        assert not self.probe_guard.scan()

    def recover_probe_tick(self):
        # A failing cleanup cannot make the supervisor exit while its probe is
        # still alive. Keep exact ownership until both process and GPU clear.
        if time.time() < self.recovery_next:
            return
        self.recovery_next = time.time() + 20
        current = self.M.identity(self.probe.pid)
        if current:
            assert all(current[k] == self.probe_identity[k] for k in ('pid', 'uid', 'start_ticks'))
            signum = signal.SIGKILL if time.time() - self.recovery_started > 120 else signal.SIGTERM
            try:
                exact_signal(self.M, current, signum)
            except (AssertionError, ProcessLookupError):
                if self.M.same(current):
                    raise
        if (Path(self.probe_cmd['cleanup_dir']) / 'ownership-anchor.json').exists():
            self.cleanup_probe('maintenance_probe_exception')
        if self.probe.poll() is None:
            return
        self.M.gpu_health(self.gpu)
        self.phase = 'NEEDS_ATTENTION'
        self.set_state('NEEDS_ATTENTION', error=self.recovery_error, probe_stopped=True)

    def run(self):
        def cancel(*_):
            self.cancel = True
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, cancel)
        self.take_over()
        while True:
            try:
                if self.phase == 'PAUSING':
                    self.pause_tick()
                elif self.phase == 'READY':
                    self.command_tick()
                elif self.phase == 'PROBE':
                    self.probe_tick()
                elif self.phase == 'PROBE_RECOVERY':
                    self.recover_probe_tick()
                for gpu, process in list(self.owner.processes.items()):
                    if gpu == self.gpu and self.phase == 'PAUSING':
                        continue
                    code = process.poll()
                    if code is None:
                        continue
                    del self.owner.processes[gpu]
                    try:
                        self.owner.return_lane(gpu, 'completed' if code == 0 else 'failed')
                    except Exception as error:
                        self.owner.state['lanes'][str(gpu)].update(state='NEEDS_ATTENTION', error=repr(error))
                        self.owner.save()
                self.owner.verify_returns()
                active = any(self.owner.state['lanes'][str(g)]['state'] == 'RLT_VERIFYING' for g in self.M.GPUS)
                if not self.owner.processes and not active and self.phase not in ('PAUSING', 'READY', 'PROBE', 'PROBE_RECOVERY'):
                    attention = any(self.owner.state['lanes'][str(g)]['state'] == 'NEEDS_ATTENTION' for g in self.M.GPUS)
                    self.owner.state['stage'] = 'NEEDS_ATTENTION' if attention else 'FINISHED'
                    self.owner.save()
                    return
            except Exception as error:
                self.M.atomic(self.directory / ('error-' + str(time.time_ns()) + '.json'), dict(time=time.time(), phase=self.phase, error=repr(error)))
                if self.phase in ('PROBE', 'PROBE_RECOVERY') or (self.probe and self.probe.poll() is None):
                    if self.phase != 'PROBE_RECOVERY':
                        self.recovery_started, self.recovery_next = time.time(), 0
                    self.recovery_error = repr(error)
                    self.phase = 'PROBE_RECOVERY'
                    self.set_state('MAINTENANCE_PROBE_STOPPING', error=repr(error))
                else:
                    # Never restart an uncertain GPU or kill a healthy peer.
                    self.phase = 'NEEDS_ATTENTION'
                    self.set_state('NEEDS_ATTENTION', error=repr(error))
            time.sleep(3)


if __name__ == '__main__':
    maintenance = Maintenance(parse_args())
    if maintenance.args.check:
        state, old, cfgs, controllers, guards, probe = maintenance.preflight()
        print(json.dumps(dict(check='PASS', gpus=maintenance.M.GPUS, pause_gpu=maintenance.gpu, owner=old, controllers=controllers, probe=probe)))
    else:
        maintenance.run()
