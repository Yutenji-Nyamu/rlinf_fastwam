"""Receipt-gated signal borrowing after existing Dojo probes finish naturally.

Private extension of the deployed maintenance owner. GPU4 borrows 4+5 under
their existing shared lock; GPU6/7 borrow one lane. Active maintenance probes
are never interrupted by this extension. It does not prepare/stop RLT again.
The frozen signal command owns one sweep/cleanup scope across borrowed GPUs.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def borrowed_gpus(gpu):
    if gpu == 4:
        return (4, 5)
    if gpu in (6, 7):
        return (gpu,)
    raise ValueError('Shared owner must be borrowed via GPU4, never GPU5 alone')


def ready_states(state, gpus):
    lanes = state.get('lanes', {})
    return all(lanes.get(str(g), {}).get('state') == 'DOJO_RUNNING' for g in gpus)


def scoped_environment(environment, gpu):
    result = dict(environment)
    result['DOJO_GPU_SCOPE'] = str(gpu)
    if 'ROBODOJO_RENDER_GPU' in result:
        result['ROBODOJO_RENDER_GPU'] = str(gpu)
    return result


def validate_command(command, gpus, project):
    if command.get('action') != 'probe' or command.get('return_to_ready'):
        raise ValueError('One frozen signal probe must return to saved Dojo')
    if command.get('priority_gpus') != list(gpus):
        raise ValueError('Signal command must name exact borrowed physical GPUs')
    argv = command.get('argv')
    if not isinstance(argv, list) or not argv or any(not isinstance(s, str) or '\0' in s for s in argv):
        raise ValueError('argv must be a nonempty string list')
    if not Path(argv[0]).is_absolute():
        raise ValueError('argv executable must be absolute')
    run = Path(command['run_dir']).resolve()
    cleanup = Path(command['cleanup_dir']).resolve()
    root = Path(project).resolve() / 'runs'
    if not run.is_relative_to(root) or run == root:
        raise ValueError('Signal run must be a fresh project/runs child')
    if not cleanup.is_relative_to(run) or cleanup == run:
        raise ValueError('Cleanup must be inside signal run')
    if command.get('sweep_id') != run.name or run.exists():
        raise ValueError('Signal scope mismatch or run already exists; no replay')
    if not 0 < command.get('timeout_s', 28800) <= 28800:
        raise ValueError('Signal timeout must be positive and at most 8 hours')


def build_class(base):
    class PriorityMaintenance(base.Maintenance):
        def __init__(self, args):
            super().__init__(args)
            if args.project.resolve() != self.project:
                raise ValueError('Project argument differs from the deployed maintenance project')
            self.borrowed = borrowed_gpus(args.gpu)
            self.signal_command = read(args.signal_command)
            self.frozen = {
                str(args.maintenance_source): args.maintenance_sha256,
                str(args.signal_command): args.signal_command_sha256,
                str(args.owner_source): digest(args.owner_source),
            }
            self.peer_processes = {}
            self.peer_results = []
            self.auto_command_sent = False
            self.original_gpu = args.gpu
            self.probe_stage = 'signal'
            self.post_signal_command = None
            self.post_started = False
            if getattr(args, 'post_signal_command', None) is not None:
                if args.gpu != 6:
                    raise ValueError('A post-signal scaling probe is supported only for single GPU6')
                self.post_signal_command = read(args.post_signal_command)
                self.frozen[str(args.post_signal_command)] = args.post_signal_command_sha256
                if self.post_signal_command['nonce'] == self.signal_command['nonce']:
                    raise ValueError('Post-signal probe requires a distinct nonce')
                if self.post_signal_command['run_dir'] == self.signal_command['run_dir']:
                    raise ValueError('Post-signal probe requires an independent fresh run scope')

        def frozen_check(self):
            for path, expected in self.frozen.items():
                if digest(path) != expected:
                    raise RuntimeError('Frozen input changed: ' + path)

        def preflight(self):
            self.frozen_check()
            state = self.M.read(self.owner_dir / 'status.json')
            if not ready_states(state, self.borrowed):
                raise RuntimeError('Wait for active probes to finish; every borrowed lane must be DOJO_RUNNING')
            validate_command(self.signal_command, self.borrowed, self.project)
            self.validate_environment(self.signal_command.get('environment', {}))
            self.validate_environment(self.signal_command.get('resume_environment', {}))
            if self.post_signal_command is not None:
                validate_command(self.post_signal_command, self.borrowed, self.project)
                self.validate_environment(self.post_signal_command.get('environment', {}))
                self.validate_environment(self.post_signal_command.get('resume_environment', {}))
            result = super().preflight()
            if result[-1] is not None:
                raise RuntimeError('Priority extension does not interrupt active probes')
            return result

        def take_over(self):
            super().take_over()
            # Remove the peer from ordinary terminal handling before signalling.
            # Otherwise its exit would make the original owner dispatch RLT.
            for gpu in self.borrowed:
                if gpu == self.gpu:
                    continue
                process = self.owner.processes.pop(gpu)
                self.peer_processes[gpu] = process
                self.owner.state['lanes'][str(gpu)].update(state='PAUSING_FOR_PRIORITY_SIGNAL')
                self.owner.save()
                if self.M.same(process.identity):
                    base.exact_signal(self.M, process.identity, signal.SIGTERM)
            self.M.atomic(self.directory / 'priority-plan.json', dict(
                time=time.time(), borrowed_gpus=self.borrowed, frozen=self.frozen,
                command=self.signal_command,
                sequence=(['signal', 'remaining_scaling', 'original_dojo', 'original_rlt']
                          if self.post_signal_command is not None else ['signal', 'original_dojo', 'original_rlt']),
                existing_scaling=('interrupted by prior owner; frozen remaining probe scheduled'
                                  if self.post_signal_command is not None else 'finished naturally before takeover')))
            if self.post_signal_command is not None:
                self.M.atomic(self.directory / 'priority-post-signal-plan.json', dict(
                    time=time.time(), command=self.post_signal_command, frozen=self.frozen,
                    sequence=['signal', 'remaining_scaling', 'original_dojo', 'original_rlt'],
                    note='Completed prior cases and remaining order are frozen by the external preparation receipt'))

        def pause_tick(self):
            for gpu, process in list(self.peer_processes.items()):
                if process.poll() is None and time.time() - self.pause_started < 180:
                    return
                guard = self.owner.guards[gpu]
                guard.cleanup(None, 'priority_signal_pause', grace=20)
                if guard.scan() or self.M.same(process.identity):
                    raise RuntimeError('Borrowed peer still alive')
                health = self.M.gpu_health(gpu)
                current = self.gpu
                try:
                    self.gpu = gpu
                    records = self.snapshot_results()
                finally:
                    self.gpu = current
                self.peer_results.extend(records)
                self.M.atomic(self.directory / f'pause-proof-gpu{gpu}.json', dict(
                    time=time.time(), gpu=health, results=records))
                del self.peer_processes[gpu]
                self.owner.state['lanes'][str(gpu)].update(state='PRIORITY_SIGNAL_READY')
                self.owner.save()
            super().pause_tick()
            if self.phase == 'READY':
                proof = self.M.read(self.directory / 'pause-proof.json')
                proof['results'].extend(self.peer_results)
                proof['borrowed_gpus'] = list(self.borrowed)
                self.M.atomic(self.directory / 'pause-proof.json', proof)

        def command_tick(self):
            if not self.auto_command_sent:
                self.frozen_check()
                for gpu in self.borrowed:
                    self.M.gpu_health(gpu)
                self.M.atomic(self.directory / 'command.json', self.signal_command)
                self.auto_command_sent = True
            super().command_tick()

        def cleanup_probe(self, reason):
            super().cleanup_probe(reason)
            for gpu in self.borrowed:
                self.M.gpu_health(gpu)

        def post_attention(self, message):
            self.phase = 'NEEDS_ATTENTION'
            self.owner.state['lanes'][str(self.gpu)].update(state='NEEDS_ATTENTION', error=message)
            self.owner.state['stage'] = 'NEEDS_ATTENTION'
            self.owner.save()
            self.M.atomic(self.directory / 'post-signal-needs-attention.json', dict(
                time=time.time(), stage=getattr(self, 'probe_stage', 'signal'), error=message,
                implicit_dojo_resume=False, implicit_rlt_return=False))

        def start_post_probe(self):
            if self.borrowed != (6,) or self.post_started or self.cancel:
                raise RuntimeError('Post-signal stage cannot be replayed, cancelled, or run on another scope')
            self.frozen_check()
            validate_command(self.post_signal_command, self.borrowed, self.project)
            self.M.gpu_health(6)
            self.post_started = True
            self.probe_stage = 'post_signal'
            self.phase, self.ready_since = 'READY', time.time()
            self.M.atomic(self.directory / 'command.json', self.post_signal_command)
            self.M.atomic(self.directory / 'post-signal-starting.json', dict(
                time=time.time(), command=self.post_signal_command))
            # Use the same pipe-gated asynchronous launch and a fresh Guard.
            # Do not call this class's auto-command path, which owns stage one.
            base.Maintenance.command_tick(self)
            if self.phase != 'PROBE':
                raise RuntimeError('Post-signal command was not dispatched as a new probe')

        def resume_formal(self, environment):
            self.frozen_check()
            for gpu in self.borrowed:
                self.M.gpu_health(gpu)
            current, config = self.gpu, self.args.config_path
            original_read = self.M.read
            proof_path = self.directory / 'pause-proof.json'
            resumed = []
            try:
                proof = original_read(proof_path)
                # Check the whole paused scope before either lane can write.
                # Once a lane resumes, its ordinary result updates must not
                # invalidate the still-paused peer's independent resume gate.
                for row in proof['results']:
                    if self.M.sha(Path(row['path'])) != row['sha256']:
                        raise RuntimeError('Saved formal result changed before priority resume')
                peer_paths = {row['path'] for row in self.peer_results}
                for gpu in self.borrowed:
                    self.gpu = gpu
                    self.args.config_path = (config if gpu == current else
                        self.project / f'scripts/lanes/configs/gpu{gpu}.json')
                    records = ([row for row in proof['results'] if row['path'] not in peer_paths]
                               if gpu == current else self.peer_results)
                    if not records:
                        raise RuntimeError('No saved formal results for priority lane ' + str(gpu))

                    def lane_read(path, *args, **kwargs):
                        if Path(path) == proof_path:
                            return {**proof, 'results': records}
                        return original_read(path, *args, **kwargs)

                    self.M.read = lane_read
                    try:
                        super().resume_formal(scoped_environment(environment, gpu))
                    finally:
                        self.M.read = original_read
                    resumed.append(gpu)
                    self.M.atomic(self.directory / f'formal-resumed-gpu{gpu}.json',
                                  self.M.read(self.directory / 'formal-resumed.json'))
            except Exception as error:
                for gpu in self.borrowed:
                    if gpu not in resumed:
                        self.owner.state['lanes'][str(gpu)].update(
                            state='NEEDS_ATTENTION', error='Priority Dojo resume failed: ' + repr(error))
                self.owner.save()
                self.M.atomic(self.directory / 'priority-resume-failed.json', dict(
                    time=time.time(), resumed_gpus=resumed, error=repr(error)))
                raise
            finally:
                self.M.read = original_read
                self.gpu, self.args.config_path = current, config
            self.M.atomic(self.directory / 'priority-dojo-resumed.json', dict(
                time=time.time(), borrowed_gpus=list(self.borrowed), resumed_gpus=resumed,
                note='Dispatch receipts; actual Dojo actions must still be verified'))

        def probe_tick(self):
            # Same bounded cleanup and fatal policy as the existing maintenance
            # owner, extended to all GPUs borrowed under the shared owner lock.
            command = self.probe_cmd
            if self.probe.poll() is None:
                if self.cancel or time.time() - self.probe_started > command.get('timeout_s', 28800):
                    current = self.M.identity(self.probe.pid)
                    if current:
                        base.exact_signal(self.M, current, signal.SIGTERM)
                    if (Path(command['cleanup_dir']) / 'ownership-anchor.json').exists():
                        self.cleanup_probe('priority_signal_stop')
                    self.probe.wait(timeout=30)
                else:
                    return
            self.cleanup_probe('priority_signal_terminal')
            health = [self.M.gpu_health(g) for g in self.borrowed]
            fatal = self.probe.returncode in (99, 134, 139, -6, -11)
            paths = list(Path(command['run_dir']).glob('**/*.log'))
            paths.append(self.directory / ('probe-' + command['nonce'] + '.log'))
            for path in paths:
                content = path.read_text(errors='replace')
                fatal |= bool(self.fatal_pattern.search(content) or 'Invalid PhysX transform' in content)
            self.M.atomic(self.directory / ('probe-' + command['nonce'] + '-finished.json'), dict(
                time=time.time(), returncode=self.probe.returncode, fatal=fatal, gpus=health,
                stage=getattr(self, 'probe_stage', 'signal')))
            if getattr(self, 'post_signal_command', None) is not None:
                stage = getattr(self, 'probe_stage', 'signal')
                if fatal or self.probe.returncode != 0 or self.cancel:
                    self.post_attention('Priority stage failed or cancelled after exact cleanup; '
                                        f'stage={stage}, rc={self.probe.returncode}, fatal={fatal}')
                    return
                if stage == 'signal':
                    self.start_post_probe()
                    return
                if stage != 'post_signal' or not self.post_started:
                    raise RuntimeError('Unexpected post-signal stage transition')
                summary_path = Path(command['run_dir']) / 'summary.json'
                summary = self.M.read(summary_path) if summary_path.is_file() else {}
                if summary.get('state') != 'COMPLETE' or summary.get('released') is not True:
                    self.post_attention('Remaining scaling lacks COMPLETE and released receipts')
                    return
                self.M.atomic(self.directory / 'post-signal-complete.json', dict(
                    time=time.time(), summary=str(summary_path), sha256=digest(summary_path)))
                self.resume_formal(self.resume_environment)
                return
            if fatal:
                self.phase = 'RLT_RETURNING'
                failures = []
                for gpu in self.borrowed:
                    try:
                        self.owner.return_lane(gpu, 'failed')
                    except Exception as error:
                        self.owner.state['lanes'][str(gpu)].update(
                            state='NEEDS_ATTENTION', error='Priority fatal return failed: ' + repr(error))
                        self.owner.save()
                        failures.append((gpu, repr(error)))
                if failures:
                    raise RuntimeError('Some frozen RLT returns require attention: ' + repr(failures))
            else:
                self.resume_formal(self.resume_environment)

        def run(self):
            try:
                return super().run()
            except BaseException as error:
                # The deployed base invokes take_over before its guarded loop.
                # If a post-handoff peer operation fails, never drop the new
                # lock and leave adopted live children without an owner.
                if not hasattr(self, 'owner'):
                    raise
                self.phase = 'NEEDS_ATTENTION'
                for gpu in self.borrowed:
                    self.owner.state['lanes'][str(gpu)].update(
                        state='NEEDS_ATTENTION', error='Priority owner escaped its loop: ' + repr(error))
                self.owner.state['stage'] = 'NEEDS_ATTENTION'
                self.owner.save()
                while True:
                    self.M.atomic(self.directory / 'priority-owner-needs-attention.json', dict(
                        time=time.time(), phase=self.phase, error=repr(error),
                        identity=self.owner.state['identity'], borrowed_gpus=list(self.borrowed),
                        note='Owner lock retained; no further dispatch or implicit RLT return. Revalidate exact children before operator recovery.'))
                    time.sleep(10)

    return PriorityMaintenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--maintenance-source', type=Path, required=True)
    parser.add_argument('--maintenance-sha256', required=True)
    parser.add_argument('--signal-command', type=Path, required=True)
    parser.add_argument('--signal-command-sha256', required=True)
    parser.add_argument('--post-signal-command', type=Path)
    parser.add_argument('--post-signal-command-sha256')
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--gpu', type=int, choices=(4, 6, 7), required=True)
    for name in ('owner-source', 'owner-dir', 'config-path', 'directory', 'expected-owner-json'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--wait-seconds', type=int, default=0)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if bool(args.post_signal_command) != bool(args.post_signal_command_sha256):
        parser.error('Both post-signal command and SHA256 must be supplied together')
    if args.post_signal_command is not None and args.gpu != 6:
        parser.error('Post-signal scaling is restricted to GPU6')
    if not 0 <= args.wait_seconds <= 28800:
        parser.error('wait-seconds must be 0..28800')
    if digest(args.maintenance_source) != args.maintenance_sha256:
        raise RuntimeError('Maintenance source hash mismatch before import')
    if digest(args.signal_command) != args.signal_command_sha256:
        raise RuntimeError('Signal command hash mismatch')
    if args.post_signal_command is not None and digest(args.post_signal_command) != args.post_signal_command_sha256:
        raise RuntimeError('Post-signal command hash mismatch')
    spec = importlib.util.spec_from_file_location('frozen_maintenance_base', args.maintenance_source)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    cls = build_class(base)
    owner = cls(args)
    deadline = time.monotonic() + args.wait_seconds
    while True:
        state = read(args.owner_dir / 'status.json')
        expected = read(args.expected_owner_json)
        if state['identity'] != expected or owner.M.identity(expected['pid']) != expected:
            raise RuntimeError('Waiting owner identity changed; revalidate instead of adopting another owner')
        if ready_states(state, owner.borrowed):
            break
        states = [state['lanes'][str(g)]['state'] for g in owner.borrowed]
        if any(s.startswith('RLT_') or s == 'NEEDS_ATTENTION' for s in states):
            raise RuntimeError('Prior lane became terminal or needs attention; do not steal its RLT return')
        if args.check or time.monotonic() >= deadline:
            raise RuntimeError('Existing probe still active; nothing was paused')
        print(json.dumps(dict(time=time.time(), phase='WAITING_EXISTING_PROBES',
                              gpus=owner.borrowed, states={str(g):state['lanes'][str(g)]['state'] for g in owner.borrowed})), flush=True)
        time.sleep(10)
    if args.check:
        state, old, configs, controllers, guards, probe = owner.preflight()
        print(json.dumps(dict(check='PASS', gpus=owner.borrowed, owner=old, controllers=controllers)))
        return
    owner.run()


if __name__ == '__main__':
    main()
