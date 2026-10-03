"""Shared claim queue under an existing exact-owner maintenance probe."""
import argparse
import ast
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
import traceback
import xml.etree.ElementTree as ET


CORE_FILES = ('run_batch.py', 'signal_math.py', 'signal_observer.py', 'validate_batch.py')
WRAPPER = 'rlinf/models/embodiment/openpi/openpi_action_model.py'
FROZEN_SCHEMA = 'pi05-frozen-smoke-v1'
FATAL_CODES = (99, 134, 139, -6, -11)
FATAL_PATTERN = re.compile(r'CUDA\s*error|illegal memory access|device-side assert|Invalid PhysX transform', re.I)


def read(p):
    return json.loads(Path(p).read_text())


def atomic(p, data):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.' + str(threading.get_ident()) + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, p)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def child_fatal(returncode, log):
    text = Path(log).read_text(errors='replace') if Path(log).is_file() else ''
    return returncode in FATAL_CODES or bool(FATAL_PATTERN.search(text))


def manifest_configs(manifest):
    configs, smoke = manifest.get('configs'), manifest.get('smoke')
    require(isinstance(configs, list) and configs and isinstance(smoke, list) and smoke,
            'Manifest needs nonempty configs and smoke lists')
    require(all(isinstance(p, str) and Path(p).is_absolute() for p in configs + smoke),
            'Manifest config paths must be absolute strings')
    require(len(set(configs)) == len(configs) and len({Path(p).stem for p in configs}) == len(configs),
            'Manifest config paths and claim stems must be unique')
    require(len(set(smoke)) == len(smoke) and set(smoke).issubset(configs),
            'Smoke must be a nonempty unique subset of formal configs')
    return configs, smoke


def signal_fingerprint(output, repo):
    """Controller-only repairs do not invalidate unchanged model/signal data."""
    output, repo = Path(output).resolve(), Path(repo).resolve()
    configs, _ = manifest_configs(read(output / 'manifest.json'))
    paths = [repo / WRAPPER, *(repo / 'tools/pi05_signals' / name for name in CORE_FILES),
             output / 'manifest.json', output / 'environment.json', *(Path(p) for p in configs)]
    files = {str(path.resolve()): digest(path) for path in paths}
    checksum = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return {'sha256': checksum, 'files': files}


def _same_config(left, right):
    # Dynamic GPU assignment is the queue's only permitted config change.
    return {k: v for k, v in left.items() if k != 'gpu'} == {k: v for k, v in right.items() if k != 'gpu'}


def freeze_existing_smoke(output, repo, smoke_run):
    """Freeze completed smoke evidence without launching models/environments.

    Legacy plans lack a wrapper runtime hash. Require their exact planned patch
    script and its locked output identity, explicitly recording that limitation.
    """
    output, repo, smoke_run = (Path(p).resolve() for p in (output, repo, smoke_run))
    manifest = read(output / 'manifest.json')
    _, smoke = manifest_configs(manifest)
    fingerprint = signal_fingerprint(output, repo)
    plan_path, summary_path = smoke_run / 'plan.json', smoke_run / 'summary.json'
    plan, summary = read(plan_path), read(summary_path)
    require(plan.get('phase') == summary.get('phase') == 'smoke' and
            Path(plan['output']).resolve() == output, 'Smoke plan/output mismatch')
    require(summary.get('passed') is True and not summary.get('failures') and
            not summary.get('cancelled') and not summary.get('fatal'), 'Smoke controller did not pass cleanly')
    files = {str(plan_path): digest(plan_path), str(summary_path): digest(summary_path)}
    planned = plan['source_sha256']
    tool_dir = repo / 'tools/pi05_signals'
    for name in CORE_FILES:
        path = tool_dir / name
        require(planned.get(str(path)) == digest(path), 'Smoke plan source drift: ' + name)
    wrapper = repo / WRAPPER
    if plan.get('model_wrapper_sha256') is not None:
        require(plan['model_wrapper_sha256'] == digest(wrapper), 'Smoke wrapper source drift')
        wrapper_provenance = 'wrapper SHA256 recorded by smoke controller before launch'
    else:
        patch = tool_dir / 'patch_model.py'
        require(planned.get(str(patch)) == digest(patch), 'Legacy smoke lacks matching planned patch identity')
        constants = {node.targets[0].id: ast.literal_eval(node.value)
                     for node in ast.parse(patch.read_text()).body
                     if isinstance(node, ast.Assign) and len(node.targets) == 1 and
                     isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'PATCHED_LF_SHA256'}
        normalized = wrapper.read_bytes().decode('utf-8').replace('\r\n', '\n').encode('utf-8')
        require(hashlib.sha256(normalized).hexdigest() == constants.get('PATCHED_LF_SHA256'),
                'Current wrapper does not match legacy planned patch output')
        files[str(patch)] = digest(patch)
        wrapper_provenance = ('legacy plan patch-source SHA256 + current exact locked patch output; '
                              'historical runtime wrapper hash was not recorded')
    validations = []
    for name in smoke:
        original = Path(name)
        config = read(original)
        batch = Path(config['output']).resolve()
        claim_path = output / 'claims' / (original.stem + '.json')
        claim = read(claim_path)
        require(claim.get('state') == 'VALIDATED' and claim.get('returncode') == 0 and
                not claim.get('fatal') and Path(claim['owner_run']).resolve() == smoke_run and
                Path(claim['output']).resolve() == batch, 'Smoke claim was not validated by this controller')
        assigned = Path(claim['assigned']).resolve()
        require(assigned.is_relative_to(smoke_run / 'configs'), 'Assigned config escaped smoke run')
        require(_same_config(config, read(assigned)) and _same_config(config, read(batch / 'config.json')),
                'Smoke config drift beyond assigned GPU: ' + name)
        schema = read(batch / 'schema.json')
        require(not schema.get('synthetic') and schema.get('shape') == {'B': 16, 'M': 10, 'H': 50, 'D': 14},
                'Acceptance requires real B16/M10/H50 episode records')
        require(schema.get('source_hashes') == {key: digest(tool_dir / key) for key in CORE_FILES},
                'Recorded smoke collector sources differ from current sources')
        validation = read(batch / 'validation.json')
        done = read(batch / 'done.json')
        require(validation.get('passed') is True and Path(validation['output']).resolve() == batch and
                validation.get('episodes') == done.get('episodes') == 16 and
                validation.get('queries') and len(validation['queries']) == done.get('queries') and
                len(validation.get('videos', [])) == 16 and not validation.get('errors') and
                not (batch / 'error.json').exists(), 'Real smoke validation incomplete')
        paths = [claim_path, assigned, batch / 'config.json', batch / 'schema.json',
                 batch / 'model-smoke.json', batch / 'done.json', batch / 'validation.json']
        cleanup_path = Path(claim['cleanup_receipt'])
        require(read(cleanup_path).get('state') == 'EMPTY', 'Smoke batch cleanup not proven empty')
        paths.append(cleanup_path)
        for path in paths:
            files[str(path)] = digest(path)
        validations.append(str(batch / 'validation.json'))
    cleanup_path = Path(summary['cleanup'])
    require(read(cleanup_path).get('state') == 'EMPTY', 'Smoke controller cleanup not proven empty')
    files[str(cleanup_path)] = digest(cleanup_path)
    require(signal_fingerprint(output, repo) == fingerprint, 'Sources changed during smoke acceptance')
    marker = {'schema': FROZEN_SCHEMA, 'passed': True, 'time': time.time(),
              'smoke_run': str(smoke_run), 'output': str(output), 'repo': str(repo),
              'fingerprint': fingerprint, 'evidence_sha256': files,
              'wrapper_provenance': wrapper_provenance, 'validations': validations}
    destination = output / 'frozen-smoke-acceptance.json'
    if destination.exists():
        previous = read(destination)
        require(all(previous.get(key) == marker[key] for key in marker if key != 'time'),
                'Existing frozen acceptance differs; do not overwrite evidence')
        return previous
    atomic(destination, marker)
    return marker


def verify_frozen_smoke(output, repo):
    output, repo = Path(output).resolve(), Path(repo).resolve()
    marker = read(output / 'frozen-smoke-acceptance.json')
    require(marker.get('schema') == FROZEN_SCHEMA and marker.get('passed') is True and
            marker.get('output') == str(output) and marker.get('repo') == str(repo), 'Invalid frozen smoke acceptance')
    require(marker.get('fingerprint') == signal_fingerprint(output, repo), 'Signal code/config drift since smoke acceptance')
    require(bool(marker.get('evidence_sha256')) and bool(marker.get('validations')), 'Empty smoke evidence')
    for path, expected in marker['evidence_sha256'].items():
        require(digest(path) == expected, 'Smoke evidence changed: ' + path)
    return marker


class Queue:
    def __init__(self, args):
        self.a = args
        self.stop = threading.Event()
        self.cancelled = threading.Event()
        self.fatal = threading.Event()
        self.lock = threading.Lock()
        self.failures = []
        self.manifest = read(args.output / 'manifest.json')
        configs, smoke = manifest_configs(self.manifest)
        self.env = read(args.output / 'environment.json')
        self.configs = smoke if args.phase == 'smoke' else configs
        self.fingerprint = signal_fingerprint(args.output, args.repo)
        if args.phase == 'formal':
            verify_frozen_smoke(args.output, args.repo)
        self.run = args.run_dir.resolve()
        if self.run.exists():
            raise RuntimeError('Fresh run required; prior ownership receipts must not be replayed')
        self.run.mkdir(parents=True)
        spec = importlib.util.spec_from_file_location('signal_process_guard', args.guard_source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.guard = module.ProcessGuard(self.run.name, 1003, self.run / 'cleanup')
        self.status = {str(g): dict(phase='READY') for g in args.gpus}
        atomic(self.run / 'plan.json', dict(time=time.time(), argv=sys.argv, gpus=args.gpus,
               phase=args.phase, output=str(args.output), priority=['signals','Dojo','RLT'],
               signal_fingerprint=self.fingerprint, model_wrapper_sha256=digest(args.repo / WRAPPER),
               source_sha256={str(p): digest(p) for p in (args.repo / 'tools/pi05_signals').glob('*.py')}))

    def update(self, gpu, **data):
        with self.lock:
            self.status[str(gpu)].update(time=time.time(), **data)
            atomic(self.run / 'status.json', dict(time=time.time(), pid=os.getpid(), lanes=self.status))

    def cancel(self, *_):
        self.cancelled.set()
        self.stop.set()

    def claim(self, gpu):
        with (self.a.output / 'queue.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if self.stop.is_set():
                return None
            require(signal_fingerprint(self.a.output, self.a.repo) == self.fingerprint,
                    'Signal inputs changed while queue was running')
            for name in self.configs:
                config = Path(name)
                claim = self.a.output / 'claims' / (config.stem + '.json')
                cfg = read(config)
                out = Path(cfg['output'])
                if claim.exists() or (out / 'started.json').exists():
                    continue
                cfg['gpu'] = gpu
                assigned = self.run / 'configs' / config.name
                atomic(assigned, cfg)
                atomic(claim, dict(time=time.time(), gpu=gpu, owner_run=str(self.run),
                                  assigned=str(assigned), output=str(out), state='CLAIMED'))
                return assigned, claim, cfg
        return None

    def health(self, gpu, run_id):
        owned = {r['pid']:r for r in self.guard.scan(run_id)}
        result = subprocess.run(['nvidia-smi','-q','-x'], capture_output=True, text=True, timeout=20, check=True)
        root = ET.fromstring(result.stdout)
        rows = []
        for index, device in enumerate(root.findall('gpu')):
            for proc in device.findall('./processes/process_info'):
                pid = int(proc.findtext('pid'))
                if pid in owned:
                    rows.append(dict(gpu=index, pid=pid, type=proc.findtext('type'),
                                     memory=proc.findtext('used_memory')))
                    if index != gpu:
                        raise RuntimeError('Owned process escaped GPU scope: ' + str(rows[-1]))
        atomic(self.run / f'gpu{gpu}-health.json', dict(time=time.time(), gpu=gpu, owned_processes=rows))

    def execute(self, argv, env, log, gpu, run_id, timeout):
        require(not self.stop.is_set(), 'Queue stopped before child launch')
        with log.open('xb') as stream:
            proc = subprocess.Popen(argv, env=env, cwd=self.a.repo, stdin=subprocess.DEVNULL,
                                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        start = time.time()
        try:
            while proc.poll() is None:
                self.health(gpu, run_id)
                self.update(gpu, heartbeat=time.time(), child_pid=proc.pid, elapsed_s=time.time()-start)
                if self.stop.wait(10) or time.time() - start > timeout:
                    raise RuntimeError('Stopped or batch deadline exceeded')
        except BaseException:
            # Exact-owner cleanup only; bounded wait even if cleanup itself fails.
            try:
                self.guard.cleanup(run_id, 'signal_queue_stop_or_exception', grace=20)
            finally:
                try:
                    proc.wait(timeout=30)
                finally:
                    if child_fatal(proc.returncode, log):
                        self.fatal.set()
            raise
        if child_fatal(proc.returncode, log):
            self.fatal.set()
            self.stop.set()
        return proc.returncode

    def lane(self, gpu):
        claim = None
        logs = []
        rc = None
        try:
            while not self.stop.is_set():
                item = self.claim(gpu)
                if item is None:
                    break
                config, claim, cfg = item
                logs = []
                rc = None
                run_id = self.run.name + ':' + config.stem
                env = {**os.environ, **self.env, 'CUDA_VISIBLE_DEVICES':str(gpu),
                       'DOJO_GPU_SCOPE':str(gpu), 'DOJO_SWEEP_ID':self.run.name,
                       'ROBODOJO_RUN_ID':run_id, 'DOJO_ROLE':'signal_batch'}
                env.pop('RAY_ADDRESS', None)
                self.update(gpu, phase='BATCH', config=str(config), output=cfg['output'])
                log = self.run / (config.stem + '.log')
                logs.append(log)
                rc = self.execute([self.a.python,'-u','-B',str(self.a.repo/'tools/pi05_signals/run_batch.py'),str(config)],
                                  env, log, gpu, run_id, 3600)
                fatal = child_fatal(rc, log)
                if fatal:
                    self.fatal.set()
                    self.stop.set()
                receipt = self.guard.cleanup(run_id, 'signal_batch_terminal', grace=20)
                validation_receipt = None
                if rc == 0 and not fatal and not self.stop.is_set():
                    self.update(gpu, phase='VALIDATING')
                    log = self.run / (config.stem + '-validation.log')
                    logs.append(log)
                    rc = self.execute([self.a.python,'-u','-B',str(self.a.repo/'tools/pi05_signals/validate_batch.py'),str(config)],
                                      env, log, gpu, run_id, 600)
                    fatal = child_fatal(rc, log)
                    if fatal:
                        self.fatal.set()
                        self.stop.set()
                    validation_receipt = self.guard.cleanup(run_id, 'signal_validation_terminal', grace=5)
                accepted = (not fatal and not self.stop.is_set() and rc == 0 and
                            read(Path(cfg['output'])/'validation.json').get('passed') is True)
                record = read(claim)
                record.update(state='VALIDATED' if accepted else 'FAILED', returncode=rc,
                              cleanup_receipt=receipt, validation_cleanup_receipt=validation_receipt,
                              finished_at=time.time(), fatal=fatal)
                atomic(claim, record)
                claim = None
                if not accepted:
                    with self.lock:
                        self.failures.append(record)
                    if self.a.phase == 'smoke' or fatal or self.stop.is_set():
                        self.stop.set()
                        break
                self.update(gpu, phase='READY', last_result=record['state'])
            self.update(gpu, phase='FINISHED')
        except BaseException:
            self.stop.set()
            error = traceback.format_exc()
            if any(child_fatal(rc, log) for log in logs):
                self.fatal.set()
            record = dict(gpu=gpu, error=error, fatal=self.fatal.is_set())
            if claim is not None:
                record = {**read(claim), **record, 'state': 'ABORTED', 'returncode': rc, 'finished_at': time.time()}
                atomic(claim, record)
            with self.lock:
                self.failures.append(record)
            self.update(gpu,phase='ERROR',error=error)

    def run_all(self):
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self.cancel)
        threads = [threading.Thread(target=self.lane,args=(gpu,)) for gpu in self.a.gpus]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        cleanup = self.guard.cleanup(None, 'signal_queue_terminal', grace=20)
        counts = {}
        for name in self.manifest['configs']:
            claim = self.a.output/'claims'/(Path(name).stem+'.json')
            state = read(claim)['state'] if claim.exists() else 'PENDING'
            counts[state] = counts.get(state,0)+1
        passed = not self.failures and not self.cancelled.is_set() and not self.stop.is_set()
        if passed and signal_fingerprint(self.a.output, self.a.repo) != self.fingerprint:
            self.failures.append({'error': 'Signal inputs changed before controller completion'})
            passed = False
        summary = dict(time=time.time(), passed=passed, phase=self.a.phase,
                       collection_finished_normally=(self.a.phase == 'formal' and
                           not self.cancelled.is_set() and not self.stop.is_set()),
                       cancelled=self.cancelled.is_set(), fatal=self.fatal.is_set(),
                       acceptance_scope='this controller only', global_counts=counts,
                       global_all_validated=counts.get('VALIDATED',0)==len(self.manifest['configs']),
                       failures=self.failures, cleanup=cleanup, lanes=self.status)
        atomic(self.run/'summary.json',summary)
        if self.a.phase == 'smoke' and passed:
            try:
                frozen = freeze_existing_smoke(self.a.output, self.a.repo, self.run)
                atomic(self.a.output/'smoke-passed.json', {**summary, 'batches': self.configs,
                       'frozen_acceptance': str(self.a.output/'frozen-smoke-acceptance.json'),
                       'fingerprint': frozen['fingerprint']})
            except BaseException:
                self.failures.append({'error': traceback.format_exc()})
                summary.update(passed=False, failures=self.failures)
                atomic(self.run/'summary.json', summary)
        print(json.dumps(summary),flush=True)
        # Native task-script failures are retained as FAILED batches; after the
        # formal queue drains normally they must not strand the saved Dojo lane.
        return 99 if self.fatal.is_set() else (0 if summary['passed'] or summary['collection_finished_normally'] else 1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--freeze-smoke',type=Path,help='Freeze completed smoke evidence; no model/environment launch')
    p.add_argument('--phase',choices=['smoke','formal'])
    p.add_argument('--gpus',type=int,nargs='+')
    p.add_argument('--run-dir',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--guard-source',type=Path)
    p.add_argument('--python')
    args=p.parse_args()
    args.output, args.repo = args.output.resolve(), args.repo.resolve()
    if args.freeze_smoke:
        if any(value is not None for value in (args.phase, args.gpus, args.run_dir, args.guard_source, args.python)):
            p.error('--freeze-smoke only takes --output and --repo')
        print(json.dumps(freeze_existing_smoke(args.output, args.repo, args.freeze_smoke)), flush=True)
        return
    if not all((args.phase, args.gpus, args.run_dir, args.guard_source, args.python)):
        p.error('queue launch requires --phase, --gpus, --run-dir, --guard-source and --python')
    if any(g not in (4,5,6,7) for g in args.gpus) or len(set(args.gpus))!=len(args.gpus):
        raise ValueError('Only physical GPU4/5/6/7, each once')
    sys.exit(Queue(args).run_all())


if __name__=='__main__':
    main()
