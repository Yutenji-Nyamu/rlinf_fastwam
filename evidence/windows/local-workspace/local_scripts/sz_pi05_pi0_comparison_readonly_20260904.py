"""Read-only audit: three scoped GRPO controls, exact configs and source text."""
from pathlib import Path
import datetime, hashlib, json, re, subprocess
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

base = Path('/data/chenyiteng/results/rlinf-shenzhen')
wt = Path('/data/chenyiteng/projects/rlinf-shenzhen/worktrees')
runs = {
    'pi0': ('grpo', 'grpo-control-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v2'),
    'official_pi05': ('pi05', 'pi05-grpo-control-formal100-2gpu64x4-g8-b1024-u2-m5-fixed32-eval5-phys45-localshard-v2'),
    'sidney': ('pi05-sidney', 'move-pillbottle-pad-grpo-formal100-2gpu64x4-g8-b1024-u2-m10-noise0p5-h200-fixed32-eval5-phys45-localshard-v1'),
}
def emit(kind, value):
    print(kind, json.dumps(value, ensure_ascii=False), flush=True)
def command(args):
    p = subprocess.run(args, capture_output=True, text=True, timeout=30)
    return {'rc': p.returncode, 'out': p.stdout, 'err': p.stderr}
def source(root, rel):
    p = root / rel
    if p.is_file():
        raw = p.read_bytes()
        emit('SOURCE_JSON', {'root': str(root), 'rel': rel,
                            'sha256': hashlib.sha256(raw).hexdigest(),
                            'text': raw.decode(errors='replace')})
emit('TIME_JSON', {'cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()})
for key, (family, name) in runs.items():
    root = base / family / 'runs' / name
    d = {'key': key, 'path': str(root)}
    d['config'] = yaml.safe_load((root / 'runtime/resolved.yaml').read_text())
    ea = EventAccumulator(str(root / 'tensorboard'), size_guidance={'scalars': 0})
    ea.Reload()
    d['scalars'] = {tag: [{'step': p.step + 1, 'value': p.value, 'wall_time': p.wall_time}
                         for p in ea.Scalars(tag)] for tag in ea.Tags()['scalars']}
    log = (root / 'runtime/driver.log').read_text(errors='replace')
    d['progress'] = re.findall(r'Global Step:\s*\d+/\d+', log)[-1:]
    d['log_start'] = log.splitlines()[:70]
    d['phase'] = [x for x in log.splitlines() if 'Generating Rollout Epochs:' in x][-3:]
    d['state'] = {p.name: p.read_text().strip() for p in (root / 'runtime').iterdir()
                  if p.name in ['wrapper.pid', 'exit_code.txt', 'started_at.txt', 'finished_at.txt']}
    d['wrapper_alive'] = Path('/proc', d['state'].get('wrapper.pid', '-1')).exists()
    d['errors'] = {x: log.lower().count(x.lower()) for x in ['Fatal Python error', 'OIDN Error', 'CUDA out of memory', 'Traceback', 'RuntimeError:', 'non-finite']}
    d['trainable_logs'] = [x for x in log.splitlines() if re.search('trainable|freeze|denois|num_steps|model_path', x, re.I)][:35]
    d['runtime_names'] = [p.name for p in (root / 'runtime').iterdir() if p.is_file()]
    d['checkpoints'] = [{'path': str(p), 'bytes': p.stat().st_size} for p in root.glob('*/checkpoints/global_step_*/actor/**/*') if p.is_file()]
    emit('RUN_JSON', d)

for name in ['pi0-dvac-grpo-current', 'pi05-robotwin-rl', 'sidney-pi05-current-rlinf']:
    root = wt / name
    emit('GIT_JSON', {'root': str(root), 'head': command(['git', '--no-optional-locks', '-C', str(root), 'rev-parse', 'HEAD']), 'dirty': command(['git', '--no-optional-locks', '-C', str(root), 'status', '--short'])})
    files = ['rlinf/algorithms/advantages.py', 'rlinf/algorithms/losses.py',
             'rlinf/workers/actor/fsdp_actor_worker.py', 'rlinf/envs/robotwin/robotwin_env.py',
             'rlinf/workers/env/env_worker.py', 'rlinf/utils/utils.py']
    files += [str(p.relative_to(root)) for p in (root / 'rlinf/models/embodiment/openpi').glob('*.py')]
    files += [str(p.relative_to(root)) for p in (root / 'rlinf/models/embodiment/openpi/dataconfig').glob('*robotwin*')]
    for rel in files:
        source(root, rel)
    for split in ['train', 'eval']:
        p = root / f'rlinf/envs/robotwin/seeds/{split}_seeds.json'
        data = json.loads(p.read_text())
        emit('SEEDS_JSON', {'root': name, 'split': split, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                           'tasks': {task: data.get(task) for task in ['adjust_bottle', 'move_pillbottle_pad']}})

robotwin = Path('/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support')
for rel in ['envs/adjust_bottle.py', 'envs/move_pillbottle_pad.py', 'envs/_base_task.py',
            'envs/robotwin_env.py', 'envs/vector_env.py', 'robotwin/envs/robotwin_env.py',
            'robotwin/envs/vector_env.py', 'env_cfg/task_config/_eval_step_limit.yml',
            'task_config/_eval_step_limit.yml']:
    source(robotwin, rel)
emit('ROBOTWIN_FILES_JSON', command(['find', str(robotwin / 'robotwin'), '-maxdepth', '3', '-type', 'f', '-name', '*.py']))
