"""Read-only ETA inputs; stream via verified SSH, with no server output files."""
import datetime, json, os, re, statistics, subprocess
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

assert os.getuid() == 1003
tz = datetime.timezone(datetime.timedelta(hours=8))
base = Path('/data/chenyiteng/results/rlinf-shenzhen')
paths = {
    'fastwam': base/'fastwam-grpo/runs/fastwam-grpo-control-fresh100-2gpu32x8-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-clean-nooidn-scene-fence-v3',
    'sidney': base/'pi05-sidney/runs/move-pillbottle-pad-grpo-formal100-2gpu64x4-g8-b1024-u2-m10-noise0p5-h200-fixed32-eval5-phys45-localshard-v1',
}
result = {'time': datetime.datetime.now(tz).isoformat(), 'uid': os.getuid(), 'runs': {}}
for key, root in paths.items():
    rt = root/'runtime'
    log = (rt/'driver.log').read_text(errors='replace')
    ea = EventAccumulator(str(root/'tensorboard'), size_guidance={'scalars': 0})
    ea.Reload()
    scalars = {tag: [{'step': x.step+1, 'value': x.value, 'time': x.wall_time} for x in ea.Scalars(tag)]
               for tag in ea.Tags()['scalars'] if tag.startswith('time/') or 'success_once' in tag}
    state = {name: (rt/name).read_text().strip() if (rt/name).exists() else None
             for name in ['wrapper.pid', 'started_at.txt', 'finished_at.txt', 'exit_code.txt']}
    state['alive'] = Path('/proc', state['wrapper.pid']).exists()
    steps = re.findall(r'Global Step:\s*(\d+)\s*/', log)
    state['completed_step'] = int(steps[-1]) if steps else 0
    state['phase'] = [line for line in log.splitlines() if 'Generating Rollout Epochs:' in line][-2:]
    state['log_time'] = datetime.datetime.fromtimestamp((rt/'driver.log').stat().st_mtime,tz).isoformat()
    errors = {x: log.count(x) for x in ['Fatal Python error','CUDA out of memory','OIDN Error','Traceback','RuntimeError:']}
    result['runs'][key] = {'path': str(root), 'state': state, 'errors': errors, 'scalars': scalars}
print('ETA_JSON '+json.dumps(result, ensure_ascii=False))
