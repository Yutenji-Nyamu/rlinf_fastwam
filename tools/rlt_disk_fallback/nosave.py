"""Last-priority, actual RLT training with in-memory replay and no new checkpoints."""
import copy
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import subprocess
import sys
import time
from common import read, save


def latest_complete(q):
    root = Path(q['checkpoint']).parent
    candidates = sorted(root.glob('global_step_*'), key=lambda p: int(p.name.rsplit('_', 1)[1]), reverse=True)
    for cp in candidates:
        marker = cp / 'actor/sac_components/rlt_trainer_state/complete.json'
        try:
            meta = read(marker)
        except (OSError, ValueError):
            continue
        weights = ((cp / 'actor/dcp_checkpoint/.metadata').is_file()
                   and (cp / 'actor/dcp_checkpoint/__0_0.distcp').is_file())
        weights |= (cp / 'actor/local_shard_checkpoint/checkpoint_rank_0.pt').is_file()
        if meta.get('complete') and weights:
            return cp
    raise RuntimeError('No complete RLT resume checkpoint; refusing a silent fresh start')


def make_config(cfg, rt, cp):
    cfg = copy.deepcopy(cfg)
    runner = cfg['runner']
    runner['save_interval'] = -1  # Native check_progress disables periodic AND final saves.
    runner['resume_dir'] = str(cp)
    # Keep the regular budget unless it has already ended; tail service then continues RLT.
    resume_step = int(cp.name.rsplit('_', 1)[1])
    if resume_step >= runner['max_epochs']:
        runner['max_epochs'] = resume_step + 3000
    if 0 <= runner.get('max_steps', -1) <= resume_step:
        runner['max_steps'] = -1
    runner['logger'].update(log_path=str(rt / 'logs'), logger_backends=[])
    runner['per_worker_log'] = False
    runner['per_worker_log_path'] = None
    cfg['algorithm']['dv_observe'] = False  # Extra diagnostic arrays, not a training signal.
    for name in ('replay_buffer', 'demo_buffer'):
        if cfg['algorithm'].get(name):
            cfg['algorithm'][name]['auto_save'] = False
    for phase in ('train', 'eval'):
        env = cfg['env'][phase]
        env['video_cfg'].update(save_video=False, video_base_dir=str(rt / 'unused-video' / phase))
        env['task_config'].update(save_path=str(rt / 'unused-data' / phase), collect_data=False, eval_video_log=False)
    return cfg


def prepare(plan, slot, gpu):
    normal_request = slot['fallback']
    q = read(normal_request)
    if q.get('normal_request'):
        normal_request = q['normal_request']
        q = read(normal_request)
    assert q['kind'] == 'rlt' and q['gpu'] == int(gpu)
    root = Path(plan['no_checkpoint_root'])
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    assert root.stat().st_uid == os.getuid() and not root.is_symlink()
    rt = root / f'g{gpu}-{time.time_ns()}'
    rt.mkdir()
    cfg = read(Path(q['runtime']) / 'resolved.yaml')
    cp = latest_complete(q)
    cfg = make_config(cfg, rt, cp)
    env = read(Path(q['runtime']) / 'environment.json')
    env.update(RLT_LOG_ROOT=str(rt / 'logs'), TMPDIR=str(rt / 'tmp'))
    env.update(CUDA_CACHE_PATH=str(rt / 'cuda-cache'), TRITON_CACHE_DIR=str(rt / 'triton-cache'))
    (rt / 'tmp').mkdir()
    save(rt / 'resolved.yaml', cfg)
    save(rt / 'environment.json', env)
    q.update(runtime=str(rt), namespace=f'rlt-nosave-g{gpu}-{time.time_ns()}',
             checkpoint=str(cp), step=int(cp.name.rsplit('_', 1)[1]),
             no_checkpoint=True, normal_request=normal_request)
    # q.run keeps the normal checkpoint source for existing preemption tools.
    save(rt / 'request.json', q)
    return str(rt / 'request.json')


def drain_log(pipe, path):
    """Independent CPU log sink survives owner restarts and exits on driver EOF."""
    child = subprocess.Popen([sys.executable, '-u', '-B', __file__, 'sink', str(path)],
                             stdin=pipe, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             start_new_session=True)
    pipe.close()
    return child


def log_sink(path):
    # At most 2 x 8 MiB per driver. Logging failure must not block actual training.
    handler = None
    logging.raiseExceptions = False
    try:
        handler = RotatingFileHandler(path, maxBytes=8*1024**2, backupCount=1, encoding='utf-8')
    except OSError:
        pass
    try:
        while True:
            chunk = sys.stdin.buffer.read(8192)
            if not chunk:
                break
            if handler:
                record = logging.LogRecord('rlt', logging.INFO, '', 0, chunk.decode(errors='replace'), (), None)
                handler.emit(record)
    finally:
        if handler:
            handler.close()


if __name__ == '__main__':
    assert sys.argv[1] == 'sink'
    log_sink(Path(sys.argv[2]))
