"""One full-state rollback on a lost simulator. Never retry a partially executed step."""
import json
import os
import time
from pathlib import Path

from omegaconf import OmegaConf
from common import admission_card, gpus


def prepare_retry(directory, plan, output):
    message=output.read_text(errors='replace')
    if not any(key in message for key in ('SimulatorTransportError:', 'RoboTwin simulator RPC exceeded',
                                           'Fatal Python error: PyGILState_Release')):
        return False
    from wait_owner import checkpoint_ready
    config_path=directory/'config/formal.yaml'
    cfg=OmegaConf.load(config_path)
    prior=Path(cfg.runner.resume_dir)
    root=Path(cfg.runner.logger.log_path)/cfg.runner.logger.experiment_name/'checkpoints'
    valid=[cp for cp in [prior,*root.glob('global_step_*')] if checkpoint_ready(cp,'rlt')]
    if not valid:
        return False
    cp=max(valid,key=lambda c:int(c.name[12:]))
    until=time.monotonic()+400
    while admission_card(gpus()[plan['gpu']],plan['gpu_admission'])['processes']:
        if time.monotonic()>=until:
            return False
        time.sleep(.5)
    cfg.runner.resume_dir=str(cp)
    tmp=config_path.with_suffix('.retry.yaml')
    OmegaConf.save(cfg,tmp)
    os.replace(tmp,config_path)
    (directory/'simulator-retry.json').write_text(json.dumps(dict(checkpoint=str(cp),time=time.time())))
    return True
