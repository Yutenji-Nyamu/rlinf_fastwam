"""Actual load method: device restore order and existing pool/counter contract."""
import ast
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch


@pytest.mark.parametrize('offloaded', [False, True])
def test_actual_bc_load_device_and_state(tmp_path, offloaded):
    source = Path('rlinf/workers/actor/fsdp_online_bc_policy_worker.py')
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'EmbodiedOnlineBCFSDPPolicy')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'load_checkpoint')
    scope = {'Path': Path, 'torch': torch, 'audit_online_bc_resume': lambda worker, path: worker.calls.append('audit')}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), scope)
    target = tmp_path/'online_bc/rank_0';target.mkdir(parents=True)
    torch.save({'update_step':500}, target/'learner.pt')
    is_new = 'dvac_new' in source.read_text()
    settings = {'version':1,'normalization':'two_level_batch'}
    if is_new: torch.save(settings, target/'dvac_new.pt')
    worker = SimpleNamespace(_rank=0, calls=[], is_weight_offloaded=offloaded,
      is_optimizer_offloaded=offloaded, device='cuda:0', model=object(), optimizer=object(),
      lr_scheduler=object(), checkpoint_format='local_shard', update_step=0, dvac=None,
      dvac_normalization='two_level_batch' if is_new else 'recent', dvac_new_state=lambda:settings,
      validate_dvac_record=lambda record:None)
    worker.load_param_and_grad=lambda device:worker.calls.append('parameters')
    worker.load_optimizer=lambda device:worker.calls.append('optimizer_device')
    def load_model(**kwargs):
        assert not worker.is_weight_offloaded and not worker.is_optimizer_offloaded
        assert kwargs['optimizers']==[worker.optimizer] and kwargs['lr_schedulers']==[worker.lr_scheduler]
        worker.calls.append('load_model_optimizer_scheduler')
    worker._strategy=SimpleNamespace(load_checkpoint=load_model)
    worker.replay_buffer=SimpleNamespace(records=[{'example':1}],load_checkpoint=lambda p:worker.calls.append('pool'))
    scope['load_checkpoint'](worker,str(tmp_path))
    assert worker.update_step==500 and worker.calls[-1]=='audit'
    if offloaded: assert worker.calls[:2]==['parameters','optimizer_device']
    else: assert 'parameters' not in worker.calls and 'optimizer_device' not in worker.calls
    assert worker.calls.count('load_model_optimizer_scheduler')==worker.calls.count('pool')==1
