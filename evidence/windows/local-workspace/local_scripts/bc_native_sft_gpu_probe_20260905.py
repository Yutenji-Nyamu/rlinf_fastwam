"""One real FSDP optimizer update on existing BC data; no rollout or saved policy."""
import collections
import json
import os
import tempfile
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.distributed.device_mesh import init_device_mesh

from rlinf.data.online_bc import SuccessReplay
from rlinf.hybrid_engines.fsdp.strategy.base import FSDPStrategyBase
from rlinf.models import get_model
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.utils.logging import get_logger

assert os.getuid() == 1003
base = Path('/data/chenyiteng/results/rlinf-shenzhen/online-bc')
cfg = OmegaConf.load(base/'implementation-20260905/smoke-resolved-compose.yaml')
assert cfg.actor.model.precision is None
assert cfg.actor.fsdp_config.mixed_precision.param_dtype is None
assert not cfg.actor.model.openpi.image_augmentation
assert cfg.actor.micro_batch_size == 32 and cfg.actor.global_batch_size == 1024
os.environ['LOCAL_RANK'] = '6'
torch.cuda.set_device(6)
torch.manual_seed(cfg.actor.seed)
diag = base/'native-sft-probe-20260905'
diag.mkdir(exist_ok=False)
(diag/'resolved.yaml').write_text(OmegaConf.to_yaml(cfg, resolve=True))
episodes = torch.load(
    base/'pi0-adjust-bottle-smoke32x1-m4-gpu6-20260905-v3/success_data/rank_0/batch_000000.pt',
    map_location='cpu', weights_only=True,
)
replay = SuccessReplay(cfg.actor.seed, str(diag/'unused_archive'))
replay.records = [row for episode in episodes for row in episode]
print('SOURCE_SUCCESS_EPISODES', len(episodes), 'QUERIES', len(replay), flush=True)

with tempfile.TemporaryDirectory(prefix='bc-fsdp-', dir=diag) as rendezvous:
    torch.distributed.init_process_group(
        'nccl', init_method='file://'+str(Path(rendezvous)/'store'), rank=0, world_size=1,
    )
    try:
        model = get_model(cfg.actor.model)
        trainable = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
        assert not any('paligemma_with_expert.paligemma.' in n for n,p in trainable)
        print('NATIVE_TRAINABLE_DTYPES', dict(collections.Counter({
            str(dtype):sum(p.numel() for n,p in trainable if p.dtype==dtype)
            for dtype in {p.dtype for n,p in trainable}
        })), flush=True)
        strategy = FSDPStrategyBase.create(cfg.actor, 1, None, get_logger())
        model = strategy.wrap_model(model, init_device_mesh('cuda', (1,), mesh_dim_names=('fsdp',)))
        model.train()
        optim_cfg = cfg.actor.optim
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=optim_cfg.lr,
            betas=(optim_cfg.adam_beta1, optim_cfg.adam_beta2),
            eps=optim_cfg.adam_eps, weight_decay=optim_cfg.weight_decay,
        )
        batches = replay.sample(1024)['forward_inputs']
        batches = {k:v.to('cuda:6') for k,v in batches.items()}
        watched = next(p for n,p in model.named_parameters() if 'action_out_proj' in n and n.endswith('weight'))
        before = watched.detach().clone()
        optimizer.zero_grad()
        losses=[]
        for index in range(32):
            batch={k:v[index*32:(index+1)*32] for k,v in batches.items()}
            prepared=model.prepare_dagger_sft_batch(batch)
            if index == 0:
                print('SFT_INPUT_DTYPES', {k:str(v.dtype) for k,v in prepared['observation'].images.items()}, flush=True)
            loss=model(forward_type=ForwardType.SFT, data=prepared, use_action_chunk_loss=True,
                       action_valid_mask=batch['action_valid_mask'])
            assert torch.isfinite(loss)
            (loss/32).backward()
            losses.append(loss.item())
            if index in (0,15,31): print('MICRO_BATCH_DONE',index+1,'LOSS',loss.item(),flush=True)
        norm=model.clip_grad_norm_(optim_cfg.clip_grad)
        assert torch.isfinite(norm)
        assert all(p.grad is None for n,p in model.named_parameters() if not p.requires_grad)
        optimizer.step()
        change=(watched.detach()-before).abs().max().item()
        assert change > 0
        stats={'passed':True,'optimizer_updates':1,'micro_batch':32,'global_batch':1024,
               'mean_fm_loss':sum(losses)/len(losses),'grad_norm':norm.item(),
               'action_out_projection_max_delta':change,
               'cuda_peak_allocated_gib':torch.cuda.max_memory_allocated(6)/1024**3,
               'cuda_peak_reserved_gib':torch.cuda.max_memory_reserved(6)/1024**3,
               'checkpoint_saved':False,'environment_created':False}
        (diag/'result.json').write_text(json.dumps(stats,indent=2)+'\n')
        print('REAL_SFT_PROBE_RESULT',json.dumps(stats),flush=True)
    finally:
        torch.distributed.destroy_process_group()
