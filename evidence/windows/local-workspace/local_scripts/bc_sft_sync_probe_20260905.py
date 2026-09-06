"""Focused real-model update -> export -> checkpoint round trip; no environment."""
import collections
import json
import os
import tempfile
import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--use-orig-params', choices=['true','false'],required=True)
args=parser.parse_args()
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.distributed.device_mesh import init_device_mesh

from rlinf.data.online_bc import SuccessReplay
from rlinf.hybrid_engines.fsdp.strategy.base import FSDPStrategyBase
from rlinf.models import get_model
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.utils.logging import get_logger
from rlinf.utils.utils import warmup_optimizer_state

assert os.getuid() == 1003
base = Path('/data/chenyiteng/results/rlinf-shenzhen/online-bc')
cfg = OmegaConf.load(base/'implementation-20260905/smoke-resolved-compose.yaml')
assert cfg.actor.model.precision is None
assert cfg.actor.fsdp_config.mixed_precision.param_dtype is None
assert not cfg.actor.model.openpi.image_augmentation
cfg.actor.fsdp_config.use_orig_params = args.use_orig_params == 'true'
cfg.actor.fsdp_config.wrap_policy = dict(
    transformer_layer_cls_to_wrap=['GemmaMLP','SiglipVisionEmbeddings','GemmaRMSNorm','GemmaRotaryEmbedding'],
    no_split_names=['q_proj','k_proj','v_proj','o_proj','action_in_proj','action_out_proj',
                    'lm_head','state_proj','action_time_mlp_in','action_time_mlp_out',
                    'time_mlp_in','time_mlp_out'],
)
assert cfg.actor.micro_batch_size == 32 and cfg.actor.global_batch_size == 1024
os.environ['LOCAL_RANK'] = '6'
torch.cuda.set_device(6)
torch.manual_seed(cfg.actor.seed)
diag = base/('sft-leaf-wrap-nativeopt-local-orig-'+args.use_orig_params+'-20260905')
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
        expected = set(strategy.get_model_state_dict(model, False, False))
        print('INITIAL_EXPORT_KEYS',len(expected),flush=True)
        strategy.offload_param_and_grad(model,False)
        strategy.onload_param_and_grad(model,torch.device('cuda:6'),False)
        assert set(strategy.get_model_state_dict(model,False,False))==expected
        strategy.offload_param_and_grad(model,False)
        strategy.onload_param_and_grad(model,torch.device('cuda:6'),False)
        model.train()
        optim_cfg = cfg.actor.optim
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=optim_cfg.lr,
            betas=(optim_cfg.adam_beta1, optim_cfg.adam_beta2),
            eps=optim_cfg.adam_eps, weight_decay=optim_cfg.weight_decay,
        )
        # Match RLinf's production optimizer setup, including unused head state.
        warmup_optimizer_state(optimizer)
        batches = replay.sample(1024)['forward_inputs']
        batches = {k:v.to('cuda:6') for k,v in batches.items()}
        watched = next(p for n,p in model.named_parameters() if 'action_out_proj' in n)
        before = watched.detach().clone()
        optimizer.zero_grad()
        losses=[]
        for index in range(64):
            offset=index%32
            batch={k:v[offset*32:(offset+1)*32] for k,v in batches.items()}
            prepared=model.prepare_dagger_sft_batch(batch)
            if index == 0:
                print('SFT_INPUT_DTYPES', {k:str(v.dtype) for k,v in prepared['observation'].images.items()}, flush=True)
            loss=model(forward_type=ForwardType.SFT, data=prepared, use_action_chunk_loss=True,
                       action_valid_mask=batch['action_valid_mask'])
            assert torch.isfinite(loss)
            (loss/32).backward()
            losses.append(loss.item())
            if index in (0,15,31,32,47,63): print('MICRO_BATCH_DONE',index+1,'LOSS',loss.item(),flush=True)
            if offset==31:
                norm=model.clip_grad_norm_(optim_cfg.clip_grad)
                assert torch.isfinite(norm)
                optimizer.step()
                print('OPTIMIZER_UPDATE_DONE',(index+1)//32,flush=True)
                optimizer.zero_grad()
        assert all(p.grad is None for n,p in model.named_parameters() if not p.requires_grad)
        change=(watched.detach()-before).abs().max().item()
        assert change > 0
        print('POST_UPDATE_EXPORT_BEGIN',flush=True)
        sd=strategy.get_model_state_dict(model,False,False)
        assert set(sd)==expected
        del sd
        print('POST_UPDATE_EXPORT_PASSED',flush=True)
        strategy.offload_optimizer(optimizer)
        strategy.offload_param_and_grad(model,False)
        strategy.onload_param_and_grad(model,torch.device('cuda:6'),False)
        strategy.onload_optimizer(optimizer,torch.device('cuda:6'))
        sd=strategy.get_model_state_dict(model,False,False)
        assert set(sd)==expected
        del sd
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.0)
        checkpoint_format='local_shard'
        strategy.save_checkpoint(model,[optimizer],[scheduler],str(diag/'checkpoint'),checkpoint_format=checkpoint_format)
        saved=watched.detach().clone()
        saved_opt={k:v.detach().clone() if torch.is_tensor(v) else v for k,v in optimizer.state[watched].items()}
        with torch.no_grad(): watched.add_(0.1)
        strategy.load_checkpoint(model,[optimizer],[scheduler],str(diag/'checkpoint'),checkpoint_format=checkpoint_format)
        assert torch.equal(watched.detach(),saved)
        for k,v in saved_opt.items():
            if torch.is_tensor(v): assert torch.equal(optimizer.state[watched][k],v)
            else: assert optimizer.state[watched][k]==v
        assert set(strategy.get_model_state_dict(model,False,False))==expected
        print('CHECKPOINT_RESTORE_PASSED',flush=True)
        stats={'passed':True,'optimizer_updates':2,'micro_batch':32,'global_batch':1024,
               'mean_fm_loss':sum(losses)/len(losses),'grad_norm':norm.item(),
               'action_out_projection_max_delta':change,
               'cuda_peak_allocated_gib':torch.cuda.max_memory_allocated(6)/1024**3,
               'cuda_peak_reserved_gib':torch.cuda.max_memory_reserved(6)/1024**3,
               'checkpoint_saved':True,'checkpoint_restore_equal':True,'export_keys':len(expected),
               'use_orig_params':cfg.actor.fsdp_config.use_orig_params,'environment_created':False}
        (diag/'result.json').write_text(json.dumps(stats,indent=2)+'\n')
        print('REAL_SFT_PROBE_RESULT',json.dumps(stats),flush=True)
    finally:
        torch.distributed.destroy_process_group()
