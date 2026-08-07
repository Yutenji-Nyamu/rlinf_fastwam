"""Two-rank FSDP fixture for OGPO online/EMA shard pairing and FP32 shadow."""

from __future__ import annotations

import os
from types import SimpleNamespace

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

from rlinf.models.embodiment.modules.ogpo_modules import OGPOEMAActionExpert


class ToyDecoderLayer(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.proj = nn.Linear(width, width)

    def forward(self, inputs):
        return torch.tanh(self.proj(inputs))


class ToyExpertModel(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.layers = nn.ModuleList([ToyDecoderLayer(width) for _ in range(2)])
        self.norm = nn.LayerNorm(width)
        self.config = SimpleNamespace(_attn_implementation=None)

    def forward(self, inputs_embeds, **_):
        hidden = inputs_embeds
        for layer in self.layers:
            hidden = layer(hidden)
        return SimpleNamespace(last_hidden_state=self.norm(hidden))


class ToyGemmaExpert(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.model = ToyExpertModel(width)
        self.lm_head = nn.Linear(width, width)


class ToyPaliWithExpert(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.gemma_expert = ToyGemmaExpert(width)


class ToyOnlinePi0(nn.Module):
    def __init__(self):
        super().__init__()
        width = 32
        action_dim = 8
        self.pi05 = False
        self.config = SimpleNamespace(action_horizon=4)
        self.paligemma_with_expert = ToyPaliWithExpert(width)
        self.action_in_proj = nn.Linear(action_dim, width)
        self.action_out_proj = nn.Linear(width, action_dim)
        self.state_proj = nn.Linear(action_dim, width)
        self.action_time_mlp_in = nn.Linear(2 * width, width)
        self.action_time_mlp_out = nn.Linear(width, width)
        self.to(dtype=torch.bfloat16)
        self.ogpo_target = OGPOEMAActionExpert(self)
        self.ogpo_target.match_online_dtypes_(self)
        self.ogpo_target.copy_from_online_(self)

    def forward(self, operation: str, **kwargs):
        if operation == "ema":
            self.ogpo_target.polyak_update_from_online_(self, float(kwargs["tau"]))
            return None
        if operation == "shadow":
            return self.ogpo_target.ema_shadow_state()
        if operation == "load_shadow":
            self.ogpo_target.load_ema_shadow_state_(kwargs["state"], self)
            return None
        raise ValueError(operation)


def main() -> None:
    torch.distributed.init_process_group("nccl")
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    if world_size != 2:
        raise RuntimeError(f"fixture requires two ranks, got {world_size}")
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)

    torch.manual_seed(1234)
    module = ToyOnlinePi0().to(device)
    model = FSDP(
        module,
        auto_wrap_policy=ModuleWrapPolicy(
            {ToyDecoderLayer, nn.LayerNorm, nn.Linear}
        ),
        use_orig_params=True,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=device,
        sync_module_states=True,
    )

    local_pairs = list(
        model.module.ogpo_target._named_parameter_pairs(model.module)
    )
    if not local_pairs:
        raise RuntimeError("FSDP fixture found no EMA parameter pairs")
    old_targets = {
        name: target.detach().float().clone()
        for name, target, _ in local_pairs
    }
    with torch.no_grad():
        for _, _, source in local_pairs:
            source.add_(torch.full_like(source, 0.125))

    model(operation="ema", tau=0.25)
    updated_pairs = list(
        model.module.ogpo_target._named_parameter_pairs(model.module)
    )
    max_update_error = 0.0
    for name, target, _ in updated_pairs:
        expected = old_targets[name] + 0.03125
        if target.numel():
            max_update_error = max(
                max_update_error,
                float((target.float() - expected).abs().max().item()),
            )
    if max_update_error > 0.01:
        raise RuntimeError(f"visible EMA update differs: {max_update_error}")

    shadow = model(operation="shadow")
    if not shadow or any(value.dtype != torch.float32 for value in shadow.values()):
        raise RuntimeError("fixture did not create an FP32 EMA shadow")
    snapshot_path = f"/tmp/ogpo_fsdp_ema_shadow_rank_{rank}.pt"
    torch.save(shadow, snapshot_path)
    restored = torch.load(snapshot_path, map_location="cpu", weights_only=False)
    model(operation="load_shadow", state={})
    model(operation="load_shadow", state=restored)
    roundtrip = model(operation="shadow")
    if roundtrip.keys() != shadow.keys():
        raise RuntimeError("EMA shadow keys changed after round-trip")
    for name, value in shadow.items():
        torch.testing.assert_close(roundtrip[name], value)

    result = {
        "rank": rank,
        "pairs": len(local_pairs),
        "shadow_tensors": len(shadow),
        "shadow_elements": sum(value.numel() for value in shadow.values()),
        "max_update_error": max_update_error,
    }
    gathered = [None] * world_size
    torch.distributed.all_gather_object(gathered, result)
    if rank == 0:
        print(f"FSDP_EMA_FIXTURE_OK results={gathered}", flush=True)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
