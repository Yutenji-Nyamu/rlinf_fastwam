#!/usr/bin/env python3
"""One real success-query FM oracle/backward/update check on an authorized GPU.

Usage: python tools/fastwam_bc_real_query_probe.py --config runtime/resolved.yaml
       --success-data RUN/success_data --out RUN/fastwam-bc-probe.json
Alternatively --demo-data uses a real clean50 demonstration for an offline input
contract check. That mode explicitly does not verify online success admission.
"""
import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.fastwam.builder import build_fastwam_policy
from rlinf.models.embodiment.fastwam.fastwam_rl import (
    build_action_conditioning, predict_action_velocity,
)
from rlinf.models.embodiment.fastwam.online_bc import native_action_fm_loss
from rlinf.models.embodiment.fastwam.rlt_stage1_data import lerobot_frame_to_env_obs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--success-data")
    source.add_argument("--demo-data")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    policy = build_fastwam_policy(config.actor.model, torch_dtype=torch.bfloat16)
    if args.success_data:
        paths = sorted(Path(args.success_data).rglob("batch_*.pt"))
        assert paths, "A real successful collection archive is required"
        episodes = torch.load(paths[0], map_location="cpu", weights_only=True)
        record = episodes[0][0]
        batch = {k: v.unsqueeze(0) for k, v in record.items()}
        source_record = dict(source_archive=str(paths[0]),
            query_idx=int(record["query_idx"]), online_success_chain_verified=True)
    else:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        from rlinf.data.storage.lerobot import resolve_lerobot_repo_id
        root = Path(args.demo_data)
        fps = float(json.loads((root / "meta/info.json").read_text())["fps"])
        dataset = LeRobotDataset(repo_id=resolve_lerobot_repo_id(str(root)),
            root=root, download_videos=False,
            delta_timestamps={"action": [i / fps for i in range(24)]})
        frame = dataset[0]
        env_obs = lerobot_frame_to_env_obs(frame, "Adjust the bottle.")
        action = torch.as_tensor(frame["action"]).float()
        assert action.shape == (24, 14), action.shape
        padding = torch.as_tensor(frame.get("action_is_pad", torch.zeros(24))).bool()
        assert not padding.any(), "Choose a full first demonstration command chunk"
        policy.eval()
        torch.manual_seed(42)
        predicted, result = policy.predict_action_batch(env_obs, online_bc=True)
        assert predicted.shape == (1, 24, 14) and torch.isfinite(predicted).all()
        # The ODE output checks the inference path; supervision uses the actual
        # demonstration commands, never the model's unexecuted prediction.
        batch = dict(result["forward_inputs"])
        batch["action"] = action.reshape(1, -1)
        batch["action_valid_mask"] = torch.ones(1, 24, 14).bool()
        source_record = dict(source_demonstration=str(root), frame_index=0,
            online_success_chain_verified=False, offline_demonstration_contract=True,
            ode_prediction_shape=list(predicted.shape))
    policy.train()
    prepared = policy.prepare_dagger_sft_batch(batch)
    action = prepared["targets"].to(policy.device, policy.model_dtype)
    scheduler = policy.model.train_action_scheduler
    torch.manual_seed(1234)
    noise = torch.randn_like(action)
    timestep = scheduler.sample_training_t(1, policy.device, action.dtype)
    noisy = scheduler.add_noise(action, noise, timestep)
    target = scheduler.training_target(action, noise, timestep)
    cond = build_action_conditioning(policy.model, input_image=prepared["image"],
        text_context=prepared["text_context"], text_context_mask=prepared["text_context_mask"],
        proprio=prepared["proprio"], action_horizon=32)
    with torch.no_grad():
        oracle_prediction = policy.model._denoise_action_with_video_cache(
            latents_action=noisy, timestep_action=timestep,
            context=cond.context, context_mask=cond.context_mask,
            video_cache_k=cond.video_cache_k, video_cache_v=cond.video_cache_v,
            action_attention_mask=cond.action_attention_mask).float()
        prediction = predict_action_velocity(policy.model, x=noisy,
            raw_timestep=timestep, conditioning=cond)
        torch.testing.assert_close(prediction, oracle_prediction, atol=1e-5, rtol=1e-5)
        oracle_loss = native_action_fm_loss(oracle_prediction, target,
            prepared["valid_mask"].to(policy.device), timestep, scheduler)
        oracle_loss = float(policy.model.loss_lambda_action) * oracle_loss
    del cond
    trainable = [(n, p) for n, p in policy.named_parameters() if p.requires_grad]
    assert trainable and all("mot.mixtures.action." in n for n, p in trainable)
    optimizer = torch.optim.Adam([p for n, p in trainable], lr=2.5e-5,
        betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-10)
    optimizer.zero_grad(set_to_none=True)
    loss = policy(forward_type=ForwardType.SFT, data=prepared, noise=noise, timestep=timestep)
    torch.testing.assert_close(loss, oracle_loss, atol=1e-5, rtol=1e-5)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is None for p in policy.parameters() if not p.requires_grad)
    grad_norm = torch.nn.utils.clip_grad_norm_([p for n, p in trainable], 1.0)
    assert torch.isfinite(grad_norm) and grad_norm > 0
    # Keep only a small action projection copy, not another model state in VRAM.
    name, changed = next((n, p) for n, p in trainable
        if p.grad is not None and p.grad.abs().max() > 0 and p.numel() < 2000000
        and n.endswith("bias"))
    before = changed.detach().clone()
    optimizer.step()
    max_delta = (changed.detach().float() - before.float()).abs().max().item()
    assert max_delta > 0, "Adam must update a real action-expert parameter"
    out = dict(status="pass", **source_record,
        targets=list(action.shape), valid_positions=int(prepared["valid_mask"][0].any(-1).sum()),
        loss=float(loss), oracle_loss=float(oracle_loss), grad_norm=float(grad_norm),
        updated_parameter=name, parameter_max_delta=max_delta,
        frozen_gradients_absent=True, trainable_parameters=sum(p.numel() for n, p in trainable),
        cuda_peak_gib=torch.cuda.max_memory_allocated()/2**30)
    Path(args.out).write_text(json.dumps(out, indent=2)+"\n")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
