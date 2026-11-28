"""FastWAM action-only model bridge for clean EXPO-FT on RoboTwin.

The EXPO learner owns Q, the residual editor and replay. FastWAM owns the
camera/prompt and action transforms, its ODE, and its action FM scheduler.
"""
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from rlinf.algorithms.expo_ft.backend import Pi05Backend, stack_env_observations
from rlinf.models.embodiment.fastwam.builder import build_fastwam_policy
from rlinf.models.embodiment.fastwam.fastwam_rl import (
    build_action_conditioning, flow_sde_rollout, predict_action_velocity,
)
from rlinf.models.embodiment.fastwam.robotwin_adapter import (
    adapt_robotwin_observation, normalize_proprio,
)


def _slice(obs, begin, end):
    return {k: (v[begin:end] if v is not None else None) for k, v in obs.items()}


def _split(size, workers):
    return [(i * size // workers, (i + 1) * size // workers) for i in range(workers)]


def _select_cache(cache, ids):
    # Q and K caches have [batch, sequence, ...] at each layer. The mask
    # spans tokens across all examples and has no batch dimension.
    return replace(cache, context=cache.context.index_select(0, ids),
                   context_mask=cache.context_mask.index_select(0, ids),
                   video_cache_k=[v.index_select(0, ids) for v in cache.video_cache_k],
                   video_cache_v=[v.index_select(0, ids) for v in cache.video_cache_v])


class FastWAMBackend(Pi05Backend):
    HORIZON, CHUNK, ENV_DIM, MODEL_DIM = 32, 24, 14, 14

    def __init__(self, model_cfg, *, device="cuda:0", lr=2.5e-5,
                 betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-10, clip_grad=1.0,
                 candidate_microbatch=2, observation_microbatch=2, fm_microbatch=2,
                 parallel_devices=1, image_augmentation=False, source_head=None,
                 critic_camera_keys=("main", "left_wrist", "right_wrist")):
        if image_augmentation or parallel_devices not in (1, 2):
            raise ValueError("This clean experiment uses one or two GPUs and unaugmented observations")
        self.device = torch.device(device)
        self.parallel_devices = parallel_devices
        self.device_ids = [torch.device("cuda", (self.device.index or 0) + i)
                           for i in range(parallel_devices)]
        self.policies = []
        self.experts = []
        self.prompt_cache = [{} for _ in self.device_ids]
        self.critic_camera_keys = critic_camera_keys
        self.batch_cap = int(model_cfg.model_forward_batch_size)
        self.obs_batch = min(self.batch_cap, int(observation_microbatch))
        self.candidate_batch = int(candidate_microbatch)
        self.ode_batch_per_gpu = min(self.batch_cap, self.candidate_batch)
        self.fm_batch = min(self.batch_cap, int(fm_microbatch))
        if min(self.obs_batch, self.candidate_batch, self.fm_batch) < 1:
            raise ValueError("All microbatch sizes must be positive")
        for dev in self.device_ids:
            with torch.cuda.device(dev):
                policy = build_fastwam_policy(model_cfg, torch_dtype=torch.bfloat16)
                policy.model.device = dev
                self.policies.append(policy)
                self.experts.append({n: p for n, p in policy.model.named_parameters() if p.requires_grad})
        self.model = self.policies[0].model
        # Keep the released BF16 calculation identical to the GRPO evaluation.
        # Store the trainable action expert and Adam updates in FP32.
        self.trainable = {n: torch.nn.Parameter(p.detach().float().clone())
                          for n, p in self.experts[0].items()}
        self.pool = ThreadPoolExecutor(max_workers=parallel_devices)
        self.optimizer = torch.optim.AdamW(self.trainable.values(), lr=lr, betas=betas,
                                           eps=eps, weight_decay=weight_decay)
        self.clip_grad = clip_grad
        self.base_updates = 0
        self.contract = dict(model="FastWAM", checkpoint=Path(model_cfg.model_path).name,
                             normalization=Path(model_cfg.dataset_stats_path).name,
                             horizon=self.HORIZON, chunk=self.CHUNK, dim=self.ENV_DIM,
                             source_head=source_head, steps=self.policies[0].config.num_inference_steps,
                             sigma_shift=self.policies[0].config.sigma_shift, lr=lr, betas=list(betas),
                             eps=eps, weight_decay=weight_decay, clip_grad=clip_grad)
        self.contract_sha256 = hashlib.sha256(json.dumps(self.contract, sort_keys=True).encode()).hexdigest()

    def apply_runtime(self, runtime):
        value = runtime["ode_batch_per_gpu"]
        if type(value) is not int or value not in (2, 4, 8):
            raise ValueError("Unsupported calibrated ODE microbatch")
        self.ode_batch_per_gpu = value
        self.contract["ode_batch_per_gpu"] = value
        self.contract_sha256 = hashlib.sha256(json.dumps(self.contract, sort_keys=True).encode()).hexdigest()

    def _prepare(self, env_obs):
        # The inherited EXPO critic camera transform uses this state.
        return {"state": normalize_proprio(env_obs["states"], self.policies[0].processor).to(self.device)}

    @torch.no_grad()
    def _condition(self, rank, obs):
        policy = self.policies[rank]
        dev = self.device_ids[rank]
        image, state, prompts = adapt_robotwin_observation(
            obs, policy.processor, device=dev, dtype=policy.model_dtype)
        texts = self.prompt_cache[rank]
        for prompt in dict.fromkeys(prompts):
            if prompt not in texts:
                texts[prompt] = policy.model.encode_prompt([prompt])
        context, mask = (torch.cat([texts[p][j] for p in prompts]) for j in (0, 1))
        return build_action_conditioning(policy.model, input_image=image, text_context=context,
                                         text_context_mask=mask, proprio=state,
                                         action_horizon=self.HORIZON, tiled=policy.config.tiled)

    @torch.no_grad()
    def _predict(self, rank, obs, noises):
        dev = self.device_ids[rank]
        policy = self.policies[rank]
        size, count = noises.shape[:2]
        out = []
        with torch.cuda.device(dev):
            for start in range(0, size, self.obs_batch):
                stop = min(size, start + self.obs_batch)
                cond = self._condition(rank, _slice(obs, start, stop))
                noise = noises[start:stop].flatten(0, 1)
                # Produce every EXPO candidate while respecting the measured
                # FastWAM forward ceiling on each device.
                batch = self.ode_batch_per_gpu
                for begin in range(0, len(noise), batch):
                    end = min(len(noise), begin + batch)
                    ids = torch.arange(begin, end, device=dev) // count
                    data = _select_cache(cond, ids)
                    result = flow_sde_rollout(policy.model, conditioning=data,
                        initial_latents=noise[begin:end].to(dev), deterministic=True,
                        num_inference_steps=policy.config.num_inference_steps,
                        sigma_shift=policy.config.sigma_shift, noise_level=policy.config.noise_level)
                    out.append(result.actions)
        return torch.cat(out).reshape(size, count, self.HORIZON, self.ENV_DIM)

    def sample_normalized(self, env_obs, num_candidates=8, *, generator=None, signal_kind=None):
        if num_candidates < 1 or signal_kind is not None:
            raise ValueError("Clean EXPO requires ordinary FastWAM candidates")
        size = len(env_obs["states"])
        noise = torch.randn(size, num_candidates, self.HORIZON, self.MODEL_DIM, generator=generator,
                            dtype=torch.float32, device="cpu")
        split_candidates = size == 1
        axis_size = num_candidates if split_candidates else size
        slots = min(axis_size, self.parallel_devices)
        jobs = []
        for rank, (left, right) in enumerate(_split(axis_size, slots)):
            obs = env_obs if split_candidates else _slice(env_obs, left, right)
            z = noise[:, left:right] if split_candidates else noise[left:right]
            jobs.append(self.pool.submit(self._predict, rank, obs, z))
        wait(jobs)
        return torch.cat([job.result().to(self.device) for job in jobs], dim=int(split_candidates))

    def decode(self, env_obs, normalized_chunk):
        chunk = normalized_chunk.detach().cpu().float()
        if chunk.shape != (len(env_obs["states"]), self.CHUNK, self.ENV_DIM):
            raise ValueError("Expected a FastWAM selected [B,24,14] prefix")
        return self.policies[0].processor.normalizer.normalizers["action"]["default"].backward(chunk)

    def encode_executed(self, env_obs, canonical_actions):
        actions = torch.as_tensor(canonical_actions).detach().cpu().float()
        if (actions.ndim != 3 or actions.shape[0] != len(env_obs["states"])
                or actions.shape[-1] != self.ENV_DIM or not 1 <= actions.shape[1] <= self.HORIZON):
            raise ValueError("Invalid executed FastWAM action window")
        return self.policies[0].processor.normalizer.normalizers["action"]["default"].forward(actions)

    def _fm_shard(self, rank, obs, targets, global_batch):
        policy, dev = self.policies[rank], self.device_ids[rank]
        scheduler = policy.model.train_action_scheduler
        loss_sum = 0.
        gradients = {}
        with torch.cuda.device(dev):
            policy.train(True)
            try:
                for start in range(0, len(targets), self.fm_batch):
                    stop = min(len(targets), start + self.fm_batch)
                    cond = self._condition(rank, _slice(obs, start, stop))
                    target = targets[start:stop].to(dev, dtype=policy.model_dtype)
                    noise = torch.randn_like(target)
                    t = scheduler.sample_training_t(len(target), device=dev, dtype=target.dtype)
                    x = scheduler.add_noise(target, noise, t)
                    pred = predict_action_velocity(policy.model, x=x, raw_timestep=t, conditioning=cond)
                    per_sample = F.mse_loss(pred.float(), scheduler.training_target(target, noise, t).float(),
                                            reduction="none").mean((1, 2))
                    loss = (per_sample * scheduler.training_weight(t)).sum() / global_batch
                    loss.backward()
                    for name, p in self.experts[rank].items():
                        if p.grad is None:
                            continue
                        if name in gradients:
                            gradients[name].add_(p.grad)
                        else:
                            gradients[name] = p.grad.detach().to(dtype=torch.float32, copy=True)
                        p.grad = None
                    loss_sum += loss.detach()
            finally:
                policy.eval()
            return float(loss_sum), gradients

    @torch.no_grad()
    def _sync_experts(self):
        for replica in self.experts:
            for name, p in replica.items():
                p.copy_(self.trainable[name])

    def fm_update(self, observations, canonical_full_actions, action_weights=None):
        if action_weights is not None:
            raise ValueError("Only clean EXPO FM is enabled")
        obs = stack_env_observations(observations) if isinstance(observations, (list, tuple)) else observations
        targets = self.encode_executed(obs, canonical_full_actions)
        size = len(targets)
        if tuple(targets.shape[1:]) != (self.HORIZON, self.MODEL_DIM) or size < 1:
            raise ValueError("Expected genuine 32-step FM windows")
        slots = min(size, self.parallel_devices)
        for policy in self.policies:
            policy.zero_grad(set_to_none=True)
        jobs = [self.pool.submit(self._fm_shard, rank, _slice(obs, left, right), targets[left:right], size)
                for rank, (left, right) in enumerate(_split(size, slots))]
        loss = 0.
        with torch.no_grad():
            for job in jobs:
                local_loss, gradients = job.result()
                loss += local_loss
                for name in list(gradients):
                    grad = gradients.pop(name).to(self.device)
                    master = self.trainable[name]
                    if master.grad is None:
                        master.grad = grad
                    else:
                        master.grad.add_(grad)
            norm = torch.nn.utils.clip_grad_norm_(self.trainable.values(), self.clip_grad, error_if_nonfinite=True)
            if float(norm) <= 0:
                raise RuntimeError("FastWAM action FM produced no gradient")
            witness = max((p for p in self.trainable.values() if p.grad is not None),
                          key=lambda p: p.numel())
            before = witness.flatten()[:1024].clone()
            self.optimizer.step()
            change = float((witness.flatten()[:1024] - before).abs().max())
            self._sync_experts()
            self.optimizer.zero_grad(set_to_none=True)
        self.base_updates += 1
        return dict(base_fm_loss=loss, base_grad_norm=float(norm), base_updates=self.base_updates,
                    base_global_batch=size, base_parallel_devices=slots,
                    base_sampled_parameter_delta_max=change)

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self._sync_experts()

