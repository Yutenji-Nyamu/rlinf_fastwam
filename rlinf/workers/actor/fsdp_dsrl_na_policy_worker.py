# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""DSRL-NA: fresh replay minibatches, VLA features, and delayed actor updates.

The three update types follow the public pi0 NA implementation. Every gradient
step samples replay anew; deterministic VLA inference is microbatched only.
"""
import math
import os
import time
from pathlib import Path
import torch
from rlinf.algorithms.dsrl_na import action_td_loss, noise_distillation_loss, reduce_q, index_tree, detach_cpu
from rlinf.data.storage.replay import project_dsrl_trajectory
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.scheduler import Worker
from rlinf.utils import drq
from rlinf.utils.metric_utils import append_to_dict, compute_split_num
from rlinf.utils.nested_dict_process import put_tensor_device
from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy


class EmbodiedDSRLNAFSDPPolicy(EmbodiedSACFSDPPolicy):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.na_counts = {'action_q': 0, 'noise_q': 0, 'actor': 0, 'blocks': 0}

    def setup_sac_components(self):
        super().setup_sac_components()
        c = self.cfg.algorithm.dsrl_na
        if not (self.use_dsrl_flat_replay and self.cfg.actor.model.openpi.dsrl_na_enabled
                and self.replay_buffer.schema_version == 5 and self._world_size == 1):
            raise ValueError('NA requires single-rank full-noise replay schema 5')
        if self.cfg.algorithm.get('dsrl_cql_weight', 0) or self.use_dsrl_u or self.alpha_optimizer is not None:
            raise ValueError('NA uses supervised noise Q; this integration requires clean, fixed-alpha-zero training')
        if self.cfg.algorithm.entropy_tuning.initial_alpha != 0 or self.cfg.algorithm.backup_entropy:
            raise ValueError('NA targets are reward-only')
        if not self.cfg.actor.model.openpi.dsrl_full_noise:
            raise ValueError('NA decoding requires full independent VLA noise')
        for name in ['blocks_per_round', 'action_q_steps', 'noise_q_steps', 'actor_steps', 'decode_batch_size']:
            if int(c[name]) <= 0: raise ValueError(name)
        critic_ids = {id(p) for group in self.qf_optimizer.param_groups for p in group['params']}
        actor_ids = {id(p) for group in self.optimizer.param_groups for p in group['params']}
        assert not critic_ids & actor_ids
        for name, parameter in self.model.named_parameters():
            if self._is_dsrl_target_q_parameter(name):
                assert parameter.dtype == torch.float32 and id(parameter) in critic_ids, name
        self.na_replay_decode_error = None

    @staticmethod
    def _is_dsrl_target_q_parameter(name):
        return EmbodiedSACFSDPPolicy._is_dsrl_target_q_parameter(name) or bool(
            {'action_critic_image_encoder', 'action_critic_state_encoder', 'action_q_head'} & set(name.split('.')))

    def _named_dsrl_target_q_parameters(self, model):
        expected = {'critic_image_encoder', 'critic_state_encoder', 'q_head',
                    'action_critic_image_encoder', 'action_critic_state_encoder', 'action_q_head'}
        params = {n: p for n, p in model.named_parameters() if self._is_dsrl_target_q_parameter(n)}
        found = set().union(*(set(n.split('.')) & expected for n in params))
        if found != expected: raise RuntimeError(f'Incomplete NA target critic: {found}')
        return params

    async def recv_rollout_trajectories(self, input_channel):
        send_num = self._component_placement.get_world_size('env') * self.stage_num
        recv_num = self._component_placement.get_world_size('actor')
        model_cfg = self.cfg.actor.model.openpi
        for _ in range(compute_split_num(send_num, recv_num)):
            trajectory = await input_channel.get(async_op=True).async_wait()
            def pack(raw, tokens):
                return detach_cpu(self.model(forward_type=ForwardType.SAC, na_operation='pack', obs=raw, tokens=tokens))
            projected = project_dsrl_trajectory(trajectory, action_horizon=model_cfg.action_horizon,
                latent_dim=model_cfg.dsrl_action_noise_dim, state_dim=model_cfg.dsrl_state_dim,
                num_action_chunks=self.cfg.actor.model.num_action_chunks, gamma=self.cfg.algorithm.gamma,
                full_noise=True, na_pack=pack)
            if model_cfg.dsrl_vla_features:
                # The collector already computed current features before selecting noise.
                packed = projected['next_vla']; rows = projected['actions'].shape[0]
                outputs = []
                self.model.eval()
                for start in range(0, rows, int(self.cfg.algorithm.dsrl_na.decode_batch_size)):
                    ix = torch.arange(start, min(rows, start + int(self.cfg.algorithm.dsrl_na.decode_batch_size)))
                    outputs.append(self.model(forward_type=ForwardType.SAC, na_operation='features',
                        obs=index_tree(packed, ix)).detach().float().cpu())
                projected['next_obs']['vla_features'] = torch.cat(outputs)
                self.model.train()
            self._local_new_transitions += self.replay_buffer.add_batch(projected)

    @torch.no_grad()
    def _decode(self, packed, noise):
        outputs = []; size = int(self.cfg.algorithm.dsrl_na.decode_batch_size)
        for start in range(0, noise.shape[0], size):
            ix = torch.arange(start, min(start + size, noise.shape[0]))
            decoded = self.model(forward_type=ForwardType.SAC, na_operation='decode',
                obs=index_tree(packed, ix), actions=noise[start:start + size])
            outputs.append(decoded.to(device=self.device, dtype=torch.float32))
        return torch.cat(outputs)

    @torch.no_grad()
    def _fresh_batch(self, warmup, kind):
        raw = self.replay_buffer.sample(int(self.cfg.actor.global_batch_size))
        packed_curr, packed_next = raw.pop('curr_vla'), raw.pop('next_vla')
        batch = put_tensor_device(raw, device=self.device)
        self.model.eval()
        try:
            if self.na_replay_decode_error is None:
                ix = torch.arange(min(4, batch['actions'].shape[0]))
                replay_actions = self._decode(index_tree(packed_curr, ix), batch['actions'][:len(ix)])
                error = (replay_actions - batch['env_actions'][:len(ix)]).abs()
                self.na_replay_decode_error = {'mean': error.mean().item(), 'max': error.max().item()}
                self.log_on_first_rank(f'NA replay action reconstruction: {self.na_replay_decode_error}')
            if kind == 'action_q':
                noise = torch.randn_like(batch['actions'], dtype=torch.float32) if warmup else self.model(
                    forward_type=ForwardType.SAC, obs=batch['next_obs'], train=False)[0].flatten(1)
                batch['next_env_actions'] = self._decode(packed_next, noise)
            elif kind == 'noise_q':
                prior = torch.randn_like(batch['actions'], dtype=torch.float32)
                current = prior if warmup else self.model(
                    forward_type=ForwardType.SAC, obs=batch['curr_obs'], train=False)[0].flatten(1)
                mask = (torch.arange(len(prior), device=self.device) % 2 == 0).unsqueeze(1)
                batch['distill_noise'] = torch.where(mask, prior, current).detach()
                batch['distill_actions'] = self._decode(packed_curr, batch['distill_noise'])
        finally:
            self.model.train()
        if self.enable_drq:
            drq.apply_drq(batch['curr_obs'], pad=4)
            drq.apply_drq(batch['next_obs'], pad=4)
        return batch

    def _action_q_loss(self, batch):
        with torch.no_grad():
            next_q = self.target_model(forward_type=ForwardType.SAC_Q, na_action_critic=True,
                obs=batch['next_obs'], actions=batch['next_env_actions'])
            next_q = reduce_q(next_q, self.cfg.algorithm.agg_q)
        q = self.model(forward_type=ForwardType.SAC_Q, na_action_critic=True,
                       obs=batch['curr_obs'], actions=batch['env_actions'])
        loss, target = action_td_loss(q, batch['rewards'], batch['continuations'], batch['discounts'], next_q)
        return loss, {'na/action_q_loss': loss.detach(), 'na/action_q': q.detach().mean(),
                      'na/action_td_target': target.mean()}

    def _noise_q_loss(self, batch):
        with torch.no_grad():
            action_q = self.model(forward_type=ForwardType.SAC_Q, na_action_critic=True,
                obs=batch['curr_obs'], actions=batch['distill_actions']).clone()
        noise_q = self.model(forward_type=ForwardType.SAC_Q, obs=batch['curr_obs'], actions=batch['distill_noise'])
        loss, target = noise_distillation_loss(noise_q, action_q, self.cfg.algorithm.agg_q)
        return loss, {'na/noise_q_loss': loss.detach(), 'na/noise_q': noise_q.detach().mean(),
                      'na/distill_target': target.mean(), 'na/distill_abs_error': (noise_q.detach().mean(-1, keepdim=True)-target).abs().mean()}

    def _critic_step(self, batch, kind):
        self.qf_optimizer.zero_grad(set_to_none=True)
        self._clear_dsrl_actor_grads_before_critic_clip()
        loss, metrics = self._action_q_loss(batch) if kind == 'action_q' else self._noise_q_loss(batch)
        loss.backward()
        norm = self.model.clip_grad_norm_(max_norm=self.cfg.actor.critic_optim.clip_grad)
        self.qf_optimizer.step(); self.qf_lr_scheduler.step()
        if kind == 'action_q': self.soft_update_target_model()
        self.na_counts[kind] += 1; self.update_step += 1
        metrics['na/' + kind + '_grad_norm'] = norm
        return metrics

    def _actor_step(self, batch):
        self.optimizer.zero_grad(set_to_none=True)
        loss, entropy, metrics = self.forward_actor(batch)
        loss.backward()
        self._clear_dsrl_critic_grads_before_actor_clip()
        norm = self.model.clip_grad_norm_(max_norm=self.cfg.actor.optim.clip_grad)
        self.optimizer.step(); self.lr_scheduler.step()
        self.na_counts['actor'] += 1; self.update_step += 1
        return {'na/actor_loss': loss.detach(), 'na/actor_grad_norm': norm, 'na/actor_entropy': entropy.detach(),
                **{'actor/' + key: value for key, value in metrics.items()}}

    @Worker.timer('run_training')
    def run_training(self):
        c = self.cfg.algorithm.dsrl_na
        new, resident, minimum = self._consume_global_dsrl_replay_counts()
        metrics = {'sac/global_new_transitions': [new], 'sac/global_resident_transitions': [resident]}
        if resident < self.cfg.algorithm.replay_buffer.warmup_size or not new:
            return self.process_train_metrics(metrics)
        if minimum <= 0: raise RuntimeError('NA local replay is empty')
        self.model.train()
        required = int(self.cfg.algorithm.dsrl_critic_warmup_updates)
        qa_steps, qw_steps, actor_steps = (int(c[k]) for k in ['action_q_steps', 'noise_q_steps', 'actor_steps'])
        blocks = max(int(c.blocks_per_round), math.ceil(max(0, required-self.dsrl_critic_warmup_done)/qa_steps))
        decode_seconds = 0.0; update_seconds = 0.0
        for block_id in range(blocks):
            warmup = self.dsrl_critic_warmup_done < required
            # Fresh replay draws and freshly decoded target actions for every update.
            for kind, count in [('action_q', qa_steps), ('noise_q', qw_steps),
                                ('actor', 0 if warmup else actor_steps)]:
                for _ in range(count):
                    start = time.monotonic()
                    batch = self._fresh_batch(warmup, kind)
                    decode_seconds += time.monotonic() - start
                    start = time.monotonic()
                    value = self._actor_step(batch) if kind == 'actor' else self._critic_step(batch, kind)
                    append_to_dict(metrics, value)
                    if kind == 'action_q' and warmup: self.dsrl_critic_warmup_done += 1
                    update_seconds += time.monotonic() - start
                    del batch
            if self.dsrl_critic_warmup_done >= required: self._set_dsrl_policy_phase(1)
            self.na_counts['blocks'] += 1
            Worker.torch_platform.synchronize()
            self.log_on_first_rank(f'NA block {block_id+1}/{blocks}: warmup={warmup}, counts={self.na_counts}')
        append_to_dict(metrics, {'na/decode_seconds': decode_seconds, 'na/update_seconds': update_seconds,
            'warmup/critic_updates_done': self.dsrl_critic_warmup_done,
            **{'na/total_' + k: v for k, v in self.na_counts.items()}})
        result = self.process_train_metrics(metrics)
        Worker.torch_platform.synchronize(); torch.distributed.barrier(); Worker.torch_platform.empty_cache()
        return result

    def _save_dsrl_trainer_state(self, save_base_path):
        super()._save_dsrl_trainer_state(save_base_path)
        path = self._dsrl_trainer_state_path(save_base_path)
        state = torch.load(path, map_location='cpu', weights_only=False)
        state['na_counts'] = self.na_counts
        state['na_version'] = 2
        temporary = path + '.na-tmp'
        torch.save(state, temporary); os.replace(temporary, path)

    def _load_dsrl_trainer_state(self, load_base_path):
        super()._load_dsrl_trainer_state(load_base_path)
        state = torch.load(self._dsrl_trainer_state_path(load_base_path), map_location='cpu', weights_only=False)
        if state.get('na_version') != 2: raise ValueError('NA continuation requires a matching full-state checkpoint')
        self.na_counts = dict(state['na_counts'])
