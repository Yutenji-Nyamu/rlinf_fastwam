# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Narrow optional losses/replay changes on fresh copies of accepted exports."""

import hashlib
import json


def replace_once(text, old, new):
    assert text.count(old) == 1, (old[:90], text.count(old))
    return text.replace(old, new, 1)


def apply(root):
    receipts = []
    p = root / "dsrl/rlinf/workers/actor/fsdp_sac_policy_worker.py"
    before = p.read_text()
    s = before
    s = replace_once(
        s,
        "        entropy = -log_pi.mean()\n        return actor_loss, entropy, metrics",
        """        prior_weight = float(self.cfg.algorithm.get("noise_prior_kl_weight", 0.0))
        if prior_weight:
            if not self.use_dsrl or getattr(self, "use_dsrl_u", False):
                raise ValueError("Noise prior regularization requires clean DSRL")
            from rlinf.algorithms.noise_prior import gaussian_prior_kl

            prior_kl = gaussian_prior_kl(pi, log_pi)
            actor_loss = actor_loss + prior_weight * prior_kl.mean()
            metrics["noise_prior_kl_per_dim"] = self._metric(prior_kl.mean())
            metrics["noise_prior_penalty"] = self._metric(prior_weight * prior_kl.mean())

        entropy = -log_pi.mean()
        return actor_loss, entropy, metrics""",
    )
    p.write_text(s)
    receipts.append(
        {
            "file": str(p),
            "before": hashlib.sha256(before.encode()).hexdigest(),
            "after": hashlib.sha256(s.encode()).hexdigest(),
        }
    )
    (
        root / "dsrl/rlinf/algorithms/noise_prior.py"
    ).write_text('''# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Gaussian behavior prior in the physical DSRL noise coordinates.

Inspired by BRAC policy regularization. This is an online noise-space adaptation,
not a claim that BRAC evaluated this policy or benchmark. log_prob must include
all tanh and scale Jacobians. Average over dimensions to keep the coefficient
independent of the latent dimension; do not regress to random Gaussian samples.
"""
import math


def gaussian_prior_kl(noise, log_prob):
    z = noise.float().reshape(noise.shape[0], -1)
    lp = log_prob.float().reshape(z.shape[0], -1).sum(-1, keepdim=True)
    log_prior = -0.5 * (z.square() + math.log(2.0 * math.pi)).sum(-1, keepdim=True)
    return (lp - log_prior) / z.shape[-1]
''')
    p = root / "rlt/rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py"
    before = p.read_text()
    s = before
    s = replace_once(
        s,
        "    def _next_actions_for_critic_target(self, next_obs):\n        return self.model(",
        """    def _next_actions_for_critic_target(self, next_obs):
        if self.cfg.algorithm.get("rlt_reference_bootstrap", False):
            ref = self._ref_chunk(next_obs)
            return ref, torch.zeros((ref.shape[0], 1), device=ref.device), None
        return self.model(""",
    )
    s = replace_once(
        s,
        "            if self.demo_buffer is not None:\n                intervene_traj_list = [",
        """            reference_limit = int(self.cfg.algorithm.get("rlt_reference_replay_size", 0))
            if reference_limit:
                # Initial precollection is produced by the frozen reference policy.
                # Fill once, then keep this buffer immutable during online learning.
                assert self.demo_buffer is not None
                remaining = max(0, reference_limit - self.demo_buffer.total_samples)
                if remaining:
                    self.demo_buffer.add_trajectories(replay_list[:remaining])
                self._last_replay_metrics["replay/reference_retained"] = float(
                    self.demo_buffer.total_samples
                )
            elif self.demo_buffer is not None:
                intervene_traj_list = [""",
    )
    s = replace_once(
        s,
        "        super().save_checkpoint(save_base_path, step)\n        if self.use_rlt_resume:",
        """        super().save_checkpoint(save_base_path, step)
        if self.cfg.algorithm.get("rlt_reference_replay_size", 0):
            self.demo_buffer.save_checkpoint(os.path.join(
                save_base_path, f"sac_components/reference_replay/rank_{self._rank}"
            ))
        if self.use_rlt_resume:""",
    )
    s = replace_once(
        s,
        "        super().load_checkpoint(load_base_path)\n        if state is not None:",
        """        super().load_checkpoint(load_base_path)
        if self.cfg.algorithm.get("rlt_reference_replay_size", 0):
            self.demo_buffer.load_checkpoint(os.path.join(
                load_base_path, f"sac_components/reference_replay/rank_{self._rank}"
            ))
        if state is not None:""",
    )
    p.write_text(s)
    receipts.append(
        {
            "file": str(p),
            "before": hashlib.sha256(before.encode()).hexdigest(),
            "after": hashlib.sha256(s.encode()).hexdigest(),
        }
    )
    (root / "patch-receipts.json").write_text(json.dumps(receipts, indent=2))
