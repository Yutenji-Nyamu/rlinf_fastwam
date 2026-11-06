# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Small production-method checks for pretraining scheduling and compact noise."""

import json
import os
import types

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
from omegaconf import OmegaConf

from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy


def main():
    w = object.__new__(EmbodiedSACFSDPPolicy)
    w.cfg = OmegaConf.create(
        {
            "algorithm": {"dsrl_critic_warmup_updates": 7},
            "actor": {
                "model": {
                    "openpi": {
                        "dsrl_noise_groups": 5,
                        "action_horizon": 50,
                        "dsrl_action_noise_dim": 160,
                    }
                }
            },
        }
    )
    w.use_dsrl = True
    w.dsrl_critic_warmup_done = 0
    w.update_step = 0
    w._dsrl_critic_warmup_active = False
    phases = []
    calls = []
    w._set_dsrl_policy_phase = lambda x: phases.append(x)
    w.log_on_first_rank = lambda x: None

    def update(self, train_actor=True):
        assert not train_actor and self._dsrl_critic_warmup_active
        calls.append(1)
        return {"sac/critic_loss": 0.5}

    w.update_one_epoch = types.MethodType(update, w)
    out = w._run_dsrl_critic_warmup()
    assert len(calls) == 7 and w.update_step == 7 and w.dsrl_critic_warmup_done == 7
    assert not w._dsrl_critic_warmup_active and phases == [0]
    w._run_dsrl_critic_warmup()
    assert len(calls) == 7
    w.dsrl_critic_warmup_done = 3
    w._run_dsrl_critic_warmup()
    assert len(calls) == 11
    for g in [5, 10, 50]:
        z = torch.randn(4, g, 32, requires_grad=True)
        expanded = z.repeat_interleave(50 // g, dim=1)
        assert expanded.shape == (4, 50, 32)
        assert torch.equal(expanded[:, :: 50 // g], z)
        expanded.square().sum().backward()
        torch.testing.assert_close(z.grad, 2 * z.detach() * (50 // g))
    print(
        json.dumps(
            {
                "critic_only_calls": len(calls),
                "resume_remaining_verified": True,
                "compact_group_gradients": [5, 10, 50],
                "metrics": out,
            }
        )
    )


if __name__ == "__main__":
    main()
