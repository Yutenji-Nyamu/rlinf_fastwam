"""Bounded GPU interface probe using real Fast features and original RLT losses.

This is not an environment success test: its bootstrap transition is synthetic.
The sequential Ray smoke separately checks the real environment/replay path.
"""

import argparse
import copy
import json
from dataclasses import asdict
from pathlib import Path

import torch
from omegaconf import OmegaConf

from rlinf.models.embodiment.fastwam.builder import build_fastwam_policy
from rlinf.models.embodiment.fastwam.rlt_features import build_rlt_conditioning
from rlinf.models.embodiment.fastwam.rlt_policy import FastWAMRLTConfig, FastWAMRLTPolicy, feature_identity
from rlinf.models.embodiment.fastwam.robotwin_adapter import adapt_robotwin_observation
from rlinf.models.embodiment.mlp_policy.rlt_mlp_policy import RLTMLPPolicy
from rlinf.workers.actor.fsdp_rlt_ac_policy_worker import RLTACLossMixin


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--stage1", required=True)
    p.add_argument("--observation", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    cfg = OmegaConf.load(args.config)
    teacher = build_fastwam_policy(cfg.model, torch_dtype=torch.bfloat16)
    identity = feature_identity(cfg.model.checkpoint_path, cfg.model.dataset_stats_path,
                                OmegaConf.to_container(cfg.token, resolve=True))
    feature = FastWAMRLTPolicy(
        model=teacher.model, processor=teacher.processor,
        config=FastWAMRLTConfig(**asdict(teacher.config)),
        stage1_checkpoint=args.stage1, identity=identity,
    )
    obs = torch.load(args.observation, weights_only=False)
    image, proprio, prompts = adapt_robotwin_observation(
        obs, teacher.processor, device=teacher.device, dtype=teacher.model_dtype,
    )
    with torch.no_grad():
        text, mask = teacher.model.encode_prompt(prompts)
        condition, hidden = build_rlt_conditioning(
            teacher.model, input_image=image, text_context=text,
            text_context_mask=mask, proprio=proprio, verify_oracle=True,
        )
    del condition
    rlt_obs, context = feature.extract_rlt_obs(obs, return_decode_context=True)
    physical, _ = teacher.predict_action_batch(obs, mode="eval")
    decoded_reference = feature.decode_rlt_action(rlt_obs["ref_chunk"], context)
    torch.testing.assert_close(decoded_reference, physical, rtol=0, atol=0)
    student = RLTMLPPolicy(2048, 14, 14, 24, output_activation="identity").cuda()
    target = copy.deepcopy(student).eval().requires_grad_(False)
    harness = RLTACLossMixin()
    harness.model, harness.target_model = student, target
    harness.update_step, harness.torch_dtype = 0, torch.float32
    harness.cfg = OmegaConf.create({
        "actor": {"model": {"num_action_chunks": 24, "action_dim": 14}},
        "env": {"train": {"env_type": "robotwin"}},
        "algorithm": {
            "rlt_transition_replay": {"enable": True, "compact": True, "bootstrap_on_truncation": True},
            "rlt_route": {"type": "full_task"},
            "q_head_type": "default", "bootstrap_type": "standard", "gamma": .99,
            "reference_dropout_prob": .5, "bc_weight": 7., "q_weight": .05,
        },
    })
    batch_obs = {k: v.cuda() for k, v in rlt_obs.items()}
    batch = {"curr_obs": batch_obs, "next_obs": batch_obs,
             "actions": batch_obs["ref_chunk"].flatten(1),
             "rewards": torch.zeros(1, 24, device="cuda"),
             "terminations": torch.zeros(1, 24, device="cuda", dtype=torch.bool),
             "dones": torch.zeros(1, 24, device="cuda", dtype=torch.bool)}
    critic_opt = torch.optim.Adam(student.q_head.parameters(), lr=1e-4)
    actor_params = list(student.backbone.parameters()) + list(student.actor_mean.parameters())
    actor_opt = torch.optim.Adam(actor_params, lr=1e-4)
    before_actor = [p.detach().clone() for p in actor_params]
    before_critic = [p.detach().clone() for p in student.q_head.parameters()]
    metrics = []
    for _ in range(2):
        critic_opt.zero_grad(set_to_none=True)
        q_loss, q_metrics = RLTACLossMixin.forward_critic.__wrapped__(harness, batch)
        q_loss.backward()
        q_norm = torch.nn.utils.clip_grad_norm_(student.q_head.parameters(), 10.)
        critic_opt.step()
        student.q_head.requires_grad_(False)
        actor_opt.zero_grad(set_to_none=True)
        a_loss, _, a_metrics = RLTACLossMixin.forward_actor.__wrapped__(harness, batch)
        a_loss.backward()
        a_norm = torch.nn.utils.clip_grad_norm_(actor_params, 10.)
        actor_opt.step()
        student.q_head.requires_grad_(True)
        if not torch.isfinite(torch.stack([q_loss, a_loss, q_norm, a_norm])).all():
            raise RuntimeError("Non-finite original RLT loss/gradient")
        metrics.append({"q_loss": float(q_loss), "actor_loss": float(a_loss), **q_metrics, **a_metrics})
        harness.update_step += 1
    assert any(not torch.equal(a, b) for a, b in zip(before_actor, actor_params))
    assert any(not torch.equal(a, b) for a, b in zip(before_critic, student.q_head.parameters()))
    assert not any(p.requires_grad for p in feature.parameters())
    report = {
        "hidden_shape": list(hidden.shape), "stage1_steps": feature.stage1_steps,
        "prefill_oracle_exact": True, "physical_teacher_decode_exact": True,
        "reference_outside_unit_fraction": float((rlt_obs["ref_chunk"].abs() > 1).float().mean()),
        "actor_updated": True, "critic_updated": True, "teacher_frozen": True,
        "transition": "synthetic bootstrap; real raw observation and teacher reference",
        "metrics": metrics,
    }
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
