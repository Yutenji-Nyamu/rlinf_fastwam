"""Small SAC actor/Q microbatch benchmark, called by model_probe.py.

Uses the actual worker actor loss and actual image/state encoders, including
configured DRQ. It does not update parameters, change GB256, or run an env.
Timing excludes critic TD, optimizer states/steps, target model, and FSDP.
"""

import inspect
import time
from types import SimpleNamespace
from unittest.mock import patch

import torch


def benchmark(model, cfg):
    from rlinf.algorithms.dsrl_ugrow import build_dsrl_u_weights
    from rlinf.utils.drq import apply_drq
    from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy

    global_batch_size = int(cfg.actor.global_batch_size)
    if global_batch_size != 256:
        raise ValueError("This comparison preserves the frozen GB256 recipe")
    device = torch.device("cuda:0")
    enable_drq = bool(cfg.actor.get("enable_drq", False))
    worker = EmbodiedSACFSDPPolicy.__new__(EmbodiedSACFSDPPolicy)
    worker.cfg = cfg
    worker.model = model
    worker.use_dsrl = True
    worker.use_dsrl_u = bool(cfg.algorithm.dsrl_u.enabled)
    worker.entropy_temp = SimpleNamespace(
        alpha=torch.tensor(float(cfg.algorithm.entropy_tuning.initial_alpha), device=device)
    )
    actual_forward_actor = inspect.unwrap(EmbodiedSACFSDPPolicy.forward_actor)

    # Exact compact replay layout; no cached image or state encoder features.
    main = torch.arange(4 * 3 * 64 * 64).remainder(255).float() / 127.5 - 1
    main = main.reshape(4, 3, 64, 64).repeat(64, 1, 1, 1).to(torch.bfloat16)
    state = torch.linspace(-0.3, 0.3, 4 * 14).reshape(4, 14).repeat(64, 1)
    raw_u = torch.linspace(0.001, 0.8, 2560).reshape(256, 10)
    weights, _ = build_dsrl_u_weights(
        raw_u, torch.ones_like(raw_u, dtype=torch.bool),
        temperature=float(cfg.algorithm.dsrl_u.temperature),
        log_eps=float(cfg.algorithm.dsrl_u.log_eps),
        minmax_eps=float(cfg.algorithm.dsrl_u.minmax_eps),
    )
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    previous_modes = [(module, module.training) for name, module in model.named_children() if name in {
        "dsrl_action_noise_net", "actor_image_encoder", "actor_state_encoder",
        "critic_image_encoder", "critic_state_encoder", "q_head",
    }]
    for module, _ in previous_modes:
        module.train()

    def gradient_vector():
        return torch.cat([
            torch.zeros(parameter.numel(), dtype=torch.float32)
            if parameter.grad is None else parameter.grad.detach().float().flatten().cpu()
            for _, parameter in trainable
        ])

    # A separate controlled-gradient comparison fixes the actual DRQ outputs
    # and the per-row Normal epsilon; timing retains the unmodified stochastic
    # path. Seed alone does not align random draws across different microbatches.
    torch.manual_seed(9106)
    torch.cuda.manual_seed(9106)
    fixed_observation = {"main_images": main.to(device), "states": state.to(device)}
    if enable_drq:
        apply_drq(fixed_observation, pad=4)
    fixed_main = fixed_observation["main_images"].cpu()
    fixed_epsilon = torch.randn(256, 32, device=device, dtype=torch.bfloat16)
    del fixed_observation

    def trial(micro_batch_size, controlled=False):
        model.zero_grad(set_to_none=True)
        torch.manual_seed(9206)
        torch.cuda.manual_seed(9206)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        resident_bytes = torch.cuda.memory_allocated(device)
        started = time.perf_counter()
        batches = []
        for start in range(0, 256, micro_batch_size):
            stop = start + micro_batch_size
            image = fixed_main[start:stop] if controlled else main[start:stop]
            batch = {
                "curr_obs": {"main_images": image.to(device).clone(), "states": state[start:stop].to(device)},
                "next_obs": {"main_images": main[start:stop].to(device).clone(), "states": state[start:stop].to(device)},
                "dsrl_u_weight": weights[start:stop].to(device),
            }
            if enable_drq and not controlled:
                apply_drq(batch["curr_obs"], pad=4)
                apply_drq(batch["next_obs"], pad=4)
            batches.append(batch)
        loss_sum = 0.0
        for index, batch in enumerate(batches):
            if controlled:
                epsilon = fixed_epsilon[index * micro_batch_size : (index + 1) * micro_batch_size]

                def fixed_rsample(normal, sample_shape=torch.Size()):
                    if sample_shape or normal.loc.shape != epsilon.shape:
                        raise AssertionError("Unexpected Normal draw in controlled comparison")
                    return normal.loc + normal.scale * epsilon.to(normal.loc.dtype)

                with patch.object(torch.distributions.Normal, "rsample", fixed_rsample):
                    loss, _, _ = actual_forward_actor(worker, batch)
            else:
                loss, _, _ = actual_forward_actor(worker, batch)
            if not torch.isfinite(loss):
                raise AssertionError("Microbatch actor loss is nonfinite")
            (loss / len(batches)).backward()
            loss_sum += float(loss.detach().item()) / len(batches)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        peak = torch.cuda.max_memory_allocated(device)
        gradients = gradient_vector()
        if not torch.isfinite(gradients).all():
            raise AssertionError("Microbatch actor/Q gradient is nonfinite")
        return {
            "seconds": elapsed, "peak_allocated_bytes": peak,
            "incremental_peak_bytes": peak - resident_bytes, "mean_actor_loss": loss_sum,
            "gradient_l2": float(gradients.double().norm().item()),
        }, gradients

    records, controlled_gradients = [], {}
    try:
        for micro_batch_size in (64, 128, 256):
            trial(micro_batch_size)  # One warm-up for this shape.
            measured = [trial(micro_batch_size)[0] for _ in range(2)]
            controlled_metrics, controlled_gradient = trial(micro_batch_size, controlled=True)
            controlled_gradients[micro_batch_size] = controlled_gradient
            records.append({
                "micro_batch_size": micro_batch_size, "measured_trials": measured,
                "mean_seconds": sum(item["seconds"] for item in measured) / len(measured),
                "controlled_actor_loss": controlled_metrics["mean_actor_loss"],
            })
        reference = controlled_gradients[256].double()
        denominator = reference.norm().clamp_min(1e-12)
        for record in records:
            gradient = controlled_gradients[record["micro_batch_size"]].double()
            difference = gradient - reference
            record["controlled_gradient_relative_l2_vs_mb256"] = float((difference.norm() / denominator).item())
            record["controlled_gradient_max_abs_vs_mb256"] = float(difference.abs().max().item())
    finally:
        model.zero_grad(set_to_none=True)
        for module, was_training in previous_modes:
            module.train(was_training)
    return {
        "global_batch_size": 256, "train_env_total_unchanged": int(cfg.env.train.total_num_envs),
        "eval_env_total_unchanged": int(cfg.env.eval.total_num_envs), "drq_enabled": enable_drq,
        "actual_worker_actor_loss": True, "encoder_features_cached": False, "optimizer_steps": 0,
        "timing_scope": "compact H2D, configured DRQ curr/next, actual actor+Q forward/backward",
        "excluded": ["critic TD pass", "alpha update", "optimizer states/step", "target model", "FSDP", "environment", "VLA generation"],
        "gradient_comparison": "same augmented inputs and per-row Normal epsilon; native BF16 accumulation may differ",
        "records": records,
    }
