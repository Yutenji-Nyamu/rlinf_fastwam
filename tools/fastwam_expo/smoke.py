"""One disposable real-demo FastWAM EXPO integration check on the two reserved GPUs."""
import argparse
import json
import math
import statistics
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import torch
from omegaconf import OmegaConf
from rlinf.algorithms.expo_ft.core import ExpoConfig, ExpoLearner
from rlinf.algorithms.expo_ft.backend import stack_env_observations
from rlinf.algorithms.expo_ft.fastwam_backend import FastWAMBackend
from rlinf.algorithms.expo_ft.formal_replay import FormalReplay
from rlinf.models.embodiment.fastwam.robotwin_adapter import adapt_robotwin_observation


def choose_ode_batch(base, core, observations):
    obs = stack_env_observations(observations)
    b = len(observations)
    critic_obs = base.critic_observation(obs)
    normalizer = base.policies[0].processor.normalizer.normalizers["action"]["default"]
    editor_rng = core.generator.get_state()
    reference = indices = None
    trials = []
    seed = 77
    for width in (2, 4, 8):
        base.ode_batch_per_gpu = width
        trial = dict(ode_batch_per_gpu=width)
        for rank in range(2):
            with torch.cuda.device(rank):
                torch.cuda.empty_cache()
        memory = [torch.cuda.mem_get_info(i) for i in range(2)]
        initial_reserved = [torch.cuda.memory_reserved(i) for i in range(2)]
        try:
            for i in range(2):
                torch.cuda.reset_peak_memory_stats(i)
            durations = []
            for _ in range(4):
                before = time.perf_counter()
                result = base.sample_normalized(obs, 8, generator=torch.Generator().manual_seed(seed))[:, :, :24]
                for rank in range(2):
                    torch.cuda.synchronize(rank)
                durations.append(time.perf_counter()-before)
            physical = normalizer.backward(result.cpu())
            core.generator.set_state(editor_rng)
            picked = core.select_actions(critic_obs, result)["index"].cpu()
            if reference is None:
                reference, indices = physical, picked
            delta = (physical-reference).abs()
            peak = [torch.cuda.max_memory_reserved(i) for i in range(2)]
            left = [min(torch.cuda.mem_get_info(i)[0],
                        memory[i][0]-max(0,peak[i]-initial_reserved[i])) for i in range(2)]
            trial.update(median_seconds=statistics.median(durations[1:]), timed_passes=3,
                         examples=b, candidates=b*8, peak_reserved_gib=[x/2**30 for x in peak],
                         free_fraction_at_peak=[left[i]/memory[i][1] for i in range(2)],
                         physical_qpos_max_abs=delta.max().item(), physical_qpos_mean_abs=delta.mean().item(),
                         relative_max=(delta.max()/reference.abs().max().clamp_min(1e-6)).item(),
                         matched_q_choices=(indices==picked).sum().item())
            # Same cross-batch qpos tolerances as the FastWAM GRPO B2/B1 test.
            trial["eligible"] = (min(trial["free_fraction_at_peak"])>=.10 and
                                 trial["physical_qpos_max_abs"]<=.01 and trial["physical_qpos_mean_abs"]<=.0015
                                 and trial["relative_max"]<=.02 and trial["matched_q_choices"]==b)
        except torch.OutOfMemoryError:
            trial.update(eligible=False, reason="cuda_oom")
        trials.append(trial)
    core.generator.set_state(editor_rng)
    valid = [row for row in trials if row["eligible"]]
    if not valid:
        raise RuntimeError("No candidate batch with sufficient memory/parity: "+json.dumps(trials))
    fastest = min(valid, key=lambda row: row["median_seconds"])
    return dict(ode_batch_per_gpu=fastest["ode_batch_per_gpu"], trials=trials)


def run(inputs, output):
    with open(inputs) as stream:
        cfg = json.load(stream)
    start = time.perf_counter()
    if torch.cuda.device_count() != 2:
        raise ValueError("Smoke requires the two reserved visible devices")
    torch.cuda.init()
    for rank in range(2):
        with torch.cuda.device(rank):
            torch.empty(1, device=torch.device("cuda", rank))
        torch.cuda.reset_peak_memory_stats(rank)
    torch.manual_seed(42)
    settings = cfg["formal"]
    base = FastWAMBackend(OmegaConf.create(cfg["model"]), device="cuda:0", parallel_devices=2,
                          candidate_microbatch=settings["candidate_microbatch"],
                          observation_microbatch=settings["observation_microbatch"],
                          fm_microbatch=settings["fm_microbatch"], lr=settings["base_lr"],
                          source_head=cfg["source_head"])
    for p,dev in zip(base.policies,base.device_ids):
        devices={parameter.device for parameter in p.model.parameters() if parameter.is_cuda}
        if devices != {dev} or any(q.device != dev for q in p.model.mot.mixtures["action"].parameters()):
            raise RuntimeError(f"FastWAM replica spans unexpected devices: {devices}, expected {dev}")
    core = ExpoLearner(replace(ExpoConfig(**cfg["core"]), critic_updates=1), device="cuda:0", seed=42)
    init_s = time.perf_counter() - start
    noise = torch.Generator(device="cpu").manual_seed(42)
    with tempfile.TemporaryDirectory(prefix="expo-smoke-", dir=Path(output).parent) as scratch:
        replay = FormalReplay(scratch, cfg["demo_path"], seed=42)
        observations, target, sources = replay.sample_fm(2)
        obs = observations[0]
        p = base.policies[0]
        seed = p.config.eval_seed
        # Compare official native singleton and our singleton using the same CPU noise.
        with torch.no_grad(), torch.cuda.device(0):
            image, state, prompts = adapt_robotwin_observation(obs, p.processor,
                                device=base.device, dtype=p.model_dtype)
            official = p.model.infer_action(prompt=prompts[0], input_image=image,
                action_horizon=32, proprio=state, seed=seed, num_inference_steps=10,
                sigma_shift=p.config.sigma_shift, tiled=False, rand_device="cpu")["action"]
            ours = base.sample_normalized(obs, 1, generator=torch.Generator().manual_seed(seed))[0, 0]
            expected = base.decode(obs, official[:24].unsqueeze(0))
            predicted = base.decode(obs, ours[:24].unsqueeze(0))
            oracle_error = (predicted - expected).abs().max().item()
            # The existing FastWAM B1 qualification compares physical qpos (2e-3).
            if oracle_error > .002:
                raise RuntimeError(f"FastWAM B1 official parity failed: {oracle_error:.6g}")
        normalized = base.encode_executed(obs, target[:1, :24])
        decoded = base.decode(obs, normalized)
        # The official normalizer clips values outside [-5,5]. Test the round
        # trip on representable physical coordinates.
        interior = normalized.abs() < 5
        roundtrip = (decoded - target[:1, :24]).abs()[interior].max().item()
        if roundtrip > 1e-5:
            raise RuntimeError(f"Executed-action roundtrip failed: {roundtrip}")
        before = time.perf_counter()
        candidates = base.sample_normalized(obs, 8, generator=noise)[:, :, :24]
        selection = core.select_actions(base.critic_observation(obs), candidates)
        if (candidates.shape != (1, 8, 24, 14) or not torch.isfinite(candidates).all()
                or float(candidates.std(dim=1).mean()) == 0 or selection["actions"].shape != (1,24,14)):
            raise RuntimeError("Real 8-candidate collection or edit/Q selection failed")
        candidate_s = time.perf_counter() - before
        # B2/Q1 is solely a pipeline check. Both GPU replicas receive real FM gradients.
        def sample_expert(rank):
            return torch.cat([p.detach().reshape(-1)[:128].float() for p in base.experts[rank].values()]).cpu()
        initial = sample_expert(0)
        before = time.perf_counter()
        metrics = core.update_call(
            lambda: replay.sample(2, base, "cuda:0"),
            lambda o: base.sample_normalized(o["env_obs"], 8, generator=noise)[:, :, :24],
            lambda: base.fm_update(observations, target),
        )
        for rank in range(2):
            torch.cuda.synchronize(rank)
        update_s = time.perf_counter() - before
        changed = sample_expert(0)
        changed_values = int((changed!=initial).sum())
        if not changed_values or not torch.equal(changed, sample_expert(1)):
            raise RuntimeError("Updated BF16 action weights or synchronization failed")
        update_peak = [torch.cuda.max_memory_reserved(i)/2**30 for i in range(2)]
        update_alloc = [torch.cuda.max_memory_allocated(i)/2**30 for i in range(2)]
        # Four scenes ensure that even with two cards both VAE batches reach 2
        # and the ODE candidate batching can really reach 8 on each device.
        calib_observations, _, _ = replay.sample_fm(4)
        measured = choose_ode_batch(base, core, calib_observations)
    if any(not math.isfinite(v) for v in metrics.values()):
        raise RuntimeError("Non-finite smoke metric")
    if (core.critic_steps, base.base_updates, core.editor_steps, core.temperature_steps) != (1,1,1,1):
        raise RuntimeError("Missing EXPO update")
    if metrics["base/base_sampled_parameter_delta_max"] <= 0:
        raise RuntimeError("FastWAM FP32 master weight did not change")
    result = dict(source=sources, global_batch=2, critic_updates=1, fm_updates=1,
                  editor_updates=1, temperature_updates=1, oracle_qpos_max_abs=oracle_error,
                  normalize_roundtrip_max_abs=roundtrip, sampled_bf16_weights_changed=changed_values,
                  peak_reserved_gib=update_peak, peak_allocated_gib=update_alloc,
                  ode_batch_calibration=measured,
                  model_init_seconds=round(init_s,2), eight_candidates_seconds=round(candidate_s,2),
                  update_seconds=round(update_s,2), metrics=metrics)
    runtime = Path(inputs).with_name("fastwam-runtime.json")
    chosen = dict(ode_batch_per_gpu=measured["ode_batch_per_gpu"])
    if runtime.exists():
        chosen = json.loads(runtime.read_text())
        base.apply_runtime(chosen)
    else:
        # Atomic creation; future resumes use the original selection.
        with tempfile.NamedTemporaryFile(mode="w", dir=runtime.parent, delete=False) as tmp:
            tmp.write(json.dumps(chosen)+"\n")
        import os
        try:
            os.link(tmp.name,runtime)
        finally:
            os.unlink(tmp.name)
    result["pinned_runtime"] = chosen
    Path(output).write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--output", required=True)
    args=parser.parse_args()
    run(args.inputs, args.output)

