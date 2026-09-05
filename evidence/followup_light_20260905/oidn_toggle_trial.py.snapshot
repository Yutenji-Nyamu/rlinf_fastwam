"""Small, isolated OIDN on/off inference trial; no Ray or optimizer."""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--denoiser", choices=["oidn", "none"], required=True)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    out = Path(args.output) / args.denoiser
    out.mkdir(parents=True, exist_ok=True)
    base = OmegaConf.load(args.source_config)
    env_cfg = OmegaConf.create(OmegaConf.to_container(base.env.eval, resolve=True))
    env_cfg.total_num_envs = 1
    env_cfg.group_size = 1
    env_cfg.auto_reset = False
    env_cfg.enable_offload = False
    env_cfg.video_cfg.save_video = False
    env_cfg.video_cfg.video_base_dir = str(out / "video")
    env_cfg.task_config.save_path = str(out / "robotwin_data")
    env_cfg.task_config.ray_tracing_denoiser = args.denoiser
    seed_data = json.loads(Path(env_cfg.seeds_path).read_text())
    seeds = seed_data[env_cfg.task_config.task_name]["success_seeds"][:3]
    config = OmegaConf.create({
        "mode": "isolated_inference_no_training",
        "denoiser": args.denoiser,
        "seeds": seeds,
        "max_episodes": 3,
        "max_action_slots_per_episode": 192,
        "max_policy_queries_per_episode": 8,
        "optimizer_updates": 0,
        "model": OmegaConf.to_container(base.rollout.model, resolve=True),
        "env": OmegaConf.to_container(env_cfg, resolve=True),
        "render": {"shader": "rt", "spp": 32, "path_depth": 8},
        "physical_gpu": 6,
    })
    OmegaConf.save(config, out / "resolved.yaml", resolve=True)
    print("RESOLVED", OmegaConf.to_yaml(config), flush=True)
    from rlinf.models.embodiment.fastwam.builder import (
        build_fastwam_policy, compose_official_robotwin_cfg,
    )
    official = compose_official_robotwin_cfg(config.model)
    official.model.load_text_encoder = True
    inference_config = OmegaConf.create({
        "model": OmegaConf.to_container(official.model, resolve=True),
        "processor": OmegaConf.to_container(official.data.train.processor, resolve=True),
        "model_runtime_overrides": {"model_dtype": "bfloat16", "device": "cuda"},
    })
    OmegaConf.save(inference_config, out / "official_resolved.yaml", resolve=True)
    if args.prepare:
        print("PREPARED", out, flush=True)
        return

    import random
    import sapien
    from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv
    from rlinf.models.embodiment.fastwam.robotwin_adapter import compose_three_camera_image
    from envs._base_task import Base_Task
    import inspect

    print("SOURCE_BASE_TASK", inspect.getfile(Base_Task), flush=True)
    print("CUDA", torch.cuda.get_device_name(), os.environ.get("CUDA_VISIBLE_DEVICES"), flush=True)
    # Passthrough audit only: do not replace the selected value or render behavior.
    denoiser_calls = []
    set_denoiser = sapien.render.set_ray_tracing_denoiser
    def record_denoiser(value):
        denoiser_calls.append(value)
        print("DENOISER_SET", value, flush=True)
        assert value == args.denoiser, (value, args.denoiser)
        return set_denoiser(value)
    sapien.render.set_ray_tracing_denoiser = record_denoiser
    torch.set_num_threads(4)
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    t0 = time.monotonic()
    print("MODEL_LOAD_BEGIN", flush=True)
    policy = build_fastwam_policy(config.model, torch_dtype=torch.bfloat16)
    policy.eval().requires_grad_(False)
    print("MODEL_LOAD_DONE", time.monotonic() - t0, flush=True)
    env = RoboTwinEnv(env_cfg, 1, 0, 1, None)
    results = []

    def snapshot(obs, episode_dir, query):
        images = {
            "head": obs["main_images"][0].cpu().numpy(),
            "left": obs["wrist_images"][0, 0].cpu().numpy(),
            "right": obs["wrist_images"][0, 1].cpu().numpy(),
        }
        composite = compose_three_camera_image(obs["main_images"], obs["wrist_images"], device="cpu", dtype=torch.float32)
        images["model_input"] = ((composite[0].permute(1, 2, 0).numpy() + 1) * 127.5).round().clip(0, 255).astype(np.uint8)
        for name, rgb in images.items():
            Image.fromarray(rgb).save(episode_dir / f"q{query:02d}_{name}.png")

    try:
        for episode, seed in enumerate(seeds):
            episode_dir = out / f"episode_{episode}_seed_{seed}"
            episode_dir.mkdir(exist_ok=True)
            print("EPISODE_BEGIN", episode, seed, flush=True)
            random.seed(1234 + episode)
            obs, _ = env.reset(env_seeds=[int(seed)])
            assert env.venv.envs[0].task.ray_tracing_denoiser == args.denoiser
            row = {"episode": episode, "requested_seed": int(seed), "instruction": obs["task_descriptions"], "initial_state": obs["states"].tolist(), "queries": [], "success": False}
            snapshot(obs, episode_dir, 0)
            for query in range(8):
                start = time.monotonic()
                with torch.inference_mode():
                    actions, _ = policy.predict_action_batch(obs, mode="eval")
                assert tuple(actions.shape) == (1, 24, 14), actions.shape
                assert bool(torch.isfinite(actions).all()), "Non-finite policy action"
                np.save(episode_dir / f"q{query:02d}_actions.npy", actions.cpu().numpy())
                if query == 0:
                    row["first_actions"] = actions[0].cpu().tolist()
                obs, reward, terminated, truncated, info = env.step(actions, auto_reset=False)
                success = bool(env.success_once[0].item())
                row["success"] = row["success"] or success
                entry = {"query": query + 1, "seconds": time.monotonic() - start, "success_once": success, "terminated": bool(terminated[0]), "truncated": bool(truncated[0]), "run_steps": int(env.venv.envs[0].task.run_steps)}
                row["queries"].append(entry)
                snapshot(obs, episode_dir, query + 1)
                print("QUERY", json.dumps(entry), flush=True)
                (episode_dir / "result.json").write_text(json.dumps(row, indent=2))
                if success or bool(terminated[0]) or bool(truncated[0]):
                    break
            results.append(row)
            print("EPISODE_DONE", episode, row["success"], len(row["queries"]), flush=True)
    finally:
        env.offload()
        env.venv.env_thread_pool.shutdown(wait=True)
    summary = {"denoiser": args.denoiser, "elapsed_seconds": time.monotonic() - t0, "episodes": results, "denoiser_calls": denoiser_calls, "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_torch_reserved_bytes": torch.cuda.max_memory_reserved()}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("TRIAL_COMPLETE", args.denoiser, len(results), sum(x["success"] for x in results), flush=True)


if __name__ == "__main__":
    main()
