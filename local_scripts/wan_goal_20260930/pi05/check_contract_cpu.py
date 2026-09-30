"""Server CPU checks for the actual patched transforms and resolved Hydra recipes.

Uses the real pinned norm_stats and installed OpenPI transforms. Only the text
tokenizer is stubbed, to avoid downloading tokenizer assets. No network, model
weights, model instantiation, GPU, training, or simulator is used.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--wm-path", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve(strict=True)
    model_path = args.model_path.resolve(strict=True)
    stats_path = model_path / "physical-intelligence/libero/norm_stats.json"
    stats_bytes = stats_path.read_bytes()
    stats_sha = hashlib.sha256(stats_bytes).hexdigest()
    expected_stats_sha = "dae37d79a22108af83df9189c6710a3ec8e077d65b28e34bdf1da724e5ae30f1"
    assert stats_sha == expected_stats_sha, "Use the fixed pi05 HF revision's unmodified norm_stats.json"
    stats_json = json.loads(stats_bytes)["norm_stats"]

    # This affects only this CPU check process, not the caller or launch controller.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["EMBODIED_PATH"] = str(repo / "examples/embodiment")
    os.environ["WAN_GOAL_PI05_PATH"] = str(model_path)
    os.environ["WAN_GOAL_WM_PATH"] = str(args.wm_path.resolve(strict=True))
    os.environ["WAN_GOAL_RUN_DIR"] = "/tmp/wan-goal-pi05-cpu-config-only"
    os.environ["ROBOT_PLATFORM"] = "LIBERO"
    sys.path.insert(0, str(repo))

    from apply_interface import build_plan
    source_rows = build_plan(repo)
    assert all(row["status"] == "already_applied" for *_, row in source_rows), "Apply the reviewed adapter first"

    import numpy as np
    import torch
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from openpi import transforms
    from openpi.models import model as upstream_model
    from rlinf.models.embodiment.openpi import _resolve_action_horizon_and_chunk
    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
    from rlinf.models.embodiment.openpi.modules.model import Observation, preprocess_observation
    from rlinf.models.embodiment.openpi.pi0 import make_attn_mask
    from rlinf.models.embodiment.openpi.policies.libero_policy import LiberoInputs
    from rlinf.models.embodiment.openpi.transforms.apply import apply_input_transform, apply_output_transform
    from rlinf.models.embodiment.openpi.transforms.env.default import repack_env_obs
    from rlinf.models.embodiment.openpi.transforms.pipeline import build_openpi_transforms

    results = []
    configs = {}
    config_dir = repo / "examples/embodiment/config"
    with initialize_config_dir(version_base="1.1", config_dir=str(config_dir)):
        official = compose(config_name="libero_goal_grpo_openpi_pi05")
        OmegaConf.resolve(official)
        for mode in ("smoke", "formal"):
            name = f"wan_goal_pi05_headonly_{mode}_sz3"
            primary = OmegaConf.load(config_dir / (name + ".yaml"))
            # A full parent recipe owns hydra.searchpath and cannot be inherited.
            # Both launch configs are independent primary configs with group defaults.
            defaults = OmegaConf.to_container(primary.defaults)
            assert "env/wan_libero_goal@env.train" in defaults
            assert "env/libero_goal@env.eval" in defaults
            assert "model/pi0_5@actor.model" in defaults
            assert "hybrid_engines/fsdp@actor.fsdp_config" in defaults
            assert "weight_syncer/patch_syncer@weight_syncer" in defaults
            assert "libero_goal_grpo_openpi_pi05" not in defaults
            assert "wan_goal_pi05_headonly_formal_sz3" not in defaults
            assert primary.hydra.searchpath
            cfg = compose(config_name=name)
            OmegaConf.resolve(cfg)
            configs[mode] = cfg
    for mode, cfg in configs.items():
        assert OmegaConf.to_container(cfg.algorithm) == OmegaConf.to_container(official.algorithm)
        assert OmegaConf.to_container(cfg.actor.optim) == OmegaConf.to_container(official.actor.optim)
        assert OmegaConf.to_container(cfg.actor.fsdp_config) == OmegaConf.to_container(official.actor.fsdp_config)
        assert cfg.cluster.component_placement["actor,env,rollout"] == "4-7"
        assert cfg.env.train.env_type == "world_model" and cfg.env.train.backend == "wan"
        assert cfg.env.train.wm_env_type == "libero" and cfg.env.train.task_suite_name == "libero_goal"
        assert cfg.env.eval.env_type == "libero" and cfg.env.eval.task_suite_name == "libero_goal"
        assert cfg.actor.model.openpi.config_name == "pi05_libero"
        assert cfg.actor.model.openpi_data.wrist_mode == "disabled"
        assert cfg.actor.model.num_action_chunks == cfg.actor.model.openpi.action_chunk == cfg.env.train.chunk == 8
        assert _resolve_action_horizon_and_chunk(cfg.actor.model, cfg.actor.model.openpi) == (10, 8)
        assert cfg.actor.model.num_steps == cfg.actor.model.openpi.num_steps == 5
        assert cfg.actor.model.action_dim == 7 and cfg.actor.model.openpi.model_action_dim == 32
        assert cfg.actor.model.openpi.noise_method == "flow_sde" and cfg.actor.model.openpi.noise_level == 0.3
        assert cfg.actor.model.openpi.train_expert_only and not cfg.actor.model.add_value_head
        assert cfg.algorithm.adv_type == "grpo" and cfg.algorithm.group_size == cfg.env.train.group_size == 8
        assert cfg.algorithm.update_epoch == 1
        assert cfg.algorithm.reward_type == cfg.algorithm.logprob_type == "chunk_level"
        assert cfg.algorithm.reward_coef == cfg.env.train.reward_coef == 1.0
        assert cfg.algorithm.filter_rewards and cfg.algorithm.rewards_lower_bound == 0.1
        assert cfg.algorithm.rewards_upper_bound == 0.9
        assert cfg.actor.optim.lr == 5e-6 and cfg.actor.seed == 42
        assert cfg.env.train.condition_frame_length == 5 and cfg.env.train.num_frames == 13
        assert cfg.env.train.num_inference_steps == 5 and cfg.env.train.enable_kir
        assert cfg.env.train.max_episode_steps == cfg.env.train.max_steps_per_rollout_epoch == 320
        assert cfg.actor.enable_offload and cfg.rollout.enable_offload and cfg.env.train.enable_offload
        assert not cfg.actor.fsdp_config.gradient_checkpointing
        assert cfg.actor.fsdp_config.sharding_strategy == "no_shard"
        assert cfg.rollout.model.model_path == cfg.actor.model.model_path == str(model_path)
        assert cfg.runner.resume_dir is None and cfg.runner.ckpt_path is None
        expected = (32, 1, 1280, 64, 2) if mode == "smoke" else (64, 8, 2048, 128, 1000)
        actual = (cfg.env.train.total_num_envs, cfg.env.train.rollout_epoch,
                  cfg.actor.global_batch_size, cfg.actor.micro_batch_size, cfg.runner.max_epochs)
        assert actual == expected, (mode, actual)
        n, r, batch, micro, epochs = actual
        assert n % (4 * 8) == 0 and batch % (4 * micro) == 0
        samples = n * r * (320 // 8)
        assert samples % batch == 0
        results.append({"check": f"resolved_{mode}", "chunk_samples_per_epoch": samples,
                        "optimizer_steps_per_epoch": samples // batch,
                        "microbatches_per_rank_per_optimizer_step": batch // (4 * micro),
                        "epochs": epochs})

    # The default adapter remains strict and retains real proprio/wrist data.
    main_image = np.full((224, 224, 3), 91, np.uint8)
    real_wrist = np.full((224, 224, 3), 172, np.uint8)
    real_state = np.linspace(-1.0, 1.0, 8, dtype=np.float32)
    sample = {"observation/image": main_image, "observation/state": real_state,
              "observation/wrist_image": real_wrist, "prompt": "pick the bowl"}
    required = LiberoInputs(model_type=upstream_model.ModelType.PI05)
    default_output = required(copy.deepcopy(sample))
    np.testing.assert_array_equal(default_output["state"], real_state)
    np.testing.assert_array_equal(default_output["image"]["left_wrist_0_rgb"], real_wrist)
    assert default_output["image_mask"]["left_wrist_0_rgb"]
    missing = dict(sample); missing.pop("observation/wrist_image")
    try:
        required(missing)
    except KeyError:
        pass
    else:
        raise AssertionError("Default required mode silently accepted a missing wrist")
    results.append({"check": "required_mode_preserved", "passed": True})

    class TokenizerStub:
        calls = []

        def __init__(self, max_len):
            self.max_len = max_len

        def tokenize(self, text, state=None):
            assert state is None, "State unexpectedly entered PI05 prompt conditioning"
            self.calls.append(text)
            values = list(text.encode("utf-8")[: self.max_len])
            tokens = np.zeros(self.max_len, dtype=np.int32)
            mask = np.zeros(self.max_len, dtype=np.bool_)
            tokens[:len(values)] = values
            mask[:len(values)] = True
            return tokens, mask

    data_kwargs = OmegaConf.to_container(configs["formal"].actor.model.openpi_data, resolve=True)
    with patch("openpi.models.tokenizer.PaligemmaTokenizer", TokenizerStub):
        train_config = get_openpi_config("pi05_libero", model_path=str(model_path), data_kwargs=data_kwargs)
        assert train_config.model.action_horizon == 10 and not train_config.model.discrete_state_input
        assert not train_config.data.extra_delta_transform
        input_steps, output_steps = build_openpi_transforms(str(model_path), "pi05_libero", data_kwargs)
        assert any(isinstance(step, transforms.Normalize) and step.norm_stats is not None for step in input_steps)
        input_fn, output_fn = transforms.compose(input_steps), transforms.compose(output_steps)
        raw = {"main_images": torch.full((2, 256, 256, 3), 91, dtype=torch.uint8),
               "wrist_images": None, "states": torch.arange(32, dtype=torch.float32).reshape(2, 16),
               "task_descriptions": ["pick the bowl", "open the drawer"]}
        repacked = repack_env_obs(raw, select_state=lambda state: state)
        processed = apply_input_transform(input_fn, repacked)
        assert processed["state"].shape == (2, 32) and torch.isfinite(processed["state"]).all()
        assert processed["image_mask"]["base_0_rgb"].all()
        assert not processed["image_mask"]["left_wrist_0_rgb"].any()
        assert not processed["image_mask"]["right_wrist_0_rgb"].any()
        assert tuple(processed["image"]["base_0_rgb"].shape) == (2, 224, 224, 3)
        assert not processed["image"]["left_wrist_0_rgb"].any()

        alternate = copy.deepcopy(raw)
        alternate["wrist_images"] = torch.full((2, 256, 256, 3), 241, dtype=torch.uint8)
        alternate["states"] = torch.full((2, 8), 1000.0)
        other = apply_input_transform(input_fn, repack_env_obs(alternate, select_state=lambda state: state))

        def same_tree(a, b):
            if isinstance(a, dict):
                assert a.keys() == b.keys()
                for key in a:
                    same_tree(a[key], b[key])
            else:
                assert torch.equal(a, b), "Disabled wrist/state changed actual transformed policy inputs"

        same_tree(processed, other)
        replay = {key: value for key, value in repacked.items() if key != "prompt"}
        replay.update(tokenized_prompt=processed["tokenized_prompt"],
                      tokenized_prompt_mask=processed["tokenized_prompt_mask"])
        same_tree(processed, apply_input_transform(input_fn, replay))
        assert TokenizerStub.calls

        obs = preprocess_observation(Observation.from_dict(copy.deepcopy(processed)), train=False)
        assert not obs.image_masks["left_wrist_0_rgb"].any()
        prefix_mask = torch.stack([obs.image_masks[key] for key in
                                   ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")], dim=1)
        attention = make_attn_mask(prefix_mask, torch.zeros(3, dtype=torch.bool))
        assert not attention[:, :, 1:].any() and attention[:, 0, 0].all()

        model_actions = torch.zeros(2, 10, 32)
        output = apply_output_transform(output_fn, {"actions": model_actions, "state": processed["state"]}, action_chunk=8)
        changed_state_output = apply_output_transform(output_fn, {"actions": model_actions, "state": processed["state"] + 1000}, action_chunk=8)
        assert output["actions"].shape == (2, 8, 7) and torch.isfinite(output["actions"]).all()
        assert torch.equal(output["actions"], changed_state_output["actions"])
        results.append({"check": "real_transform_chain_with_stub_tokenizer", "passed": True,
                        "state_before_norm": 8, "state_after_pad": 32, "valid_cameras": 1,
                        "action_output": [2, 8, 7], "actor_replay_matches_rollout": True})

    result = {"status": "passed", "scope": "CPU source/config/real-transform checks; no GPU policy or GRPO update",
              "norm_stats_sha256": stats_sha,
              "norm_stats_dimensions": {key: len(value["mean"]) for key, value in stats_json.items()},
              "source": [row for *_, row in source_rows], "checks": results}
    payload = json.dumps(result, indent=2) + "\n"
    if args.receipt:
        args.receipt.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
