"""Create private click_bell sources from the working OpenDW B16 sources.

This command only writes new files. It never edits the donor, starts a service,
loads a model, borrows a GPU, or changes RLT ownership.
"""
import argparse
import hashlib
import json
from pathlib import Path


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError("Donor source differs at required hook: " + old[:100])
    return source.replace(old, new, 1)


def adapter_source(source):
    source = replace_once(source, "    success_threshold=0.9,\n):", "    success_threshold=0.9,\n    reward_mode=\"frame_score\",\n):")
    source = replace_once(source, "    reward_frames = scores.copy()\n", "    if reward_mode not in {\"frame_score\", \"first_success_binary\"}:\n        raise ValueError(\"Unknown reward mode\")\n    if reward_mode == \"first_success_binary\" and (relative_reward or float(reward_coef) != 1.0):\n        raise ValueError(\"Binary success uses no score differences and coefficient one\")\n    reward_frames = scores.copy()\n")
    source = replace_once(source, "    success = hits.any(axis=1)\n", "    success = hits.any(axis=1)\n    if reward_mode == \"first_success_binary\":\n        # The complete C32 block was generated/executed; reward and done are\n        # aligned at its boundary. The env stops this row after its first hit.\n        rewards.fill(0.0)\n        rewards[:, -1] = success.astype(np.float32)\n")
    return source


def env_source(source):
    if 'self.use_rynn' in source:
        raise ValueError("Use the working pre-Rynn OpenDW env donor")
    if 'self.reward_mode' in source:
        raise ValueError("Reward mode hook is already present; do not patch twice")
    source = replace_once(source, '        self.auto_reset = bool(cfg.get("auto_reset", False))\n', '        self.auto_reset = bool(cfg.get("auto_reset", False))\n        self.reward_mode = str(cfg.get("reward_mode", "frame_score"))\n        if self.reward_mode not in {"frame_score", "first_success_binary"}:\n            raise ValueError("Unknown reward mode")\n        if self.reward_mode == "first_success_binary" and (\n            cfg.get("ignore_terminations", False) or cfg.get("use_rel_reward", True)\n            or float(cfg.get("reward_coef", 1.0)) != 1.0\n        ):\n            raise ValueError("Binary success requires termination, no shaping, coefficient one")\n')
    source = replace_once(source, '                success_threshold=self.cfg.get("success_reward_threshold", 0.9),\n', '                success_threshold=self.cfg.get("success_reward_threshold", 0.9),\n                reward_mode=self.reward_mode,\n')
    source = replace_once(source, '            "first_success_action": torch.from_numpy(first_success),\n', '            "first_success_action": torch.from_numpy(first_success),\n            "reward_score_last": torch.from_numpy(self.prev_step_reward.copy()),\n')
    return source


def base_owner_source(source):
    source = replace_once(source, "env['task_name'] == 'adjust_bottle'", "env['task_name'] == 'click_bell'")
    source = replace_once(source, "    assert env['use_rel_reward'] is True and env['reward_coef'] == 1.0\n", "    assert env['use_rel_reward'] is False and env['reward_coef'] == 1.0\n    assert env['reward_mode'] == 'first_success_binary'\n    assert env['reward_source'] == 'worldarena_t5_classifier'\n")
    source = replace_once(source, "        assert one_arg(argv, '--physical-gpu') == str(service['physical_gpu'])\n", "        assert one_arg(argv, '--physical-gpu') == str(service['physical_gpu'])\n        assert one_arg(argv, '--execution-mode') == 'batched'\n        assert one_arg(argv, '--wm-batch-size') == '16'\n        assert one_arg(argv, '--reward-checkpoint') == service['reward_checkpoint']\n        assert sha(owned_path(service['reward_checkpoint'])) == service['reward_checkpoint_sha256']\n")
    return source


def formal_owner_source(source):
    source = replace_once(source, "native['task_config']['task_name'] == 'adjust_bottle'", "native['task_config']['task_name'] == 'click_bell'")
    source = replace_once(source, "read(native['seeds_path'])['adjust_bottle']['success_seeds']", "read(native['seeds_path'])['click_bell']['success_seeds']")
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('adapter', 'env', 'base-owner', 'formal-owner'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = [
        (args.adapter, 'rlinf/envs/world_model/opendw_adapter.py', adapter_source),
        (args.env, 'rlinf/envs/world_model/opendw_robotwin_env.py', env_source),
        (args.base_owner, 'owner/opendw_multigpu_owner.py', base_owner_source),
        (args.formal_owner, 'owner/opendw_formal_owner.py', formal_owner_source),
    ]
    generated = []
    for donor, relative, transform in rows:
        source = transform(donor.read_text(encoding='utf-8'))
        compile(source, relative, 'exec')
        generated.append((donor, relative, source))
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {'schema': 1, 'task': 'click_bell', 'launched': False, 'files': []}
    for donor, relative, source in generated:
        target = args.output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source, encoding='utf-8', newline='\n')
        manifest['files'].append({'donor': str(donor), 'donor_sha256': hashlib.sha256(donor.read_bytes()).hexdigest(),
                                 'target': str(target), 'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
    (args.output / 'source-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()
