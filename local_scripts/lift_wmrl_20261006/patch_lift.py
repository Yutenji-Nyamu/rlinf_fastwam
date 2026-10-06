"""Make minimal private lift_pot sources from the working click_bell sources.

These pure transformations neither overwrite donors nor load GPU models. The
driver must freeze donor/target hashes and install only into a new private run.
Unchanged: WM batching, eight-frame score ordering, first-success semantics,
offload barriers, training budget, process lifecycle, and RLT return.
"""
import argparse
import hashlib
import json
from pathlib import Path


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError("Donor source differs at required hook: " + old[:120])
    return source.replace(old, new, 1)


def service_source(source):
    source = replace_once(source, "        from opendw_reward import RoboTwinT5Reward\n",
                          "        from rm_adapter import LiftPotReward\n")
    source = replace_once(source, "        self.reward = RoboTwinT5Reward(args.reward_checkpoint, args.t5_path)\n",
                          "        self.reward = LiftPotReward(args.reward_checkpoint)\n")
    source = replace_once(source, '            "t5_path": str(args.t5_path), "norm_stats_sha256":',
                          '            **self.reward.deployment_metadata(), "norm_stats_sha256":')
    source = replace_once(source, '    parser.add_argument("--t5-path", type=Path, required=True)\n', "")
    source = source.replace("    --t5-path /path/t5-base --physical-gpu 6 --port 18941 --output-dir /path/run \\\n",
                            "    --physical-gpu 6 --port 18941 --output-dir /path/run \\\n", 1)
    if "RoboTwinT5Reward" in source or "args.t5_path" in source:
        raise ValueError("Residual task RM T5 loading found")
    compile(source, "opendw_service_batched.py", "exec")
    return source


def base_owner_source(source):
    source = replace_once(source, "env['task_name'] == 'click_bell'", "env['task_name'] == 'lift_pot'")
    source = replace_once(source, "env['reward_source'] == 'worldarena_t5_classifier'",
                          "env['reward_source'] == 'single_task_resnet_classifier'")
    source = replace_once(source, "env['success_reward_threshold'] == 0.9",
                          "env['success_reward_threshold'] == 0.7082200646400452")
    compile(source, "opendw_multigpu_owner.py", "exec")
    return source


def formal_owner_source(source):
    source = replace_once(source, "native['task_config']['task_name'] == 'click_bell'",
                          "native['task_config']['task_name'] == 'lift_pot'")
    source = replace_once(source, "read(native['seeds_path'])['click_bell']['success_seeds']",
                          "read(native['seeds_path'])['lift_pot']['success_seeds']")
    compile(source, "opendw_formal_owner.py", "exec")
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("service", "base-owner", "formal-owner"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [(args.service, "service/opendw_service_batched.py", service_source),
            (args.base_owner, "owner/opendw_multigpu_owner.py", base_owner_source),
            (args.formal_owner, "owner/opendw_formal_owner.py", formal_owner_source)]
    generated = [(source, relative, transform(source.read_text(encoding="utf-8")))
                 for source, relative, transform in rows]
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(schema_version=1, task="lift_pot", launched=False, files=[])
    for source, relative, text in generated:
        target = args.output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8", newline="\n")
        manifest["files"].append(dict(donor=str(source), donor_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                                      target=str(target), sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    (args.output / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
