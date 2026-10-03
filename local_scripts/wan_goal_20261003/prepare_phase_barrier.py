"""Build an exact-SHA reviewed patch; default does not write the target checkout.

--apply is for a new isolated, approved checkout only. Never run on the old r6
checkout. Configuration, optimizer schedule, rollout geometry and budgets stay fixed.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from pathlib import Path


EXPECTED = {
    "rlinf/runners/embodied_runner.py": "94619101397255b04ec301d63d143570031d13f813e429a813df7a2e479dc0bd",
    "rlinf/workers/env/env_worker.py": "c622571777474b8f2d08f42216f97ed419a0a32581cf9a8cc1c661c96dfe6023",
    "rlinf/workers/actor/embodied_fsdp_actor_worker.py": "9b8abf1f6ca0f90cbf2ee6969bb6969033d465ebb6765ce7aa405e807bf29d76",
    "rlinf/workers/rollout/hf/huggingface_worker.py": "0abb32c7b6a743a0057551410489c8505a76c7cdaf8a79aa33262e1070086e3c",
}


def replace_once(text, old, new):
    assert text.count(old) == 1, (old, text.count(old))
    return text.replace(old, new, 1)


def build(repo):
    changes = []
    for relative, expected in EXPECTED.items():
        path = repo / relative
        before = path.read_bytes()
        assert hashlib.sha256(before).hexdigest() == expected, relative
        text = before.decode()
        if relative.endswith("embodied_runner.py"):
            text = replace_once(text,
                "                    self.actor.recv_rollout_trajectories(\n"
                "                        input_channel=self.actor_channel\n"
                "                    ).wait()\n"
                "                    rollout_handle.wait()\n                    if self.reward is not None:\n",
                "                    self.actor.recv_rollout_trajectories(\n"
                "                        input_channel=self.actor_channel\n"
                "                    ).wait()\n"
                "                    rollout_handle.wait()\n"
                "                    # A final trajectory may arrive before env video flush/offload.\n"
                "                    # Colocated actor training must wait for the environment phase.\n"
                "                    env_handle.wait()\n"
                "                    if self.reward is not None:\n")
        else:
            # Add an ordinary import before the first rlinf import, retaining module headers.
            anchor = next(line for line in text.splitlines(True) if line.startswith("from rlinf."))
            text = replace_once(text, anchor,
                "from rlinf.utils.resource_telemetry import record_resource_boundary\n" + anchor)
            if relative.endswith("env_worker.py"):
                text = replace_once(text,
                    "        for env in self.env_list:\n            if self.train_enable_offload:\n                get_env_attr(env, \"offload\")()\n\n        return env_metrics\n",
                    "        record_resource_boundary(self, \"env_before_offload\")\n"
                    "        for env in self.env_list:\n            if self.train_enable_offload:\n                get_env_attr(env, \"offload\")()\n"
                    "        record_resource_boundary(self, \"env_after_offload\")\n\n        return env_metrics\n")
            elif relative.endswith("embodied_fsdp_actor_worker.py"):
                text = replace_once(text,
                    "        Run the training process using the received rollout batch.\n        \"\"\"\n        if self.is_weight_offloaded:",
                    "        Run the training process using the received rollout batch.\n        \"\"\"\n"
                    "        record_resource_boundary(self, \"actor_before_onload\", reset_peak=True)\n"
                    "        if self.is_weight_offloaded:")
                text = replace_once(text,
                    "            self.load_optimizer(self.device)\n\n        if self.cfg.algorithm.loss_type == \"opd\":",
                    "            self.load_optimizer(self.device)\n"
                    "        record_resource_boundary(self, \"actor_after_onload\")\n\n"
                    "        if self.cfg.algorithm.loss_type == \"opd\":")
                text = replace_once(text,
                    "        return mean_metric_dict\n\n    def train_micro_batch(",
                    "        record_resource_boundary(self, \"actor_after_training\")\n"
                    "        return mean_metric_dict\n\n    def train_micro_batch(")
            else:
                text = replace_once(text,
                    "        actor_channel: Channel,\n    ):\n        if self.enable_offload:\n            self.reload_model()\n",
                    "        actor_channel: Channel,\n    ):\n"
                    "        record_resource_boundary(self, \"rollout_before_onload\", reset_peak=True)\n"
                    "        if self.enable_offload:\n            self.reload_model()\n"
                    "        record_resource_boundary(self, \"rollout_after_onload\")\n")
                text = replace_once(text,
                    "        if self.enable_offload:\n            self.offload_model()\n\n    @Worker.timer(\"evaluate\")",
                    "        record_resource_boundary(self, \"rollout_before_offload\")\n"
                    "        if self.enable_offload:\n            self.offload_model()\n"
                    "        record_resource_boundary(self, \"rollout_after_offload\")\n\n"
                    "    @Worker.timer(\"evaluate\")")
        compile(text, relative, "exec")
        changes.append((relative, before, text.encode()))
    helper = Path(__file__).with_name("resource_telemetry.py").read_bytes()
    compile(helper, "resource_telemetry.py", "exec")
    relative = "rlinf/utils/resource_telemetry.py"
    assert not (repo / relative).exists(), relative
    changes.append((relative, b"", helper))
    return changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    if args.apply:
        assert repo.name != "RLinf-pi05", "Preserve the original r6 checkout"
    changes = build(repo)
    args.output.mkdir(parents=True, exist_ok=True)
    patch = "".join("".join(difflib.unified_diff(before.decode().splitlines(True),
                   after.decode().splitlines(True), fromfile="a/"+relative,
                   tofile="b/"+relative)) for relative, before, after in changes)
    (args.output / "phase-barrier.patch").write_text(patch, encoding="utf-8")
    manifest = []
    for relative, before, after in changes:
        target = args.output / "reviewed-source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(after)
        manifest.append(dict(path=relative, before_sha256=hashlib.sha256(before).hexdigest(),
                             after_sha256=hashlib.sha256(after).hexdigest(), bytes=len(after)))
    if args.apply:
        # All sources, replacements and syntax have been checked before any target write.
        for relative, before, after in changes:
            target = repo / relative
            target.write_bytes(after)
    (args.output / "patch-manifest.json").write_text(json.dumps({"applied":args.apply,
        "repo":str(repo),"changes":manifest},indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"applied":args.apply,"files":len(changes),"output":str(args.output)}))


if __name__ == "__main__":
    main()
