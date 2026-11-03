# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Apply the accepted Dojo boundary to an exported, pinned DSRL source tree.

No changes to SAC losses, replay projection, target updates or optimizer logic.
"""

import json
import shutil
import sys
from pathlib import Path

DSRL_BASE = "ff64209f8ee668b35837ae6469509b7f0257b89d"


def enable_offloaded_resume(root):
    """Reuse FSDPModelManager's onload prelude before SAC strict resume."""
    p = Path(root) / "rlinf/workers/actor/fsdp_sac_policy_worker.py"
    text = p.read_text()
    old = "    def load_checkpoint(self, load_base_path):\n        # load model\n"
    new = (
        "    def load_checkpoint(self, load_base_path):\n"
        "        if self.is_weight_offloaded:\n"
        "            self.load_param_and_grad(self.device)\n"
        "            self.is_weight_offloaded = False\n"
        "        if self.is_optimizer_offloaded:\n"
        "            self.load_optimizer(self.device)\n"
        "            self.is_optimizer_offloaded = False\n\n"
        "        # load model\n"
    )
    if new not in text:
        assert text.count(old) == 1
        p.write_text(text.replace(old, new, 1))


def integrate(root, accepted):
    root, accepted = Path(root), Path(accepted)
    for name in (
        "rlinf/envs/robodojo",
        "tools/dojo_grpo",
        "tools/rlt_disk_fallback",
        "tools/dojo_offpolicy",
    ):
        shutil.copytree(accepted / name, root / name, dirs_exist_ok=True)
    p = root / "rlinf/envs/__init__.py"
    text = p.read_text()
    if 'ROBODOJO = "robodojo"' not in text:
        text = text.replace(
            'ROBOTWIN = "robotwin"', 'ROBOTWIN = "robotwin"\n    ROBODOJO = "robodojo"'
        )
        token = "    elif env_type == SupportedEnvType.ROBOTWIN:"
        assert token in text
        text = text.replace(
            token,
            "    elif env_type == SupportedEnvType.ROBODOJO:\n        from rlinf.envs.robodojo.robodojo_env import RoboDojoEnv\n\n        return RoboDojoEnv\n"
            + token,
        )
        p.write_text(text)
    p = root / "rlinf/models/embodiment/openpi/dataconfig/__init__.py"
    text = p.read_text()
    source = (accepted / p.relative_to(root)).read_text()
    begin = source.index('    TrainConfig(\n        name="pi05_robodojo"')
    end = source.index("    TrainConfig(", begin + 20)
    block = source[begin:end]
    assert 'name="pi05_robodojo"' not in text
    p.write_text(text.replace("_CONFIGS = [\n", "_CONFIGS = [\n" + block, 1))
    p = root / "rlinf/models/embodiment/openpi/__init__.py"
    text = p.read_text()
    old = 'config_name == "pi05_sidney_robotwin" and actor_model_config.use_dsrl'
    assert old in text
    p.write_text(
        text.replace(
            old,
            'config_name in ("pi05_sidney_robotwin", "pi05_robodojo") and actor_model_config.use_dsrl',
        )
    )
    enable_offloaded_resume(root)
    (root / "dojo-offpolicy-source.json").write_text(
        json.dumps(
            {
                "dsrl_base": DSRL_BASE,
                "dojo_source": str(accepted),
                "method_core_changed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    integrate(*sys.argv[1:])
