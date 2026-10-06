"""One bounded teacher-collection smoke to cover real successful U-weighted BC.

Input is the frozen v1 *actual smoke* configuration. Only three smoke fields
change: runner.max_steps 2 -> 4, runner.save_interval 2 -> 4, and
algorithm.rlt_schedule.warmup_post_collect_updates 2 -> 8. With at most two
critic updates per round, collection starts at learner versions 0, 2, 4, 6;
all four training rounds therefore use the original teacher. Any successes
must come naturally from this fixed run and the original seed stream.

The caller separately creates a fresh v2 run/namespace and resource receipt.
This helper does not launch, resume, mutate the input or write any files.
It returns (revised_config, changed_leaves). The operation is authorized for
one v2 attempt only; do not repeat it to search for a favorable outcome.

Formal remains fresh Stage2 from the original complete Stage1 CP2000, N4,
800 rounds, B512/micro256/U5, warmup 10k and 15k initialization updates.
No smoke checkpoint or replay is used to initialize formal training.
"""

import copy
from collections.abc import Mapping
from typing import Any


EXPECTED_CHANGES = {
    "runner.max_steps": [2, 4],
    "runner.save_interval": [2, 4],
    "algorithm.rlt_schedule.warmup_post_collect_updates": [2, 8],
}


def _leaves(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {
            leaf: item
            for key, child in value.items()
            for leaf, item in _leaves(
                child, f"{prefix}.{key}" if prefix else str(key)
            ).items()
        }
    return {prefix: value}


def revise_smoke(cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, list[Any]]]:
    """Copy the frozen smoke recipe and return its exact three-leaf delta.

    Args:
        cfg: Plain resolved dictionary read from the frozen v1 actual smoke.

    Returns:
        A deep copy with the three smoke-only overrides, and a mapping from
        changed dotted paths to [original_value, revised_value].

    Raises:
        AssertionError: The input is not the intended v1 smoke, or any other
        configuration value changes while constructing the v2 recipe.
    """
    assert isinstance(cfg, dict), "Use the resolved v1 smoke dictionary."
    before = _leaves(cfg)
    for path, (old, _) in EXPECTED_CHANGES.items():
        assert before.get(path) == old, f"Unexpected frozen v1 value: {path}."
    assert cfg["runner"].get("resume_dir") is None, "v2 must start fresh."
    assert cfg["runner"].get("ckpt_path") is None, "Do not restore smoke weights."
    assert cfg["env"]["train"]["total_num_envs"] == 4, "Preserve N4."
    assert cfg["env"]["train"]["max_episode_steps"] == 200, "Preserve 200 actions."
    assert cfg["actor"]["global_batch_size"] == 32, "Use the historical smoke batch."
    assert cfg["actor"]["micro_batch_size"] == 16, "Use the historical smoke microbatch."
    assert cfg["algorithm"]["update_epoch"] == 1, "Preserve historical smoke U1."
    schedule = cfg["algorithm"]["rlt_schedule"]
    assert schedule["enable"] is True, "Teacher gating requires the RLT schedule."
    assert schedule["warmup_min_size"] == 4, "Preserve the smoke replay threshold."
    assert schedule["max_updates_per_train_step"] == 2, "Preserve two updates per round."
    assert cfg["algorithm"]["rlt_route"]["type"] == "full_task", "Use the original full-task route."
    method = cfg["algorithm"]["rlt_dvac"]
    assert method["mode"] == "apply", "The smoke must exercise applied weights."
    assert method["signal_source"] == "ugrow_10_5", "Use the frozen U signal."

    revised = copy.deepcopy(cfg)
    revised["runner"]["max_steps"] = 4
    revised["runner"]["save_interval"] = 4
    revised["algorithm"]["rlt_schedule"]["warmup_post_collect_updates"] = 8
    after = _leaves(revised)
    assert before.keys() == after.keys(), "Keep the frozen recipe's field set."
    changes = {
        key: [before.get(key), after.get(key)]
        for key in before.keys() | after.keys()
        if before.get(key) != after.get(key)
    }
    assert changes == EXPECTED_CHANGES, "Only the three authorized smoke leaves may change."
    assert _leaves(cfg) == before, "The frozen input must remain unchanged."
    return revised, changes
