from __future__ import annotations

import sys
from collections import defaultdict

sys.path.insert(0, "/root/autodl-tmp")
from qam_real_model_basic_probe import configure_qam  # noqa: E402

from rlinf.models import get_model  # noqa: E402


cfg = configure_qam()
model = get_model(cfg.actor.model)
groups: dict[int, list[str]] = defaultdict(list)
for name, parameter in model.named_parameters(remove_duplicate=False):
    groups[id(parameter)].append(name)
shared = [names for names in groups.values() if len(names) > 1]
print("NO_SPLIT_MODULES", model._no_split_modules)
print("NO_SPLIT_NAMES", model._no_split_names)
print("SHARED_GROUP_COUNT", len(shared))
for names in shared:
    print("SHARED", " | ".join(names))
for name, module in model.named_modules():
    if name.endswith("embed_tokens") or name.endswith("lm_head"):
        weight = getattr(module, "weight", None)
        print(
            "MODULE",
            name,
            type(module).__name__,
            None if weight is None else tuple(weight.shape),
            None if weight is None else id(weight),
        )
