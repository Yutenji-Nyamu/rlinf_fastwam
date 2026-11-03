"""Exercise the actual Clean batch mask and GRPO score for early success."""

import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rlinf.utils.utils import preprocess_embodied_batch
from rlinf.algorithms.utils import (
    preprocess_embodied_advantages_inputs,
    calculate_scores,
)

rewards = torch.zeros(3, 8, 50)
dones = torch.zeros(4, 8, 50, dtype=torch.bool)
rewards[0, 0, 16] = 1
dones[1, 0, 16:] = True
dones[2:, 0, :] = True
dones[-1, 1:, -1] = True
batch = preprocess_embodied_batch(
    {"rewards": rewards, "dones": dones},
    rollout_epoch=1,
    auto_reset=False,
    ignore_terminations=False,
    reward_type="chunk_level",
    filter_rewards=True,
    group_size=8,
    rewards_lower_bound=0.1,
    rewards_upper_bound=0.9,
)
assert batch["loss_mask"][0, 0].all()
assert not batch["loss_mask"][1:, 0].any()
assert float((batch["rewards"] * batch["loss_mask"]).sum()) == 1.0
inputs = preprocess_embodied_advantages_inputs(
    **batch, reward_type="chunk_level", adv_type="grpo", group_size=8
)
scores = calculate_scores(**inputs)["rewards"]
torch.testing.assert_close(
    scores, torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
)
print(
    "CREDIT_PASS: action17 success preserved; later chunks masked; group return exactly1"
)
