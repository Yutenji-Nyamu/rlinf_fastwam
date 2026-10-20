import unittest
import numpy as np
import torch
from objects import annotate
from analysis import reconstruct


class ObjectTests(unittest.TestCase):
    def test_repeated_words_are_union_and_missing_roles_are_invalid(self):
        text = "Task: bottle to bottle, State: 0;\nAction: "
        row = dict(full_prompt=text, instruction_span=[6, 22],
                   offsets=[[0, 0], [6, 12], [13, 15], [16, 22]])
        a = annotate(row)
        self.assertEqual(a["token_indices"]["all_noun"], [1, 3])
        self.assertEqual(a["token_indices"]["source_noun"], [1])
        self.assertEqual(a["token_indices"]["goal_noun"], [3])

    def test_uniform_full_softmax_and_layer_product(self):
        P, V, H = 4, 2, 50
        row = dict(full_prompt="Task: bottle, State: 0;\nAction: ", instruction_span=[6, 12],
                   offsets=[[0, 0], [6, 12]])
        raw = dict(q_prefix=torch.zeros(1, 3, 2, P, 4), k_prefix=torch.zeros(1, 3, 1, P, 4),
                   q_action=torch.zeros(1, 10, 3, 2, H, 4), k_action=torch.zeros(1, 10, 3, 1, H, 4),
                   prefix_attention_mask=torch.zeros(1, 1, P, P),
                   attention_mask=torch.zeros(1, 10, 1, H, P + H),
                   prefix_scaling=[.5] * 3, scaling=[.5] * 3,
                   prefix_valid=torch.ones(1, P, dtype=torch.bool), token_ids=torch.zeros(1, 2, dtype=torch.long),
                   key_valid=torch.ones(1, P + H, dtype=torch.bool),
                   source=torch.tensor([[0, 0, 4, 1] + [3] * H]),
                   native_dv_telemetry={"z_endpoint": torch.zeros(1, 10, H, 14)})
        z, report = reconstruct(raw, dict(token_rows=[row], language_start=V), device="cpu")
        np.testing.assert_allclose(z["I_direct"][..., 0], 1 / (P + H), rtol=1e-6)
        np.testing.assert_allclose(z["I_via_visual"][..., 0], V / (P + H) / P, rtol=1e-6)
        self.assertTrue(np.isnan(z["I_direct"][..., 2]).all())
        self.assertTrue(report["passed"])


if __name__ == "__main__":
    unittest.main()
