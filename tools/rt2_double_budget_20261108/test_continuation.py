# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""CPU regression for actual DSRL/RLT restore layouts and stale save markers."""

import tempfile
import unittest
from pathlib import Path

from continue_rt2 import checkpoint_ready


class CheckpointLayoutTest(unittest.TestCase):
    def test_stale_marker_and_distinct_formats(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)

            def put(name):
                f = p / name
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_bytes(b"fixture")

            put("dojo-complete.json")
            self.assertFalse(checkpoint_ready(p, "dsrl"))
            self.assertFalse(checkpoint_ready(p, "rlt"))
            for name in [
                "actor/sac_components/target_model/checkpoint_rank_0.pt",
                "actor/dcp_checkpoint/.metadata",
                "actor/dcp_checkpoint/__0_0.distcp",
                "actor/model_state_dict/full_weights.pt",
                "actor/sac_components/rlt_trainer_state/checkpoint_rank_0.pt",
                "actor/sac_components/rlt_trainer_state/complete.json",
                "actor/sac_components/replay_buffer/rank_0/metadata.json",
            ]:
                put(name)
            self.assertTrue(checkpoint_ready(p, "rlt"))
            self.assertFalse(checkpoint_ready(p, "dsrl"))
            (p / "actor/dcp_checkpoint/__0_0.distcp").unlink()
            self.assertFalse(checkpoint_ready(p, "rlt"))
            for name in [
                "actor/local_shard_checkpoint/checkpoint_rank_0.pt",
                "actor/sac_components/dsrl_trainer_state_rank_0.pt",
                "actor/sac_components/replay_buffer/rank_0/dsrl_transition_replay.pt",
                "actor/sac_components/alpha/dcp_checkpoint/.metadata",
                "actor/sac_components/alpha/dcp_checkpoint/__0_0.distcp",
            ]:
                put(name)
            self.assertTrue(checkpoint_ready(p, "dsrl"))
            (
                p
                / "actor/sac_components/replay_buffer/rank_0/dsrl_transition_replay.pt"
            ).write_bytes(b"")
            self.assertFalse(checkpoint_ready(p, "dsrl"))


if __name__ == "__main__":
    unittest.main()
