"""Regression: the old 400/600-second admission windows expired under contention."""
import subprocess
import unittest
from unittest.mock import patch
from pathlib import Path
import queue_admission as admission


class AdmissionTests(unittest.TestCase):
    def test_waits_through_slow_release_then_admits(self):
        held = {4: {"processes": [{"pid": 77, "type": "C", "mib": 24000}]},
                7: {"processes": []}}
        tiny = {4: {"processes": [{"pid": 72, "type": "G", "mib": 7}]},
                7: {"processes": []}}
        slots = {4: {"tolerate_tiny_graphics_mib": 8}, 7: {}}
        with patch.object(admission, "gpus", side_effect=[held]*202+[tiny]), \
             patch.object(admission.time, "sleep") as delay, patch.object(admission,"save") as save:
            admission.wait_for_cards(Path("."), slots)
        self.assertEqual(sum(call.args[0] for call in delay.call_args_list),606)
        self.assertEqual(save.call_count,1)

    def test_unavailable_sample_keeps_reservation(self):
        with patch.object(admission,"gpus",side_effect=[subprocess.TimeoutExpired("nvidia-smi",25),{7:{"processes":[]}}]), \
             patch.object(admission.time,"sleep"), patch.object(admission,"save") as save:
            admission.wait_for_cards(Path("."), {7:{}})
        self.assertEqual(save.call_args.args[1]["gpu_query"],"TimeoutExpired")


if __name__ == "__main__":
    unittest.main()

