"""Frozen lift_pot ResNet reward for the existing OpenDW score interface.

Copy this file beside the exact training ``rm_inference.py`` in the new private
service directory. This model produces probabilities only; the existing RLinf
environment owns the fixed threshold, first-success reward, done, and mask.
"""
from pathlib import Path
import re

from rm_inference import SingleTaskReward, sha256


TASK_NAME = "lift_pot"
TASK_CONFIG = "demo_clean"
CHECKPOINT_SHA256 = "0363474ef67e87db1f6f9456bba9b17aeb8c3bb801a536a037413de0b23f232f"
SUCCESS_THRESHOLD = 0.7082200646400452


class LiftPotReward(SingleTaskReward):
    """RGB tensor or ``main_images`` dict -> float success probabilities [B]."""

    def __init__(self, checkpoint_path, *, expected_sha256=CHECKPOINT_SHA256):
        checkpoint_path = Path(checkpoint_path)
        if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
            raise ValueError("Require the reviewed task checkpoint SHA256")
        if sha256(checkpoint_path) != expected_sha256:
            raise ValueError("Task reward checkpoint differs from the reviewed file")
        super().__init__(checkpoint_path=checkpoint_path)
        if (self.metadata["task_name"], self.metadata.get("task_config")) != (TASK_NAME, TASK_CONFIG):
            raise ValueError("Require the lift_pot/demo_clean reward checkpoint")
        self.float().eval().requires_grad_(False)
        self.checkpoint_sha256 = expected_sha256

    def deployment_metadata(self):
        return dict(reward_architecture="resnet18_mlp256", reward_task=TASK_NAME,
                    reward_task_config=TASK_CONFIG,
                    reward_checkpoint_sha256=self.checkpoint_sha256,
                    reward_threshold_owner="RLinf environment",
                    reward_instructions="accepted; ignored by the single-task model")
