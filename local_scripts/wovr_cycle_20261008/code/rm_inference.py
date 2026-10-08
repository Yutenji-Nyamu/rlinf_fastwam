# Copyright 2026 The RLinf Authors and task-reward contributors.
# Licensed under the Apache License, Version 2.0.
"""Single-task ResNet RM with the deployed WorldArena adapter's score interface.

Architecture: RLinf/WorldArena resnet_reward_model.py (ResNet18 + 256 head).
Pixels: the deployed opendw_reward.py contract: RGB, bilinear 224, ImageNet
normalization. Instructions are accepted and ignored for a task-specific model.
No constructor downloads weights; training requires an explicit local backbone.
"""
import hashlib
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


PREPROCESS = dict(color="RGB", size=[224, 224], interpolation="bilinear",
                  align_corners=False, mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225], input="uint8 [0,255] or float [0,1] / [0,255]",
                  crop=None, instruction="accepted but ignored; one task per checkpoint")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SingleTaskReward(nn.Module):
    def __init__(self, checkpoint_path=None, pretrained_path=None):
        super().__init__()
        from torchvision.models import resnet18
        self.backbone = resnet18(weights=None)
        self.metadata = None
        if checkpoint_path is not None and pretrained_path is not None:
            raise ValueError("Choose a task checkpoint or an ImageNet initializer")
        if pretrained_path is not None:
            self.backbone.load_state_dict(torch.load(pretrained_path, map_location="cpu", weights_only=True), strict=True)
        self.backbone.fc = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 1))
        for layer in self.backbone.fc.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        self.register_buffer("_mean", torch.tensor(PREPROCESS["mean"]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_std", torch.tensor(PREPROCESS["std"]).view(1, 3, 1, 1), persistent=False)
        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if checkpoint.get("schema_version") != 1 or checkpoint.get("architecture") != "resnet18_mlp256":
                raise ValueError("Unsupported task reward checkpoint")
            if checkpoint.get("preprocess") != PREPROCESS or not checkpoint.get("task_name"):
                raise ValueError("Checkpoint preprocessing/task metadata mismatch")
            self.load_state_dict(checkpoint["model_state_dict"], strict=True)
            self.metadata = {key: value for key, value in checkpoint.items() if key != "model_state_dict"}
            self.eval().requires_grad_(False)

    def preprocess_images(self, images):
        if not torch.is_tensor(images):
            images = torch.as_tensor(images)
        if images.ndim != 4:
            raise ValueError("Reward images must be a batch of RGB images")
        if images.shape[-1] == 3:
            images = images.permute(0, 3, 1, 2)
        if images.shape[1] != 3:
            raise ValueError("Reward requires exactly three RGB channels")
        images = images.to(self._mean.device)
        if images.dtype == torch.uint8:
            images = images.float().div(255)
        elif images.is_floating_point():
            images = images.float()
            if not bool(torch.isfinite(images).all()) or float(images.min()) < 0 or float(images.max()) > 255:
                raise ValueError("Nonfinite/out-of-range reward pixels")
            if float(images.max()) > 1:
                images = images.div(255)
        else:
            raise TypeError("Expected uint8 or floating-point RGB pixels")
        if images.shape[-2:] != (224, 224):
            images = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
        return (images - self._mean) / self._std

    def forward(self, images, preprocessed=False):
        if not preprocessed:
            images = self.preprocess_images(images)
        return self.backbone(images).squeeze(-1)

    @torch.no_grad()
    def compute_reward(self, images, instructions=None):
        """Return float scores [B], matching RoboTwinT5Reward.compute_reward.

        Threshold/done/first-success masking remain owned by the environment.
        """
        if self.training:
            raise RuntimeError("Reward inference requires model.eval()")
        if isinstance(images, dict):
            if "main_images" not in images:
                raise ValueError("Observation dictionary lacks main_images")
            images = images["main_images"]
        return torch.sigmoid(self.forward(images))
