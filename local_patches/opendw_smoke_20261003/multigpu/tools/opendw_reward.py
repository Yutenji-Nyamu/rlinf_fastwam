# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Minimal inference-only WorldArena RoboTwin T5 reward model.

Architecture and preprocessing follow WorldArena-2.0 commit
5978ce5c81e55b8c8358f4f5966a13ce385ff155:
RL_env_benchmark/rlinf/models/embodiment/reward/{robotwin_reward_model,
base_image_reward_model}.py. Training/framework wrappers are omitted. ResNet is
created without downloading ImageNet weights because the full task state dict
is required and loaded strictly. T5/tokenizer load from an explicit local path.
"""

from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


class RoboTwinT5Reward(nn.Module):
    def __init__(self, checkpoint_path, t5_path):
        super().__init__()
        from torchvision.models import resnet18
        from transformers import AutoTokenizer, T5EncoderModel

        t5_path = Path(t5_path)
        if not t5_path.is_dir():
            raise FileNotFoundError(f"Local T5 directory does not exist: {t5_path}")
        backbone = resnet18(weights=None)
        self.visual_encoder = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
        )
        self.t5_tokenizer = AutoTokenizer.from_pretrained(str(t5_path), local_files_only=True)
        self.t5_encoder = T5EncoderModel.from_pretrained(str(t5_path), local_files_only=True)
        self.text_proj = nn.Linear(self.t5_encoder.config.d_model, 512)
        self.cross_attn = nn.MultiheadAttention(512, 8, dropout=0.0, batch_first=True)
        self.ln_attn = nn.LayerNorm(512)
        self.reward_head = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 1))
        self.register_buffer("_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
        if not isinstance(checkpoint, dict) or not checkpoint:
            raise ValueError("Reward checkpoint must contain the full state dictionary")
        cleaned = {}
        for original_key, value in checkpoint.items():
            key = original_key
            while key.startswith(("module.", "_orig_mod.", "model.")):
                key = key.split(".", 1)[1]
            if key in cleaned:
                raise ValueError(f"Reward checkpoint duplicate normalized key: {key}")
            cleaned[key] = value
        self.load_state_dict(cleaned, strict=True)
        self.eval().requires_grad_(False)
        self.float()

    def preprocess_images(self, images):
        if images.ndim != 4:
            raise ValueError("Reward images must have rank 4")
        if images.shape[-1] in (1, 3, 4):
            images = images.permute(0, 3, 1, 2)
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        elif images.max() > 1.0:
            images = images / 255.0
        if images.shape[-2:] != (224, 224):
            images = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
        return (images - self._mean) / self._std

    @torch.no_grad()
    def compute_reward(self, images, instructions):
        device = next(self.parameters()).device
        images = self.preprocess_images(images.to(device))
        feat = self.visual_encoder(images)
        batch, channels, height, width = feat.shape
        visual_tokens = feat.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        encoded = self.t5_tokenizer(instructions, padding=True, truncation=True, max_length=64, return_tensors="pt")
        ids = encoded["input_ids"].to(device)
        mask = encoded["attention_mask"].to(device)
        text_tokens = self.text_proj(self.t5_encoder(input_ids=ids, attention_mask=mask).last_hidden_state)
        attention, _ = self.cross_attn(query=visual_tokens, key=text_tokens, value=text_tokens, key_padding_mask=(mask == 0))
        pooled = self.ln_attn(visual_tokens + attention).mean(dim=1)
        return torch.sigmoid(self.reward_head(pooled).squeeze(-1))
