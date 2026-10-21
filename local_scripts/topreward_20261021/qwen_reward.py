# Copyright 2026 The RLinf Authors and task-reward contributors.
# Licensed under the Apache License, Version 2.0.
"""Frozen Qwen3-VL score using PR #1554's exact prompt and token definition.

This class is stateless: the caller supplies an episode-local window. It is
used by both offline calibration and the optional OpenDW reward backend.
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

PROMPT = (
    "<|vision_start|><|video_pad|><|vision_end|>"
    "The above video shows a robot manipulation trajectory that completes the "
    "following task: {instruction} Decide whether the above statement is True "
    "or not. The answer is:"
)
SOURCE_REVISION = "cbb4e4fb19aae42693bb130b7ae27102345c7667"


def window_frames(
    history: np.ndarray, future: np.ndarray, limit: int = 16
) -> np.ndarray:
    """Join consecutive valid RGB frames, keeping the latest scoring window."""
    if future.dtype != np.uint8 or future.ndim != 4 or future.shape[-1] != 3:
        raise ValueError("future must be uint8 [T,H,W,3]")
    if not len(future) or limit < 1:
        raise ValueError("A scoring window requires at least one frame")
    if len(history):
        if history.dtype != np.uint8 or history.shape[1:] != future.shape[1:]:
            raise ValueError("History and new frames must use the same RGB layout")
        future = np.concatenate((history, future))
    return future[-limit:]


class QwenReward(nn.Module):
    """Return P(next token = ' True'); thresholds remain outside the VLM."""

    def __init__(self, model_path: str | Path, fps: float = 2.0):
        super().__init__()
        os.environ.setdefault("USE_TF", "0")
        os.environ.setdefault("USE_FLAX", "0")
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.fps = float(fps)
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be positive")
        self.processor = AutoProcessor.from_pretrained(
            str(model_path), local_files_only=True
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            str(model_path),
            dtype=torch.bfloat16,
            attn_implementation="eager",
            local_files_only=True,
        )
        self.model.eval().requires_grad_(False)
        ids = self.processor.tokenizer.encode(" True", add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError("The reference score requires exactly one ' True' token")
        self.true_id = ids[0]
        self.last_details: dict = {}

    @torch.inference_mode()
    def score(self, frames: np.ndarray, instruction: str) -> dict:
        """Score one explicit window, reporting the actual processor frame count."""
        if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError("Expected uint8 [T,H,W,3]")
        if not 1 <= len(frames) <= 16 or not instruction.strip():
            raise ValueError("Expected 1..16 frames and a task instruction")
        started = time.monotonic()
        pil = [Image.fromarray(f).convert("RGB") for f in frames]
        # Qwen's resize rejects T=1 before its normal temporal padding. An
        # immobile/failed native first chunk can contain just one valid frame.
        # Make that required duplicate explicit in the returned evidence.
        if len(pil) == 1:
            pil = pil * 2
        inputs = self.processor(
            text=[PROMPT.format(instruction=instruction)],
            videos=[pil],
            fps=self.fps,
            video_metadata=[
                {
                    "fps": self.fps,
                    "total_num_frames": len(pil),
                    "duration": len(pil) / self.fps,
                }
            ],
            return_tensors="pt",
        ).to(self.model.device)
        grid = inputs["video_grid_thw"].detach().cpu().tolist()
        patch = int(self.model.config.vision_config.temporal_patch_size)
        actual_frames = int(grid[0][0]) * patch
        expected_frames = math.ceil(len(frames) / patch) * patch
        if actual_frames != expected_frames:
            raise RuntimeError(
                f"Processor changed frame count: {len(frames)} -> {actual_frames}"
            )
        # The next-token probability is unchanged; do not materialize logits for
        # every visual token or allocate a KV cache that no generation will use.
        logits = (
            self.model(**inputs, use_cache=False, logits_to_keep=1)
            .logits[0, -1]
            .float()
        )
        logp = float(torch.log_softmax(logits, dim=-1)[self.true_id])
        if not math.isfinite(logp):
            raise RuntimeError("Nonfinite VLM token probability")
        self.last_details = {
            "probability": math.exp(logp),
            "log_probability": logp,
            "input_frames": len(frames),
            "processed_frames": actual_frames,
            "explicit_temporal_padding": max(0, 2 - len(frames)),
            "video_grid_thw": grid,
            "seconds": time.monotonic() - started,
        }
        return dict(self.last_details)
