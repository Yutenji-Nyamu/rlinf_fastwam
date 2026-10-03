"""CPU-only contracts for Sidney H50 / OpenDW C32 RoboTwin smoke runs.

The policy keeps its H50 denoising chain and exposes only C32 to this adapter.
Raw order is left joints 0:6, left gripper 6, right joints 7:13, right gripper 13.
No action/stat normalization takes place here: each model owns its own stats.
"""

from dataclasses import dataclass

import numpy as np
from PIL import Image


ACTION_CHUNK = 32
ACTION_DIM = 14
FRAME_STRIDE = 4
CANVAS_HW = (384, 320)


def prepare_raw_actions(actions, chunk=ACTION_CHUNK):
    """Copy the executed raw command chunk, matching native gripper clipping."""
    if hasattr(actions, "detach"):
        actions = actions.detach().float().cpu().numpy()
    result = np.asarray(actions, dtype=np.float32).copy()
    if result.ndim != 3 or result.shape[1:] != (chunk, ACTION_DIM):
        raise ValueError(f"Expected [B,{chunk},14] raw actions, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError("Non-finite raw actions")
    result[..., [6, 13]] = np.clip(result[..., [6, 13]], 0.0, 1.0)
    return result


def next_command_state(executed_actions):
    """Our WM convention is the final accepted absolute command, not qpos."""
    return prepare_raw_actions(executed_actions)[:, -1].copy()


def _rgb(image):
    result = np.asarray(image)
    if result.ndim != 3 or result.shape[-1] != 3 or result.dtype != np.uint8:
        raise ValueError(f"Expected uint8 HWC RGB, got {result.shape}/{result.dtype}")
    return result


def _resize(image, width, height):
    return np.asarray(
        Image.fromarray(_rgb(image)).resize((width, height), Image.Resampling.BILINEAR),
        dtype=np.uint8,
    )


def compose_robotwin_views(head, left_wrist, right_wrist):
    """DW05 Robotwin bundle uses robotwin_resize, not letterbox."""
    head = _resize(head, 320, 256)
    wrists = np.concatenate(
        [_resize(left_wrist, 160, 128), _resize(right_wrist, 160, 128)], axis=1
    )
    return np.concatenate([head, wrists], axis=0)


def resize_policy_views(head, left_wrist, right_wrist, image_size=(256, 256)):
    """Resize raw views directly; do not round-trip native resets through WM."""
    height, width = (int(value) for value in image_size)
    return tuple(_resize(image, width, height) for image in (head, left_wrist, right_wrist))


def split_robotwin_views(canvas, image_size=(256, 256)):
    canvas = _rgb(canvas)
    if canvas.shape != (384, 320, 3):
        raise ValueError(f"Unexpected DW05 canvas {canvas.shape}")
    return resize_policy_views(canvas[:256], canvas[256:, :160], canvas[256:, 160:], image_size)


@dataclass
class ChunkFeedback:
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    next_score: np.ndarray
    first_success_action: np.ndarray


def map_frame_scores(
    frame_scores,
    previous_score,
    elapsed_steps,
    *,
    max_episode_steps,
    reward_coef=1.0,
    relative_reward=True,
    success_threshold=0.9,
):
    """Map 8 future frame scores to 32 actions without multiplying rewards by 4.

As in WorldArena, success is detected in any predicted frame but termination
occurs at the chunk boundary. A full 32-command transition is therefore valid.
This smoke adapter rejects partial chunks rather than silently exceeding a
native 400-action budget or training on unexecuted actions.
"""
    scores = np.asarray(frame_scores, dtype=np.float32)
    if scores.ndim != 2 or scores.shape[1] != 8:
        raise ValueError(f"Expected [B,8] future frame scores, got {scores.shape}")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Reward model scores must be finite probabilities in [0,1]")
    batch = scores.shape[0]
    previous = np.asarray(previous_score, dtype=np.float32).reshape(batch)
    elapsed = np.asarray(elapsed_steps, dtype=np.int64).reshape(batch)
    if np.any(elapsed + ACTION_CHUNK > int(max_episode_steps)):
        raise ValueError("Partial C32 chunk requires a separate valid-action mask adapter")
    reward_frames = scores.copy()
    if relative_reward:
        reward_frames = np.diff(np.concatenate([previous[:, None], scores], axis=1), axis=1)
    reward_frames *= float(reward_coef)
    rewards = np.zeros((batch, ACTION_CHUNK), dtype=np.float32)
    rewards[:, FRAME_STRIDE - 1 :: FRAME_STRIDE] = reward_frames
    hits = scores >= float(success_threshold)
    success = hits.any(axis=1)
    first_success = np.where(success, (hits.argmax(axis=1) + 1) * FRAME_STRIDE, -1)
    terminated = np.zeros_like(rewards, dtype=bool)
    terminated[:, -1] = success
    truncated = np.zeros_like(terminated)
    truncated[:, -1] = elapsed + ACTION_CHUNK >= int(max_episode_steps)
    return ChunkFeedback(rewards, terminated, truncated, scores[:, -1].copy(), first_success)
