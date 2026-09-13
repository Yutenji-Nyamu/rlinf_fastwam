"""Raw clean50 LeRobot frames to the existing Fast-WAM observation adapter."""

from __future__ import annotations

import torch


def lerobot_frame_to_env_obs(frame, default_prompt="adjust the bottle"):
    def rgb(key):
        image = torch.as_tensor(frame[key])
        if image.ndim != 3:
            raise ValueError(f"{key}: expected CHW image")
        if image.shape[0] == 3:
            image = image.permute(1, 2, 0)
        if image.dtype != torch.uint8:
            if image.min() < 0 or image.max() > 1:
                raise ValueError(f"{key}: expected raw LeRobot RGB in [0,1]")
            image = (image * 255).round().to(torch.uint8)
        return image.contiguous()

    task = frame.get("task", default_prompt)
    if not isinstance(task, str) or not task.strip():
        raise ValueError("LeRobot task must be nonempty text")
    state = torch.as_tensor(frame["observation.state"]).float()
    if state.shape != (14,):
        raise ValueError(f"Expected physical 14D state, got {tuple(state.shape)}")
    return {
        "main_images": rgb("observation.images.cam_high").unsqueeze(0),
        "wrist_images": torch.stack([
            rgb("observation.images.cam_left_wrist"),
            rgb("observation.images.cam_right_wrist"),
        ]).unsqueeze(0),
        "states": state.unsqueeze(0),
        "task_descriptions": [task],
    }
