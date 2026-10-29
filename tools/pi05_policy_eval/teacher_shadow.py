"""G-family teacher actions and accepted noise4 signals on the student trajectory."""

from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/pi05_u_video"))
sys.path.insert(0, str(ROOT / "tools/pi05_signals"))
from noise_samples import collect_noise_samples, numpy_exact
from signal_math import noise_disagreement


def compare(noise4, teacher):
    student = noise4[:, 0, :, :14].astype(np.float32)
    teacher = teacher[..., :14].astype(np.float32)
    difference = student - teacher
    dispersion = noise_disagreement(noise4[..., :14])
    independent = noise_disagreement(noise4[:, 1:, :, :14])
    assert dispersion.valid.all() and independent.valid.all()
    return {
        "noise4_u": dispersion.score,
        "noise3_without_main_u": independent.score,
        "teacher_student_difference": difference,
        "teacher_student_rmse": np.sqrt(np.mean(difference**2, axis=-1)),
    }


def observe(weak, strong, obs, trace, actions, seed, max_extra_batch=None):
    data = collect_noise_samples(
        weak, obs, trace, actions, seed, max_extra_batch=max_extra_batch
    )
    initial = trace["initial_noise_full"]
    devices = [initial.device.index] if initial.device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices), torch.inference_mode():
        teacher_actions, result = strong.predict_action_batch(
            obs,
            mode="eval",
            compute_values=False,
            num_steps=10,
            noise=trace["initial_noise_full"].clone(),
            return_dvac_telemetry=True,
        )
    tt = result["dvac_telemetry"]
    assert torch.equal(tt["initial_noise_full"], trace["initial_noise_full"])
    data.update(
        teacher_model_actions=numpy_exact(tt["final_model_action"]),
        teacher_env_actions=numpy_exact(teacher_actions),
        teacher_initial_noise=numpy_exact(tt["initial_noise_full"]),
    )
    data.update(compare(data["noise4_model_actions"], data["teacher_model_actions"]))
    return data


def save_observation(path, obs):
    """The original three cameras and state, once per actual query."""
    arrays = {
        k: numpy_exact(v) if torch.is_tensor(v) else np.asarray(v)
        for k, v in obs.items()
        if v is not None
    }
    np.savez_compressed(path, **arrays)
